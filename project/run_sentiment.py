import random
import embeddings
import minitorch
from datasets import load_dataset

# Set up tensor backend for minitorch
BACKEND = minitorch.TensorBackend(minitorch.FastOps)


def RParam(*shape):
    """Generate a random parameter tensor with values between -0.05 and 0.05.

    This function helps initialize weights and biases with small random values.

    Args:
    ----
        shape: The shape of the tensor to be created.

    Returns:
    -------
        A Parameter tensor with random values in the specified shape.
    """
    r = 0.1 * (minitorch.rand(shape, backend=BACKEND) - 0.5)
    return minitorch.Parameter(r)


class Linear(minitorch.Module):
    """A simple linear layer (fully connected layer) that performs a weighted sum.

    This layer computes the output as `output = input @ weights + bias`.

    Args:
    ----
        in_size: The size of the input (number of input features).
        out_size: The size of the output (number of output features).
    """
    def __init__(self, in_size, out_size):
        super().__init__()
        self.weights = RParam(in_size, out_size)  # Initialize weights
        self.bias = RParam(out_size)  # Initialize bias
        self.out_size = out_size

    def forward(self, x):
        """Forward pass: Apply linear transformation to input."""
        batch, in_size = x.shape
        return (
            x.view(batch, in_size) @ self.weights.value.view(in_size, self.out_size)
        ).view(batch, self.out_size) + self.bias.value


class Conv1d(minitorch.Module):
    """1D Convolution layer. This layer slides filters (kernels) over input to extract features.

    Args:
    ----
        in_channels: Number of input channels (e.g., the number of features).
        out_channels: Number of output channels (number of filters).
        kernel_width: Width of each filter.
    """
    def __init__(self, in_channels, out_channels, kernel_width):
        super().__init__()
        self.weights = RParam(out_channels, in_channels, kernel_width)  # Filter weights
        self.bias = RParam(1, out_channels, 1)  # Bias for each filter

    def forward(self, input):
        """Forward pass: Apply 1D convolution on input."""
        return minitorch.conv1d(input, self.weights.value) + self.bias.value


class CNNSentimentKim(minitorch.Module):
    """CNN model for sentiment classification based on Y. Kim (2014).

    This model applies convolution, max-pooling, and fully connected layers to classify sentiment.

    Args:
    ----
        feature_map_size: Number of filters in each convolution layer.
        embedding_size: The size of each word embedding.
        filter_sizes: Sizes of the filters to be used for convolution.
        dropout: Dropout rate for regularization to prevent overfitting.
    """
    def __init__(
        self,
        feature_map_size=100,
        embedding_size=50,
        filter_sizes=[3, 4, 5],
        dropout=0.25,
    ):
        super().__init__()
        self.feature_map_size = feature_map_size
        self.conv1 = Conv1d(embedding_size, feature_map_size, filter_sizes[0])
        self.conv2 = Conv1d(embedding_size, feature_map_size, filter_sizes[1])
        self.conv3 = Conv1d(embedding_size, feature_map_size, filter_sizes[2])
        self.final = Linear(feature_map_size, 1)  # Final fully connected layer
        self.dropout = dropout

    def forward(self, embeddings):
        """Forward pass: Apply convolution, max-pooling, and fully connected layers."""
        embeddings = embeddings.permute(0, 2, 1)  # Reorder dimensions for convolution

        # Apply convolutions with ReLU activation
        x1 = self.conv1.forward(embeddings).relu()
        x2 = self.conv2.forward(embeddings).relu()
        x3 = self.conv3.forward(embeddings).relu()

        # Max-pooling across each feature map (filter)
        x_mid = (
            minitorch.nn.max(x1, 2) + minitorch.nn.max(x2, 2) + minitorch.nn.max(x3, 2)
        )

        # Fully connected layer (Linear)
        x = self.final(x_mid.view(x_mid.shape[0], x_mid.shape[1]))

        # Apply dropout and sigmoid for final prediction
        x = minitorch.nn.dropout(x, self.dropout)
        return x.sigmoid().view(embeddings.shape[0])


# Helper functions for evaluation
def get_predictions_array(y_true, model_output):
    """Generate predictions from model output and compare to true labels.

    Args:
    ----
        y_true: Actual labels.
        model_output: Model predictions.

    Returns:
    -------
        List of tuples with true labels, predicted labels, and logits.
    """
    predictions_array = []
    for j, logit in enumerate(model_output.to_numpy()):
        true_label = y_true[j]
        predicted_label = 1.0 if logit > 0.5 else 0  # Convert logits to binary predictions
        predictions_array.append((true_label, predicted_label, logit))
    return predictions_array


def get_accuracy(predictions_array):
    """Compute accuracy from predictions.

    Args:
    ----
        predictions_array: List of true labels, predicted labels, and logits.

    Returns:
    -------
        Accuracy as a percentage.
    """
    correct = sum(1 for y_true, y_pred, _ in predictions_array if y_true == y_pred)
    return correct / len(predictions_array)


# Tracking the best validation accuracy
best_val = 0.0


def default_log_fn(
    epoch,
    train_loss,
    losses,
    train_predictions,
    train_accuracy,
    validation_predictions,
    validation_accuracy,
):
    """Log function for each epoch during training."""
    global best_val
    best_val = max(best_val, validation_accuracy[-1])
    log_line = f"Epoch {epoch}, loss {train_loss}, train accuracy: {train_accuracy[-1]:.2%}"
    if validation_predictions:
        log_line += f", validation accuracy: {validation_accuracy[-1]:.2%}"
    print(log_line)

    # Save to sentiment.txt
    with open('sentiment.txt', 'a') as f:
        f.write(log_line + '\n')


class SentenceSentimentTrain:
    """Train a sentiment analysis model."""

    def __init__(self, model):
        self.model = model

    def train(
        self,
        data_train,
        learning_rate,
        batch_size=10,
        max_epochs=500,
        data_val=None,
        log_fn=default_log_fn,
    ):
        """Train the model with training data and evaluate on validation data."""
        model = self.model
        (X_train, y_train) = data_train
        n_training_samples = len(X_train)
        optim = minitorch.SGD(self.model.parameters(), learning_rate)
        losses = []
        train_accuracy = []
        validation_accuracy = []
        for epoch in range(1, max_epochs + 1):
            total_loss = 0.0

            model.train()  # Set model to training mode
            train_predictions = []
            batch_size = min(batch_size, n_training_samples)
            for batch_num, example_num in enumerate(
                range(0, n_training_samples, batch_size)
            ):
                # Prepare batch
                y = minitorch.tensor(y_train[example_num: example_num + batch_size], backend=BACKEND)
                x = minitorch.tensor(X_train[example_num: example_num + batch_size], backend=BACKEND)
                x.requires_grad_(True)
                y.requires_grad_(True)

                # Forward pass
                out = model.forward(x)
                prob = (out * y) + (out - 1.0) * (y - 1.0)
                loss = -(prob.log() / y.shape[0]).sum()
                loss.view(1).backward()

                # Save predictions
                train_predictions += get_predictions_array(y, out)
                total_loss += loss[0]

                # Update model parameters
                optim.step()

            # Evaluate on validation set
            validation_predictions = []
            if data_val:
                (X_val, y_val) = data_val
                model.eval()  # Set model to evaluation mode
                y = minitorch.tensor(y_val, backend=BACKEND)
                x = minitorch.tensor(X_val, backend=BACKEND)
                out = model.forward(x)
                validation_predictions += get_predictions_array(y, out)
                validation_accuracy.append(get_accuracy(validation_predictions))
                model.train()  # Switch back to training mode

            train_accuracy.append(get_accuracy(train_predictions))
            losses.append(total_loss)
            log_fn(
                epoch,
                total_loss,
                losses,
                train_predictions,
                train_accuracy,
                validation_predictions,
                validation_accuracy,
            )


def encode_sentences(
    dataset, N, max_sentence_len, embeddings_lookup, unk_embedding, unks
):
    """Encode sentences into word embeddings, padding short sentences."""
    Xs = []
    ys = []
    for sentence in dataset["sentence"][:N]:
        # Pad sentences to the same length for batch processing
        sentence_embedding = [[0] * embeddings_lookup.d_emb] * max_sentence_len
        for i, w in enumerate(sentence.split()):
            if w in embeddings_lookup:
                sentence_embedding[i] = embeddings_lookup.emb(w)
            else:
                unks.add(w)  # Add unknown words to the set
                sentence_embedding[i] = unk_embedding
        Xs.append(sentence_embedding)

    ys = dataset["label"][:N]  # Load labels
    return Xs, ys


def encode_sentiment_data(dataset, pretrained_embeddings, N_train, N_val=0):
    """Encode sentiment data using pre-trained embeddings."""
    max_sentence_len = max(len(sentence.split()) for sentence in dataset["train"]["sentence"] + dataset["validation"]["sentence"])

    unks = set()
    unk_embedding = [0.1 * (random.random() - 0.5) for _ in range(pretrained_embeddings.d_emb)]
    X_train, y_train = encode_sentences(
        dataset["train"],
        N_train,
        max_sentence_len,
        pretrained_embeddings,
        unk_embedding,
        unks,
    )
    X_val, y_val = encode_sentences(
        dataset["validation"],
        N_val,
        max_sentence_len,
        pretrained_embeddings,
        unk_embedding,
        unks,
    )
    print(f"Missing pre-trained embedding for {len(unks)} unknown words")

    return (X_train, y_train), (X_val, y_val)


if __name__ == "__main__":
    open('sentiment.txt', 'w').close()
    train_size = 450
    validation_size = 100
    learning_rate = 0.01
    max_epochs = 250

    # Load dataset and encode it
    (X_train, y_train), (X_val, y_val) = encode_sentiment_data(
        load_dataset("glue", "sst2"),
        embeddings.GloveEmbedding("wikipedia_gigaword", d_emb=50, show_progress=True),
        train_size,
        validation_size,
    )

    # Train the model
    model_trainer = SentenceSentimentTrain(
        CNNSentimentKim(feature_map_size=100, filter_sizes=[3, 4, 5], dropout=0.25)
    )
    model_trainer.train(
        (X_train, y_train),
        learning_rate,
        max_epochs=max_epochs,
        data_val=(X_val, y_val),
    )

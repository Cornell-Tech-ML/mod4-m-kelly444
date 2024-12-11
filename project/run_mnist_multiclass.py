from mnist import MNIST
import minitorch

# Load the MNIST dataset from the specified path
mndata = MNIST("project/data/")
images, labels = mndata.load_training()

# Set up the backend for computations
BACKEND = minitorch.TensorBackend(minitorch.FastOps)

# Batch size for training
BATCH = 16

# Number of possible classes (digits 0-9)
C = 10

# Size of each image (28x28 pixels)
H, W = 28, 28


def RParam(*shape):
    """Generate a tensor with random values for parameters (weights and biases).

    The tensor values are between -0.05 and 0.05 to start with, which helps with learning.

    Args:
        shape: The shape of the tensor (e.g., the number of rows and columns).

    Returns:
        A tensor filled with small random values.
    """
    r = 0.1 * (minitorch.rand(shape, backend=BACKEND) - 0.5)
    return minitorch.Parameter(r)


class Linear(minitorch.Module):
    """A simple linear layer that connects inputs to outputs using weights and biases.

    Args:
        in_size: Number of input features.
        out_size: Number of output features (or neurons).
    """
    def __init__(self, in_size, out_size):
        super().__init__()
        self.weights = RParam(in_size, out_size)  # Initialize weights
        self.bias = RParam(out_size)  # Initialize bias
        self.out_size = out_size

    def forward(self, x):
        """Perform the linear transformation on the input."""
        batch, in_size = x.shape
        return (
            x.view(batch, in_size) @ self.weights.value.view(in_size, self.out_size)
        ).view(batch, self.out_size) + self.bias.value


class Conv2d(minitorch.Module):
    """A 2D convolution layer that detects patterns in images using filters.

    Args:
        in_channels: Number of input channels (1 for grayscale images).
        out_channels: Number of output channels (number of filters).
        kh: Height of the filter (kernel).
        kw: Width of the filter (kernel).
    """
    def __init__(self, in_channels, out_channels, kh, kw):
        super().__init__()
        self.weights = RParam(out_channels, in_channels, kh, kw)  # Filters
        self.bias = RParam(out_channels, 1, 1)  # Bias for each filter

    def forward(self, input):
        """Apply the 2D convolution on the input image."""
        return minitorch.Conv2dFun.apply(input, self.weights.value) + self.bias.value


class Network(minitorch.Module):
    """A Convolutional Neural Network (CNN) for classifying MNIST digits.

    This network uses the LeNet architecture with convolutional layers followed by fully connected layers.

    1. Convolve with a 3x3 filter, 4 output channels.
    2. Convolve again with a 3x3 filter, 8 output channels.
    3. Pool the result with a 4x4 max-pool.
    4. Flatten the data.
    5. Apply a fully connected layer to reduce the size to 64 neurons.
    6. Apply another fully connected layer to output predictions for 10 classes (digits 0-9).
    7. Apply log-softmax to get probabilities for each class.
    """
    def __init__(self):
        super().__init__()

        # Layers of the network
        self.conv1 = Conv2d(1, 4, 3, 3)  # First convolution layer
        self.conv2 = Conv2d(4, 8, 3, 3)  # Second convolution layer
        self.linear1 = Linear(392, 64)  # First fully connected layer
        self.linear2 = Linear(64, C)  # Second fully connected layer (output layer)

    def forward(self, x):
        """Perform a forward pass through the network: convolution, pooling, and classification."""
        # Apply the first convolution and ReLU activation
        self.mid = self.conv1(x).relu()
        # Apply the second convolution and ReLU activation
        self.out = self.conv2(self.mid).relu()
        # Apply max-pooling to reduce the spatial size
        po = minitorch.nn.maxpool2d(self.out, (4, 4))

        # Flatten the pooled output for the fully connected layers
        bs = po.shape[0]  # Batch size
        fo = po.view(bs, 392)

        # Pass through the first fully connected layer and apply ReLU
        ho = self.linear1(fo).relu()
        # Apply dropout to prevent overfitting
        do = minitorch.dropout(ho, 0.25)

        # Pass through the second fully connected layer and apply log-softmax
        return minitorch.logsoftmax(self.linear2(do), dim=1)


def make_mnist(start, stop):
    """Prepare the MNIST data for training by converting labels to one-hot encoding
    and reshaping the images to the correct format.

    Args:
        start: Starting index of the training data.
        stop: Ending index of the training data.

    Returns:
        X: List of image data.
        ys: List of one-hot encoded labels.
    """
    ys = []
    X = []
    for i in range(start, stop):
        y = labels[i]
        # One-hot encoding of the label (e.g., label 3 -> [0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
        vals = [0.0] * 10
        vals[y] = 1.0
        ys.append(vals)
        # Reshape the image data into a 28x28 matrix
        X.append([[images[i][h * W + w] for w in range(W)] for h in range(H)])
    return X, ys


def default_log_fn(epoch, total_loss, correct, total, losses, model):
    """Log the training progress, including loss and accuracy."""
    log_line = f"Epoch {epoch} loss {total_loss} valid acc {correct}/{total}"
    print(log_line)

    # Save to mnist.txt
    with open('mnist.txt', 'a') as f:
        f.write(log_line + '\n')

class ImageTrain:
    """Handles the training process for the image classification model."""

    def __init__(self):
        self.model = Network()

    def run_one(self, x):
        """Run a single forward pass through the model."""
        return self.model.forward(minitorch.tensor([x], backend=BACKEND))

    def train(self, data_train, data_val, learning_rate, max_epochs=500, log_fn=default_log_fn):
        """Train the model on the MNIST dataset."""
        (X_train, y_train) = data_train
        (X_val, y_val) = data_val
        self.model = Network()  # Reset the model for each training session
        model = self.model
        n_training_samples = len(X_train)
        optim = minitorch.SGD(self.model.parameters(), learning_rate)  # Stochastic Gradient Descent optimizer

        losses = []
        for epoch in range(1, max_epochs + 1):
            total_loss = 0.0
            model.train()  # Set the model to training mode
            for batch_num, example_num in enumerate(range(0, n_training_samples, BATCH)):
                if n_training_samples - example_num <= BATCH:
                    continue
                # Get the next batch of training data
                y = minitorch.tensor(y_train[example_num : example_num + BATCH], backend=BACKEND)
                x = minitorch.tensor(X_train[example_num : example_num + BATCH], backend=BACKEND)
                x.requires_grad_(True)
                y.requires_grad_(True)

                # Perform a forward pass through the model
                out = model.forward(x.view(BATCH, 1, H, W)).view(BATCH, C)
                # Compute the loss (cross-entropy between output and true labels)
                prob = (out * y).sum(1)
                loss = -(prob / y.shape[0]).sum()

                loss.view(1).backward()  # Backpropagation
                total_loss += loss[0]
                losses.append(total_loss)

                # Update model parameters using the optimizer
                optim.step()

                # Every 5 batches, evaluate on the validation set
                if batch_num % 5 == 0:
                    model.eval()  # Set the model to evaluation mode
                    correct = 0
                    for val_example_num in range(0, 1 * BATCH, BATCH):
                        y = minitorch.tensor(y_val[val_example_num : val_example_num + BATCH], backend=BACKEND)
                        x = minitorch.tensor(X_val[val_example_num : val_example_num + BATCH], backend=BACKEND)
                        out = model.forward(x.view(BATCH, 1, H, W)).view(BATCH, C)

                        # Check predictions
                        for i in range(BATCH):
                            m = -1000
                            ind = -1
                            for j in range(C):
                                if out[i, j] > m:
                                    ind = j
                                    m = out[i, j]
                            if y[i, ind] == 1.0:
                                correct += 1
                    log_fn(epoch, total_loss, correct, BATCH, losses, model)

                    total_loss = 0.0
                    model.train()  # Set the model back to training mode


if __name__ == "__main__":
    open('mnist.txt', 'w').close()
    # Prepare training and validation data
    data_train, data_val = (make_mnist(0, 5000), make_mnist(10000, 10500))

    # Start training
    ImageTrain().train(data_train, data_val, learning_rate=0.01)

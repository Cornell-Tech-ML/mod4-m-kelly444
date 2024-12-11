from typing import Tuple

from . import operators
from .autodiff import Context
from .fast_ops import FastOps
from .tensor import Tensor
from .tensor_functions import Function, rand

# Functions in this file:
# - avgpool2d: Apply average pooling on an image to reduce size by averaging nearby pixels
# - argmax: Find the position of the highest value and mark it as 1, others as 0
# - Max: Compute the maximum value along a specified dimension
# - max: Apply max reduction (find max value) along a specific axis
# - softmax: Convert values to probabilities (sum equals 1)
# - logsoftmax: Compute the log of softmax values for numerical stability
# - maxpool2d: Apply max pooling to an image, reducing size by selecting the maximum value
# - dropout: Randomly "drop" some values from the tensor to prevent overfitting during training

max_reduce = FastOps.reduce(
    operators.max, -1e9
)  # Efficient max reduction with a large negative number for the initial comparison


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """Split an image into small tiles (chunks) for 2D pooling operations.

    This function divides the input tensor into smaller tiles, making it easier to apply pooling operations on each tile.

    Args:
    ----
        input: The image tensor with shape (batch, channel, height, width)
        kernel: Size of the pooling filter (height and width), e.g., (2, 2)

    Returns:
    -------
        A reshaped tensor ready for pooling, plus the new height and width after pooling.

    """
    batch, channel, height, width = input.shape  # Get the input tensor dimensions
    kh, kw = kernel  # Extract the kernel size (height, width)

    # Ensure that the height and width are divisible by the kernel size
    assert height % kh == 0
    assert width % kw == 0

    new_height = height // kh  # New height after pooling
    new_width = width // kw  # New width after pooling

    # Reshape input into tiles by splitting height and width
    output = (
        input.contiguous()
        .view(batch, channel, new_height, kh, new_width, kw)
        .permute(0, 1, 2, 4, 3, 5)  # Reorder the dimensions for pooling
    )

    # Combine the kernel dimensions into a single dimension
    output = output.contiguous().view(batch, channel, new_height, new_width, kh * kw)

    return output, new_height, new_width  # Return reshaped tensor and new dimensions


def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Apply average pooling on an image tensor. This reduces the size by averaging pixels in each region.

    Args:
    ----
        input: The input tensor with shape (batch, channel, height, width)
        kernel: Size of the pooling filter (height, width)

    Returns:
    -------
        A tensor with reduced size after applying average pooling.

    """
    batch, channel, _, _ = input.shape  # Get the dimensions of the input tensor
    tiled_input, _, _ = tile(input, kernel)  # Split the image into tiles
    pooled_tensor = tiled_input.mean(dim=4)  # Compute the mean (average) of each tile
    pooled_tensor = pooled_tensor.view(
        batch,
        channel,
        pooled_tensor.shape[2],
        pooled_tensor.shape[3],  # Reshape to remove extra dimensions
    )
    return pooled_tensor  # Return the pooled tensor


def argmax(input: Tensor, dim: int) -> Tensor:
    """Find the position of the highest value and mark it as 1, others as 0.

    This creates a tensor where the maximum value's position is marked as 1, and others are 0 (one-hot encoding).

    Args:
    ----
        input: The tensor to find the max value in
        dim: The dimension (axis) along which to find the argmax

    Returns:
    -------
        A tensor where the max value is 1 and others are 0.

    """
    out = max_reduce(input, dim)  # Find the max value along the specified dimension
    return out == input  # Create a tensor with 1 at the max positions and 0 elsewhere


class Max(Function):
    """Function to compute the maximum value along a dimension of a tensor."""

    @staticmethod
    def forward(ctx: Context, input_tensor: Tensor, dimension: Tensor) -> Tensor:
        """Compute the maximum value along a given dimension of the input tensor.

        Args:
        ----
            ctx: The context object to save intermediate values for later use
            input_tensor: The input tensor to find the maximum value from
            dimension: The dimension along which to apply the max operation

        Returns:
        -------
            The tensor with maximum values along the specified dimension.

        """
        max_reduction = max_reduce(
            input_tensor, int(dimension.item())
        )  # Apply max reduction
        ctx.save_for_backward(input_tensor, max_reduction)  # Save for backward pass
        return max_reduction  # Return the reduced tensor (max values)

    @staticmethod
    def backward(ctx: Context, gradient_output: Tensor) -> Tuple[Tensor, float]:
        """Compute the gradient during the backward pass for max operation.

        Args:
        ----
            ctx: The context object containing saved values
            gradient_output: The gradient of the output tensor

        Returns:
        -------
            A tuple with the gradient w.r.t the input tensor and 0 (no gradient for dimension)

        """
        input_tensor, max_reduction = ctx.saved_values  # Retrieve saved values
        return gradient_output * (
            max_reduction == input_tensor
        ), 0.0  # Only propagate gradient for max values


def max(input: Tensor, dim: int) -> Tensor:
    """Find the maximum value along a specific dimension of a tensor.

    Args:
    ----
        input: The input tensor
        dim: The dimension (axis) to apply max operation on

    Returns:
    -------
        A tensor containing the maximum values along the specified dimension.

    """
    return Max.apply(input, input._ensure_tensor(dim))  # Apply the Max function


def softmax(input: Tensor, dim: int) -> Tensor:
    """Convert the input tensor into probabilities using the softmax function.

    Softmax transforms each value into a probability between 0 and 1, where all probabilities sum to 1.

    Args:
    ----
        input: The tensor to apply softmax on
        dim: The dimension along which to apply softmax

    Returns:
    -------
        A tensor where each value is transformed into a probability.

    """
    exp_input = input.exp()  # Apply exponential to each element in the tensor
    sum_exp = exp_input.sum(dim)  # Sum the exponentials along the specified dimension
    return exp_input / sum_exp  # Normalize to get probabilities


def logsoftmax(input: Tensor, dim: int) -> Tensor:
    """Compute the log of the softmax values for numerical stability.

    This is useful for computing log-probabilities in tasks like classification.

    Args:
    ----
        input: The tensor to apply log-softmax on
        dim: The dimension along which to apply log-softmax

    Returns:
    -------
        A tensor with log-softmax values for better numerical stability.

    """
    exp_input = input.exp()  # Apply exponential to each element in the tensor
    sum_exp_input = exp_input.sum(dim)  # Sum the exponentials along the dimension
    log_sum_exp_input = sum_exp_input.log()  # Take the log of the sum
    return input - log_sum_exp_input  # Subtract to get log-softmax


def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Apply max pooling on a 2D tensor (image). This reduces the size by selecting the maximum value in each region.

    Args:
    ----
        input: The image tensor with shape (batch, channel, height, width)
        kernel: Size of the pooling filter (height x width)

    Returns:
    -------
        A tensor with reduced size after applying max pooling.

    """
    batch_size, num_channels, _, _ = input.shape  # Get input tensor dimensions
    tiled_input, pooled_height, pooled_width = tile(
        input, kernel
    )  # Reshape the input tensor
    pooled_input = max_reduce(tiled_input, 4)  # Apply max reduction on each tile
    return pooled_input.contiguous().view(
        batch_size,
        num_channels,
        pooled_height,
        pooled_width,  # Reshape to remove extra dimensions
    )


def dropout(input: Tensor, rate: float, ignore: bool = False) -> Tensor:
    """Apply dropout to a tensor. This randomly sets some values to zero to prevent overfitting during training.

    Args:
    ----
        input: The input tensor
        rate: The probability of dropping each element (e.g., 0.5 means 50% chance)
        ignore: If True, no dropout will be applied (used for inference)

    Returns:
    -------
        A tensor with dropout applied (some values set to zero)

    """
    if not ignore:
        rand_tensor = rand(input.shape)  # Generate random values
        random_drop = rand_tensor > rate  # Create a mask to randomly drop values
        return input * random_drop  # Apply the dropout mask to the tensor
    else:
        return input  # Return the input unchanged if ignore is True

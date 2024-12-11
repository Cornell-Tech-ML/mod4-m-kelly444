from typing import Tuple, TypeVar, Any
import numpy as np
from numba import prange
from numba import njit as _njit

from .autodiff import Context
from .tensor import Tensor
from .tensor_data import (
    MAX_DIMS,
    Shape,
    Strides,
    Storage,
    broadcast_index,
    index_to_position,
    to_index,
)
from .tensor_functions import Function

Fn = TypeVar("Fn")


def njit(fn: Fn, **kwargs: Any) -> Fn:
    """Compiles a function using Numba's JIT (Just-In-Time) compiler for faster execution.

    This decorator optimizes the input function by compiling it into machine code
    for faster execution at runtime. The `**kwargs` allows passing additional
    parameters to Numba's JIT compiler.

    Args:
    ----
        fn: The function to be compiled.
        **kwargs: Additional keyword arguments passed to Numba's JIT compiler (e.g., `parallel`, `fastmath`).

    Returns:
    -------
        A faster, compiled version of the input function.

    """
    return _njit(inline="always", **kwargs)(fn)  # type: ignore


# Apply JIT compilation for specific tensor functions to speed up execution.
to_index = njit(to_index)
index_to_position = njit(index_to_position)
broadcast_index = njit(broadcast_index)


def _tensor_conv1d(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    input: Storage,
    input_shape: Shape,
    input_strides: Strides,
    weight: Storage,
    weight_shape: Shape,
    weight_strides: Strides,
    reverse: bool,
) -> None:
    """Performs a 1D convolution on input data using a filter (kernel).

    This applies the filter to the input data (tensor) to produce an output tensor.
    The `reverse` flag determines if the filter should be applied in reverse (for backward convolution).

    Args:
    ----
        out: The output tensor where the result will be stored.
        out_shape: The shape (dimensions) of the output tensor.
        out_strides: The strides (step sizes) for accessing elements in the output tensor.
        out_size: Total number of elements in the output tensor.
        input: The input tensor (data to be convolved).
        input_shape: The shape of the input tensor.
        input_strides: The strides for the input tensor.
        weight: The filter (kernel) tensor used for the convolution.
        weight_shape: The shape of the weight tensor.
        weight_strides: The strides for the weight tensor.
        reverse: If True, applies backward convolution (filter in reverse).

    """
    batch_, out_channels, out_width = out_shape
    batch, in_channels, width = input_shape
    out_channels_, in_channels_, kw = weight_shape

    # Ensure the dimensions match across the tensors
    assert (
        batch == batch_
        and in_channels == in_channels_
        and out_channels == out_channels_
    )

    # Get the strides for each tensor
    s1 = input_strides
    s2 = weight_strides

    # Loop through all the positions in the output tensor
    for i in prange(out_size):
        out_index = np.empty(MAX_DIMS, np.int32)
        to_index(i, out_shape, out_index)
        out_batch = out_index[0]
        out_channel = out_index[1]
        out_width = out_index[2]
        val = 0.0

        # Loop through the input channels and kernel width
        for j in prange(in_channels):
            for k in range(kw):
                weight_index = np.array([out_channel, j, k])
                w_pos = index_to_position(weight_index, s2)

                if reverse:  # Backward convolution
                    if out_width - k >= 0:
                        in_index = np.array([out_batch, j, out_width - k])
                        in_pos = index_to_position(in_index, s1)
                        val += input[in_pos] * weight[w_pos]
                else:  # Forward convolution
                    if width > out_width + k:
                        in_index = np.array([out_batch, j, out_width + k])
                        in_pos = index_to_position(in_index, s1)
                        val += input[in_pos] * weight[w_pos]

        # Store the result in the output tensor
        out[i] = val


# Apply JIT compilation for the 1D convolution function
tensor_conv1d = njit(_tensor_conv1d, parallel=True)


class Conv1dFun(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, weight: Tensor) -> Tensor:
        """Computes the result of a 1D convolution.

        This function performs a forward pass of a 1D convolution on the input tensor
        using the given filter tensor (weights) and stores the result.

        Args:
        ----
            ctx: The context object that stores information for the backward pass.
            input: The input tensor with shape (batch, in_channels, width).
            weight: The weight tensor (filter) with shape (out_channels, in_channels, kernel_width).

        Returns:
        -------
            The result of the convolution (output tensor) with shape (batch, out_channels, width).

        """
        ctx.save_for_backward(input, weight)
        batch, in_channels, w = input.shape
        out_channels, in_channels2, kw = weight.shape
        assert in_channels == in_channels2

        # Initialize an empty output tensor with the same width as the input
        output = input.zeros((batch, out_channels, w))

        # Perform the 1D convolution
        tensor_conv1d(
            *output.tuple(), output.size, *input.tuple(), *weight.tuple(), False
        )
        return output

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Computes the gradients for the input and weight tensors during the backward pass.

        This function computes the gradient of the loss with respect to both the input and the filter (weights)
        using the chain rule.

        Args:
        ----
            ctx: The context that stores saved tensors for backpropagation.
            grad_output: The gradient of the output tensor (from the loss function).

        Returns:
        -------
            grad_input: The gradient with respect to the input tensor.
            grad_weight: The gradient with respect to the weight tensor.

        """
        input, weight = ctx.saved_values
        batch, in_channels, w = input.shape
        out_channels, in_channels, kw = weight.shape

        # Compute gradient with respect to the filter (weight)
        grad_weight = grad_output.zeros((in_channels, out_channels, kw))
        new_input = input.permute(1, 0, 2)
        new_grad_output = grad_output.permute(1, 0, 2)
        tensor_conv1d(  # type: ignore
            *grad_weight.tuple(),
            grad_weight.size,
            *new_input.tuple(),
            *new_grad_output.tuple(),
            False,  # type: ignore
        )
        grad_weight = grad_weight.permute(1, 0, 2)

        # Compute gradient with respect to the input tensor
        grad_input = input.zeros((batch, in_channels, w))
        new_weight = weight.permute(1, 0, 2)
        tensor_conv1d(  # type: ignore
            *grad_input.tuple(),
            grad_input.size,  # type: ignore
            *grad_output.tuple(),
            *new_weight.tuple(),
            True,  # type: ignore
        )
        return grad_input, grad_weight


# Convenience function to call the forward and backward pass for 1D convolution
conv1d = Conv1dFun.apply


def _tensor_conv2d(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    input: Storage,
    input_shape: Shape,
    input_strides: Strides,
    weight: Storage,
    weight_shape: Shape,
    weight_strides: Strides,
    reverse: bool,
) -> None:
    """Performs a 2D convolution on input data using a 2D filter.

    Similar to 1D convolution, but this applies a 2D filter on input data that has 2D spatial dimensions
    (height and width).

    Args:
    ----
        out: The output tensor to store the result of the convolution.
        out_shape: The shape of the output tensor.
        out_strides: The strides for the output tensor.
        out_size: The number of elements in the output tensor.
        input: The input tensor (data to be convolved).
        input_shape: The shape of the input tensor.
        input_strides: The strides for the input tensor.
        weight: The filter tensor (kernel) to convolve with the input.
        weight_shape: The shape of the filter tensor.
        weight_strides: The strides for the filter tensor.
        reverse: If True, performs bottom-right to top-left convolution; otherwise, top-left to bottom-right.

    """
    batch_, out_channels, out_height, out_width = out_shape
    batch, in_channels, height, width = input_shape
    out_channels_, in_channels_, kh, kw = weight_shape

    # Ensure dimensions match across tensors
    assert (
        batch == batch_
        and in_channels == in_channels_
        and out_channels == out_channels_
    )

    # Get the strides for the tensors
    s1 = input_strides
    s2 = weight_strides

    # Iterate over each element in the output tensor
    for b_idx in prange(batch_):
        for oc_idx in prange(out_channels):
            for oh_idx in prange(out_height):
                for ow_idx in prange(out_width):
                    o = (
                        b_idx * out_strides[0]
                        + oc_idx * out_strides[1]
                        + oh_idx * out_strides[2]
                        + ow_idx * out_strides[3]
                    )

                    # Apply filter (weight) to the input tensor
                    for ic_idx in prange(in_channels):
                        hw, ww = 0, 0
                        if reverse:  # Bottom-right convolution
                            for hi in prange(
                                max(oh_idx - kh + 1, 0), min(oh_idx + 1, height)
                            ):
                                for wi in prange(
                                    max(ow_idx - kw + 1, 0), min(ow_idx + 1, width)
                                ):
                                    out[o] += (
                                        input[
                                            b_idx * s1[0]
                                            + ic_idx * s1[1]
                                            + hi * s1[2]
                                            + wi * s1[3]
                                        ]
                                        * weight[
                                            oc_idx * s2[0]
                                            + ic_idx * s2[1]
                                            + hw * s2[2]
                                            + ww * s2[3]
                                        ]
                                    )
                                    ww += 1
                                ww = 0
                                hw += 1
                        else:  # Top-left convolution
                            for hi in prange(
                                min(oh_idx, height - 1), min(oh_idx + kh, height)
                            ):
                                for wi in prange(
                                    min(ow_idx, width - 1), min(ow_idx + kw, width)
                                ):
                                    out[o] += (
                                        input[
                                            b_idx * s1[0]
                                            + ic_idx * s1[1]
                                            + hi * s1[2]
                                            + wi * s1[3]
                                        ]
                                        * weight[
                                            oc_idx * s2[0]
                                            + ic_idx * s2[1]
                                            + hw * s2[2]
                                            + ww * s2[3]
                                        ]
                                    )
                                    ww += 1
                                ww = 0
                                hw += 1


# Apply JIT compilation for the 2D convolution function
tensor_conv2d = njit(_tensor_conv2d, parallel=True, fastmath=True)


class Conv2dFun(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, weight: Tensor) -> Tensor:
        """Computes the result of a 2D convolution.

        Args:
        ----
            ctx: Context object to store intermediate results for the backward pass.
            input: The input tensor with shape (batch, in_channels, height, width).
            weight: The filter tensor with shape (out_channels, in_channels, kernel_height, kernel_width).

        Returns:
        -------
            The output tensor after the 2D convolution with shape (batch, out_channels, height, width).

        """
        ctx.save_for_backward(input, weight)
        batch, in_channels, h, w = input.shape
        out_channels, in_channels2, kh, kw = weight.shape
        assert in_channels == in_channels2

        # Initialize an empty output tensor
        output = input.zeros((batch, out_channels, h, w))

        # Perform the 2D convolution
        tensor_conv2d(
            *output.tuple(), output.size, *input.tuple(), *weight.tuple(), False
        )
        return output

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Computes gradients for input and weight tensors during the backward pass.

        Args:
        ----
            ctx: Context that holds saved tensors for backpropagation.
            grad_output: Gradient of the output tensor.

        Returns:
        -------
            grad_input: Gradient of the input tensor.
            grad_weight: Gradient of the weight tensor.

        """
        input, weight = ctx.saved_values
        batch, in_channels, h, w = input.shape
        out_channels, in_channels, kh, kw = weight.shape

        # Compute gradient of the weight (filter)
        grad_weight = grad_output.zeros((in_channels, out_channels, kh, kw))
        new_input = input.permute(1, 0, 2, 3)
        new_grad_output = grad_output.permute(1, 0, 2, 3)
        tensor_conv2d(  # type: ignore
            *grad_weight.tuple(),
            grad_weight.size,
            *new_input.tuple(),
            *new_grad_output.tuple(),
            False,  # type: ignore
        )
        grad_weight = grad_weight.permute(1, 0, 2, 3)

        # Compute gradient of the input
        grad_input = input.zeros((batch, in_channels, h, w))
        new_weight = weight.permute(1, 0, 2, 3)
        tensor_conv2d(  # type: ignore
            *grad_input.tuple(),
            grad_input.size,  # type: ignore
            *grad_output.tuple(),
            *new_weight.tuple(),
            True,  # type: ignore
        )
        return grad_input, grad_weight


# Convenience function to call forward and backward pass for 2D convolution
conv2d = Conv2dFun.apply

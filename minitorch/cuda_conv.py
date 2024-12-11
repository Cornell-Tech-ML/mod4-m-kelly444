from typing import Tuple, Any
import numpy as np
from numba import cuda
import math

from .tensor import Tensor
from .tensor_data import Storage, Shape, Strides
from .tensor_functions import Function
from .autodiff import Context
from . import operators

def _assert_shape_match(
    batch: int, batch_: int,
    in_channels: int, in_channels_: int,
    out_channels: int, out_channels_: int
) -> None:
    """Verify tensor shapes match for convolution operations."""
    assert (
        batch == batch_
        and in_channels == in_channels_
        and out_channels == out_channels_
    )

@cuda.jit
def _conv1d_forward_kernel(
    out: Storage,
    out_size: int,
    out_strides: Strides,
    in_storage: Storage,
    in_shape: Shape,
    in_strides: Strides,
    weight_storage: Storage,
    weight_shape: Shape,
    weight_strides: Strides,
) -> None:
    """CUDA kernel for 1D convolution forward pass."""
    idx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    if idx >= out_size:
        return
    
    batch = in_shape[0]
    in_channels = in_shape[1]
    width = in_shape[2]
    kw = weight_shape[2]

    # Calculate position
    out_batch = idx // (weight_shape[0] * width)
    out_channel = (idx // width) % weight_shape[0]
    out_pos = idx % width

    acc = 0.0
    for c in range(in_channels):
        for k in range(kw):
            if out_pos + k < width:
                in_idx = (
                    out_batch * in_strides[0] +
                    c * in_strides[1] + 
                    (out_pos + k) * in_strides[2]
                )
                w_idx = (
                    out_channel * weight_strides[0] +
                    c * weight_strides[1] +
                    k * weight_strides[2]
                )
                acc += in_storage[in_idx] * weight_storage[w_idx]
    
    out[idx] = acc

@cuda.jit
def _conv1d_backward_kernel(
    out: Storage,
    out_size: int,
    out_strides: Strides,
    in_storage: Storage,
    in_shape: Shape,
    in_strides: Strides,
    weight_storage: Storage,
    weight_shape: Shape,
    weight_strides: Strides,
) -> None:
    """CUDA kernel for 1D convolution backward pass."""
    idx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    if idx >= out_size:
        return

    batch = in_shape[0]
    in_channels = in_shape[1]
    width = in_shape[2]
    kw = weight_shape[2]

    # Calculate position
    out_batch = idx // (in_channels * width)
    out_channel = (idx // width) % in_channels
    out_pos = idx % width

    acc = 0.0
    for c in range(weight_shape[0]):
        for k in range(kw):
            if out_pos >= k:
                in_idx = (
                    out_batch * in_strides[0] +
                    c * in_strides[1] + 
                    (out_pos - k) * in_strides[2]
                )
                w_idx = (
                    c * weight_strides[0] +
                    out_channel * weight_strides[1] +
                    k * weight_strides[2]
                )
                acc += in_storage[in_idx] * weight_storage[w_idx]
    
    out[idx] = acc

@cuda.jit
def _conv2d_forward_kernel(
    out: Storage,
    out_size: int,
    out_strides: Strides,
    in_storage: Storage,
    in_shape: Shape,
    in_strides: Strides,
    weight_storage: Storage,
    weight_shape: Shape,
    weight_strides: Strides,
) -> None:
    """CUDA kernel for 2D convolution forward pass."""
    idx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    if idx >= out_size:
        return

    batch = in_shape[0]
    in_channels = in_shape[1]
    height = in_shape[2]
    width = in_shape[3]
    kh = weight_shape[2]
    kw = weight_shape[3]

    # Calculate output position
    out_batch = idx // (weight_shape[0] * height * width)
    out_channel = (idx // (height * width)) % weight_shape[0]
    out_h = (idx // width) % height
    out_w = idx % width

    acc = 0.0
    for c in range(in_channels):
        for h in range(kh):
            for w in range(kw):
                if out_h + h < height and out_w + w < width:
                    in_idx = (
                        out_batch * in_strides[0] +
                        c * in_strides[1] +
                        (out_h + h) * in_strides[2] +
                        (out_w + w) * in_strides[3]
                    )
                    w_idx = (
                        out_channel * weight_strides[0] +
                        c * weight_strides[1] +
                        h * weight_strides[2] +
                        w * weight_strides[3]
                    )
                    acc += in_storage[in_idx] * weight_storage[w_idx]

    out[idx] = acc

@cuda.jit
def _conv2d_backward_kernel(
    out: Storage,
    out_size: int,
    out_strides: Strides,
    in_storage: Storage,
    in_shape: Shape,
    in_strides: Strides,
    weight_storage: Storage,
    weight_shape: Shape,
    weight_strides: Strides,
) -> None:
    """CUDA kernel for 2D convolution backward pass."""
    idx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    if idx >= out_size:
        return

    batch = in_shape[0]
    in_channels = in_shape[1]
    height = in_shape[2]
    width = in_shape[3]
    kh = weight_shape[2]
    kw = weight_shape[3]

    # Calculate output position
    out_batch = idx // (in_channels * height * width)
    out_channel = (idx // (height * width)) % in_channels
    out_h = (idx // width) % height
    out_w = idx % width

    acc = 0.0
    for c in range(weight_shape[0]):
        for h in range(kh):
            for w in range(kw):
                if out_h >= h and out_w >= w:
                    in_idx = (
                        out_batch * in_strides[0] +
                        c * in_strides[1] +
                        (out_h - h) * in_strides[2] +
                        (out_w - w) * in_strides[3]
                    )
                    w_idx = (
                        c * weight_strides[0] +
                        out_channel * weight_strides[1] +
                        h * weight_strides[2] +
                        w * weight_strides[3]
                    )
                    acc += in_storage[in_idx] * weight_storage[w_idx]

    out[idx] = acc

class CudaConv1dFunction(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, weight: Tensor) -> Tensor:
        """Forward pass for CUDA 1D convolution."""
        ctx.save_for_backward(input, weight)
        batch, in_channels, w = input.shape
        out_channels, in_channels2, kw = weight.shape
        assert in_channels == in_channels2

        # Create output tensor
        output = input.zeros((batch, out_channels, w))
        
        # Setup CUDA grid
        threads_per_block = 256
        blocks = (output.size + threads_per_block - 1) // threads_per_block

        # Launch kernel
        _conv1d_forward_kernel[blocks, threads_per_block](
            output._tensor._storage,
            output.size,
            output._tensor._strides,
            input._tensor._storage,
            input.shape,
            input._tensor._strides,
            weight._tensor._storage,
            weight.shape,
            weight._tensor._strides,
        )

        return output

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Backward pass for CUDA 1D convolution."""
        input, weight = ctx.saved_values
        batch, in_channels, w = input.shape
        out_channels, in_channels, kw = weight.shape

        # Compute gradients
        grad_weight = grad_output.zeros((in_channels, out_channels, kw))
        grad_input = grad_output.zeros((batch, in_channels, w))

        # Setup CUDA grid
        threads_per_block = 256
        blocks_weight = (grad_weight.size + threads_per_block - 1) // threads_per_block
        blocks_input = (grad_input.size + threads_per_block - 1) // threads_per_block

        # Launch kernels
        _conv1d_backward_kernel[blocks_weight, threads_per_block](
            grad_weight._tensor._storage,
            grad_weight.size,
            grad_weight._tensor._strides,
            input.permute(1, 0, 2)._tensor._storage,
            input.permute(1, 0, 2).shape,
            input.permute(1, 0, 2)._tensor._strides,
            grad_output.permute(1, 0, 2)._tensor._storage,
            grad_output.permute(1, 0, 2).shape,
            grad_output.permute(1, 0, 2)._tensor._strides,
        )
        grad_weight = grad_weight.permute(1, 0, 2)

        _conv1d_backward_kernel[blocks_input, threads_per_block](
            grad_input._tensor._storage,
            grad_input.size,
            grad_input._tensor._strides,
            grad_output._tensor._storage,
            grad_output.shape,
            grad_output._tensor._strides,
            weight.permute(1, 0, 2)._tensor._storage,
            weight.permute(1, 0, 2).shape,
            weight.permute(1, 0, 2)._tensor._strides,
        )

        return grad_input, grad_weight

class CudaConv2dFunction(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, weight: Tensor) -> Tensor:
        """Forward pass for CUDA 2D convolution."""
        ctx.save_for_backward(input, weight)
        batch, in_channels, h, w = input.shape
        out_channels, in_channels2, kh, kw = weight.shape
        assert in_channels == in_channels2

        # Create output tensor
        output = input.zeros((batch, out_channels, h, w))
        
        # Setup CUDA grid
        threads_per_block = 256
        blocks = (output.size + threads_per_block - 1) // threads_per_block

        # Launch kernel
        _conv2d_forward_kernel[blocks, threads_per_block](
            output._tensor._storage,
            output.size,
            output._tensor._strides,
            input._tensor._storage,
            input.shape,
            input._tensor._strides,
            weight._tensor._storage,
            weight.shape,
            weight._tensor._strides,
        )

        return output

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Backward pass for CUDA 2D convolution."""
        input, weight = ctx.saved_values
        batch, in_channels, h, w = input.shape
        out_channels, in_channels, kh, kw = weight.shape

        # Compute gradients
        grad_weight = grad_output.zeros((in_channels, out_channels, kh, kw))
        grad_input = grad_output.zeros((batch, in_channels, h, w))

        # Setup CUDA grid
        threads_per_block = 256
        blocks_weight = (grad_weight.size + threads_per_block - 1) // threads_per_block
        blocks_input = (grad_input.size + threads_per_block - 1) // threads_per_block

        # Launch kernels
        _conv2d_backward_kernel[blocks_weight, threads_per_block](
            grad_weight._tensor._storage,
            grad_weight.size,
            grad_weight._tensor._strides,
            input.permute(1, 0, 2, 3)._tensor._storage,
            input.permute(1, 0, 2, 3).shape,
            input.permute(1, 0, 2, 3)._tensor._strides,
            grad_output.permute(1, 0, 2, 3)._tensor._storage,
            grad_output.permute(1, 0, 2, 3).shape,
            grad_output.permute(1, 0, 2, 3)._tensor._strides,
        )
        grad_weight = grad_weight.permute(1, 0, 2, 3)

        _conv2d_backward_kernel[blocks_input, threads_per_block](
            grad_input._tensor._storage,
            grad_input.size,
            grad_input._tensor._strides,
            grad_output._tensor._storage,
            grad_output.shape,
            grad_output._tensor._strides,
            weight.permute(1, 0, 2, 3)._tensor._storage,
            weight.permute(1, 0, 2, 3).shape,
            weight.permute(1, 0, 2, 3)._tensor._strides,
        )

        return grad_input, grad_weight

# Register as tensor operations
cuda_conv1d = CudaConv1dFunction.apply
cuda_conv2d = CudaConv2dFunction.apply

def use_cuda_conv(tensor: Tensor) -> bool:
    """Check if CUDA convolution should be used for given tensor."""
    try:
        return tensor.backend.cuda and cuda.is_available()
    except:
        return False

def has_cuda() -> bool:
    """Check if CUDA is available on the system."""
    try:
        return cuda.is_available()
    except:
        return False
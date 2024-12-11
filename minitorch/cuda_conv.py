from __future__ import annotations
from typing import Tuple
import numpy as np
from numba import cuda

from .tensor_data import Shape, Storage, Strides, MAX_DIMS
from .tensor import Tensor
from .tensor_functions import Function

# CUDA kernel for 1D convolution
@cuda.jit
def _cuda_conv1d_kernel(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    input: Storage,
    input_shape: Shape,
    input_strides: Strides,
    weight: Storage,
    weight_shape: Shape,
    weight_strides: Strides,
    reverse: bool
):
    # Get thread and block indices
    idx = cuda.grid(1)
    
    # Check if thread is within bounds
    if idx >= out.size:
        return
        
    # Calculate output indices
    batch = idx // (out_shape[1] * out_shape[2])
    out_channel = (idx // out_shape[2]) % out_shape[1]
    out_width = idx % out_shape[2]
    
    # Initialize output value
    val = 0.0
    
    # Loop through input channels and kernel width
    for in_channel in range(input_shape[1]):
        for k in range(weight_shape[2]):
            if reverse:
                if out_width - k >= 0:
                    in_pos = (
                        batch * input_strides[0] +
                        in_channel * input_strides[1] +
                        (out_width - k) * input_strides[2]
                    )
                    w_pos = (
                        out_channel * weight_strides[0] +
                        in_channel * weight_strides[1] +
                        k * weight_strides[2]
                    )
                    val += input[in_pos] * weight[w_pos]
            else:
                if out_width + k < input_shape[2]:
                    in_pos = (
                        batch * input_strides[0] +
                        in_channel * input_strides[1] +
                        (out_width + k) * input_strides[2]
                    )
                    w_pos = (
                        out_channel * weight_strides[0] +
                        in_channel * weight_strides[1] +
                        k * weight_strides[2]
                    )
                    val += input[in_pos] * weight[w_pos]
    
    # Store result
    out_pos = (
        batch * out_strides[0] +
        out_channel * out_strides[1] +
        out_width * out_strides[2]
    )
    out[out_pos] = val

# CUDA kernel for 2D convolution
@cuda.jit
def _cuda_conv2d_kernel(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    input: Storage,
    input_shape: Shape,
    input_strides: Strides,
    weight: Storage,
    weight_shape: Shape,
    weight_strides: Strides,
    reverse: bool
):
    # Get thread indices for 2D grid
    tx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    ty = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    
    # Check if thread is within bounds
    if tx >= out_shape[2] or ty >= out_shape[3]:
        return
        
    # Process each batch and output channel
    for batch in range(out_shape[0]):
        for out_channel in range(out_shape[1]):
            val = 0.0
            
            # Loop through input channels and kernel dimensions
            for in_channel in range(input_shape[1]):
                for kh in range(weight_shape[2]):
                    for kw in range(weight_shape[3]):
                        if reverse:
                            h, w = ty - kh, tx - kw
                            if h >= 0 and w >= 0:
                                in_pos = (
                                    batch * input_strides[0] +
                                    in_channel * input_strides[1] +
                                    h * input_strides[2] +
                                    w * input_strides[3]
                                )
                                w_pos = (
                                    out_channel * weight_strides[0] +
                                    in_channel * weight_strides[1] +
                                    kh * weight_strides[2] +
                                    kw * weight_strides[3]
                                )
                                val += input[in_pos] * weight[w_pos]
                        else:
                            h, w = ty + kh, tx + kw
                            if h < input_shape[2] and w < input_shape[3]:
                                in_pos = (
                                    batch * input_strides[0] +
                                    in_channel * input_strides[1] +
                                    h * input_strides[2] +
                                    w * input_strides[3]
                                )
                                w_pos = (
                                    out_channel * weight_strides[0] +
                                    in_channel * weight_strides[1] +
                                    kh * weight_strides[2] +
                                    kw * weight_strides[3]
                                )
                                val += input[in_pos] * weight[w_pos]
            
            # Store result
            out_pos = (
                batch * out_strides[0] +
                out_channel * out_strides[1] +
                ty * out_strides[2] +
                tx * out_strides[3]
            )
            out[out_pos] = val

class CudaConv1dFunction(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, weight: Tensor) -> Tensor:
        ctx.save_for_backward(input, weight)
        
        # Get shapes
        batch, in_channels, width = input.shape
        out_channels, in_channels2, kernel_width = weight.shape
        assert in_channels == in_channels2
        
        # Create output tensor
        output = input.zeros((batch, out_channels, width))
        
        # Configure CUDA grid
        threads_per_block = 256
        blocks = (output.size + threads_per_block - 1) // threads_per_block
        
        # Launch kernel
        _cuda_conv1d_kernel[blocks, threads_per_block](
            *output.tuple(), output.size,
            *input.tuple(),
            *weight.tuple(),
            False
        )
        
        return output

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        input, weight = ctx.saved_values
        batch, in_channels, width = input.shape
        out_channels, in_channels, kernel_width = weight.shape
        
        # Initialize gradients
        grad_weight = grad_output.zeros((in_channels, out_channels, kernel_width))
        grad_input = input.zeros((batch, in_channels, width))
        
        # Configure CUDA grid
        threads_per_block = 256
        blocks = (grad_weight.size + threads_per_block - 1) // threads_per_block
        
        # Calculate weight gradients
        new_input = input.permute(1, 0, 2)
        new_grad_output = grad_output.permute(1, 0, 2)
        _cuda_conv1d_kernel[blocks, threads_per_block](
            *grad_weight.tuple(), grad_weight.size,
            *new_input.tuple(),
            *new_grad_output.tuple(),
            False
        )
        grad_weight = grad_weight.permute(1, 0, 2)
        
        # Calculate input gradients
        blocks = (grad_input.size + threads_per_block - 1) // threads_per_block
        new_weight = weight.permute(1, 0, 2)
        _cuda_conv1d_kernel[blocks, threads_per_block](
            *grad_input.tuple(), grad_input.size,
            *grad_output.tuple(),
            *new_weight.tuple(),
            True
        )
        
        return grad_input, grad_weight

class CudaConv2dFunction(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, weight: Tensor) -> Tensor:
        ctx.save_for_backward(input, weight)
        
        # Get shapes
        batch, in_channels, height, width = input.shape
        out_channels, in_channels2, kernel_height, kernel_width = weight.shape
        assert in_channels == in_channels2
        
        # Create output tensor
        output = input.zeros((batch, out_channels, height, width))
        
        # Configure CUDA grid
        threads_per_block = (16, 16)
        blocks = (
            (width + threads_per_block[0] - 1) // threads_per_block[0],
            (height + threads_per_block[1] - 1) // threads_per_block[1]
        )
        
        # Launch kernel
        _cuda_conv2d_kernel[blocks, threads_per_block](
            *output.tuple(),
            *input.tuple(),
            *weight.tuple(),
            False
        )
        
        return output

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        input, weight = ctx.saved_values
        batch, in_channels, height, width = input.shape
        out_channels, in_channels, kernel_height, kernel_width = weight.shape
        
        # Initialize gradients
        grad_weight = grad_output.zeros(
            (in_channels, out_channels, kernel_height, kernel_width)
        )
        grad_input = input.zeros((batch, in_channels, height, width))
        
        # Configure CUDA grid
        threads_per_block = (16, 16)
        blocks = (
            (width + threads_per_block[0] - 1) // threads_per_block[0],
            (height + threads_per_block[1] - 1) // threads_per_block[1]
        )
        
        # Calculate weight gradients
        new_input = input.permute(1, 0, 2, 3)
        new_grad_output = grad_output.permute(1, 0, 2, 3)
        _cuda_conv2d_kernel[blocks, threads_per_block](
            *grad_weight.tuple(),
            *new_input.tuple(),
            *new_grad_output.tuple(),
            False
        )
        grad_weight = grad_weight.permute(1, 0, 2, 3)
        
        # Calculate input gradients
        new_weight = weight.permute(1, 0, 2, 3)
        _cuda_conv2d_kernel[blocks, threads_per_block](
            *grad_input.tuple(),
            *grad_output.tuple(),
            *new_weight.tuple(),
            True
        )
        
        return grad_input, grad_weight

# Convenience functions for using the CUDA convolutions
cuda_conv1d = CudaConv1dFunction.apply
cuda_conv2d = CudaConv2dFunction.apply
# type: ignore
from typing import Callable, Optional, TypeVar, Any
import numpy as np
import numba
from numba import cuda
from numba.cuda import jit as _jit
from .tensor import Tensor
from .tensor_data import (
    MAX_DIMS,
    Shape,
    Storage,
    Strides,
    TensorData,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps

FakeCUDAKernel = Any


def use_cuda() -> bool:
    """Checks if CUDA (GPU) is available for computation.

    Returns
    -------
        bool: True if CUDA is available, False otherwise.

    """
    try:
        return cuda.is_available()
    except (numba.cuda.cudadrv.driver.CudaSupportError, RuntimeError):
        return False


Fn = TypeVar("Fn")


def device_jit(fn: Fn, **kwargs: Any) -> Fn:
    """Compiles a function for execution on a GPU using Numba's JIT compiler.

    Args:
    ----
        fn: The function to compile for GPU execution.
        kwargs: Additional arguments to be passed to the JIT compiler.

    Returns:
    -------
        The compiled GPU function.

    """
    return _jit(device=True, **kwargs)(fn)  # type: ignore


def jit(fn: Fn, **kwargs: Any) -> FakeCUDAKernel:
    """Compiles a function using Numba's JIT compiler for general use.

    Args:
    ----
        fn: The function to compile.
        kwargs: Additional arguments for JIT compilation.

    Returns:
    -------
        A compiled version of the function.

    """
    return _jit(**kwargs)(fn)  # type: ignore


to_index = device_jit(to_index)
index_to_position = device_jit(index_to_position)
broadcast_index = device_jit(broadcast_index)
THREADS_PER_BLOCK = 32


class CudaOps(TensorOps):
    """Contains operations that are optimized for execution on a GPU using CUDA.
    Inherits from TensorOps and overrides certain methods to enable GPU computation.
    """

    cuda = True

    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """Applies a function to each element of a tensor.

        Args:
        ----
            fn: A function that operates on each element of the tensor (e.g., square, log).

        Returns:
        -------
            A function that applies `fn` to a tensor and returns the result.

        """
        cufn: Callable[[float], float] = device_jit(fn)
        f = tensor_map(cufn)

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            """Applies the function to a tensor and returns the result.

            Args:
            ----
                a: The input tensor.
                out: Optional output tensor to store the result. If None, a new tensor is created.

            Returns:
            -------
                A new tensor with the function applied to each element.

            """
            if out is None:
                out = a.zeros(a.shape)
            if use_cuda():
                threadsperblock = THREADS_PER_BLOCK
                blockspergrid = (out.size + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
                f[blockspergrid, threadsperblock](*out.tuple(), out.size, *a.tuple())  # type: ignore
            else:
                out._tensor._storage = np.array([fn(x) for x in a._tensor._storage])
            return out

        return ret


@staticmethod
def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
    """Takes two tensors and applies a function element-wise to pairs of elements from both tensors.

    Args:
    ----
        fn: A function that takes two float values and returns a float (e.g., addition, multiplication).

    Returns:
    -------
        A function that applies `fn` element-wise to two tensors.

    """
    cufn: Callable[[float, float], float] = device_jit(fn)
    f = tensor_zip(cufn)

    def ret(a: Tensor, b: Tensor) -> Tensor:
        """Applies the function to each pair of elements from two tensors and returns the result.

        Args:
        ----
            a: The first input tensor.
            b: The second input tensor.

        Returns:
        -------
            A new tensor with the function applied to each pair of elements.

        """
        c_shape = shape_broadcast(a.shape, b.shape)
        out = a.zeros(c_shape)
        if use_cuda():
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + (threadsperblock - 1)) // threadsperblock
            f[blockspergrid, threadsperblock](  # type: ignore
                *out.tuple(), out.size, *a.tuple(), *b.tuple()
            )
        else:
            out._tensor._storage = np.array(
                [
                    fn(a._tensor._storage[i], b._tensor._storage[i])
                    for i in range(
                        min(len(a._tensor._storage), len(b._tensor._storage))
                    )
                ]
            )
        return out

    return ret


def tensor_map(
    fn: Callable[[float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """Helper function that maps a function over the elements of a tensor on the GPU.

    Args:
    ----
        fn: A function that operates on individual elements of the tensor.

    Returns:
    -------
        A function that performs the element-wise operation on the GPU.

    """

    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        in_index = cuda.local.array(MAX_DIMS, numba.int32)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        if i < out_size:
            to_index(i, out_shape, out_index)
            broadcast_index(out_index, out_shape, in_shape, in_index)
            out_pos = index_to_position(out_index, out_strides)
            in_pos = index_to_position(in_index, in_strides)
            out[out_pos] = fn(in_storage[in_pos])

    return cuda.jit()(_map)  # type: ignore


def tensor_zip(
    fn: Callable[[float, float], float],
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """Helper function that applies a function element-wise to two tensors on the GPU.

    Args:
    ----
        fn: A function that takes two float values and returns a float.

    Returns:
    -------
        A function that applies the operation to two tensors in parallel on the GPU.

    """

    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        a_index = cuda.local.array(MAX_DIMS, numba.int32)
        b_index = cuda.local.array(MAX_DIMS, numba.int32)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        if i < out_size:
            to_index(i, out_shape, out_index)
            broadcast_index(out_index, out_shape, a_shape, a_index)
            broadcast_index(out_index, out_shape, b_shape, b_index)
            out_pos = index_to_position(out_index, out_strides)
            a_pos = index_to_position(a_index, a_strides)
            b_pos = index_to_position(b_index, b_strides)
            out[out_pos] = fn(a_storage[a_pos], b_storage[b_pos])

    return cuda.jit()(_zip)  # type: ignore


def _sum_practice(out: Storage, a: Storage, size: int) -> None:
    """Helper function to sum elements of a tensor in parallel on the GPU.

    Args:
    ----
        out: The output storage where the sum result will be stored.
        a: The input tensor storage.
        size: The number of elements in the input tensor.

    """
    BLOCK_DIM = 32
    cache = cuda.shared.array(BLOCK_DIM, numba.float64)
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    pos = cuda.threadIdx.x
    if i < size:
        cache[pos] = a[i]
    else:
        cache[pos] = 0
    cuda.syncthreads()
    stride = BLOCK_DIM // 2
    while stride > 0:
        if pos < stride:
            cache[pos] += cache[pos + stride]
        cuda.syncthreads()
        stride //= 2
    if pos == 0:
        out[cuda.blockIdx.x] = cache[0]


jit_sum_practice = cuda.jit()(_sum_practice)


def sum_practice(a: Tensor) -> TensorData:
    """Computes the sum of the elements in a tensor. Uses GPU for large tensors, or CPU for small ones.

    Args:
    ----
        a: The input tensor.

    Returns:
    -------
        A `TensorData` object containing the sum of the tensor's elements.

    """
    if use_cuda():
        (size,) = a.shape
        threadsperblock = THREADS_PER_BLOCK
        blockspergrid = (size // THREADS_PER_BLOCK) + 1
        out = TensorData([0.0 for i in range(2)], (2,))
        out.to_cuda_()
        jit_sum_practice[blockspergrid, threadsperblock](
            out.tuple()[0], a._tensor._storage, size
        )
        return out
    else:
        total = float(np.sum(a._tensor._storage))
        if len(a._tensor._storage) <= 16:
            return TensorData([total, 0.0], (2,))
        else:
            half_sum = total / 2
            return TensorData([half_sum, half_sum], (2,))


def tensor_reduce(
    fn: Callable[[float, float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """Reduces (combines) elements of a tensor along a specific axis using a binary function (e.g., sum, max).
    Uses GPU for parallel computation.

    Args:
    ----
        fn: A binary function that combines two elements (e.g., addition, multiplication).

    Returns:
    -------
        A function that performs the reduction on the tensor along a specified axis.

    """

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
        reduce_value: float,
    ) -> None:
        BLOCK_DIM = 1024
        cache = cuda.shared.array(BLOCK_DIM, numba.float64)
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        out_pos = cuda.blockIdx.x
        pos = cuda.threadIdx.x
        to_index(out_pos, out_shape, out_index)
        reduce_size = a_shape[reduce_dim]
        if pos < reduce_size:
            out_index[reduce_dim] = pos
            a_pos = index_to_position(out_index, a_strides)
            cache[pos] = a_storage[a_pos]
        else:
            cache[pos] = reduce_value
        cuda.syncthreads()
        stride = BLOCK_DIM // 2
        while stride > 0:
            if pos < stride and pos + stride < reduce_size:
                cache[pos] = fn(cache[pos], cache[pos + stride])
            cuda.syncthreads()
            stride //= 2
        if pos == 0:
            out[out_pos] = cache[0]

    return cuda.jit()(_reduce)  # type: ignore


def _mm_practice(out: Storage, a: Storage, b: Storage, size: int) -> None:
    BLOCK_DIM = 32
    a_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    b_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    i = cuda.threadIdx.x
    j = cuda.threadIdx.y
    if i >= size or j >= size:
        return
    a_shared[i, j] = a[size * i + j]
    b_shared[i, j] = b[size * i + j]
    cuda.syncthreads()
    accum = 0.0
    for k in range(size):
        accum += a_shared[i, k] * b_shared[k, j]
    out[size * i + j] = accum


jit_mm_practice = jit(_mm_practice)


def mm_practice(a: Tensor, b: Tensor) -> TensorData:
    """Performs matrix multiplication between two tensors.

    Args:
    ----
        a: The first input tensor.
        b: The second input tensor.

    Returns:
    -------
        A `TensorData` object containing the result of the matrix multiplication.

    """
    if use_cuda():
        # If CUDA is available, perform matrix multiplication on the GPU
        (size, _) = a.shape  # Assume a is a square matrix
        threadsperblock = (
            THREADS_PER_BLOCK,
            THREADS_PER_BLOCK,
        )  # Number of threads per block (for 2D grid)
        blockspergrid = 1  # Number of blocks in the grid (1 block for simplicity)

        # Create an output tensor and move it to the GPU
        out = TensorData([0.0 for i in range(size * size)], (size, size))
        out.to_cuda_()

        # Perform the matrix multiplication on the GPU
        jit_mm_practice[blockspergrid, threadsperblock](
            out.tuple()[0], a._tensor._storage, b._tensor._storage, size
        )

        return out
    else:
        # If CUDA is not available, perform matrix multiplication on the CPU
        a_array = a._tensor._storage.reshape(
            a.shape
        )  # Reshape the input tensor a to a 2D array
        b_array = b._tensor._storage.reshape(
            b.shape
        )  # Reshape the input tensor b to a 2D array

        # Perform matrix multiplication using NumPy
        result = np.matmul(a_array, b_array)

        # Return the result as a TensorData object
        return TensorData(result.flatten().tolist(), result.shape)


def _tensor_matrix_multiply(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    b_storage: Storage,
    b_shape: Shape,
    b_strides: Strides,
) -> None:
    """Low-level CUDA kernel for performing matrix multiplication between two tensors.
    This function is designed to run on the GPU using shared memory for better performance.

    Args:
    ----
        out: The output storage where the result of the matrix multiplication will be stored.
        out_shape: The shape of the output tensor.
        out_strides: The strides to access elements in the output tensor.
        out_size: The total number of elements in the output tensor.
        a_storage: The storage of the first input tensor.
        a_shape: The shape of the first input tensor.
        a_strides: The strides to access elements in the first input tensor.
        b_storage: The storage of the second input tensor.
        b_shape: The shape of the second input tensor.
        b_strides: The strides to access elements in the second input tensor.

    Returns:
    -------
        None: The result is written directly to the `out` storage.

    """
    BLOCK_DIM = 32  # The block size (32x32 threads per block)

    # If the tensors are batched, this defines the batch stride
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0
    batch = cuda.blockIdx.z  # The batch index

    # Shared memory for storing sub-matrices of a and b
    a_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    b_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)

    # Thread and block index calculations
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    pi = cuda.threadIdx.x
    pj = cuda.threadIdx.y

    accum = 0.0  # Initialize accumulator for the result of the matrix multiplication

    # Loop over the 'k' dimension of the matrices in chunks (blocked processing)
    for phase in range(0, a_shape[2], BLOCK_DIM):
        # Load a sub-matrix from tensor 'a' into shared memory
        k = phase + pj
        if i < a_shape[1] and k < a_shape[2]:
            a_pos = batch * a_batch_stride + i * a_strides[1] + k * a_strides[2]
            a_shared[pi, pj] = a_storage[a_pos]
        else:
            a_shared[pi, pj] = 0.0  # Padding with zeros if out of bounds

        # Load a sub-matrix from tensor 'b' into shared memory
        k = phase + pi
        if k < b_shape[1] and j < b_shape[2]:
            b_pos = batch * b_batch_stride + k * b_strides[1] + j * b_strides[2]
            b_shared[pi, pj] = b_storage[b_pos]
        else:
            b_shared[pi, pj] = 0.0  # Padding with zeros if out of bounds

        # Synchronize threads to make sure shared memory is fully loaded
        cuda.syncthreads()

        # Perform matrix multiplication for the current block
        if i < out_shape[1] and j < out_shape[2]:
            for k in range(min(BLOCK_DIM, a_shape[2] - phase)):
                accum += a_shared[pi, k] * b_shared[k, pj]

        # Synchronize threads again before the next phase
        cuda.syncthreads()

    # Write the result to the output storage at the correct position
    if i < out_shape[1] and j < out_shape[2]:
        out_pos = batch * out_strides[0] + i * out_strides[1] + j * out_strides[2]
        out[out_pos] = accum


# JIT compile the low-level matrix multiplication function for GPU
tensor_matrix_multiply = jit(_tensor_matrix_multiply)

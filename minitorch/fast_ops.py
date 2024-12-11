from __future__ import annotations
from typing import TYPE_CHECKING, TypeVar, Any
import numpy as np
from numba import prange
from numba import njit as _njit
from .tensor_data import (
    MAX_DIMS,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps

if TYPE_CHECKING:
    from typing import Callable, Optional

    from .tensor import Tensor
    from .tensor_data import Shape, Storage, Strides

Fn = TypeVar("Fn")


def njit(fn: Fn, **kwargs: Any) -> Fn:
    """Decorates a function with Just-In-Time (JIT) compilation for optimization.

    Args:
    ----
        fn: The function to be optimized.
        kwargs: Additional arguments passed to the `njit` decorator.

    Returns:
    -------
        The JIT-compiled function.

    """
    return _njit(inline="always", **kwargs)(fn)  # type: ignore


# JIT compile some utility functions for better performance
to_index = njit(to_index)
index_to_position = njit(index_to_position)
broadcast_index = njit(broadcast_index)


class FastOps(TensorOps):
    """FastOps provides optimized tensor operations (like map, zip, reduce, and matrix multiplication)
    with support for GPU (CUDA) acceleration using JIT compilation.
    """

    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """Applies a function element-wise to a tensor.

        Args:
        ----
            fn: A function that takes a float and returns a float (applied element-wise).

        Returns:
        -------
            A function that applies `fn` to every element of a tensor.

        """
        f = tensor_map(njit(fn))

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)  # Initialize output tensor if not provided
            f(*out.tuple(), *a.tuple())  # Perform the map operation
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        """Applies a binary function element-wise to two tensors, element-by-element.

        Args:
        ----
            fn: A function that takes two floats and returns a float (applied element-wise).

        Returns:
        -------
            A function that applies `fn` to each corresponding pair of elements from two tensors.

        """
        f = tensor_zip(njit(fn))

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)  # Get the broadcasted shape
            out = a.zeros(
                c_shape
            )  # Initialize the output tensor with the broadcasted shape
            f(*out.tuple(), *a.tuple(), *b.tuple())  # Perform the zip operation
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        """Reduces a tensor along a specified dimension using a binary function.

        Args:
        ----
            fn: A function that combines two values and returns a single value (applied in a reduction).
            start: The starting value for the reduction.

        Returns:
        -------
            A function that reduces a tensor along a specified dimension.

        """
        f = tensor_reduce(njit(fn))

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = 1  # Set the size of the reduced dimension to 1
            out = a.zeros(tuple(out_shape))  # Initialize the output tensor
            out._tensor._storage[:] = start  # Set the starting value for reduction
            f(*out.tuple(), *a.tuple(), dim)  # Perform the reduction
            return out

        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        """Performs matrix multiplication between two tensors.

        Args:
        ----
            a: The first input tensor (matrix).
            b: The second input tensor (matrix).

        Returns:
        -------
            The result of multiplying the two tensors.

        """
        both_2d = 0
        if len(a.shape) == 2:
            a = a.contiguous().view(
                1, a.shape[0], a.shape[1]
            )  # Ensure a is 3D (batch dimension)
            both_2d += 1
        if len(b.shape) == 2:
            b = b.contiguous().view(
                1, b.shape[0], b.shape[1]
            )  # Ensure b is 3D (batch dimension)
            both_2d += 1
        both_2d = both_2d == 2
        ls = list(
            shape_broadcast(a.shape[:-2], b.shape[:-2])
        )  # Broadcast the shapes (excluding last two dims)
        ls.append(a.shape[-2])  # Append rows from a
        ls.append(b.shape[-1])  # Append columns from b
        assert a.shape[-1] == b.shape[-2]  # Ensure inner dimensions match
        out = a.zeros(tuple(ls))  # Initialize the output tensor
        tensor_matrix_multiply(
            *out.tuple(), *a.tuple(), *b.tuple()
        )  # Perform matrix multiplication
        if both_2d:
            out = out.view(
                out.shape[1], out.shape[2]
            )  # Reshape back to 2D if necessary
        return out

    cuda = False


def tensor_map(
    fn: Callable[[float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """Applies a function element-wise to a tensor, using parallel processing.

    Args:
    ----
        fn: A function that takes a float and returns a float (applied element-wise).

    Returns:
    -------
        A function that performs the mapping operation using the provided function.

    """

    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        for i in prange(len(out)):  # Loop over each element in the output tensor
            out_index = np.empty(MAX_DIMS, np.int32)
            in_index = np.empty(MAX_DIMS, np.int32)

            to_index(i, out_shape, out_index)  # Convert index to position
            broadcast_index(
                out_index, out_shape, in_shape, in_index
            )  # Broadcast indices
            o = index_to_position(
                out_index, out_strides
            )  # Get position in output tensor
            j = index_to_position(in_index, in_strides)  # Get position in input tensor
            out[o] = fn(in_storage[j])  # Apply function element-wise

    return njit(_map, parallel=True)  # JIT compile the function with parallel execution


def tensor_zip(
    fn: Callable[[float, float], float],
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """Applies a binary function element-wise to two tensors.

    Args:
    ----
        fn: A function that takes two floats and returns a float (applied element-wise).

    Returns:
    -------
        A function that performs the zip operation between two tensors.

    """

    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:
        for i in prange(len(out)):  # Loop over each element in the output tensor
            out_index = np.empty(MAX_DIMS, np.int32)
            a_index = np.empty(MAX_DIMS, np.int32)
            b_index = np.empty(MAX_DIMS, np.int32)

            to_index(i, out_shape, out_index)  # Convert index to position
            o = index_to_position(
                out_index, out_strides
            )  # Get position in output tensor
            broadcast_index(
                out_index, out_shape, a_shape, a_index
            )  # Broadcast index for a
            j = index_to_position(a_index, a_strides)  # Get position in input tensor a
            broadcast_index(
                out_index, out_shape, b_shape, b_index
            )  # Broadcast index for b
            k = index_to_position(b_index, b_strides)  # Get position in input tensor b
            out[o] = fn(
                a_storage[j], b_storage[k]
            )  # Apply the binary function element-wise

    return njit(_zip, parallel=True)  # JIT compile the function with parallel execution


def tensor_reduce(
    fn: Callable[[float, float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """Reduces a tensor along a specified dimension using a binary function.

    Args:
    ----
        fn: A function that combines two values and returns a single value (applied in reduction).

    Returns:
    -------
        A function that performs the reduction operation along a given dimension.

    """

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
    ) -> None:
        for i in prange(len(out)):  # Loop over each element in the output tensor
            out_index = np.empty(MAX_DIMS, np.int32)
            local_index = np.empty(MAX_DIMS, np.int32)
            to_index(i, out_shape, out_index)  # Convert index to position
            o = index_to_position(
                out_index, out_strides
            )  # Get position in output tensor
            for j in range(len(out_shape)):
                local_index[j] = out_index[j]
            for s in range(a_shape[reduce_dim]):  # Loop along the reduce dimension
                local_index[reduce_dim] = s
                j = index_to_position(
                    local_index, a_strides
                )  # Get position in input tensor
                out[o] = fn(out[o], a_storage[j])  # Apply the reduction function

    return njit(
        _reduce, parallel=True
    )  # JIT compile the function with parallel execution


def _tensor_matrix_multiply(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    b_storage: Storage,
    b_shape: Shape,
    b_strides: Strides,
) -> None:
    """Performs matrix multiplication between two tensors (low-level implementation).

    Args:
    ----
        out: The output tensor where the result will be stored.
        out_shape: Shape of the output tensor.
        out_strides: Strides for accessing elements in the output tensor.
        a_storage: Storage for the first input tensor.
        a_shape: Shape of the first input tensor.
        a_strides: Strides for accessing elements in the first input tensor.
        b_storage: Storage for the second input tensor.
        b_shape: Shape of the second input tensor.
        b_strides: Strides for accessing elements in the second input tensor.

    Returns:
    -------
        None: The result is stored directly in the output tensor.

    """
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0
    blocks = a_shape[-1]
    for row_i in prange(0, out_shape[0]):  # Loop over output rows
        for col_j in range(0, out_shape[1]):  # Loop over output columns
            for block_k in range(0, out_shape[2]):  # Loop over blocks (inner product)
                row_s = row_i * a_batch_stride + col_j * a_strides[1]
                col_s = row_i * b_batch_stride + block_k * b_strides[2]
                temp = 0.0  # Accumulator for the sum of products
                for _ in range(0, blocks):  # Inner loop for matrix multiplication
                    temp += a_storage[row_s] * b_storage[col_s]
                    row_s += a_strides[-1]
                    col_s += b_strides[-2]
                out[
                    row_i * out_strides[0]
                    + col_j * out_strides[1]
                    + block_k * out_strides[2]
                ] = temp


# JIT compile the matrix multiplication function for GPU acceleration
tensor_matrix_multiply = njit(_tensor_matrix_multiply, parallel=True)
assert tensor_matrix_multiply is not None

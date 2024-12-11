from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING, Callable, Optional, Type
from typing_extensions import Protocol
from . import operators
from .tensor_data import (
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
    Index,
    Shape,
    Storage,
    Strides,
)

MAX_DIMS = 32  # Maximum number of dimensions supported for tensors

if TYPE_CHECKING:
    # These are type hints for better code understanding (used only for checking types)
    from .tensor import Tensor
    from .tensor_data import Shape, Storage, Strides


class MapProto(Protocol):
    """A protocol that defines the structure for a function that maps an operation
    across all elements in a tensor.

    The function takes a tensor as input, applies some operation to each element,
    and returns the result as a new tensor.
    """

    def __call__(self, x: Tensor, out: Optional[Tensor] = ..., /) -> Tensor:
        """Apply an operation to every number in the input tensor.

        Args:
        ----
            x (Tensor): The input tensor (a grid of numbers).
            out (Optional[Tensor], optional): An optional tensor to store the result. If not provided, a new tensor is created.

        Returns:
        -------
            Tensor: A new tensor with the operation applied to every number.

        """
        ...


class TensorOps:
    """A class that provides various operations for manipulating tensors.
    These operations include element-wise functions, reductions, and matrix operations.
    """

    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """Maps an operation (fn) over all elements of a tensor.
        For example, applying a function like square or sin to each element.
        """
        ...

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        """Combines two tensors element-wise using a function (fn).
        For example, adding corresponding elements from two tensors.
        """
        ...

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        """Reduces a tensor along a specific dimension using a function (fn).
        For example, summing or multiplying elements along a specific axis (like summing rows of a matrix).
        """
        ...

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        """Performs matrix multiplication between two tensors.
        This is not implemented in this version.
        """
        raise NotImplementedError("Not implemented in this assignment")

    cuda = False  # Specifies if CUDA (GPU support) is available


class TensorBackend:
    """A class that handles the application of various operations on tensors.
    It maps operations like negation, sigmoid, etc., using the provided operations (ops).
    """

    def __init__(self, ops: Type[TensorOps]):
        """Initializes various operations (like negation, addition, etc.) using the provided ops class.
        These operations can be applied on tensors for mathematical computation.
        """
        self.neg_map = ops.map(operators.neg)
        self.sigmoid_map = ops.map(operators.sigmoid)
        self.relu_map = ops.map(operators.relu)
        self.log_map = ops.map(operators.log)
        self.exp_map = ops.map(operators.exp)
        self.id_map = ops.map(operators.id)
        self.inv_map = ops.map(operators.inv)
        self.add_zip = ops.zip(operators.add)
        self.mul_zip = ops.zip(operators.mul)
        self.lt_zip = ops.zip(operators.lt)
        self.eq_zip = ops.zip(operators.eq)
        self.is_close_zip = ops.zip(operators.is_close)
        self.relu_back_zip = ops.zip(operators.relu_back)
        self.log_back_zip = ops.zip(operators.log_back)
        self.inv_back_zip = ops.zip(operators.inv_back)
        self.add_reduce = ops.reduce(operators.add, 0.0)
        self.mul_reduce = ops.reduce(operators.mul, 1.0)
        self.matrix_multiply = ops.matrix_multiply
        self.cuda = ops.cuda


class SimpleOps(TensorOps):
    """A simple implementation of tensor operations, providing basic element-wise and reduction operations."""

    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """Applies a given function (fn) to each element in the tensor.
        Creates a new tensor where each element is the result of applying fn to the corresponding element in the original tensor.
        """
        f = tensor_map(fn)

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)  # Create a new tensor with the same shape as 'a'
            f(*out.tuple(), *a.tuple())  # Apply the function element-wise
            return out

        return ret

    @staticmethod
    def zip(
        fn: Callable[[float, float], float],
    ) -> Callable[["Tensor", "Tensor"], "Tensor"]:
        """Combines two tensors element-wise using a given function (fn).
        The function is applied to corresponding elements from both tensors.
        """
        f = tensor_zip(fn)

        def ret(a: "Tensor", b: "Tensor") -> "Tensor":
            if a.shape != b.shape:
                c_shape = shape_broadcast(
                    a.shape, b.shape
                )  # If shapes differ, find a common shape
            else:
                c_shape = a.shape  # If shapes are the same, use the original shape
            out = a.zeros(c_shape)  # Create a new tensor for the result
            f(*out.tuple(), *a.tuple(), *b.tuple())  # Apply the function element-wise
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[["Tensor", int], "Tensor"]:
        """Reduces a tensor along a specific dimension (e.g., summing along rows).
        Applies the function (fn) repeatedly to elements in the tensor along a specified axis.
        """
        f = tensor_reduce(fn)

        def ret(a: "Tensor", dim: int) -> "Tensor":
            out_shape = list(a.shape)
            out_shape[dim] = 1  # Reduce the specified dimension to size 1
            out = a.zeros(tuple(out_shape))  # Create a new tensor for the result
            out._tensor._storage[:] = (
                start  # Initialize the result with the starting value
            )
            f(*out.tuple(), *a.tuple(), dim)  # Apply the reduction function
            return out

        return ret

    @staticmethod
    def matrix_multiply(a: "Tensor", b: "Tensor") -> "Tensor":
        """Matrix multiplication (not implemented in this version)."""
        raise NotImplementedError("Not implemented in this assignment")

    is_cuda = False  # Specifies if CUDA (GPU support) is available


def tensor_map(
    fn: Callable[[float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """Maps an operation (like a mathematical function) over all elements of a tensor.
    The operation is applied to each element, and the result is stored in an output tensor.
    """

    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        out_index: Index = np.zeros(MAX_DIMS, np.int32)
        in_index: Index = np.zeros(MAX_DIMS, np.int32)
        for i in range(len(out)):
            to_index(
                i, out_shape, out_index
            )  # Convert linear index to multidimensional index
            broadcast_index(
                out_index, out_shape, in_shape, in_index
            )  # Handle broadcasting
            o = index_to_position(out_index, out_strides)  # Find position in the output
            j = index_to_position(in_index, in_strides)  # Find position in the input
            out[o] = fn(in_storage[j])  # Apply the function to the input element

    return _map


def tensor_zip(
    fn: Callable[[float, float], float],
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """Combines two tensors element-wise using a function (like addition or multiplication).
    It applies the function to corresponding elements in both tensors and stores the result in the output tensor.
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
        out_index: Index = np.zeros(MAX_DIMS, np.int32)
        a_index: Index = np.zeros(MAX_DIMS, np.int32)
        b_index: Index = np.zeros(MAX_DIMS, np.int32)
        for i in range(len(out)):
            to_index(
                i, out_shape, out_index
            )  # Convert linear index to multidimensional index
            o = index_to_position(out_index, out_strides)  # Find position in the output
            broadcast_index(
                out_index, out_shape, a_shape, a_index
            )  # Broadcast a tensor if needed
            j = index_to_position(a_index, a_strides)  # Find position in tensor a
            broadcast_index(
                out_index, out_shape, b_shape, b_index
            )  # Broadcast b tensor if needed
            k = index_to_position(b_index, b_strides)  # Find position in tensor b
            out[o] = fn(
                a_storage[j], b_storage[k]
            )  # Apply the function to both tensors

    return _zip


def tensor_reduce(
    fn: Callable[[float, float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """Reduces a tensor along a specific axis, applying a function (like sum or product)
    to the elements along that axis.
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
        out_index: Index = np.zeros(MAX_DIMS, np.int32)
        reduce_size = a_shape[reduce_dim]  # Get the size of the dimension to reduce
        for i in range(len(out)):
            to_index(
                i, out_shape, out_index
            )  # Convert linear index to multidimensional index
            o = index_to_position(out_index, out_strides)  # Find position in the output
            for s in range(
                reduce_size
            ):  # Loop through the elements along the reduced dimension
                out_index[reduce_dim] = s
                j = index_to_position(
                    out_index, a_strides
                )  # Find position in the input
                out[o] = fn(out[o], a_storage[j])  # Apply the function

    return _reduce


SimpleBackend = TensorBackend(SimpleOps)

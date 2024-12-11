from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING
import numpy as np
from . import operators
from .autodiff import Context, Variable, backpropagate
from .tensor_data import TensorData
from .tensor_functions import (
    EQ,
    LT,
    Add,
    All,
    Copy,
    Exp,
    Inv,
    IsClose,
    Log,
    MatMul,
    Mul,
    Neg,
    Permute,
    ReLU,
    Sigmoid,
    Sum,
    View,
    tensor,
)

if TYPE_CHECKING:
    from typing import Any, Iterable, List, Optional, Sequence, Tuple, Type, Union
    import numpy.typing as npt
    from .tensor_data import Shape, Storage, Strides, UserIndex, UserShape, UserStrides
    from .tensor_functions import Function
    from .tensor_ops import TensorBackend

    TensorLike = Union[float, int, "Tensor"]


@dataclass
class History:
    """Keeps track of the history of a tensor's operations.
    - last_fn: The last function that was applied to this tensor.
    - ctx: The context in which this function was applied.
    - inputs: A sequence of tensors that were inputs to the function.
    """

    last_fn: Optional[Type[Function]] = None
    ctx: Optional[Context] = None
    inputs: Sequence[Tensor] = ()


_tensor_count = 0


class Tensor:
    """A class representing a multi-dimensional array (tensor) and its operations.
    This class supports automatic differentiation and various mathematical operations on tensors.
    """

    backend: TensorBackend
    history: Optional[History]
    grad: Optional[Tensor]
    _tensor: TensorData
    unique_id: int
    name: str

    def __init__(
        self,
        v: TensorData,
        back: Optional[History] = None,
        name: Optional[str] = None,
        backend: Optional[TensorBackend] = None,
    ):
        """Initializes a tensor with data and optional history (for backpropagation).

        Args:
        ----
            v: The actual data of the tensor (stored as TensorData)
            back: Optional history of operations leading to this tensor
            name: Optional name for the tensor (useful for debugging)
            backend: The computational backend to use (e.g., CPU or GPU)

        """
        global _tensor_count
        _tensor_count += 1
        self.unique_id = _tensor_count
        assert isinstance(v, TensorData)
        assert backend is not None
        self._tensor = v
        self.history = back
        self.backend = backend
        self.grad = None
        if name is not None:
            self.name = name
        else:
            self.name = str(self.unique_id)
        self.f = backend

    def requires_grad_(self, x: bool) -> None:
        """Marks the tensor to track gradients for backpropagation.

        Args:
        ----
            x: Whether to track gradients (True) or not (False)

        """
        self.history = History()

    def requires_grad(self) -> bool:
        """Returns True if the tensor tracks gradients, False otherwise."""
        return self.history is not None

    def to_numpy(self) -> npt.NDArray[np.float64]:
        """Converts the tensor to a NumPy array for easy interaction with other libraries."""
        return self.contiguous()._tensor._storage.reshape(self.shape)

    def _ensure_tensor(self, b: TensorLike) -> Tensor:
        """Converts a scalar value or another tensor into a tensor of the correct type.

        Args:
        ----
            b: Input value to convert (scalar or tensor)

        Returns:
        -------
            A new tensor of the correct type

        """
        if isinstance(b, (int, float)):
            c = Tensor.make([b], (1,), backend=self.backend)
        else:
            b._type_(self.backend)
            c = b
        return c

    def item(self) -> float:
        """Returns the single value stored in a scalar tensor.

        This method assumes the tensor only has one value.
        """
        assert self.size == 1
        x: float = self._tensor._storage[0]
        return x

    def contiguous(self) -> Tensor:
        """Returns a new tensor that is contiguous in memory (for efficient operations)."""
        return Copy.apply(self)

    def __repr__(self) -> str:
        """Returns a string representation of the tensor (useful for debugging)."""
        return self._tensor.to_string()

    def __hash__(self) -> float:
        """Returns a hash value for the tensor based on its unique ID."""
        return hash(self.unique_id)

    def __getitem__(self, key: Union[int, UserIndex]) -> float:
        """Allows indexing into the tensor using a single index or tuple of indices.

        Args:
        ----
            key: Index or tuple of indices to access

        Returns:
        -------
            Value at the specified index

        """
        key2 = (key,) if isinstance(key, int) else key
        return self._tensor.get(key2)

    def __setitem__(self, key: Union[int, UserIndex], val: float) -> None:
        """Allows setting a value at a specific index in the tensor.

        Args:
        ----
            key: Index or tuple of indices to modify
            val: New value to assign

        """
        key2 = (key,) if isinstance(key, int) else key
        self._tensor.set(key2, val)

    def _type_(self, backend: TensorBackend) -> None:
        """Moves the tensor data to the specified backend.

        Args:
        ----
            backend: Backend to move the tensor to (e.g., CPU or GPU)

        """
        self.backend = backend
        if backend.cuda:  # pragma: no cover
            self._tensor.to_cuda_()

    def _new(self, tensor_data: TensorData) -> Tensor:
        """Creates a new tensor from the provided tensor data.

        Args:
        ----
            tensor_data: Data to create the new tensor from

        Returns:
        -------
            A new tensor

        """
        return Tensor(tensor_data, backend=self.backend)

    @staticmethod
    def make(
        storage: Union[Storage, List[float]],
        shape: UserShape,
        strides: Optional[UserStrides] = None,
        backend: Optional[TensorBackend] = None,
    ) -> Tensor:
        """Creates a new tensor with specified storage, shape, and optional strides.

        Args:
        ----
            storage: Data to store in the tensor
            shape: Shape (dimensions) of the tensor
            strides: Optional strides (step size) for each dimension
            backend: Optional computational backend

        Returns:
        -------
            A new tensor

        """
        return Tensor(TensorData(storage, shape, strides), backend=backend)

    def expand(self, other: Tensor) -> Tensor:
        """Expands the tensor to match the shape of another tensor.

        Args:
        ----
            other: Tensor whose shape to match (broadcasting is applied if needed)

        Returns:
        -------
            A new tensor with the expanded shape

        """
        if self.shape == other.shape:
            return other
        true_shape = TensorData.shape_broadcast(self.shape, other.shape)
        buf = self.zeros(true_shape)
        self.backend.id_map(other, buf)
        if self.shape == true_shape:
            return buf
        out = buf
        orig_shape = [1] * (len(out.shape) - len(self.shape)) + list(self.shape)
        for dim, shape in enumerate(out.shape):
            if orig_shape[dim] == 1 and shape != 1:
                out = self.backend.add_reduce(out, dim)
        assert out.size == self.size, f"{out.shape} {self.shape}"
        return Tensor.make(out._tensor._storage, self.shape, backend=self.backend)

    def zeros(self, shape: Optional[UserShape] = None) -> Tensor:
        """Creates a tensor filled with zeros.

        Args:
        ----
            shape: Optional shape for the new tensor (defaults to current tensor's shape)

        Returns:
        -------
            A new tensor filled with zeros

        """

        def zero(shape: UserShape) -> Tensor:
            return Tensor.make(
                [0.0] * int(operators.prod(list(shape))),
                shape,
                backend=self.backend,
            )

        if shape is None:
            out = zero(self.shape)
        else:
            out = zero(shape)
        out._type_(self.backend)
        return out

    def tuple(self) -> Tuple[Storage, Shape, Strides]:
        """Returns a tuple of the tensor's storage, shape, and strides."""
        return self._tensor.tuple()

    def detach(self) -> Tensor:
        """Returns a new tensor that shares the same data but doesn't track gradients."""
        return Tensor(self._tensor, backend=self.backend)

    def accumulate_derivative(self, x: Any) -> None:
        """Accumulates a gradient (derivative) into the tensor's grad attribute.

        Args:
        ----
            x: Gradient to accumulate (tensor or scalar)

        """
        assert self.is_leaf(), "Only leaf variables can have derivatives."
        if self.grad is None:
            self.grad = Tensor.make(
                [0.0] * int(operators.prod(list(self.shape))),
                self.shape,
                backend=self.backend,
            )
        self.grad += x

    def is_leaf(self) -> bool:
        """Returns True if this tensor is a leaf node in the computation graph (no operations created it)."""
        return self.history is not None and self.history.last_fn is None

    def is_constant(self) -> bool:
        """Returns True if this tensor is constant and doesn't track gradients."""
        return self.history is None

    @property
    def parents(self) -> Iterable[Variable]:
        """Returns the parent variables (inputs) of the tensor in the computation graph."""
        assert self.history is not None
        return self.history.inputs

    def chain_rule(self, d: Any) -> Iterable[Tuple[Variable, Any]]:
        """Applies the chain rule to calculate derivatives of inputs.

        Args:
        ----
            d: Derivative of the current tensor

        Returns:
        -------
            List of tuples containing each input and its derivative

        """
        h = self.history
        assert h is not None
        assert h.last_fn is not None
        assert h.ctx is not None

        x = h.last_fn._backward(h.ctx, d)
        assert len(x) == len(h.inputs), f"Bug in function {h.last_fn}"
        return [
            (inp, inp.expand(self._ensure_tensor(d_in)))
            for inp, d_in in zip(h.inputs, x)
        ]

    def backward(self, grad_output: Optional[Tensor] = None) -> None:
        """Performs backpropagation to compute gradients.

        Args:
        ----
            grad_output: Optional gradient to propagate (defaults to ones for scalar output)

        """
        if grad_output is None:
            assert self.shape == (1,), "Must provide grad_output if non-scalar"
            grad_output = Tensor.make([1.0], (1,), backend=self.backend)
        backpropagate(self, grad_output)

    def __truediv__(self, b: TensorLike) -> Tensor:
        """Division operator (self / b)."""
        result = Mul.apply(self, Inv.apply(self._ensure_tensor(b)))
        return result

    def __rtruediv__(self, b: TensorLike) -> Tensor:
        """Reverse division operator (b / self)."""
        return Mul.apply(self._ensure_tensor(b), Inv.apply(self))

    def __matmul__(self, b: Tensor) -> Tensor:
        """Matrix multiplication operator (@)."""
        return MatMul.apply(self, b)

    def __add__(self, b: TensorLike) -> Tensor:
        """Addition operator (self + b)."""
        return Add.apply(self, self._ensure_tensor(b))

    def __radd__(self, b: TensorLike) -> Tensor:
        """Reverse addition operator (b + self)."""
        return self + b

    def __sub__(self, b: TensorLike) -> Tensor:
        """Subtraction operator (self - b)."""
        return Add.apply(self, -self._ensure_tensor(b))

    def __mul__(self, b: TensorLike) -> Tensor:
        """Multiplication operator (self * b)."""
        return Mul.apply(self, self._ensure_tensor(b))

    def __rmul__(self, b: TensorLike) -> Tensor:
        """Reverse multiplication operator (b * self)."""
        return self * b

    def __neg__(self) -> Tensor:
        """Negation operator (-self)."""
        return Neg.apply(self)

    def __lt__(self, b: TensorLike) -> Tensor:
        """Less than comparison (self < b)."""
        return LT.apply(self, self._ensure_tensor(b))

    def __gt__(self, b: TensorLike) -> Tensor:
        """Greater than comparison (self > b)."""
        return LT.apply(self._ensure_tensor(b), self)

    def __eq__(self, b: TensorLike) -> Tensor:
        """Equality comparison (self == b)."""
        return EQ.apply(self, self._ensure_tensor(b))

    def is_close(self, y: Tensor) -> Tensor:
        """Checks if two tensors have similar values.

        Args:
        ----
            y: Tensor to compare against

        Returns:
        -------
            A tensor indicating whether values are close

        """
        return IsClose.apply(self, y)

    def sigmoid(self) -> Tensor:
        """Applies the sigmoid activation function to the tensor.

        Returns
        -------
        - A new tensor after applying the sigmoid function.

        """
        return Sigmoid.apply(self)

    def relu(self) -> Tensor:
        """Applies the ReLU activation function to the tensor.

        Returns
        -------
        - A new tensor after applying the ReLU function.

        """
        return ReLU.apply(self)

    def log(self) -> Tensor:
        """Applies the logarithm function to the tensor.

        Returns
        -------
        - A new tensor after applying the log function.

        """
        return Log.apply(self)

    def exp(self) -> Tensor:
        """Applies the exponential function to the tensor.

        Returns
        -------
        - A new tensor after applying the exp function.

        """
        return Exp.apply(self)

    def sum(self, dim: Optional[int] = None) -> Tensor:
        """Sums the values along specified dimension(s).

        Args:
        ----
            dim: Optional dimension to sum over (None for all dimensions)

        Returns:
        -------
            A new tensor containing the sum

        """
        if dim is None:
            return Sum.apply(self.contiguous().view(self.size), self._ensure_tensor(0))
        else:
            return Sum.apply(self, self._ensure_tensor(dim))

    def mean(self, dim: Optional[int] = None) -> Tensor:
        """Calculates mean along specified dimension(s).

        Args:
        ----
            dim: Optional dimension to average over (None for all dimensions)

        Returns:
        -------
            A new tensor containing the mean

        """
        if dim is not None:
            return self.sum(dim) / self.shape[dim]
        else:
            return self.sum() / self.size

    def permute(self, *order: int) -> Tensor:
        """Reorders the dimensions of the tensor.

        Args:
        ----
            order: New order of dimensions (e.g., (1, 0) to swap axes)

        Returns:
        -------
            A new tensor with permuted dimensions

        """
        return Permute.apply(self, tensor(list(order)))

    def view(self, *shape: int) -> Tensor:
        """Reshapes the tensor to a new shape.

        Args:
        ----
            shape: New shape for the tensor

        Returns:
        -------
            A new tensor with the specified shape

        """
        return View.apply(self, tensor(list(shape)))

    def all(self, dim: Optional[int] = None) -> Tensor:
        """Checks if all values are True along specified dimension(s).

        Args:
        ----
            dim: Optional dimension to check (None for all dimensions)

        Returns:
        -------
            A tensor indicating whether all values are True

        """
        if dim is None:
            return All.apply(self.view(self.size), self._ensure_tensor(0))
        else:
            return All.apply(self, self._ensure_tensor(dim))

    def zero_grad_(self) -> None:
        """Resets the gradients of the tensor to None."""
        self.grad = None

    @property
    def shape(self) -> UserShape:
        """Returns the shape (dimensions) of the tensor."""
        return self._tensor.shape

    @property
    def size(self) -> int:
        """Returns the total number of elements in the tensor."""
        return self._tensor.size

    @property
    def dims(self) -> int:
        """Returns the number of dimensions (axes) of the tensor."""
        return self._tensor.dims

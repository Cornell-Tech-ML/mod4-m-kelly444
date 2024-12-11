from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence, Tuple, Type, Union
import numpy as np
from dataclasses import field
from .autodiff import Context, Variable, backpropagate, central_difference
from .scalar_functions import (
    EQ,
    LT,
    Add,
    Exp,
    Inv,
    Log,
    Mul,
    Neg,
    ReLU,
    ScalarFunction,
    Sigmoid,
)

ScalarLike = Union[float, int, "Scalar"]  # Value can be a float, int, or Scalar object.


@dataclass
class ScalarHistory:
    """Stores the history of a Scalar, like the last operation that created it and its input values."""

    last_fn: Optional[Type[ScalarFunction]] = (
        None  # The last function used to create this Scalar.
    )
    ctx: Optional[Context] = None  # The context in which the operation happened.
    inputs: Sequence[Scalar] = ()  # The inputs used to create this Scalar.


_var_count = 0  # Keeps track of the number of Scalar objects created.


@dataclass
class Scalar:
    """A class representing a scalar value (a single number) and its related operations.

    This can represent any numeric value and track how it was created or changed in calculations.
    """

    data: float  # The actual numeric value of this Scalar.
    history: Optional[ScalarHistory] = field(
        default_factory=ScalarHistory
    )  # The history of how this Scalar was created.
    derivative: Optional[float] = (
        None  # The derivative (rate of change) of this Scalar, if it has one.
    )
    name: str = field(default="")  # A name for this Scalar, used for identification.
    unique_id: int = field(default=0)  # A unique identifier for this Scalar.

    def __post_init__(self):
        """Sets up some additional properties when a Scalar is created.

        It automatically gives the Scalar a unique ID and name.
        """
        global _var_count
        _var_count += 1
        object.__setattr__(self, "unique_id", _var_count)
        object.__setattr__(self, "name", str(self.unique_id))
        object.__setattr__(self, "data", float(self.data))

    def __repr__(self) -> str:
        """Returns a string representation of the Scalar.

        This is what you see when you print a Scalar.
        """
        return f"Scalar({self.data})"

    def __mul__(self, b: ScalarLike) -> Scalar:
        """Multiplies this Scalar by another value (another Scalar, or a number).

        Returns a new Scalar that is the result of the multiplication.
        """
        if isinstance(b, (int, float)):
            b = Scalar(b)  # Convert numbers into Scalar objects
        return Mul.apply(self, b)

    def __truediv__(self, b: ScalarLike) -> Scalar:
        """Divides this Scalar by another value.

        Returns a new Scalar that is the result of the division.
        """
        if isinstance(b, (int, float)):
            b = Scalar(b)
        return Mul.apply(self, Inv.apply(b))

    def __rtruediv__(self, b: ScalarLike) -> Scalar:
        """Handles division where this Scalar is on the right side.

        This is a special case for division (when the Scalar is on the right).
        """
        if isinstance(b, (int, float)):
            b = Scalar(b)
        return Mul.apply(b, Inv.apply(self))

    def __bool__(self) -> bool:
        """Converts this Scalar to a boolean value (True or False).

        If the Scalar value is non-zero, it’s True; otherwise, it’s False.
        """
        return bool(self.data)

    def __radd__(self, b: ScalarLike) -> Scalar:
        """Handles addition when this Scalar is on the right side.

        This is a special case for addition (when the Scalar is on the right).
        """
        if isinstance(b, (int, float)):
            b = Scalar(b)
        return self + b

    def __rmul__(self, b: ScalarLike) -> Scalar:
        """Handles multiplication when this Scalar is on the right side.

        This is a special case for multiplication (when the Scalar is on the right).
        """
        if isinstance(b, (int, float)):
            b = Scalar(b)
        return self * b

    def __lt__(self, b: ScalarLike) -> Scalar:
        """Compares this Scalar with another value (checks if it is less than).

        Returns a Scalar representing the result of the comparison.
        """
        if isinstance(b, (int, float)):
            b = Scalar(b)
        return LT.apply(self, b)

    def __gt__(self, b: ScalarLike) -> Scalar:
        """Compares this Scalar with another value (checks if it is greater than).

        Returns a Scalar representing the result of the comparison.
        """
        if isinstance(b, (int, float)):
            b = Scalar(b)
        return LT.apply(b, self)

    def __sub__(self, b: ScalarLike) -> Scalar:
        """Subtracts another value from this Scalar.

        Returns a new Scalar with the result of the subtraction.
        """
        if isinstance(b, (int, float)):
            b = Scalar(b)
        return Add.apply(self, Neg.apply(b))

    def __rsub__(self, b: ScalarLike) -> Scalar:
        """Handles subtraction when this Scalar is on the right side.

        This is a special case for subtraction (when the Scalar is on the right).
        """
        if isinstance(b, (int, float)):
            b = Scalar(b)
        return Add.apply(b, Neg.apply(self))

    def __neg__(self) -> Scalar:
        """Negates the value of this Scalar (makes it negative).

        Returns a new Scalar that is the negation of this one.
        """
        return Neg.apply(self)

    def __add__(self, b: ScalarLike) -> Scalar:
        """Adds another value to this Scalar.

        Returns a new Scalar with the result of the addition.
        """
        if isinstance(b, (int, float)):
            return Add.apply(self, Scalar(b))
        return Add.apply(self, b)

    def log(self) -> Scalar:
        """Computes the logarithm of this Scalar.

        Returns a new Scalar that is the log of this value.
        """
        return Log.apply(self)

    def exp(self) -> Scalar:
        """Computes the exponential (e^x) of this Scalar.

        Returns a new Scalar that is the exponential of this value.
        """
        return Exp.apply(self)

    def sigmoid(self) -> Scalar:
        """Applies the Sigmoid function to this Scalar.

        Sigmoid is a common function used in machine learning, especially for binary classification.
        """
        return Sigmoid.apply(self)

    def relu(self) -> Scalar:
        """Applies the ReLU (Rectified Linear Unit) function to this Scalar.

        ReLU is a commonly used function in neural networks, where all negative values are replaced with zero.
        """
        return ReLU.apply(self)

    def __eq__(self, b: ScalarLike) -> Scalar:
        """Checks if this Scalar is equal to another value.

        Returns a Scalar representing the result of the comparison.
        """
        if isinstance(b, (int, float)):
            b = Scalar(b)
        return EQ.apply(self, b)

    def __hash__(self) -> float:
        """Returns a unique hash value for this Scalar, based on its unique ID."""
        return hash(self.unique_id)

    def accumulate_derivative(self, x: Any) -> None:
        """Adds a derivative value to this Scalar’s existing derivative.

        This is useful in backpropagation when we want to accumulate gradients.
        """
        assert self.is_leaf(), "Only leaf variables can have derivatives."
        if self.derivative is None:
            self.__setattr__("derivative", 0.0)
        self.__setattr__("derivative", self.derivative + x)

    def is_leaf(self) -> bool:
        """Checks if this Scalar is a "leaf" in the computation graph (i.e., it wasn’t created by any function).

        Leaf nodes are the ones we start with, like the inputs to a model.
        """
        return self.history is not None and self.history.last_fn is None

    def is_constant(self) -> bool:
        """Checks if this Scalar is a constant value (i.e., it doesn’t change).

        Constants are used for values that don’t have a history or derivative.
        """
        return self.history is None

    @property
    def parents(self) -> Iterable[Variable]:
        """Returns the parent variables (inputs) that created this Scalar.

        This is part of the computation graph that tracks how each Scalar was created.
        """
        assert self.history is not None
        return self.history.inputs

    def chain_rule(self, d: Any) -> Iterable[Tuple[Variable, Any]]:
        """Applies the chain rule for backpropagation to calculate gradients for this Scalar.

        This helps in updating parameters in machine learning models.
        """
        h = self.history
        assert h is not None
        assert h.last_fn is not None
        assert h.ctx is not None

        local_grads = h.last_fn._backward(h.ctx, d)
        if not isinstance(local_grads, Iterable):
            local_grads = [local_grads]

        paired_grads = zip(h.inputs, local_grads)
        result = [(x, grad) for x, grad in paired_grads if not x.is_constant()]

        return result

    def backward(self, d_output: Optional[float] = None) -> None:
        """Runs backpropagation to compute the gradients for this Scalar.

        If no derivative is provided, it starts with a default value of 1.0.
        """
        if d_output is None:
            d_output = 1.0
        backpropagate(self, d_output)


def derivative_check(f: Any, *scalars: Scalar) -> None:
    """Checks if the derivative calculated using backpropagation matches the one calculated using central difference.

    This is a way to verify that the derivatives are correct.
    """
    out = f(*scalars)
    out.backward()
    err_msg = """
Derivative check at arguments f(%s) and received derivative f'=%f for argument %d,
but was expecting derivative f'=%f from central difference."""
    for i, x in enumerate(scalars):
        check = central_difference(f, *scalars, arg=i)
        assert x.derivative is not None
        np.testing.assert_allclose(
            x.derivative,
            check.data,
            1e-2,
            1e-2,
            err_msg=err_msg
            % (str([x.data for x in scalars]), x.derivative, i, check.data),
        )

from __future__ import annotations
from typing import TYPE_CHECKING
import minitorch
from . import operators
from .autodiff import Context

if TYPE_CHECKING:
    from typing import Tuple
    from .scalar import Scalar, ScalarLike


def wrap_tuple(x: float | Tuple[float, ...]) -> Tuple[float, ...]:
    """Ensures the input is always a tuple.

    If you give it a single number, it will turn it into a tuple with that number.
    If you give it a tuple, it will leave it as is.
    """
    if isinstance(x, tuple):
        return x
    return (x,)


class ScalarFunction:
    """A class that defines how math operations (like addition, multiplication) are done while keeping track of the steps.

    - `forward` does the math operation (like adding two numbers).
    - `backward` calculates how to adjust the inputs based on the result (helpful in training machine learning models).
    """

    @classmethod
    def _backward(cls, ctx: Context, d_out: float) -> Tuple[float, ...]:
        """Calculates how much to change the inputs to improve the result.

        This is the "backward" step in machine learning, where we figure out what to adjust.
        """
        return wrap_tuple(cls.backward(ctx, d_out))  # type: ignore

    @classmethod
    def _forward(cls: type[ScalarFunction], ctx: Context, *inps: float) -> float:
        """Performs the math operation (like adding numbers) to get the result.

        This is where the actual calculation happens.
        """
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: ScalarLike) -> Scalar:
        """Takes the input values, does the math, and returns the result wrapped in a special object.

        This ensures that we keep track of how the result was obtained for later adjustments (like when training models).
        """
        raw_vals = []
        scalars = []
        for v in vals:
            if isinstance(v, minitorch.scalar.Scalar):
                scalars.append(v)
                raw_vals.append(v.data)
            else:
                scalars.append(minitorch.scalar.Scalar(v))
                raw_vals.append(v)
        ctx = Context(False)  # This helps track the steps in the calculation.
        c = cls._forward(ctx, *raw_vals)  # Perform the math operation.
        assert isinstance(c, float), f"Expected a number, got {type(c)}"
        back = minitorch.scalar.ScalarHistory(
            cls, ctx, scalars
        )  # Keep a history of how we got the result.
        return minitorch.scalar.Scalar(c, back)


class Add(ScalarFunction):
    """Adds two numbers together."""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Performs the addition operation: a + b."""
        return operators.add(a, b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Calculates the gradient: how much to change each input to improve the result.

        Since both inputs have the same effect on the result, we return the same gradient for both.
        """
        return d_output, d_output


class Log(ScalarFunction):
    """Computes the natural logarithm (log) of a number."""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Performs the logarithm operation: log(a)."""
        ctx.save_for_backward(a)  # Save the input to use for the backward step.
        return operators.log(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Calculates the gradient based on the input value."""
        (a,) = ctx.saved_values
        grad = operators.log_back(a, d_output)
        return grad


class Mul(ScalarFunction):
    """Multiplies two numbers together."""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Performs the multiplication operation: a * b."""
        ctx.save_for_backward(a, b)  # Save both numbers for later use.
        return operators.mul(a, b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Calculates the gradient: how much to change each input to improve the result.

        The gradients are calculated as: d_output * the other input.
        """
        a, b = ctx.saved_values
        grad_a, grad_b = d_output * b, d_output * a
        return grad_a, grad_b


class Inv(ScalarFunction):
    """Computes the reciprocal (1 divided by the number)."""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Performs the reciprocal operation: 1 / a."""
        ctx.save_for_backward(a)
        return operators.inv(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Calculates the gradient: how much to change the input to improve the result."""
        (a,) = ctx.saved_values
        grad = operators.inv_back(a, d_output)
        return grad


class Neg(ScalarFunction):
    """Changes the sign of a number to its negative (multiplies it by -1)."""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Performs the negation operation: -a."""
        return operators.neg(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Calculates the gradient: how much to change the input to improve the result."""
        return operators.neg(d_output)  # The negative sign also affects the gradient.


class Sigmoid(ScalarFunction):
    """Applies the sigmoid function, which squashes a number between 0 and 1.

    Often used to model probabilities in machine learning.
    """

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Applies the sigmoid function: 1 / (1 + e^(-a))."""
        result = operators.sigmoid(a)
        ctx.save_for_backward(result)  # Save the output for use in backward.
        return result

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Calculates the gradient of the sigmoid function."""
        (a,) = ctx.saved_values
        grad = operators.sigmoid_back(a)
        return grad * d_output  # The gradient depends on the sigmoid output.


class ReLU(ScalarFunction):
    """Applies the ReLU function: if the number is negative, make it zero. If it’s positive, leave it unchanged.

    Commonly used in deep learning to avoid certain problems.
    """

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Performs the ReLU operation: max(0, a)."""
        ctx.save_for_backward(a)
        return operators.relu(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Calculates the gradient for ReLU."""
        (a,) = ctx.saved_values
        grad = operators.relu_back(a, d_output)
        return grad


class Exp(ScalarFunction):
    """Raises the number e (Euler's number) to the power of the input number."""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Performs the exponential operation: e^a."""
        ctx.save_for_backward(a)
        return operators.exp(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Calculates the gradient: how much to change the input to improve the result."""
        (a,) = ctx.saved_values
        return d_output * operators.exp(
            a
        )  # The gradient depends on the original exponential value.


class LT(ScalarFunction):
    """Checks if one number is smaller than another.

    Returns 1 if true, and 0 if false.
    """

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Performs the comparison: a < b."""
        return operators.lt(a, b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """No gradient for comparison, as it’s not a smooth function."""
        return (
            0.0,
            0.0,
        )


class EQ(ScalarFunction):
    """Checks if two numbers are equal.

    Returns 1 if true, and 0 if false.
    """

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Performs the comparison: a == b."""
        return operators.eq(a, b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """No gradient for equality, as equality is not a smooth function."""
        return (
            0.0,
            0.0,
        )

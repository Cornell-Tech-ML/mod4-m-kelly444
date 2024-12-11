from __future__ import annotations
import random
from typing import TYPE_CHECKING, Optional
import numpy as np
import minitorch
from . import operators
from .autodiff import Context
from .tensor_ops import SimpleBackend, TensorBackend

if TYPE_CHECKING:
    from typing import Any, List, Tuple
    from .tensor import Tensor
    from .tensor_data import UserIndex, UserShape


def wrap_tuple(x: Any) -> tuple:
    """Takes an input `x` and returns it as a tuple.
    If `x` is already a tuple, it just returns it.
    Otherwise, it wraps `x` into a new tuple.
    """
    if isinstance(x, tuple):
        return x
    return (x,)


class Function:
    """A base class for all mathematical functions that will be applied to tensors.
    It manages the forward pass (calculating results) and backward pass (calculating gradients).
    """

    @classmethod
    def _backward(cls, ctx: Context, grad_out: Tensor) -> Tuple[Tensor, ...]:
        """Handles the backward pass for the function.
        It calculates how the output gradients affect the input tensors.
        """
        return wrap_tuple(cls.backward(ctx, grad_out))  # type: ignore

    @classmethod
    def _forward(cls: type[Function], ctx: Context, *inps: Tensor) -> Tensor:
        """Handles the forward pass for the function.
        It calculates the output based on the input tensors.
        """
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls: type[Function], *vals: Tensor) -> Tensor:
        """Applies the function to the given tensors and returns the result.
        Also handles whether the tensors need gradients for backpropagation.
        """
        raw_vals = []
        need_grad = False
        for v in vals:
            if v.requires_grad():
                need_grad = True
            raw_vals.append(v.detach())
        ctx = Context(not need_grad)  # Determines if gradients are needed
        c = cls._forward(ctx, *raw_vals)
        back = None
        if need_grad:
            back = minitorch.History(
                cls, ctx, vals
            )  # Tracks history for gradient calculation
        return minitorch.Tensor(c._tensor, back, backend=c.backend)


class Neg(Function):
    """Negates (flips the sign of) a tensor.
    For example, turning positive values into negative ones and vice versa.
    """

    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Performs the forward pass: negates the input tensor.
        This means changing all positive values to negative, and vice versa.
        """
        return t1.f.neg_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Performs the backward pass: returns the gradient as the negative of the input gradient.
        This means multiplying the gradient by -1.
        """
        return -1.0 * grad_output


class Inv(Function):
    """Inverts a tensor (takes its reciprocal).
    For example, 1/x where x is a tensor.
    """

    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Performs the forward pass: calculates the reciprocal of the input tensor.
        This means returning 1 divided by each element of the tensor.
        """
        ctx.save_for_backward(t1)
        result = t1.f.inv_map(t1)
        return result

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Performs the backward pass: calculates how the gradient changes with the inversion.
        This means computing how the output gradient behaves with respect to the reciprocal of the input.
        """
        (t1,) = ctx.saved_values
        result = grad_output.f.inv_back_zip(t1, grad_output)
        return result


class Add(Function):
    """Adds two tensors together element-wise."""

    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Performs the forward pass: adds the two input tensors together element-wise.
        This means adding corresponding values in each tensor.
        """
        return t1.f.add_zip(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Performs the backward pass: the gradient for both inputs is the same.
        This means that the gradient with respect to each input is the same as the output gradient.
        """
        return grad_output, grad_output


class All(Function):
    """Checks if all elements in a tensor are True (or non-zero)."""

    @staticmethod
    def forward(ctx: Context, a: Tensor, dim: Optional[Tensor] = None) -> Tensor:
        """Performs the forward pass: reduces the tensor by applying an 'all' check along a specified dimension.
        This means checking if all values in the tensor are True or non-zero along a given dimension.
        """
        dim_value = int(dim.item()) if dim is not None else -1
        ctx.save_for_backward(tensor([dim_value]))
        if dim_value == -1:
            return a.f.mul_reduce(a.contiguous().view(int(operators.prod(a.shape))), 0)
        else:
            return a.f.mul_reduce(a, dim_value)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, None]:
        """Performs the backward pass: no gradient is needed for the `dim` parameter.
        This means we don't calculate gradients for the dimension used in the 'all' operation.
        """
        return grad_output, None


class Mul(Function):
    """Multiplies two tensors together element-wise."""

    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        """Performs the forward pass: multiplies the two input tensors element-wise.
        This means multiplying corresponding elements in each tensor.
        """
        ctx.save_for_backward(a, b)
        return a.f.mul_zip(a, b)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Performs the backward pass: calculates how the gradient affects each input tensor.
        This means calculating the gradient with respect to each input by multiplying it with the output gradient.
        """
        a, b = ctx.saved_values
        return (
            grad_output.f.mul_zip(b, grad_output),
            grad_output.f.mul_zip(a, grad_output),
        )


class Sigmoid(Function):
    """Applies the sigmoid function to a tensor (scales values between 0 and 1)."""

    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Performs the forward pass: applies the sigmoid function to the input tensor.
        This means scaling the values in the tensor between 0 and 1.
        """
        out = t1.f.sigmoid_map(t1)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Performs the backward pass: calculates the gradient based on the sigmoid's output.
        This means using the sigmoid output to calculate how the gradient should change.
        """
        sigma: Tensor = ctx.saved_values[0]
        return sigma * (-sigma + 1.0) * grad_output


class ReLU(Function):
    """Applies the ReLU (Rectified Linear Unit) function, which sets negative values to zero."""

    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Performs the forward pass: applies ReLU to the input tensor.
        This means setting all negative values in the tensor to zero.
        """
        ctx.save_for_backward(t1)
        return t1.f.relu_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Performs the backward pass: calculates how the gradient changes with ReLU.
        This means adjusting the gradient based on which values in the tensor were positive or negative.
        """
        (a,) = ctx.saved_values
        return grad_output.f.relu_back_zip(a, grad_output)


class Log(Function):
    """Takes the natural logarithm (log base e) of each element in the tensor."""

    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Performs the forward pass: calculates the natural log of the input tensor.
        This means calculating the logarithm (base e) of each value in the tensor.
        """
        ctx.save_for_backward(t1)
        out = t1.f.log_map(t1)
        return out

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Performs the backward pass: calculates how the gradient changes with the logarithm.
        This means adjusting the gradient based on the natural log's derivative.
        """
        (a,) = ctx.saved_values
        return grad_output.f.log_back_zip(a, grad_output)


class Exp(Function):
    """Raises `e` (the mathematical constant) to the power of each element in the tensor."""

    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Performs the forward pass: applies the exponential function to the input tensor.
        This means returning `e` raised to the power of each value in the tensor.
        """
        out = t1.f.exp_map(t1)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Performs the backward pass: calculates how the gradient changes with the exponential function.
        This means adjusting the gradient based on the exponential function's derivative.
        """
        (a,) = ctx.saved_values
        return grad_output.f.mul_zip(a, grad_output)


class Sum(Function):
    """Sums the elements of a tensor along a specified dimension."""

    @staticmethod
    def forward(ctx: Context, a: Tensor, dim: Tensor) -> Tensor:
        """Performs the forward pass: sums the tensor's elements along a specific dimension.
        This means adding up the values along the specified axis of the tensor.
        """
        ctx.save_for_backward(a.shape, dim)
        return a.f.add_reduce(a, int(dim.item()))

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Performs the backward pass: no gradient for the dimension, only for the tensor itself.
        This means we don't calculate gradients for the dimension, only for the tensor.
        """
        a_shape, dim = ctx.saved_values
        return grad_output, 0.0


class LT(Function):
    """Compares two tensors element-wise, checking if one is less than the other."""

    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        """Performs the forward pass: compares each element of `a` with `b`, checking if `a < b`.
        This means returning a tensor with `True` where `a` is less than `b`, and `False` otherwise.
        """
        ctx.save_for_backward(a.shape, b.shape)
        return a.f.lt_zip(a, b)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Performs the backward pass: no gradient for the comparison operation.
        This means returning zeros for both tensors as the comparison doesn't have gradients.
        """
        a_shape, b_shape = ctx.saved_values
        return zeros(a_shape), zeros(b_shape)


class EQ(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        """Compares two tensors element-wise for equality.
        This function checks if each element in tensor 'a' is equal to the corresponding element in tensor 'b'.
        Returns a tensor of boolean values (True/False) indicating where they are equal.

        Args:
        ----
            ctx: Context to store any necessary information for backward computation.
            a: The first tensor to compare.
            b: The second tensor to compare.

        Returns:
        -------
            Tensor: A tensor of boolean values (True or False) representing element-wise equality.

        """
        ctx.save_for_backward(a.shape, b.shape)
        return a.f.eq_zip(a, b)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Returns gradients for tensors 'a' and 'b' after performing the equality check in the forward pass.
        Since the comparison is a simple equality check, the gradients are zero.

        Args:
        ----
            ctx: Context that stores saved values from the forward pass.
            grad_output: Gradient of the output tensor.

        Returns:
        -------
            Tuple[Tensor, Tensor]: Gradients of tensors 'a' and 'b' (both are zero tensors).

        """
        a_shape, b_shape = ctx.saved_values
        return zeros(a_shape), zeros(b_shape)


class IsClose(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        """Compares two tensors element-wise to check if their values are close,
        within a small tolerance. Useful for checking near equality between floating-point values.

        Args:
        ----
            ctx: Context to store any necessary information for backward computation.
            a: The first tensor to compare.
            b: The second tensor to compare.

        Returns:
        -------
            Tensor: A tensor of boolean values (True or False) indicating where the elements in 'a' and 'b' are close.

        """
        return a.f.is_close_zip(a, b)


class Permute(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, order: Tensor) -> Tensor:
        """Reorders the dimensions of a tensor according to the specified order.
        This function rearranges the axes (dimensions) of tensor 'a' based on the given 'order'.

        Args:
        ----
            ctx: Context to store the order for backward computation.
            a: The input tensor to permute.
            order: A tensor that specifies the new order of dimensions.

        Returns:
        -------
            Tensor: A new tensor with its dimensions reordered according to the 'order'.

        """
        ctx.save_for_backward(order)
        return a._new(a._tensor.permute(*[int(order[i]) for i in range(order.size)]))

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Computes the gradient of the permuted tensor with respect to the original tensor.
        Since permuting is a simple reordering of the tensor's dimensions, the gradient is also permuted accordingly.

        Args:
        ----
            ctx: Context that stores the saved order from the forward pass.
            grad_output: The gradient of the output tensor.

        Returns:
        -------
            Tuple[Tensor, float]: The gradient of the input tensor (with permuted dimensions) and a scalar value (0.0) as a placeholder.

        """
        order: Tensor = ctx.saved_values[0]
        order2: List[int] = [
            a[0]
            for a in sorted(
                enumerate([order[i] for i in range(order.size)]), key=lambda a: a[1]
            )
        ]
        return grad_output._new(grad_output._tensor.permute(*order2)), 0.0


class View(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, shape: Tensor) -> Tensor:
        """Changes the shape of a tensor without changing its data.
        This operation 'views' the original tensor in a different shape, provided the total number of elements stays the same.

        Args:
        ----
            ctx: Context to store the original shape for backward computation.
            a: The input tensor to reshape.
            shape: The target shape to reshape the tensor into.

        Returns:
        -------
            Tensor: A new tensor with the specified shape but sharing the same data as the original tensor.

        """
        ctx.save_for_backward(a.shape)
        assert a._tensor.is_contiguous(), "Must be contiguous to view"
        shape2 = [int(shape[i]) for i in range(shape.size)]
        return minitorch.Tensor.make(
            a._tensor._storage, tuple(shape2), backend=a.backend
        )

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Computes the gradient for the tensor reshaped by the 'view' operation.
        It returns a tensor with the same shape as the original input tensor before reshaping.

        Args:
        ----
            ctx: Context that stores the original shape from the forward pass.
            grad_output: The gradient of the output tensor.

        Returns:
        -------
            Tuple[Tensor, float]: The gradient of the reshaped tensor (which is equivalent to the original tensor's gradient) and a placeholder scalar (0.0).

        """
        (original,) = ctx.saved_values
        return (
            minitorch.Tensor.make(
                grad_output._tensor._storage, original, backend=grad_output.backend
            ),
            0.0,
        )


class Copy(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor) -> Tensor:
        """Creates a copy of the input tensor.
        This operation simply returns a new tensor that has the same data as the input tensor.

        Args:
        ----
            ctx: Context for storing information related to the operation (not needed here).
            a: The input tensor to copy.

        Returns:
        -------
            Tensor: A new tensor that is a copy of the input tensor.

        """
        return a.f.id_map(a)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Since the 'copy' operation does not alter the data, the gradient of the output is just passed through unchanged.

        Args:
        ----
            ctx: Context storing any saved values from the forward pass (not used here).
            grad_output: The gradient of the output tensor.

        Returns:
        -------
            Tensor: The same gradient as the input, because no change in data occurred.

        """
        return grad_output


class MatMul(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Performs matrix multiplication between two tensors.
        This function calculates the dot product between two tensors 't1' and 't2' (e.g., a matrix multiplication).

        Args:
        ----
            ctx: Context to store the input tensors for backward computation.
            t1: The first tensor (typically a matrix or vector).
            t2: The second tensor (typically a matrix or vector).

        Returns:
        -------
            Tensor: The result of the matrix multiplication between 't1' and 't2'.

        """
        ctx.save_for_backward(t1, t2)
        return t1.f.matrix_multiply(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Computes the gradients of the input tensors with respect to the output tensor
        for the matrix multiplication operation.

        Args:
        ----
            ctx: Context storing the input tensors for backward computation.
            grad_output: The gradient of the output tensor.

        Returns:
        -------
            Tuple[Tensor, Tensor]: The gradients of the two input tensors 't1' and 't2'.

        """
        t1, t2 = ctx.saved_values

        def transpose(a: Tensor) -> Tensor:
            order = list(range(a.dims))
            order[-2], order[-1] = order[-1], order[-2]
            return a._new(a._tensor.permute(*order))

        return (
            grad_output.f.matrix_multiply(grad_output, transpose(t2)),
            grad_output.f.matrix_multiply(transpose(t1), grad_output),
        )


def zeros(shape: UserShape, backend: TensorBackend = SimpleBackend) -> Tensor:
    """Creates a tensor of the specified shape, filled with zeros.

    Args:
    ----
        shape: The shape of the tensor to create.
        backend: The backend to use for creating the tensor (defaults to SimpleBackend).

    Returns:
    -------
        Tensor: A tensor filled with zeros.

    """
    return minitorch.Tensor.make(
        [0.0] * int(operators.prod(shape)), shape, backend=backend
    )


def ones(shape: UserShape, backend: TensorBackend = SimpleBackend) -> Tensor:
    """Creates a tensor of the specified shape, filled with ones.

    Args:
    ----
        shape: The shape of the tensor to create.
        backend: The backend to use for creating the tensor (defaults to SimpleBackend).

    Returns:
    -------
        Tensor: A tensor filled with ones.

    """
    return minitorch.Tensor.make(
        [1.0] * int(operators.prod(list(map(float, shape)))), shape, backend=backend
    )


def rand(
    shape: UserShape,
    backend: TensorBackend = SimpleBackend,
    requires_grad: bool = False,
) -> Tensor:
    """Generates a tensor of random values within the range [0, 1). The shape of the tensor is specified.

    Args:
    ----
        shape: The shape of the tensor to create.
        backend: The backend to use for creating the tensor (defaults to SimpleBackend).
        requires_grad: A flag indicating whether the tensor should track gradients for automatic differentiation.

    Returns:
    -------
        Tensor: A tensor of random values between 0 and 1.

    """
    vals = [
        random.random() for _ in range(int(operators.prod(list(map(float, shape)))))
    ]
    tensor = minitorch.Tensor.make(vals, shape, backend=backend)
    tensor.requires_grad_(requires_grad)
    return tensor


def _tensor(
    ls: Any,
    shape: UserShape,
    backend: TensorBackend = SimpleBackend,
    requires_grad: bool = False,
) -> Tensor:
    """Creates a tensor from a list or nested list of values, with the specified shape and backend.

    Args:
    ----
        ls: The list or nested list of values to form the tensor.
        shape: The shape of the resulting tensor.
        backend: The backend to use for creating the tensor (defaults to SimpleBackend).
        requires_grad: A flag indicating whether the tensor should track gradients for automatic differentiation.

    Returns:
    -------
        Tensor: A tensor created from the provided values.

    """
    tensor = minitorch.Tensor.make(ls, shape, backend=backend)
    tensor.requires_grad_(requires_grad)
    return tensor


def tensor(
    ls: Any, backend: TensorBackend = SimpleBackend, requires_grad: bool = False
) -> Tensor:
    """Converts a nested list (or other iterable) into a tensor. The tensor's shape is derived from the structure of the list.

    Args:
    ----
        ls: The nested list or iterable to convert into a tensor.
        backend: The backend to use for creating the tensor (defaults to SimpleBackend).
        requires_grad: A flag indicating whether the tensor should track gradients for automatic differentiation.

    Returns:
    -------
        Tensor: A tensor representation of the input list.

    """

    def shape(ls: Any) -> List[int]:
        if isinstance(ls, (list, tuple)):
            return [len(ls)] + shape(ls[0])
        else:
            return []

    def flatten(ls: Any) -> List[float]:
        if isinstance(ls, (list, tuple)):
            return [y for x in ls for y in flatten(x)]
        else:
            return [ls]

    cur = flatten(ls)
    shape2 = shape(ls)
    return _tensor(cur, tuple(shape2), backend=backend, requires_grad=requires_grad)


def grad_central_difference(
    f: Any, *vals: Tensor, arg: int = 0, epsilon: float = 1e-6, ind: UserIndex
) -> float:
    """Approximates the gradient of a function using the central difference method.

    Args:
    ----
        f: The function whose gradient is being computed.
        vals: The input tensors to the function.
        arg: The index of the tensor to differentiate with respect to (default is 0).
        epsilon: A small value used for numerical differentiation (default is 1e-6).
        ind: The specific index in the tensor to calculate the gradient for.

    Returns:
    -------
        float: The approximate gradient of the function with respect to the specified argument.

    """
    x = vals[arg]
    up = zeros(x.shape)
    up[ind] = epsilon
    vals1 = [x if j != arg else x + up for j, x in enumerate(vals)]
    vals2 = [x if j != arg else x - up for j, x in enumerate(vals)]
    delta: Tensor = f(*vals1).sum() - f(*vals2).sum()

    return delta[0] / (2.0 * epsilon)


def grad_check(f: Any, *vals: Tensor) -> None:
    """Verifies the correctness of the gradients computed by automatic differentiation.

    Args:
    ----
        f: The function whose gradients are being checked.
        vals: The input tensors to the function.

    Returns:
    -------
        None: This function raises an assertion error if any gradient check fails.

    """
    for x in vals:
        x.requires_grad_(True)
        x.zero_grad_()
    random.seed(10)
    out = f(*vals)
    out.sum().backward()
    err_msg = """

Gradient check error for function %s.

Input %s

Received derivative %f for argument %d and index %s,
but was expecting derivative %f from central difference.

"""

    for i, x in enumerate(vals):
        ind = x._tensor.sample()
        check = grad_central_difference(f, *vals, arg=i, ind=ind)
        assert x.grad is not None
        np.testing.assert_allclose(
            x.grad[ind],
            check,
            1e-2,
            1e-2,
            err_msg=err_msg % (f, vals, x.grad[ind], i, ind, check),
        )

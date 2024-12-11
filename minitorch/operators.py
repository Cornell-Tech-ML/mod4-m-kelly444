import math
from typing import Callable, List, Sequence, Union


def mul(x: float, y: float) -> float:
    """Multiplies two numbers and returns the result.

    Example: mul(2, 3) returns 6.
    """
    return x * y


def id(x: float) -> float:
    """Returns the number as it is (identity function).

    Example: id(5) returns 5.
    """
    return x


def add(x: float, y: float) -> float:
    """Adds two numbers and returns the result.

    Example: add(2, 3) returns 5.
    """
    return float(x + y)


def neg(x: float) -> float:
    """Returns the negation (opposite) of the number.

    Example: neg(3) returns -3.
    """
    return -1.0 * x


def lt(x: float, y: float) -> float:
    """Checks if x is less than y. Returns 1.0 if true, otherwise 0.0.

    Example: lt(2, 3) returns 1.0 (because 2 is less than 3).
    """
    return 1.0 if x < y else 0.0


def eq(x: float, y: float) -> float:
    """Checks if x is equal to y. Returns 1.0 if true, otherwise 0.0.

    Example: eq(2, 2) returns 1.0 (because they are equal).
    """
    return 1.0 if x == y else 0.0


def max(x: float, y: float) -> float:
    """Returns the larger of the two numbers.

    Example: max(2, 3) returns 3.
    """
    return x if x > y else y


def is_close(x: float, y: float) -> bool:
    """Checks if two numbers are "close enough" to each other (within 0.01).

    Example: is_close(2.0001, 2.0002) returns True (because the difference is small enough).
    """
    return abs(x - y) < 1e-2


def sigmoid(x: float) -> float:
    """Computes the sigmoid of a number, which is a smooth curve used in machine learning.

    The result will be a value between 0 and 1. Example: sigmoid(0) returns 0.5.
    """
    if x >= 0:
        return 1.0 / (
            1.0 + math.exp(-x)
        )  # Standard sigmoid formula for positive numbers
    else:
        return math.exp(x) / (1.0 + math.exp(x))  # Special case for negative numbers


def sigmoid_back(x: float) -> float:
    """Computes the derivative (rate of change) of the sigmoid function.

    Example: sigmoid_back(0.5) returns 0.25, which is the rate of change of the sigmoid at 0.5.
    """
    return x * (1 - x)  # The derivative of the sigmoid function


def log(x: float) -> float:
    """Computes the natural logarithm (log base e) of a number.

    Example: log(2.718) returns approximately 1.0 (since log(e) = 1).
    """
    return math.log(x)


def exp(x: float) -> float:
    """Computes the exponential of a number (e raised to the power of x).

    Example: exp(1) returns 2.718, which is e^1.
    """
    return math.exp(x)


def log_back(x: float, d: float) -> float:
    """Computes the derivative of the logarithm function.

    This is useful for backpropagation in machine learning. Example: log_back(2, 3) returns 1.5.
    """
    return d * 1.0 / x  # The derivative of the log function


def inv(x: float) -> float:
    """Returns the reciprocal (1 divided by x) of the number.

    Example: inv(2) returns 0.5.
    """
    return 1.0 / x


def inv_back(x: float, d: float) -> float:
    """Computes the derivative of the inverse function (1/x).

    This is useful in backpropagation. Example: inv_back(2, 3) returns -0.75.
    """
    return -1.0 * d / x**2.0  # The derivative of the inverse function


def relu(x: float) -> float:
    """Applies the ReLU function, which returns 0 for negative numbers and the number itself for positive ones.

    Example: relu(-2) returns 0, relu(2) returns 2.
    """
    return float(x) if x > 0.0 else 0.0


def relu_back(x: float, d: float) -> float:
    """Computes the derivative of the ReLU function.

    Returns the gradient (rate of change) for backpropagation. Example: relu_back(-2, 3) returns 0.
    """
    return d * (1.0 if x > 0.0 else 0.0)  # Derivative of ReLU


def map(fn: Callable[[float], float], x: List[float]) -> List[float]:
    """Applies a function to each element in a list of numbers.

    Example: map(squared, [1, 2, 3]) returns [1, 4, 9] (if squared is the function).
    """
    return [fn(x[i]) for i in range(len(x))]


def zipWith(
    fn: Callable[[float, float], float], x: List[float], y: List[float]
) -> List[float]:
    """Applies a function to corresponding elements from two lists and returns a new list.

    Example: zipWith(add, [1, 2, 3], [4, 5, 6]) returns [5, 7, 9].
    """
    return [fn(x[i], y[i]) for i in range(len(x))]


def reduce(
    fn: Callable[[float, float], float], x: List[float], initial: float
) -> float:
    """Reduces a list of numbers to a single number by repeatedly applying a function.

    Example: reduce(add, [1, 2, 3], 0) returns 6 (1 + 2 + 3).
    """
    for i in range(len(x)):
        initial = fn(initial, x[i])
    return initial


def addLists(x: List[float], y: List[float]) -> List[float]:
    """Adds corresponding elements from two lists of numbers.

    Example: addLists([1, 2], [3, 4]) returns [4, 6].
    """
    return zipWith(add, x, y)


def negList(x: List[float]) -> List[float]:
    """Negates all the numbers in a list.

    Example: negList([1, -2, 3]) returns [-1, 2, -3].
    """
    return map(neg, x)


def prod(x: Sequence[Union[float, int]]) -> float:
    """Computes the product of all elements in a sequence (multiplying them together).

    Example: prod([2, 3, 4]) returns 24 (2 * 3 * 4).
    """
    float_list: List[float] = [float(item) for item in x]
    return reduce(mul, float_list, 1.0)


def sum(x: List[float]) -> float:
    """Computes the sum of all elements in a list.

    Example: sum([1, 2, 3]) returns 6.
    """
    return reduce(add, x, 0.0)

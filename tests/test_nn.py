import pytest
from hypothesis import given
import minitorch
from minitorch import Tensor

from .strategies import assert_close
from .tensor_strategies import tensors


# Testing the average pooling operation for 2D tensors (images)
@pytest.mark.task4_3
@given(tensors(shape=(1, 1, 4, 4)))  # Test on a tensor with shape (1, 1, 4, 4)
def test_avg(t: Tensor) -> None:
    """Test the average pooling operation over 2D tensors (images) with different pool sizes."""

    # Perform average pooling with a 2x2 filter (window)
    out = minitorch.avgpool2d(t, (2, 2))
    # Check if the output is correct by manually averaging the first 2x2 block of the image
    assert_close(
        out[0, 0, 0, 0], sum([t[0, 0, i, j] for i in range(2) for j in range(2)]) / 4.0
    )

    # Perform average pooling with a 2x1 filter (vertical pooling)
    out = minitorch.avgpool2d(t, (2, 1))
    # Check if the output is correct by averaging the first 2x1 block
    assert_close(
        out[0, 0, 0, 0], sum([t[0, 0, i, j] for i in range(2) for j in range(1)]) / 2.0
    )

    # Perform average pooling with a 1x2 filter (horizontal pooling)
    out = minitorch.avgpool2d(t, (1, 2))
    # Check if the output is correct by averaging the first 1x2 block
    assert_close(
        out[0, 0, 0, 0], sum([t[0, 0, i, j] for i in range(1) for j in range(2)]) / 2.0
    )

    # Check gradients (to verify the backward pass works)
    minitorch.grad_check(lambda t: minitorch.avgpool2d(t, (2, 2)), t)


# Testing the max operation along different axes for 3D tensors
@pytest.mark.task4_4
@given(tensors(shape=(2, 3, 4)))  # Test on a tensor with shape (2, 3, 4)
def test_max(t: Tensor) -> None:
    """Test the max operation along different axes of a tensor."""

    # Find the maximum value along axis 0 (across the first dimension)
    out = minitorch.nn.max(t, 0)
    assert out[0, 0, 0] == max([t[z, 0, 0] for z in range(2)])  # Max across dimension 0

    # Find the maximum value along axis 1 (across the second dimension)
    out = minitorch.nn.max(t, 1)
    assert out[0, 0, 0] == max([t[0, y, 0] for y in range(3)])  # Max across dimension 1

    # Find the maximum value along axis 2 (across the third dimension)
    out = minitorch.nn.max(t, 2)
    assert out[0, 0, 0] == max([t[0, 0, x] for x in range(4)])  # Max across dimension 2


# Testing the max pooling operation for 2D tensors (images)
@pytest.mark.task4_4
@given(tensors(shape=(1, 1, 4, 4)))  # Test on a tensor with shape (1, 1, 4, 4)
def test_max_pool(t: Tensor) -> None:
    """Test the max pooling operation on 2D tensors (images) with different pool sizes."""

    # Perform max pooling with a 2x2 filter (window)
    out = minitorch.maxpool2d(t, (2, 2))
    # Check if the output is correct by manually selecting the max value from the first 2x2 block
    assert_close(
        out[0, 0, 0, 0], max([t[0, 0, i, j] for i in range(2) for j in range(2)])
    )

    # Perform max pooling with a 2x1 filter (vertical pooling)
    out = minitorch.maxpool2d(t, (2, 1))
    # Check if the output is correct by manually selecting the max value from the first 2x1 block
    assert_close(
        out[0, 0, 0, 0], max([t[0, 0, i, j] for i in range(2) for j in range(1)])
    )

    # Perform max pooling with a 1x2 filter (horizontal pooling)
    out = minitorch.maxpool2d(t, (1, 2))
    # Check if the output is correct by manually selecting the max value from the first 1x2 block
    assert_close(
        out[0, 0, 0, 0], max([t[0, 0, i, j] for i in range(1) for j in range(2)])
    )


# Testing dropout operation for tensors (used for regularization during training)
@pytest.mark.task4_4
@given(tensors())  # Test on a tensor of any shape
def test_drop(t: Tensor) -> None:
    """Test the dropout operation, which randomly zeroes out some values for regularization."""

    # Dropout with 0% probability (no dropout, should return the original tensor)
    q = minitorch.dropout(t, 0.0)
    idx = q._tensor.sample()
    assert q[idx] == t[idx]  # The output should be the same as input

    # Dropout with 100% probability (all values should be zero)
    q = minitorch.dropout(t, 1.0)
    assert q[q._tensor.sample()] == 0.0  # All values should be zero

    # Dropout with 100% probability, but with "ignore" flag (should not affect the tensor)
    q = minitorch.dropout(t, 1.0, ignore=True)
    idx = q._tensor.sample()
    assert q[idx] == t[idx]  # The output should be the same as input


# Testing the softmax function for tensors
@pytest.mark.task4_4
@given(tensors(shape=(1, 1, 4, 4)))  # Test on a tensor with shape (1, 1, 4, 4)
def test_softmax(t: Tensor) -> None:
    """Test the softmax function, which normalizes the input into probabilities (sums to 1)."""

    # Apply softmax along the last dimension (dimension 3)
    q = minitorch.softmax(t, 3)
    x = q.sum(dim=3)  # Check if the sum of the probabilities along dim 3 is 1
    assert_close(x[0, 0, 0, 0], 1.0)

    # Apply softmax along the second dimension (dimension 1)
    q = minitorch.softmax(t, 1)
    x = q.sum(dim=1)  # Check if the sum of the probabilities along dim 1 is 1
    assert_close(x[0, 0, 0, 0], 1.0)

    # Check gradients (to verify the backward pass works)
    minitorch.grad_check(lambda a: minitorch.softmax(a, dim=2), t)


# Testing the log-softmax function (logarithmic version of softmax)
@pytest.mark.task4_4
@given(tensors(shape=(1, 1, 4, 4)))  # Test on a tensor with shape (1, 1, 4, 4)
def test_log_softmax(t: Tensor) -> None:
    """Test the log-softmax function, which returns log-probabilities from the input."""

    # Apply softmax along the last dimension (dim=3)
    q = minitorch.softmax(t, 3)
    # Apply log-softmax along the same dimension and check if exp(log(x)) equals the original softmax values
    q2 = minitorch.logsoftmax(t, 3).exp()
    for i in q._tensor.indices():
        assert_close(q[i], q2[i])

    # Check gradients (to verify the backward pass works)
    minitorch.grad_check(lambda a: minitorch.logsoftmax(a, dim=2), t)

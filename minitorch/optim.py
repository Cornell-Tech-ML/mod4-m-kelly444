from typing import Sequence
from .module import Parameter
from .scalar import Scalar


class Optimizer:
    """Base class for optimization algorithms.

    It keeps track of the parameters (like weights) that need to be updated during training.
    Specific algorithms (like SGD) will use this class to update those parameters in a way that improves the model's performance.
    """

    def __init__(self, parameters: Sequence[Parameter]):
        """Sets up the optimizer with a list of parameters to update.

        parameters: A list of the things (like model weights) that will be adjusted during training.
        """
        self.parameters = parameters


class SGD(Optimizer):
    """Implementation of the Stochastic Gradient Descent (SGD) algorithm.

    SGD is a popular way to update the model's parameters to make it perform better. It uses a learning rate
    to decide how big each update should be.
    """

    def __init__(self, parameters: Sequence[Parameter], lr: float = 1.0):
        """Sets up the SGD optimizer with parameters and a learning rate.

        parameters: A list of things (like model weights) that will be updated during training.
        lr: The learning rate controls how much the parameters are changed each time.
            The default is 1.0.
        """
        super().__init__(parameters)
        self.lr = lr

    def zero_grad(self) -> None:
        """Clears out the gradients (previous updates) from the parameters.

        We do this before each step to make sure we're not using old data when updating the parameters.
        """
        for p in self.parameters:
            if p.value is None:
                continue
            # If there's a derivative (used in some cases), clear it
            if hasattr(p.value, "derivative"):
                if p.value.derivative is not None:
                    p.value.derivative = None
            # If there's a gradient (used in some cases), clear it
            if hasattr(p.value, "grad"):
                if p.value.grad is not None:
                    p.value.grad = None

    def step(self) -> None:
        """Updates the parameters based on their gradients (or derivatives) and the learning rate.

        This is where the magic happens — the parameters are adjusted to try to make the model better.
        The learning rate controls how big each update is.
        """
        for p in self.parameters:
            if p.value is None:
                continue
            # If the parameter has a derivative, use it to update
            if hasattr(p.value, "derivative"):
                if p.value.derivative is not None:
                    # Update the parameter by moving it in the right direction based on the derivative
                    p.update(Scalar(p.value.data - self.lr * p.value.derivative))
            # If the parameter has a "grad", use that instead
            elif hasattr(p.value, "grad"):
                if p.value.grad is not None:
                    # Update the parameter by moving it in the right direction based on the gradient
                    p.update(p.value - self.lr * p.value.grad)

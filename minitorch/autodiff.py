from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple, Protocol


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    """Computes the numerical derivative of a function at a specific point using the central difference method.

    The central difference is a way to approximate the derivative (rate of change) of a function at a point.
    This method calculates the function values at two points: one slightly before and one slightly after
    the given point, and then uses the difference between these two values to estimate the derivative.

    Args:
    ----
        f (Any): The function whose derivative is being computed.
        *vals (Any): The point at which to compute the derivative, represented as a list of values for the function's inputs.
        arg (int, optional): The index in `vals` where the change will happen for calculating the derivative. Default is 0.
        epsilon (float, optional): The small value to adjust the input for the central difference method. Default is 1e-6.

    Returns:
    -------
        Any: The estimated derivative of the function at the given point.

    """
    vals1 = [v for v in vals]
    vals2 = [v for v in vals]
    vals1[arg] = (
        vals1[arg] + epsilon
    )  # Increase the value of the chosen argument by a small epsilon
    vals2[arg] = (
        vals2[arg] - epsilon
    )  # Decrease the value of the chosen argument by a small epsilon
    # Return the difference between the function values divided by 2*epsilon, which approximates the derivative
    return (f(*vals1) - f(*vals2)) / (2.0 * epsilon)


variable_count = 1  # This will be used for generating unique IDs for each variable


class Variable(Protocol):
    """A protocol that defines the basic operations of a variable in a computational graph.

    A variable can be a node in a graph that computes values. Variables may depend on other variables,
    and we want to be able to calculate how changes to the variables affect each other (using derivatives).
    """

    def accumulate_derivative(self, x: Any) -> None:
        """Adds the derivative (rate of change) of this variable with respect to some value.

        This function will store the derivative of the variable in a way that is useful for backpropagation.
        """
        ...

    @property
    def unique_id(self) -> int:
        """Returns a unique identifier for the variable.

        Every variable gets a unique ID to distinguish it from other variables.
        """
        ...

    def is_leaf(self) -> bool:
        """Checks if this variable is a leaf node in the computational graph.

        A leaf variable doesn't depend on any other variables (i.e., it's an input value).
        """
        ...

    def is_constant(self) -> bool:
        """Checks if the variable is a constant, meaning its value doesn't change during computation."""
        ...

    @property
    def parents(self) -> Iterable["Variable"]:
        """Returns a list of parent variables (i.e., the variables this variable depends on).

        This is useful for backpropagation, where we need to know which variables to propagate the derivatives to.
        """
        ...

    def chain_rule(self, d: Any) -> Iterable[Tuple["Variable", Any]]:
        """Applies the chain rule to calculate how a change in this variable will affect its parents.

        The chain rule is used during backpropagation to calculate the derivatives of variables
        with respect to the final output.
        """
        ...


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """Sorts the variables in a computational graph so that each variable appears after all of its parents.

    This ensures that we compute the values in the correct order for backpropagation,
    starting from the outputs and working backward toward the inputs.

    Args:
    ----
        variable (Variable): The starting point of the graph (usually the output variable).

    Returns:
    -------
        Iterable[Variable]: A list of variables sorted topologically (in order to compute their derivatives).

    """
    order: List[Variable] = []  # The ordered list of variables we will return
    seen = set()  # A set of already visited variables to avoid circular dependencies

    def visit(var: Variable) -> None:
        """Helper function to traverse the graph and build the topological order."""
        if var.unique_id in seen or var.is_constant():
            return  # Skip if already seen or constant (constants don't need to be backpropagated)
        if (
            not var.is_leaf()
        ):  # If the variable has parents, we need to visit them first
            for m in var.parents:
                if not m.is_constant():
                    visit(m)  # Visit parents recursively
        seen.add(var.unique_id)  # Mark this variable as visited
        order.insert(0, var)  # Add this variable to the order (starting from the end)

    visit(variable)  # Start the sorting from the given variable
    return order  # Return the ordered list of variables


def backpropagate(variable: Variable, deriv: Any) -> None:
    """Propagates the derivative (rate of change) backward through the computational graph using topological sorting.

    Backpropagation updates the derivatives of all variables based on the chain rule, starting from the output
    variable and working backwards through its parents.

    Args:
    ----
        variable (Variable): The starting variable where the derivative is known (usually the output).
        deriv (Any): The derivative (gradient) of the output variable that needs to be propagated backwards.

    """
    queue = topological_sort(variable)  # Get the variables in the correct order
    derivatives = {
        variable.unique_id: deriv
    }  # Store the derivative of the output variable
    for var in queue:
        deriv = derivatives[
            var.unique_id
        ]  # Get the derivative for the current variable
        if var.is_leaf():
            var.accumulate_derivative(deriv)  # If it's a leaf, update its derivative
        else:
            # Apply the chain rule to propagate the derivative to the parents
            for v, d in var.chain_rule(deriv):
                if v.is_constant():
                    continue  # Skip constants since their derivatives are always 0
                derivatives.setdefault(
                    v.unique_id, 0.0
                )  # Ensure the derivative exists for each variable
                derivatives[v.unique_id] += (
                    d  # Update the derivative for the parent variable
                )


@dataclass
class Context:
    """Stores additional context information, like whether to compute gradients and any values saved for backward computation.

    This is used during operations that may involve backpropagation, allowing you to save necessary data
    for the backward pass (derivative calculation).
    """

    no_grad: bool = False  # If True, do not compute gradients
    saved_values: Tuple[
        Any, ...
    ] = ()  # Store values that need to be saved for the backward pass

    def save_for_backward(self, *values: Any) -> None:
        """Saves values that are needed for the backward pass.

        These values might be used to compute gradients in the backward pass. If `no_grad` is True, it won't save anything.

        Args:
        ----
            *values (Any): The values to be saved for the backward pass.

        """
        if self.no_grad:
            return  # If gradients are disabled, don't save any values
        self.saved_values = values  # Save the values for later use

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        """Returns the saved values for the backward pass.

        This is a getter that allows you to retrieve the saved values.
        """
        return self.saved_values

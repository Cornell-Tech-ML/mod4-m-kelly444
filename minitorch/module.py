from __future__ import annotations
from typing import Any, Dict, Optional, Sequence, Tuple


class Module:
    """A Module represents a unit or building block of a larger model (e.g., layers in a neural network).

    - It can contain other modules (sub-modules) and parameters (trainable values).
    - It can be set to training or evaluation mode, which affects behavior during training (e.g., dropout).
    """

    _modules: Dict[str, Module]  # Stores sub-modules (e.g., layers in a model).
    _parameters: Dict[str, Parameter]  # Stores parameters (e.g., weights and biases).
    training: bool  # Flag indicating whether the module is in training mode.

    def __init__(self) -> None:
        """Initializes a new Module with no sub-modules, no parameters, and sets it to training mode."""
        self._modules = {}  # Dictionary to store sub-modules.
        self._parameters = {}  # Dictionary to store parameters.
        self.training = True  # Module is in training mode by default.

    def modules(self) -> Sequence[Module]:
        """Returns a list of all sub-modules in this module.

        Example: If this module has a sub-module 'layer1', it will be returned in the list.
        """
        m: Dict[str, Module] = self.__dict__["_modules"]
        return list(m.values())

    def train(self) -> None:
        """Sets the module (and all its sub-modules) to training mode.

        This is important for behaviors like dropout and batch normalization that change during training.
        """
        self.training = True
        for module in self.modules():
            module.training = True

    def eval(self) -> None:
        """Sets the module (and all its sub-modules) to evaluation mode.

        This is useful when you want to disable certain behaviors like dropout during evaluation or testing.
        """
        self.training = False
        for module in self.modules():
            module.training = False

    def named_parameters(self) -> Sequence[Tuple[str, Parameter]]:
        """Returns a list of all parameters in the module and its sub-modules, with their names.

        Example: If a module has a parameter 'weight', it will return [("weight", weight)].
        """
        params = []
        for name, param in self._parameters.items():
            params.append((name, param))
        for module_name, module in self._modules.items():
            for sub_name, sub_param in module.named_parameters():
                params.append((f"{module_name}.{sub_name}", sub_param))
        return params

    def parameters(self) -> Sequence[Parameter]:
        """Returns a list of all parameters in the module and its sub-modules, without their names."""
        return [param for _, param in self.named_parameters()]

    def add_parameter(self, k: str, v: Any) -> Parameter:
        """Adds a new parameter to the module with a given name (k) and value (v).

        This can be used to add trainable variables like weights and biases.
        """
        val = Parameter(v, k)
        self.__dict__["_parameters"][k] = val
        return val

    def __setattr__(self, key: str, val: Parameter) -> None:
        """Allows setting attributes for parameters and sub-modules.

        If the value is a Parameter, it's added to the parameters dictionary.
        If the value is a Module, it's added to the sub-modules dictionary.
        """
        if isinstance(val, Parameter):
            self.__dict__["_parameters"][key] = val
        elif isinstance(val, Module):
            self.__dict__["_modules"][key] = val
        else:
            super().__setattr__(key, val)

    def __getattr__(self, key: str) -> Any:
        """Retrieves the value of an attribute, either a parameter or sub-module.

        If the key is found in the parameters or modules, it returns the corresponding value.
        """
        if key in self.__dict__["_parameters"]:
            return self.__dict__["_parameters"][key]

        if key in self.__dict__["_modules"]:
            return self.__dict__["_modules"][key]
        return None

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Allows the module to be called like a function, forwarding the arguments to the `forward` method."""
        return self.forward(*args, **kwargs)

    def __repr__(self) -> str:
        """Returns a string representation of the module, including its sub-modules.

        This is useful for debugging, to see the structure of the module and its sub-modules.
        """

        def _addindent(s_: str, numSpaces: int) -> str:
            s2 = s_.split("\n")
            if len(s2) == 1:
                return s_
            first = s2.pop(0)
            s2 = [(numSpaces * " ") + line for line in s2]
            s = "\n".join(s2)
            s = first + "\n" + s
            return s

        child_lines = []
        for key, module in self._modules.items():
            mod_str = repr(module)
            mod_str = _addindent(mod_str, 2)
            child_lines.append("(" + key + "): " + mod_str)
        lines = child_lines
        main_str = self.__class__.__name__ + "("
        if lines:
            main_str += "\n  " + "\n  ".join(lines) + "\n"
        main_str += ")"
        return main_str


class Parameter:
    """A Parameter represents a value that can be trained, such as a weight or bias in a machine learning model.

    - It holds a value (`x`) and an optional name.
    - If the value is a tensor-like object, it will automatically be set to require gradients for backpropagation.
    """

    def __init__(self, x: Any, name: Optional[str] = None) -> None:
        """Initializes a new Parameter with a value (x) and an optional name.

        If the value requires gradients (e.g., a tensor), it will be set to track gradients for optimization.
        """
        self.value = x
        self.name = name
        if hasattr(x, "requires_grad_"):
            self.value.requires_grad_(True)
            if self.name:
                self.value.name = self.name

    def update(self, x: Any) -> None:
        """Updates the value of the parameter and re-applies gradient tracking if necessary."""
        self.value = x
        if hasattr(x, "requires_grad_"):
            self.value.requires_grad_(True)
            if self.name:
                self.value.name = self.name

    def __repr__(self) -> str:
        """Returns a string representation of the parameter value."""
        return repr(self.value)

    def __str__(self) -> str:
        """Returns the string version of the parameter value."""
        return str(self.value)

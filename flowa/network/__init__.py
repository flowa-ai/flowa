"""
```
flowa: (V10.5.5)

Create fast, optimized, and easy-to-use neural networks.
```

## Installing
```shell
# Linux/macOS
python3 pip install -U flowa

# Windows
py -3 -m pip install -U flowa
```

### FastFix:
```diff
+ Made it so for hidden layers, you can have just one layer, in or not in a list/tuple.
```

# Usage
```python
import flowa.network as nk     # or import flowa as nk

nk.Seed(52) # Optional, used for testing purposes.

x = nk.Array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = nk.Array([[0], [1], [1], [0]])
```
```python
network = nk.Network(
    nk.Input(2),
    (
        nk.Hidden(4, nk.Tanh), 
        nk.Hidden(2, nk.Sigmoid)
    ),
    nk.Output(1)
)
```
```python
network.train(x, y, epoch=500)
print(network.predict(x)
```
```javascript
/* Output (800ms Average):
>>> Epoch: 0, Error: 0.4984800733120248
>>> Epoch: 50, Error: 0.49632395442760113
>>> Epoch: 100, Error: 0.4781823668945816
>>> Epoch: 150, Error: 0.35665153383154413
>>> Epoch: 200, Error: 0.1874969659672475
>>> Epoch: 250, Error: 0.12789399797698137
>>> Epoch: 300, Error: 0.10069853802998781
>>> Epoch: 350, Error: 0.08495289503359527
>>> Epoch: 400, Error: 0.07452557528756484
>>> Epoch: 450, Error: 0.06702276126613768
[[0.10447174]
 [0.94106133]
 [0.94096653]
 [0.02281434]]
*/
```

# All activations:
```javascript
/*
    "Sigmoid",
    "Tanh",
    "ReLU",
    "LeakyReLU",
    "ELU",
    "Swish",
    "Gaussion",
    "Identity",
    "BinaryStep",
    "PReLU",
    "Exponential",
    "Softplus",
    "Softsign",
    "BentIdentity",
    "ArcTan",
    "SiLU",
    "Mish",
    "HardSigmoid",
    "HardTanh",
    "SoftExponential",
    "ISRU",
    "Sine",
    "Cosine",
    "SQNL",
    "SoftClipping",
    "BentIdentity2",
    "LogLog",
    "GELU",
    "Softmin",
*/
```

# Make your own!
```python
import flowa as nk

class MyModule(nk.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        return x

    def backward(self, x, y, outputs):
        return nk.np.ones_like(x)
```

# Seeing used modules + seed.
```python
import flowa as nk

...Defining A Neural Network Here...

print(nk.modules())
```
```javascript
/* Example:
{'Input': Input(size: 2), 'Hidden': Hidden(size: 2), 'Output': Output(size: 1), 'Network': Network(
    Input Layer:
        1 Input(size: 2)

    Hidden Layers:        
        1 Hidden(size: 3)
        2 Hidden(size: 2)

    Output Layer:
        1 Output(size: 1)
)}
*/
```
```python
print(nk.seed())
# print(nk.seed(34))
```
```javascript
/* Example:
0
# 34
*/
```
"""

import numpy as np
import pickle
import inspect
from .activations import *


class _Netwk:
    modules = []
    seed = np.random.randint(1, 99999999)
    np.random.seed(seed)


def modules(*args, **kwargs):
    return _Netwk.modules


def seed(value=None, *args, **kwargs):
    if value:
        _Netwk.seed = value
        np.random.seed(_Netwk.seed)
    return _Netwk.seed


def get(name, *args, **kwargs):
    for item in _Netwk.modules:
        if item[0] == name:
            return item


def remove(name, *args, **kwargs):
    for item in _Netwk.modules:
        if item[0] == name:
            _Netwk.modules.remove(item)
            return item

    return _Netwk.modules


class Module:
    def __init__(self, *args, **kwargs) -> None:
        self.args: tuple = args
        self.kwargs: dict = kwargs
        for key, value in self.kwargs.items():
            setattr(self, key, value)

        self.name: str = self.__class__.__name__
        self.line: int = inspect.getsourcelines(self.__class__)[1]
        self.lenmethods: int = len(self.__dir__())
        self.methods: list = [
            method for method in self.__dir__() if not method.startswith("__")
        ]
        self.methods.sort()
        _Netwk.modules.append((self.name, self))

    def __str__(self) -> str:
        return f"{self.name}(line={self.line}, methods={self.lenmethods})"

    def __repr__(self) -> str:
        return self.describe

    def __call__(self, *args, **kwargs) -> object:
        return self

    @property
    def describe(self) -> str:
        mdict: dict = {
            "name": self.name,
            "line": self.line,
            "method_info": {"total": self.lenmethods, "methods": self.methods},
        }
        return str(mdict)

    @property
    def total(self) -> int:
        return self.lenmethods

    def forward(self, x, *args, **kwargs):
        return x

    def backward(self, x, y, output, *args, **kwargs):
        return 1

    def save(self, path, *args, **kwargs):
        """Save the module to a file."""
        with open(path, "wb") as f:
            pickle.dump(self, f)

        return self

    @staticmethod
    def load(path, *args, **kwargs):
        """Load a module from a file."""
        with open(path, "rb") as f:
            return pickle.load(f)


class Network(Module):
    """
    Neural Network

    Parameters:
        input_layer (Input): Input layer.
        hidden_layers (list): List of hidden layers.
        output_layer (Output): Output layer.

    Methods:
        train(x, y, epoch=1, *args, **kwargs): Train the network.
        predict(x, *args, **kwargs): Predict the output of the network.
    """

    def __init__(self, input_layer, hidden_layers, output_layer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input = input_layer
        self.input_layer = input_layer.size
        self.output = output_layer
        self.output_layer = output_layer.size

        if not isinstance(hidden_layers, (tuple, list)):
            hidden_layers = [hidden_layers]

        self.hidden_layers = hidden_layers
        self.hidden_sizes = [layer.size for layer in hidden_layers]

        self.num_layers = len(hidden_layers)

        self.weights = [None] * (self.num_layers + 1)

        self.weights[0] = np.random.randn(self.input_layer, self.hidden_sizes[0])

        for i in range(1, self.num_layers):
            self.weights[i] = np.random.randn(
                self.hidden_sizes[i - 1], self.hidden_sizes[i]
            )

        self.weights[self.num_layers] = np.random.randn(
            self.hidden_sizes[-1], self.output_layer
        )

    def sigmoid(self, x, *args, **kwargs):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x, *args, **kwargs):
        return x * (1 - x)

    def forward(self, x, *args, **kwargs):
        self.z = [None] * (self.num_layers + 1)
        self.a = [None] * (self.num_layers + 2)

        self.a[0] = x

        for i in range(1, self.num_layers + 1):
            self.z[i] = np.dot(self.a[i - 1], self.weights[i - 1])
            self.a[i] = self.hidden_layers[i - 1].forward(self.z[i])

        self.z[self.num_layers] = np.dot(
            self.a[self.num_layers], self.weights[self.num_layers]
        )
        output = self.sigmoid(self.z[self.num_layers])

        return output

    def backward(self, x, y, output, *args, **kwargs):
        self.output_error = y - output
        self.output_delta = self.output_error * self.sigmoid_derivative(output)

        self.errors = [None] * (self.num_layers + 1)
        self.deltas = [None] * (self.num_layers + 1)

        self.errors[self.num_layers] = self.output_delta
        self.deltas[self.num_layers] = self.output_delta

        for i in range(self.num_layers, 0, -1):
            self.errors[i - 1] = self.deltas[i].dot(self.weights[i].T)
            self.deltas[i - 1] = self.errors[i - 1] * self.hidden_layers[
                i - 1
            ].backward(self.a[i])

        for i in range(self.num_layers + 1):
            self.weights[i] += self.a[i].T.dot(self.deltas[i])

    def train(self, x, y, epoch, *args, verbose=True, **kwargs):
        for i in range(epoch):
            output = self.forward(x)
            self.backward(x, y, output)
            if verbose == 1 or verbose is True:
                if i % (epoch // 10) == 0:
                    print(f"Epoch: {i}, Error: {np.mean(np.abs(y - output))}")

    def predict(self, x, *args, **kwargs):
        return self.forward(x)

    def __str__(self, *args, **kwargs):
        len_hidden = len(self.hidden_sizes)
        hidden_layer = "".join(
            f"\n        {i+1} {str(self.hidden_layers[i])}" for i in range(len_hidden)
        )
        input_layer = f"1 {str(self.input)}"
        output_layer = f"1 {str(self.output)}"

        return f"Network(\n    Input Layer:\n        {input_layer}\n\n    Hidden Layers:        {hidden_layer}\n\n    Output Layer:\n        {output_layer}\n)"


class Input(Module):
    """
    Input Layer

    Parameters:
        size (int): size of input layer
    """

    def __init__(self, size=2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.size = size


class Hidden(Module):
    """
    Hidden Layer

    Parameters:
        size (int): size of hidden layer
        activation (function): activation function
    """

    def __init__(self, size=2, activation=Sigmoid, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.size = size
        self.activation_method = activation
        self.activation_args = args
        self.activation_kwargs = kwargs
        self.activation = self.activation_method(*args, **kwargs)

    def forward(self, x, *args, **kwargs):
        return self.activation.forward(x)

    def backward(self, x, *args, **kwargs):
        return self.activation.backward(x)


class Output(Module):
    """
    Output Layer

    Parameters:
        size (int): size of output layer
    """

    def __init__(self, size=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.size = size


def Array(array, *args, **kwargs):
    """Creates an array of modules."""
    return np.array(array)


def Seed(number=0, *args, **kwargs):
    """Set the seed for the random number generator."""
    np.random.seed(number)
    _Netwk.seed = number
    return number


def Random(size, *args, **kwargs):
    """Creates a random array of size `size`"""
    return np.random.randn(size)


__all__ = [
    "Module",
    "Network",
    "Input",
    "Hidden",
    "Output",
    "Array",
    "Seed",
    "Random",
    "get",
    "remove",
    "modules",
    "seed",
    "Sigmoid",
    "Tanh",
    "ReLU",
    "LeakyReLU",
    "ELU",
    "Swish",
    "Gaussion",
    "Identity",
    "BinaryStep",
    "PReLU",
    "Exponential",
    "Softplus",
    "Softsign",
    "BentIdentity",
    "ArcTan",
    "SiLU",
    "Mish",
    "HardSigmoid",
    "HardTanh",
    "SoftExponential",
    "ISRU",
    "Sine",
    "Cosine",
    "SQNL",
    "SoftClipping",
    "BentIdentity2",
    "LogLog",
    "GELU",
    "Softmin",
]

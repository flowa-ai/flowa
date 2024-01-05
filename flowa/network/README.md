<a href="https://ibb.co/885w17s](https://i.ibb.co/bdBVcKm/flowa.jpg)"><img src="https://i.ibb.co/bdBVcKm/flowa.jpg" alt="flowa" border="0" width="145"></a>

# [flowa - Machine Learning Toolkit](https://pypi.org/project/flowa)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/flowa/flowa/blob/main/LICENSE)
[![Python Versions](https://img.shields.io/badge/python-3.7%20|%203.8%20|%203.9%20|%203.10%20|%203.11%20|%203.12%20-blue)](https://www.python.org/downloads/)

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

# Links
- [Github](https://github.com/flowa-ai)
- [Project](https://pypi.org/project/flowa/)

<a href="https://ibb.co/885w17s](https://i.ibb.co/bdBVcKm/flowa.jpg)"><img src="https://i.ibb.co/bdBVcKm/flowa.jpg" alt="flowa" border="0" width="145"></a>

# [flowa - Machine Learning Toolkit](https://pypi.org/project/flowa)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/flowa/flowa/blob/main/LICENSE)
[![Python Versions](https://img.shields.io/badge/python-3.7%20|%203.8%20|%203.9%20|%203.10%20|%203.11%20|%203.12%20-blue)](https://www.python.org/downloads/)

```
flowa: (V10.5.5)

Python Machine Learning, Image Generation, Decision Trees, Label Encoders, and more!
```

## Installing
```shell
# Linux/macOS
python3 pip install -U flowa

# Windows
py -3 -m pip install -U flowa
```

(1) -> Merge the `netwk` library.
```python
# Create neural networks:
x = flowa.Array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = flowa.Array([[0], [1], [1], [0]])

network = flowa.Network(
    flowa.Input(2),
    (
        flowa.Hidden(4, flowa.Tanh), 
        flowa.Hidden(2, flowa.Sigmoid)
    ),
    flowa.Output(1)
)

network.train(x, y, epoch=1000)
print(network.predict(x))
```

(2) -> Updated the flowa.network.Module class

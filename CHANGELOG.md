<a href="https://ibb.co/885w17s](https://i.ibb.co/bdBVcKm/flowa.jpg)"><img src="https://i.ibb.co/bdBVcKm/flowa.jpg" alt="flowa" border="0" width="145"></a>

# [flowa - Decision Trees & Label Encoding](https://pypi.org/project/flowa)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/flowa/flowa/blob/main/LICENSE)
[![Python Versions](https://img.shields.io/badge/python-3.7%20|%203.8%20|%203.9%20|%203.10%20|%203.11%20|%203.12%20-blue)](https://www.python.org/downloads/)

```
flowa: (V1.1.5)

Python Machine Learning, Decision Trees, and Label Encoders.
```

## Installing
```shell
# Linux/macOS
python3 pip install -U flowa

# Windows
py -3 -m pip install -U flowa
```

(1) -> Fixed:
```javascript
flowa.Encoder.new(): // Fixed returning values
```

(2) -> Added:
```javascript
flowa.types.Map():  // Moved Map(object) to flowa.types
flowa.types.Node(): // Moved Node(object) to flowa.types
```

(3) -> Datasets:
```
flowa/datasets
    /music_data.csv
    /play_tennis.csv
```
```python
from flowa import (
    Dataset,
    read_csv,
    convert
)

dataset = Dataset.get_play_tennis()

csv = read_csv(convert(dataset))
```
```javascript
// >>>     Outlook Temperature Humidity    Wind Play Tennis
// >>> 0  Overcast        Mild   Normal    Weak         Yes
// >>> 1     Sunny        Mild   Normal    Weak         Yes
// >>> ...   [2 rows not shown]
```

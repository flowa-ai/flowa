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

# Simple Examples
```python
from flowa import (
    Encoder,
    Tree,
    Dataset,
    read_csv,
    convert
)

classifier: Tree = Tree()
encoder: Encoder = Encoder()

dataset: str = convert(Dataset.get_music_data())
csv: object = read_csv(dataset)

dataframe: object = encoder.df(csv, 'gender')

X_matrix: object = dataframe.drop('genre', axis=1).values
y_column: object = encoder(dataframe['genre'].values)

classifier.fit(X_matrix, y_column)

age, gender = encoder.new(30, 'female')
fix: list = encoder.fix(age, gender)

prediction: list[int] = classifier.predict(fix)
print(encoder.inverse(prediction))

#>>> ['Pop']

```
String Dataset to dataframe conversion:
```python
from flowa import (
    Dataset,
    read_csv,
    convert
)

dataset: Dataset = Dataset.get_play_tennis()

converted_dataset: str = convert(dataset)

csv: Dataset = read_csv(converted_dataset)
print(csv)

#>>>     Outlook Temperature Humidity    Wind Play Tennis
#>>> 0  Overcast        Mild   Normal    Weak         Yes
#>>> 1     Sunny        Mild   Normal    Weak         Yes
#>>> ... [2 rows not shown]
```

# Links
- [Github](https://github.com/flowa-ai)

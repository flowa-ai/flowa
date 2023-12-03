"""
Flowa V1.2.0

Copyright (c)     2023 flowa 
License   (Lic.)  MIT

A package for easy and simple Decision Tree Classification
Comes with Label Encoders, Decision Trees, and Node and Map objects for both the encoders and the trees.

Classes:
  - Encoder: Encodes categorical data into numerical data.
  - Tree: Represents a Decision Tree.

  - Dataset: Class for getting pre-made datasets.

  - Node: Represents a Node in a Decision Tree.
  - Map: Represents a Map in an Encoder.

Functions:
  - convert: Converts string of text into an object that can be converted into a dataframe using read_csv()
  - read_csv: Reads a CSV file into a DataFrame.

Constants:
  - PYTHON_REQUIRED_MAJOR: Major version of Python.
  - PYTHON_REQUIRED_MINOR: Minor version of Python.

Variables:
  - datasets: Dictionary of datasets.
  - version_dict: Dictionary of version information.
  - python_tuple: Tuple of the Python version.

  - __version__: Current version of Flowa.
  - __author__: Author of Flowa.
  - __email__: Email of Flowa.
  - __discord__: Discord user for Flowa.
  - __github__: Github link for Flowa.
  - __repo__: Github link for Flowa's Repository.
  - __license__: License of Flowa.
  - __copyright__: Copyright of Flowa.
  - __all__: List of all Flowa classes.

=================================================


EXAMPLE USAGE:

```python
'''
Dataset Snippet: (music_data.csv)

age,gender,genre
25,male,Rock
30,female,Pop
22,male,HipHop
28,female,Classical
'''


from flowa import (
    Encoder,
    Tree,
    read_csv,
)

encoder: Encoder = Encoder()
classifier: Tree = Tree()

csv: object = read_csv('music_data.csv')
dataframe: object = encoder.df(csv, 'gender')

X_matrix: object = dataframe.drop('genre', axis=1).values
y_column: object = encoder(dataframe['genre'].values)

classifier.fit(X_matrix, y_column)

age, gender = encoder.new(30, 'female')
fix: list = encoder.fix(age, gender)

prediction: list[int] = classifier.predict(fix)
print(encoder.inverse(prediction))

>>> ['Pop']
```

=================================================

"""

import io
import sys
import pandas

from collections import Counter

from ._version import (
    get_version,
    get_python,
)

from .types import (
    Node,
    Map,
)
from .utils import Dataset, datasets

from .main import Encoder, Tree

version_dict: dict = get_version()
python_tuple: tuple = get_python()

PYTHON_REQUIRED_MAJOR = python_tuple[0]
PYTHON_REQUIRED_MINOR = python_tuple[1]

if (
    sys.version_info.major != PYTHON_REQUIRED_MAJOR
    or sys.version_info.minor < PYTHON_REQUIRED_MINOR
):
    import warnings

    warnings.warn(
        f"Please update python to {PYTHON_REQUIRED_MAJOR}.{PYTHON_REQUIRED_MINOR} or higher."
        f"Current Version: {sys.version}",
        category=RuntimeWarning,
    )

read_csv: object = pandas.read_csv
convert: object = io.StringIO

__version__ = version_dict.get("FLOWA_VERSION")
__author__ = "flowa (Discord: @flow.a)"
__email__ = "flowa.dev@gmail.com"
__discord__ = "@flow.a"
__github__ = "https://github.com/flowa-ai"
__repo__ = "https://github.com/flowa-ai/flowa"
__license__ = "MIT"
__copyright__ = "Copyright (c) 2023 flowa"
__all__: tuple = (
    "Encoder",
    "Dataset",
    "Tree",
    "Node",
    "Map",
)

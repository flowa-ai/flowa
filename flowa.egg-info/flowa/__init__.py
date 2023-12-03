"""
Flowa V1.1.5

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

Variables:
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

__version__ = "1.1.5"
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


import io
import numpy
import pandas
import pathlib

from collections import Counter

from .types import (
    Node,
    Map,
)

read_csv: object = pandas.read_csv
convert: object = io.StringIO


datasets: dict = {
    "music_data.csv": (
        "Dataset on which genre different people of different ages and gender like.",
        "flowa.Dataset.get_music_data()",
    ),
    "play_tennis.csv": (
        "Dataset on what weather and climate conditions is appropriate to play a tennis match.",
        "flowa.Dataset.get_play_tennis()",
    ),
}


class Dataset(object):
    """
    Get pre-made datasets from the flowa/datasets folder.

    Methods:
      get_music_data()
      get_play_tennis()
    """

    def get_music_data(*args, **kwargs) -> str:
        """Get the music_data.csv dataset."""
        path_to_datasets_folder: str = (
            str(pathlib.Path(__file__).parent.absolute()) + "/datasets/"
        )
        with open(path_to_datasets_folder + "music_data.csv", "r") as file:
            return file.read()
        return "Could not gather music_data.csv"

    def get_play_tennis(*args, **kwargs) -> str:
        """Get the play_tennis.csv dataset."""
        path_to_datasets_folder: str = (
            str(pathlib.Path(__file__).parent.absolute()) + "/datasets/"
        )
        with open(path_to_datasets_folder + "play_tennis.csv", "r") as file:
            return file.read()
        return "Could not gather play_tennis.csv"


class Tree(object):
    """
    Decision Tree Classifier

    Parameter:
      min_samples_split(int, default=2): Minimum number of samples required to split an internal node.
      max_depth(int, default=100): Maximum depth of the tree. If None, the tree is unlimit
      num_features(int, default=None): Number of features to consider when looking for the best split. If None, all
      root(Node, default=None): The root of the tree.

    Methods:
      __init__: Constructs a Tree object.
      __repr__: Returns the string representation of the tree.
      __str__: Returns the string representation of the tree.
      __call__: Returns the prediction of the tree.

    Functions:
      fit: Fits the tree to the data.
      predict: Predicts the label of the data.
    """

    def __init__(
        self,
        min_samples_split: int = 2,
        max_depth: int = 100,
        num_features: int = None,
        *args,
        **kwargs,
    ) -> None:
        """Constructs a Tree object.""" ""
        self.min_samples_split: int = min_samples_split
        self.max_depth: int = max_depth
        self.num_features: int = num_features
        self.root: Node = None

    def __repr__(self, *args, **kwargs) -> str:
        """Returns the string representation of the tree.""" ""
        return f"DecisionTree(min_samples_split={self.min_samples_split}, max_depth={self.max_depth}, num_features={self.num_features})"

    def __str__(self, *args, **kwargs) -> str:
        """Returns the string representation of the tree."""
        return self.__repr__()

    def __call__(self, X_matrix: list, *args, **kwargs) -> list:
        """Returns the prediction of the tree."""
        return self.predict(X_matrix)

    def fit(self, X_matrix: list, y_column: list, *args, **kwargs) -> None:
        """Fits the tree to the data."""
        self.num_features: int = (
            X_matrix.shape[1]
            if not self.num_features
            else min(X_matrix.shape[1], self.num_features)
        )
        self.root: Node = self._grow_tree(X_matrix, y_column)

    def _grow_tree(
        self, X_matrix: list, y_column: list, depth: int = 0, *args, **kwargs
    ) -> Node:
        """Grows the tree to the data."""
        num_samples, num_feats = X_matrix.shape
        num_labels: int = len(numpy.unique(y_column))

        if (
            depth >= self.max_depth
            or num_labels == 1
            or num_samples < self.min_samples_split
        ):
            leaf_value: int = self._most_common_label(y_column)
            return Node(value=leaf_value)

        features_indexs: int = numpy.random.choice(
            num_feats, self.num_features, replace=False
        )

        best_feature, best_threshold = self._best_split(
            X_matrix, y_column, features_indexs
        )

        left_indexs, right_indexs = self._split(
            X_matrix[:, best_feature], best_threshold
        )
        left: Node = self._grow_tree(
            X_matrix[left_indexs, :], y_column[left_indexs], depth + 1
        )
        right: Node = self._grow_tree(
            X_matrix[right_indexs, :], y_column[right_indexs], depth + 1
        )
        return Node(best_feature, best_threshold, left, right)

    def _best_split(
        self, X_matrix: list, y_column: list, features_indexs: int, *args, **kwargs
    ) -> tuple:
        """Finds the best split for the data."""
        best_gain: int = -1
        split_index, split_threshold = None, None

        for feat_index in features_indexs:
            X_column: list = X_matrix[:, feat_index]
            thresholds: int = numpy.unique(X_column)

            for threshold in thresholds:
                gain: int = self._information_gain(X_column, y_column, threshold)

                if gain > best_gain:
                    best_gain: int = gain
                    split_index: int = feat_index
                    split_threshold: int = threshold

        return split_index, split_threshold

    def _information_gain(
        self, X_column: list, y_column: list, threshold: int, *args, **kwargs
    ) -> int:
        """Calculates the information gain."""
        parent_entropy: int = self._entropy(y_column)

        left_indexs, right_indexs = self._split(X_column, threshold)

        if len(left_indexs) == 0 or len(right_indexs) == 0:
            return 0

        num: int = len(y_column)
        num_left, num_right = len(left_indexs), len(right_indexs)
        entropy_left, entropy_right = self._entropy(
            y_column[left_indexs]
        ), self._entropy(y_column[right_indexs])
        child_entropy: int = (num_left / num) * entropy_left + (
            num_right / num
        ) * entropy_right

        information_gain: int = parent_entropy - child_entropy
        return information_gain

    def _split(self, X_column: list, split_threshold: int, *args, **kwargs) -> tuple:
        """Splits the data."""
        left_indexs: int = numpy.argwhere(X_column <= split_threshold).flatten()
        right_indexs: int = numpy.argwhere(X_column > split_threshold).flatten()
        return left_indexs, right_indexs

    def _entropy(self, y_column: list, *args, **kwargs) -> int:
        """Calculates the entropy."""
        hist_y_column: list = numpy.bincount(y_column)
        probabilities: int = hist_y_column / len(y_column)
        return -numpy.sum(
            [
                propbability * numpy.log(propbability)
                for propbability in probabilities
                if propbability > 0
            ]
        )

    def _most_common_label(self, y_column: list, *args, **kwargs) -> str | None:
        """Returns the most common label."""
        counter: Counter = Counter(y_column)
        if counter:
            value: str = counter.most_common(1)[0][0]
            return value
        else:
            return None

    def predict(self, X_matrix: list, *args, **kwargs) -> list:
        """Predicts the labels."""
        return numpy.array(
            [self._traverse_tree(matrix, self.root) for matrix in X_matrix]
        )

    def _traverse_tree(self, matrix: list, node: Node, *args, **kwargs) -> int:
        """Traverses the tree."""
        if node.is_leaf_node():
            return node.value

        if int(matrix[node.feature]) <= int(node.threshold):
            return self._traverse_tree(matrix, node.left)
        return self._traverse_tree(matrix, node.right)


class Encoder(object):
    """
    A Label Encoder class for the Flowa Decision Tree.

    Attributes:
      labels: A dictionary mapping the label to the encoded label.
      map: A Map object of the label encoder.
      next_label: The next encoded label.
      inverse: Encoder.inverse_transform()
      dataframe: Encoder.df()

    Methods:
      __init__: Constructs a Label Encoder object.
      __repr__: Returns the string representation of the label encoder.
      __str__: Returns the string representation of the label encoder.
      __call__: Returns the value of the label encoder.

    Functions:
      fit: Fits the label encoder.
      transform: Transforms the label.
      fit_transform: Fits and transforms the label.
      inverse_transform: Transforms the encoded label to the original label.
      df: Returns the dataframe of the label encoder.
      new: Returns a encoded labels or creates new labels for unknown labels.
      fix: Fix the labels inputted for the Tree.predict() method.
    """

    def __init__(self, *args, **kwargs) -> None:
        """Constructs a Label Encoder object."""
        self.labels: dict = {}
        self.map: Map = Map
        self.next_label: int = 0
        self.inverse: object = self.inverse_transform
        self.dataframe: object = self.df

    def __repr__(self, *args, **kwargs) -> str:
        """Returns the string representation of the label encoder."""
        self.map: Map = Map(self.labels)
        return f"Encoder(map={self.map}, next_label={self.next_label})"

    def __str__(self, *args, **kwargs) -> str:
        """Returns the string representation of the label encoder."""
        return self.__repr__()

    def __call__(self, y_column: list) -> dict:
        """Fit and transforms the input"""
        return self.fit_transform(y_column)

    def df(self, df: list, *args, as_str: bool = False):
        """Returns the dataframe of the label encoder."""
        try:
            for arg in args:
                if as_str:
                    df[[*args]] = df[[*args]].astype(str)
                df[arg]: list = self.fit_transform(df[arg])
                self.map: Map = Map(self.labels)
            return df
        except Exception:
            raise TypeError(
                "\n\n\n\nConsider trying df(*, as_str=True) to bypass this error."
            )

    def new(self, *labels, **kwargs) -> tuple:
        """Returns a encoded labels or creates new labels for unknown labels."""
        returns: list = []
        for label in labels:
            if not isinstance(label, str):
                returns.append(label)
            else:
                if label not in self.labels:
                    self.labels[label]: int = self.next_label
                    self.next_label += 1
                    self.map: Map = Map(self.labels)
                returns.append(self.labels[label])

        return tuple(returns) if len(returns) > 1 else returns[0]

    def fix(self, *args, **kwargs) -> list:
        """Fix the labels inputted for the Tree.predict() method."""
        args_updated: list = []
        for arg in args:
            if type(arg) == float or type(arg) == int:
                args_updated.append(arg)
            else:
                args_updated.append(self.transform([arg]))

        self.map: Map = Map(self.labels)
        return [args_updated]

    def fit(self, y_column: list, *args, **kwargs) -> Map:
        """Fits the label encoder."""
        labels: list = numpy.unique(y_column)
        for label in labels:
            if label not in self.labels:
                self.labels[label]: int = self.next_label
                self.next_label += 1
        self.map: Map = Map(self.labels)

    def fit_transform(self, y_column: list, *args, **kwargs) -> list:
        """Fits and transforms the label."""
        self.fit(y_column)
        self.map: Map = Map(self.labels)
        return self.transform(y_column)

    def transform(self, y_column: list, *args, **kwargs) -> list:
        """Transforms the label."""
        self.map: Map = Map(self.labels)
        return numpy.array([self.labels[label] for label in y_column])

    def inverse_transform(self, y_column: list, *args, **kwargs) -> list:
        """Inverse Transforms the label."""
        inverse_map: dict = {value: key for key, value in self.labels.items()}
        self.map: Map = Map(self.labels)
        return numpy.array([inverse_map[label] for label in y_column])

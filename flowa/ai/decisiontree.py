import numpy

from collections import Counter
from ..types import Node


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

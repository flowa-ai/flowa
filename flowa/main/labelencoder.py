import numpy

from ..types import Map


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

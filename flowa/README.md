Flowa V1.1.5

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

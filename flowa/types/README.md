flowa.types

Classes:
    - Node: Represents a Node in a Decision Tree.
    - Map: Represents a Map in an Encoder.


-> flowa.types.Node(object, *, **)
-> flowa.types.Map(object, *, **)

# Node:
    A Node class for the Flowa Decision Tree.
    
    Attributes:
      value: The value of the node.
      feature: The feature used to split the node.
      threshold: The threshold used to split the node.
      left: The left child of the node.
      right: The right child of the node.
    
    Methods:
      __init__: Constructs a Node object.
      __repr__: Returns the string representation of the node.
      __str__: Returns the string representation of the node.
      __call__: Returns the value of the node.
    
    Functions:
      is_leaf_node: Returns True if the node is a leaf node.
      children: Returns the children of the node.



# Map:
    A Map class for the Flowa Decision Tree.
    
    Attributes:
      labels: The map of the label encoder.
    
    Methods:
      __init__: Constructs a Map object.
      __repr__: Returns the string representation of the map.
      __str__: Returns the string representation of the map.
      __call__: Returns the value of the map.

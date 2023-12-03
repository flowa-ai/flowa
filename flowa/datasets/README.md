Flowa pre-made datasets
/flowa/datasets

# Current Datasets (as of V1.1.5)
  -> music_data.csv (age,gender,genre)
      SAMPLE:
       * 48,female,HipHop
       * 53,female,R&B
       * 56,male,Classical

  -> play_tenis.csv (Outlook,Temperature,Humidity,Wind,Play Tennis)
      SAMPLE:
       * Sunny,Mild,Normal,Weak,Yes
       * Rain,Cool,High,Strong,No

## To add datasets to the next version, please:

  1. -> Fork the repository / Make a pull request with your added dataset.
  2. -> After reviewing the dataset, it may or may not be added depending on the feedback.



### Using these datasets in code:

SAMPLE:
```python
from flowa import (
    Dataset,
    read_csv,
    convert
)

dataset = Dataset.get_play_tennis()

csv = read_csv(convert(dataset))
```

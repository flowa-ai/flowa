import pathlib

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
        path_to_datasets_folder: str = str(
            pathlib.Path(__file__).parent.absolute()
        ).replace("utils", "datasets/")
        with open(path_to_datasets_folder + "music_data.csv", "r") as file:
            return file.read()
        return "Could not gather music_data.csv"

    def get_play_tennis(*args, **kwargs) -> str:
        """Get the play_tennis.csv dataset."""
        path_to_datasets_folder: str = str(
            pathlib.Path(__file__).parent.absolute()
        ).replace("utils", "datasets/")
        with open(path_to_datasets_folder + "play_tennis.csv", "r") as file:
            return file.read()
        return "Could not gather play_tennis.csv"

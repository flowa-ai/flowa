"""
Flowa Image Generation

This module contains the ImageModel class, which is used to generate images using the pollinations API.

Example usage:
from flowa import ImageModel

model: ImageModel[object] = ImageModel()
image: ImageModel[str] = model.generate(prompt='a cat', model='pixart', width=512, height=512, logo=False).save('some-file.png')
"""

import random, requests

from ..types import Image


class ImageModel:
    """
    ImageModel class

    This class is used to generate images using the pollinations API.

    Parameters:
      default_model (str): The default model to use for generating images.
      default_width (int): The default width of the generated images.
      default_height (int): The default height of the generated images.
      default_logo (bool): Whether to include a logo in the generated images.

    Attributes:
      model (str): The current model being used for generating images.
      models (dict): A dictionary of available models.
      size (dict): The current size of the generated images.
      sizes (list): A list of available sizes for the generated images.
      logo (bool): Whether to include a logo in the generated images.

    Methods:
        __init__: Initializes the ImageModel class.
        __class_getitem__: For type hints.
        __call__: Updates the parameters
        update: Updates the parameters
        set_sizes: Sets the available sizes for the generated images.
        set_size: Sets the current size of the generated images.
        get_sizes: Gets the available sizes for the generated images.
        get_size: Gets the current size of the generated images.
        generate: Generates an image using the pollinations API.
    """

    def __init__(
        self,
        default_model: str = "pixart",
        default_width: int = 512,
        default_height: int = 512,
        default_logo: bool = True,
        *args,
        **kwargs,
    ) -> None:
        """Initializes the ImageModel class."""
        self.model: str = default_model
        self.models: dict = {
            "pixart": "PixArt image generation model",
            "deliberate": "Deliberate image generation model",
            "dreamshaper": "Dreamshaper image generation model",
            "turbo": "Turbo image generation model",
        }
        self.size: dict = {"width": default_width, "height": default_height}
        self.sizes: list = [{"large": {}}, {"medium": {}}, {"small": {}}]
        self.logo: bool = default_logo

    def __class_getitem__(cls, item: object) -> object:
        return object

    def __call__(
        self,
        *,
        model: str = "pixart",
        width: int = 512,
        height: int = 512,
        logo: bool = True,
        **kwargs,
    ) -> object:
        """Update model parameters."""
        return self.update(model=model, width=width, height=height, logo=logo)

    @property
    def PixArt(self) -> str:
        return "pixart"

    @property
    def Deliberate(self) -> str:
        return "deliberate"

    @property
    def Dreamshaper(self) -> str:
        return "dreamshaper"

    @property
    def Turbo(self) -> str:
        return "turbo"

    def update(
        self,
        *,
        model: str = "pixart",
        width: int = 512,
        height: int = 512,
        logo: bool = True,
        **kwargs,
    ) -> object:
        """Update model parameters."""
        self.model: str = model
        self.size: dict = {"width": width, "height": height}
        self.logo: bool = logo
        return self

    def set_sizes(self, large: dict, medium: dict, small: dict) -> None:
        """Set the available sizes for the generated images."""
        self.sizes: list = [{"large": large}, {"medium": medium}, {"small": small}]
        return self.sizes

    def set_size(self, label: str, sizes: dict) -> None:
        """Set the current size of the generated images."""
        sizes_index: dict = {"large": 0, "medium": 1, "small": 2}
        self.sizes[sizes_index[label]] = sizes
        return self.sizes[sizes_index[label]]

    def get_sizes(self) -> list:
        """Get the available sizes for the generated images."""
        return self.sizes

    def get_size(self, label: str) -> dict:
        """Get the current size of the generated images."""
        sizes_index: dict = {"large": 0, "medium": 1, "small": 2}
        return self.sizes[sizes_index[label]]

    def generate(
        self,
        prompt: str,
        *,
        model: str = None,
        width: int = None,
        height: int = None,
        logo: bool = None,
        seed: int = None,
        **kwargs,
    ) -> str:
        """Generate an image using the pollinations API.

        Parameters:
          prompt (str): The prompt to use for generating the image.

        Optional:
            model (str): The model to use for generating the image.
            width (int): The width of the generated image.
            height (int): The height of the generated image.
            logo (bool): Whether to include a logo in the generated image.
            seed (int): The seed to use for generating the image.
        """
        update_params = {
            "model": model if model else self.model,
            "width": width if width else self.size["width"],
            "height": height if height else self.size["height"],
            "logo": logo if logo else self.logo,
            "seed": seed if seed else random.randint(0, 1000000000),
        }
        self.update(
            model=update_params["model"],
            width=update_params["width"],
            height=update_params["height"],
            logo=update_params["logo"],
        )

        image: object = self.__generate_helper(prompt, **update_params)
        return image

    def __generate_helper(self, prompt, model, width, height, logo, seed) -> str:
        """Generation helper."""
        url: str = f"https://image.pollinations.ai/prompt/{prompt}"

        image: object = requests.get(
            url,
            params={
                "model": model,
                "width": width,
                "height": height,
                "nologo": logo,
                "seed": seed,
            },
        )
        return Image(prompt, image.url, model, width, height, seed, logo, image.content)

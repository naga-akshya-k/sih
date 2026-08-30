"""
Image preprocessing for Digepath GI Foundation Model.
"""

from typing import Union
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T

# Digepath standard input size and normalization
IMAGE_SIZE = 224
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]


def get_digepath_transform() -> T.Compose:
    """
    Returns standard PyTorch torchvision transform for Digepath ViT-L/16.
    """
    return T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ])


def preprocess_image(
    image_input: Union[str, Path, Image.Image, np.ndarray]
) -> torch.Tensor:
    """
    Preprocesses an input image into a standardized 4D tensor [1, 3, 224, 224].

    Args:
        image_input: Filepath, PIL Image, or Numpy array (RGB or BGR).

    Returns:
        torch.Tensor of shape [1, 3, 224, 224] with Digepath normalization.
    """
    if isinstance(image_input, (str, Path)):
        path = Path(image_input)
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {path}")
        pil_img = Image.open(path).convert("RGB")
    elif isinstance(image_input, np.ndarray):
        if image_input.ndim == 2:  # Grayscale
            pil_img = Image.fromarray(image_input).convert("RGB")
        elif image_input.shape[2] == 4:  # RGBA
            pil_img = Image.fromarray(image_input).convert("RGB")
        else:
            pil_img = Image.fromarray(image_input)
    elif isinstance(image_input, Image.Image):
        pil_img = image_input.convert("RGB")
    else:
        raise TypeError(f"Unsupported image input type: {type(image_input)}")

    transform = get_digepath_transform()
    tensor = transform(pil_img)  # [3, 224, 224]
    return tensor.unsqueeze(0)  # [1, 3, 224, 224]

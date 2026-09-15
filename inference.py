"""Shared CPU inference, preserving pet-segmentation's predict.py transforms."""
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, UnidentifiedImageError
import torch
from torch.nn import functional as F
from torchvision.transforms import functional as TF, InterpolationMode

from model import PetUNet

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
MAX_IMAGE_PIXELS = 8_000_000


class InvalidImage(ValueError):
    """An upload cannot be safely decoded as a supported photo."""


def decode_image(stream):
    """Decode by content, not filename; bound decoded memory before loading pixels."""
    try:
        with Image.open(stream) as image:
            if image.format not in {"JPEG", "PNG", "WEBP"}:
                raise InvalidImage("Please upload a JPEG, PNG, or WebP image.")
            if image.width * image.height > MAX_IMAGE_PIXELS:
                raise InvalidImage("Image is too large. Please use at most 8 megapixels.")
            if getattr(image, "is_animated", False):
                raise InvalidImage("Please upload a still image instead of an animation.")
            # Matches predict.py: no EXIF transpose, crop, or alpha compositing.
            return image.convert("RGB")
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise InvalidImage("Image is too large. Please use at most 8 megapixels.") from exc
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        if isinstance(exc, InvalidImage):
            raise
        raise InvalidImage("This file could not be read as an image. Try another photo.") from exc


def preprocess(image, size):
    """RGB PIL bilinear resize -> CHW float [0,1] -> ImageNet normalization."""
    image = image.convert("RGB")
    tensor = TF.to_tensor(TF.resize(image, [size, size], InterpolationMode.BILINEAR))
    return TF.normalize(tensor, MEAN, STD).unsqueeze(0)


@dataclass
class Prediction:
    original: Image.Image
    mask: Image.Image
    overlay: Image.Image


class Segmenter:
    def __init__(self, checkpoint_path):
        path = Path(checkpoint_path).expanduser()
        if not path.is_file():
            raise ValueError(f"Checkpoint does not exist: {path}")
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            if not isinstance(checkpoint, dict):
                raise ValueError("Expected a checkpoint dictionary")
            config = checkpoint.get("config")
            size = config.get("size") if isinstance(config, dict) else None
            # Fail rather than silently changing the trained resolution on this small VM.
            if type(size) is not int or not 64 <= size <= 512:
                raise ValueError("checkpoint['config']['size'] must be an integer from 64 to 512")
            if not isinstance(checkpoint.get("model"), dict):
                raise ValueError("Expected checkpoint['model'] to contain a state dictionary")
            self.model = PetUNet(pretrained=False).cpu()
            self.model.load_state_dict(checkpoint["model"], strict=True)
            self.model.eval()
            self.size = size
        except Exception as exc:
            raise ValueError(f"Cannot load PetUNet checkpoint {path}: {exc}") from exc

    @torch.inference_mode()
    def predict(self, stream):
        image = decode_image(stream)
        tensor = preprocess(image, self.size)
        logits = F.interpolate(
            self.model(tensor).float(), size=(image.height, image.width),
            mode="bilinear", align_corners=False,
        )
        mask = (logits.sigmoid() >= 0.5).squeeze(0).byte().cpu() * 255
        mask_image = TF.to_pil_image(mask)
        tint = Image.blend(image, Image.new("RGB", image.size, (0, 255, 0)), 0.4)
        overlay = Image.composite(tint, image, mask_image)
        return Prediction(image, mask_image, overlay)

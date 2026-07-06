from io import BytesIO

from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def clamp_unit(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x, 0.0, 1.0)


def feature_squeeze(x: torch.Tensor, bit_depth: int = 5) -> torch.Tensor:
    levels = float(2**bit_depth - 1)
    return torch.round(clamp_unit(x) * levels) / levels


def jpeg_compress_batch(x: torch.Tensor, quality: int = 75) -> torch.Tensor:
    device = x.device
    cpu_batch = x.detach().cpu()
    restored = []

    for image in cpu_batch:
        pil_image = transforms_to_pil(image)
        buffer = BytesIO()
        pil_image.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        compressed = Image.open(buffer).convert("RGB")
        restored.append(pil_to_tensor(compressed))

    return torch.stack(restored, dim=0).to(device)


def transforms_to_pil(x: torch.Tensor) -> Image.Image:
    array = x.mul(255.0).round().byte().permute(1, 2, 0).numpy()
    return Image.fromarray(array)


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32)
    return torch.from_numpy(array).permute(2, 0, 1) / 255.0


def apply_preprocess(x: torch.Tensor, defense: str, **kwargs) -> torch.Tensor:
    key = defense.lower()
    if key == "none":
        return x
    if key == "jpeg":
        return jpeg_compress_batch(x, quality=kwargs.get("jpeg_quality", 75))
    if key == "squeeze":
        return feature_squeeze(x, bit_depth=kwargs.get("bit_depth", 5))
    raise ValueError(f"unknown defense: {defense}")


class PreprocessWrapper(nn.Module):
    def __init__(self, model: nn.Module, defense: str, **kwargs):
        super().__init__()
        self.model = model
        self.defense = defense
        self.kwargs = kwargs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = apply_preprocess(x, self.defense, **self.kwargs)
        return self.model(x)


def bpda_identity_backward(x: torch.Tensor, defense: str, **kwargs) -> torch.Tensor:
    if defense == "none":
        return x
    processed = apply_preprocess(x, defense, **kwargs)
    return x + (processed - x).detach()


def logits_with_preprocess(
    model: nn.Module, x: torch.Tensor, defense: str = "none", bpda: bool = False, **kwargs
) -> torch.Tensor:
    if bpda:
        x = bpda_identity_backward(x, defense, **kwargs)
    else:
        x = apply_preprocess(x, defense, **kwargs)
    return model(x)


def cross_entropy_with_preprocess(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    defense: str = "none",
    bpda: bool = False,
    **kwargs,
) -> torch.Tensor:
    logits = logits_with_preprocess(model, x, defense=defense, bpda=bpda, **kwargs)
    return F.cross_entropy(logits, y)

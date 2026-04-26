"""
Shared image-noise presets for robustness experiments.

These presets are model-agnostic and should be applied to document images
before each model's own preprocessing so comparisons stay fair.
"""

from __future__ import annotations

import io
from copy import deepcopy
from typing import Any

from PIL import Image, ImageFilter


NOISE_PRESETS: dict[str, dict[str, Any]] = {
    "clean": {
        "rotation_degrees": 0.0,
        "blur_radius": 0.0,
        "jpeg_quality": None,
        "scale_factor": 1.0,
    },
    "light": {
        "rotation_degrees": 2.0,
        "blur_radius": 0.6,
        "jpeg_quality": 85,
        "scale_factor": 0.95,
    },
    "medium": {
        "rotation_degrees": 5.0,
        "blur_radius": 1.2,
        "jpeg_quality": 65,
        "scale_factor": 0.8,
    },
    "heavy": {
        "rotation_degrees": 9.0,
        "blur_radius": 2.0,
        "jpeg_quality": 45,
        "scale_factor": 0.65,
    },
}


def get_noise_config(noise_setting: str) -> dict[str, Any]:
    if noise_setting not in NOISE_PRESETS:
        available = ", ".join(sorted(NOISE_PRESETS))
        raise ValueError(
            f"Unknown noise setting '{noise_setting}'. Available: {available}"
        )
    return deepcopy(NOISE_PRESETS[noise_setting])


def apply_noise(image: Image.Image, noise_config: dict[str, Any]) -> Image.Image:
    """
    Apply a deterministic noise configuration to a PIL image.
    """
    result = image.convert("RGB")

    rotation = float(noise_config.get("rotation_degrees", 0.0) or 0.0)
    blur_radius = float(noise_config.get("blur_radius", 0.0) or 0.0)
    jpeg_quality = noise_config.get("jpeg_quality")
    scale_factor = float(noise_config.get("scale_factor", 1.0) or 1.0)

    if scale_factor < 1.0:
        original_size = result.size
        resized = (
            max(1, int(original_size[0] * scale_factor)),
            max(1, int(original_size[1] * scale_factor)),
        )
        result = result.resize(resized, Image.Resampling.BILINEAR)
        result = result.resize(original_size, Image.Resampling.BILINEAR)

    if rotation:
        result = result.rotate(rotation, resample=Image.Resampling.BILINEAR)

    if blur_radius > 0.0:
        result = result.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    if jpeg_quality is not None:
        buffer = io.BytesIO()
        result.save(buffer, format="JPEG", quality=int(jpeg_quality))
        buffer.seek(0)
        result = Image.open(buffer).convert("RGB")

    return result

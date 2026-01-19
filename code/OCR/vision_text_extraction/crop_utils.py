from __future__ import annotations

from pathlib import Path
from typing import Tuple

from PIL import Image


def crop_with_percentages(
    image_path: Path,
    output_path: Path,
    bbox: Tuple[float, float, float, float],
) -> None:
    """Crop an image using percentage-based bounding box coordinates."""
    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    left_pct, top_pct, width_pct, height_pct = bbox
    if not all(0 <= v <= 1 for v in (left_pct, top_pct, width_pct, height_pct)):
        raise ValueError("All bounding box percentages must be between 0 and 1")

    with Image.open(image_path) as img:
        width, height = img.size
        left = int(round(left_pct * width))
        top = int(round(top_pct * height))
        right = int(round((left_pct + width_pct) * width))
        bottom = int(round((top_pct + height_pct) * height))

        left = max(0, min(left, width - 1))
        top = max(0, min(top, height - 1))
        right = max(left + 1, min(right, width))
        bottom = max(top + 1, min(bottom, height))

        crop = img.crop((left, top, right, bottom))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        crop.save(output_path)

        print(
            f"Cropped region saved to {output_path} ("
            f"left={left}px, top={top}px, width={right-left}px, height={bottom-top}px)"
        )

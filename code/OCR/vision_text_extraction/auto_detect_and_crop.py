"""Detect a person's bounding box with a small vision model and crop the image."""

from __future__ import annotations

import argparse
import base64
import json
import re
from pathlib import Path
from typing import Tuple

import requests

from crop_utils import crop_with_percentages

OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_PROMPT = (
    "Identify the bounding box that tightly contains the main person in this image. "
    "Return ONLY JSON with this structure: {\"bbox\":{\"left\":L,\"top\":T,\"width\":W,\"height\":H}} "
    "where L, T, W, H are decimal fractions between 0 and 1 (e.g., 0.25 = 25%). "
    "left (L) = fraction of total width from the left edge to the person's left side. "
    "top (T) = fraction of total height from the top edge to the top of the head. "
    "width (W) = fraction of the total width spanned by the person. "
    "height (H) = fraction of the total height spanned by the person. "
    "Replace L/T/W/H with the real decimal numbers for THIS image. "
    "If you output placeholders or repeat this instruction, the result is invalid. "
    "No prose, captions, or markdown—return the bare JSON only."
)


def encode_image_to_base64(image_path: Path) -> str:
    with image_path.open("rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def request_bbox(image_path: Path, model: str, prompt: str) -> Tuple[float, float, float, float]:
    image_b64 = encode_image_to_base64(image_path)
    payload = {
        "model": model,
        "prompt": prompt,
        "images": [image_b64],
        "stream": False,
    }
    response = requests.post(OLLAMA_URL, json=payload, timeout=300)
    response.raise_for_status()
    body = response.json()
    raw = body.get("response", "")
    return parse_bbox(raw)


def parse_bbox(raw: str) -> Tuple[float, float, float, float]:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json|JSON)?", "", text)
        text = re.sub(r"```$", "", text).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise ValueError(f"Model response was not valid JSON: {text}")
        data = json.loads(match.group(0))

    bbox = data.get("bbox", data)
    required = ("left", "top", "width", "height")
    if not all(k in bbox for k in required):
        raise ValueError(f"Missing keys in bbox JSON: {bbox}")

    values = tuple(float(bbox[key]) for key in required)
    if not all(0 <= v <= 1 for v in values):
        raise ValueError(f"Bounding box values outside 0-1 range: {values}")
    return values  # type: ignore[return-value]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Auto-detect person bbox and crop")
    parser.add_argument("--input", required=True, help="Input relationship-view image")
    parser.add_argument("--output", required=True, help="Path for cropped result")
    parser.add_argument(
        "--model",
        default="llava:latest",
        help="Ollama vision model to use (default: llava:latest)",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Override the detection prompt",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_path = Path(args.input)
    output_path = Path(args.output)

    bbox = request_bbox(image_path, args.model, args.prompt)
    print(f"Model returned bbox (left, top, width, height) = {bbox}")
    crop_with_percentages(image_path, output_path, bbox)


if __name__ == "__main__":
    main()

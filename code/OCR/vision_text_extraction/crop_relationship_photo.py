"""Utility script to crop the photo region from relationship view images.

Default crop coordinates come from an LLM-estimated bounding box for
`relationship_000_student_view.png` (left=2%, top=18%, width=25%, height=73%).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

from crop_utils import crop_with_percentages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crop a relationship view image")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the input relationship view image",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path for the cropped image",
    )
    parser.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        metavar=("LEFT", "TOP", "WIDTH", "HEIGHT"),
        default=(0.02, 0.18, 0.25, 0.73),
        help="Bounding box percentages as decimals (default: 0.02 0.18 0.25 0.73)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    crop_with_percentages(input_path, output_path, tuple(args.bbox))


if __name__ == "__main__":
    main()

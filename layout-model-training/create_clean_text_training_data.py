#!/usr/bin/env python3
"""Prepare cleaned text-only training data.

This script:
1) Uses all images currently present on disk
2) Recreates the medium 80/20 train/val split from annotations.json
3) Builds a text-only dataset
4) Removes suspicious text boxes that are likely page-level / portrait-containing mistakes
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create cleaned text-only training data")
    parser.add_argument(
        "--annotations",
        default="layout-model-training/data/yearbook/annotations.json",
        help="Full annotations JSON",
    )
    parser.add_argument(
        "--images-dir",
        default="layout-model-training/data/yearbook/images",
        help="Directory containing downloaded images",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split")
    parser.add_argument(
        "--max-image-fraction",
        type=float,
        default=0.05,
        help="Drop text boxes covering at least this fraction of the image when portrait-like",
    )
    parser.add_argument(
        "--min-portrait-aspect",
        type=float,
        default=0.35,
        help="Lower bound for portrait-like aspect ratio w/h",
    )
    parser.add_argument(
        "--max-portrait-aspect",
        type=float,
        default=1.30,
        help="Upper bound for portrait-like aspect ratio w/h",
    )
    parser.add_argument(
        "--photo-overlap-iou",
        type=float,
        default=0.50,
        help="Drop text boxes whose IoU with a photo box is at least this threshold",
    )
    return parser.parse_args()


def iou(box_a: list[float], box_b: list[float]) -> float:
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    ix1 = max(ax, bx)
    iy1 = max(ay, by)
    ix2 = min(ax + aw, bx + bw)
    iy2 = min(ay + ah, by + bh)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def split_existing_images(data: dict[str, Any], images_dir: Path, seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
    existing_images = [img for img in data["images"] if (images_dir / img["file_name"]).exists()]
    random.seed(seed)
    random.shuffle(existing_images)

    split_idx = int(len(existing_images) * 0.8)
    train_images = existing_images[:split_idx]
    val_images = existing_images[split_idx:]

    train_ids = {img["id"] for img in train_images}
    val_ids = {img["id"] for img in val_images}

    train_data = {
        "images": train_images,
        "annotations": [ann for ann in data["annotations"] if ann["image_id"] in train_ids],
        "categories": data["categories"],
    }
    val_data = {
        "images": val_images,
        "annotations": [ann for ann in data["annotations"] if ann["image_id"] in val_ids],
        "categories": data["categories"],
    }
    return train_data, val_data


def build_clean_text_dataset(
    split_data: dict[str, Any],
    max_image_fraction: float,
    min_portrait_aspect: float,
    max_portrait_aspect: float,
    photo_overlap_iou: float,
) -> tuple[dict[str, Any], Counter[str]]:
    images_by_id = {img["id"]: img for img in split_data["images"]}
    anns_by_image: dict[int, dict[int, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for ann in split_data["annotations"]:
        anns_by_image[ann["image_id"]][ann["category_id"]].append(ann)

    kept_annotations: list[dict[str, Any]] = []
    kept_image_ids: set[int] = set()
    removed_reasons: Counter[str] = Counter()

    for image_id, img in images_by_id.items():
        photos = anns_by_image[image_id].get(1, [])
        text_boxes = anns_by_image[image_id].get(2, [])
        image_area = img["width"] * img["height"]

        for ann in text_boxes:
            x, y, width, height = ann["bbox"]
            if width <= 0 or height <= 0 or image_area <= 0:
                removed_reasons["invalid_bbox"] += 1
                continue

            area_fraction = (width * height) / image_area
            aspect_ratio = width / height
            max_overlap = max((iou(ann["bbox"], photo["bbox"]) for photo in photos), default=0.0)

            is_large_portrait_like = (
                min_portrait_aspect <= aspect_ratio <= max_portrait_aspect
                and area_fraction >= max_image_fraction
            )
            overlaps_photo = max_overlap >= photo_overlap_iou

            if is_large_portrait_like:
                removed_reasons["large_portrait_like"] += 1
            if overlaps_photo:
                removed_reasons["photo_overlap"] += 1
            if is_large_portrait_like or overlaps_photo:
                continue

            clean_ann = dict(ann)
            clean_ann["category_id"] = 1
            kept_annotations.append(clean_ann)
            kept_image_ids.add(image_id)

    kept_images = [img for img in split_data["images"] if img["id"] in kept_image_ids]
    clean_data = {
        "images": kept_images,
        "annotations": kept_annotations,
        "categories": [{"id": 1, "name": "text"}],
    }
    return clean_data, removed_reasons


def main() -> None:
    args = parse_args()
    annotations_path = Path(args.annotations)
    images_dir = Path(args.images_dir)
    output_dir = annotations_path.parent

    data = load_json(annotations_path)
    train_data, val_data = split_existing_images(data, images_dir, args.seed)

    write_json(output_dir / "annotations-train-medium.json", train_data)
    write_json(output_dir / "annotations-val-medium.json", val_data)

    clean_train, train_removed = build_clean_text_dataset(
        train_data,
        args.max_image_fraction,
        args.min_portrait_aspect,
        args.max_portrait_aspect,
        args.photo_overlap_iou,
    )
    clean_val, val_removed = build_clean_text_dataset(
        val_data,
        args.max_image_fraction,
        args.min_portrait_aspect,
        args.max_portrait_aspect,
        args.photo_overlap_iou,
    )

    write_json(output_dir / "annotations-text-only-clean-train.json", clean_train)
    write_json(output_dir / "annotations-text-only-clean-val.json", clean_val)

    print("✅ Prepared medium split from currently downloaded images")
    print(f"   Train images: {len(train_data['images'])} | annotations: {len(train_data['annotations'])}")
    print(f"   Val images:   {len(val_data['images'])} | annotations: {len(val_data['annotations'])}")
    print()
    print("✅ Prepared cleaned text-only datasets")
    print(f"   Train images: {len(clean_train['images'])} | text boxes kept: {len(clean_train['annotations'])}")
    print(f"   Val images:   {len(clean_val['images'])} | text boxes kept: {len(clean_val['annotations'])}")
    print(f"   Train removed: {dict(train_removed)}")
    print(f"   Val removed:   {dict(val_removed)}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Create a COCO training subset from specific yearbook image identifiers."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any


DEFAULT_ANNOTATIONS = "layout-model-training/data/yearbook/annotations.json"
DEFAULT_IMAGES_DIR = "layout-model-training/data/yearbook/images"
DEFAULT_OUTPUT_DIR = "layout-model-training/data/yearbook"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a train/val subset from specific image ids, task ids, or file names."
    )
    parser.add_argument(
        "--annotations",
        default=DEFAULT_ANNOTATIONS,
        help="Path to the full COCO annotations JSON",
    )
    parser.add_argument(
        "--images-dir",
        default=DEFAULT_IMAGES_DIR,
        help="Directory containing downloaded yearbook images",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the subset JSON files should be written",
    )
    parser.add_argument(
        "--output-prefix",
        default="annotations-selected",
        help="Prefix for generated JSON files",
    )
    parser.add_argument(
        "--image-ids",
        nargs="*",
        type=int,
        default=[],
        help="COCO image ids to include",
    )
    parser.add_argument(
        "--task-ids",
        nargs="*",
        type=int,
        default=[],
        help="Label Studio task ids to include",
    )
    parser.add_argument(
        "--file-names",
        nargs="*",
        default=[],
        help="Image file names to include",
    )
    parser.add_argument(
        "--id-file",
        help="Optional text, JSON, or CSV file containing ids to include",
    )
    parser.add_argument(
        "--id-type",
        choices=["image", "task", "file"],
        help="Type of values stored in --id-file",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
        help="Fraction of selected images to place in validation",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for the split")
    parser.add_argument(
        "--require-downloaded",
        action="store_true",
        help="Only include selected images that already exist in --images-dir",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def normalize_file_name(value: str) -> str:
    return Path(value.strip()).name


def load_id_file(path: Path, id_type: str) -> tuple[set[int], set[int], set[str]]:
    image_ids: set[int] = set()
    task_ids: set[int] = set()
    file_names: set[str] = set()

    suffix = path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            values = payload.get("ids") or payload.get("items") or payload.get("values") or []
        elif isinstance(payload, list):
            values = payload
        else:
            raise ValueError(f"Unsupported JSON content in {path}")
    elif suffix == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if id_type == "image":
                column = "image_id"
            elif id_type == "task":
                column = "task_id"
            else:
                column = "file_name"
            if column not in reader.fieldnames:
                raise ValueError(f"CSV file {path} must contain column '{column}'")
            values = [row[column] for row in reader if row.get(column)]
    else:
        values = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    if id_type == "image":
        image_ids.update(int(value) for value in values)
    elif id_type == "task":
        task_ids.update(int(value) for value in values)
    else:
        file_names.update(normalize_file_name(value) for value in values)

    return image_ids, task_ids, file_names


def gather_requested_ids(args: argparse.Namespace) -> tuple[set[int], set[int], set[str]]:
    image_ids = set(args.image_ids)
    task_ids = set(args.task_ids)
    file_names = {normalize_file_name(value) for value in args.file_names}

    if args.id_file:
        if not args.id_type:
            raise ValueError("--id-type is required when using --id-file")
        file_image_ids, file_task_ids, file_names_from_file = load_id_file(Path(args.id_file), args.id_type)
        image_ids.update(file_image_ids)
        task_ids.update(file_task_ids)
        file_names.update(file_names_from_file)

    if not image_ids and not task_ids and not file_names:
        raise ValueError("Provide at least one of --image-ids, --task-ids, --file-names, or --id-file")

    return image_ids, task_ids, file_names


def select_images(
    data: dict[str, Any],
    image_ids: set[int],
    task_ids: set[int],
    file_names: set[str],
    images_dir: Path,
    require_downloaded: bool,
) -> tuple[list[dict[str, Any]], list[str]]:
    selected: list[dict[str, Any]] = []
    seen_ids: set[int] = set()

    available_image_ids = {img["id"] for img in data["images"]}
    available_task_ids = {img.get("task_id") for img in data["images"] if img.get("task_id") is not None}
    available_file_names = {img["file_name"] for img in data["images"]}

    missing: list[str] = []
    for value in sorted(image_ids - available_image_ids):
        missing.append(f"image_id:{value}")
    for value in sorted(task_ids - available_task_ids):
        missing.append(f"task_id:{value}")
    for value in sorted(file_names - available_file_names):
        missing.append(f"file_name:{value}")

    for image in data["images"]:
        matches = (
            image["id"] in image_ids
            or image.get("task_id") in task_ids
            or image["file_name"] in file_names
        )
        if not matches or image["id"] in seen_ids:
            continue
        if require_downloaded and not (images_dir / image["file_name"]).exists():
            continue
        selected.append(image)
        seen_ids.add(image["id"])

    return selected, missing


def build_subset(
    data: dict[str, Any],
    selected_images: list[dict[str, Any]],
    val_fraction: float,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if not 0 <= val_fraction < 1:
        raise ValueError("--val-fraction must be in the range [0, 1)")

    shuffled_images = list(selected_images)
    random.seed(seed)
    random.shuffle(shuffled_images)

    val_count = int(round(len(shuffled_images) * val_fraction))
    if len(shuffled_images) > 1 and val_fraction > 0 and val_count == 0:
        val_count = 1
    if val_count >= len(shuffled_images) and shuffled_images:
        val_count = len(shuffled_images) - 1

    val_images = shuffled_images[:val_count]
    train_images = shuffled_images[val_count:]

    train_ids = {img["id"] for img in train_images}
    val_ids = {img["id"] for img in val_images}
    selected_ids = train_ids | val_ids

    subset_annotations = [ann for ann in data["annotations"] if ann["image_id"] in selected_ids]
    train_annotations = [ann for ann in subset_annotations if ann["image_id"] in train_ids]
    val_annotations = [ann for ann in subset_annotations if ann["image_id"] in val_ids]

    subset_data = {
        "images": shuffled_images,
        "annotations": subset_annotations,
        "categories": data["categories"],
    }
    train_data = {
        "images": train_images,
        "annotations": train_annotations,
        "categories": data["categories"],
    }
    val_data = {
        "images": val_images,
        "annotations": val_annotations,
        "categories": data["categories"],
    }
    return subset_data, train_data, val_data


def main() -> None:
    args = parse_args()
    annotations_path = Path(args.annotations)
    images_dir = Path(args.images_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_ids, task_ids, file_names = gather_requested_ids(args)
    data = load_json(annotations_path)

    selected_images, missing = select_images(
        data,
        image_ids=image_ids,
        task_ids=task_ids,
        file_names=file_names,
        images_dir=images_dir,
        require_downloaded=args.require_downloaded,
    )

    if not selected_images:
        raise ValueError("No matching images were selected")

    subset_data, train_data, val_data = build_subset(
        data,
        selected_images=selected_images,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )

    base = output_dir / args.output_prefix
    write_json(base.with_suffix(".json"), subset_data)
    write_json(output_dir / f"{args.output_prefix}-train.json", train_data)
    write_json(output_dir / f"{args.output_prefix}-val.json", val_data)

    print("✅ Selected subset created")
    print(f"   Requested matches: {len(selected_images)} images")
    print(f"   Train: {len(train_data['images'])} images | {len(train_data['annotations'])} annotations")
    print(f"   Val:   {len(val_data['images'])} images | {len(val_data['annotations'])} annotations")
    print(f"   Wrote: {base.with_suffix('.json')}")
    print(f"   Wrote: {output_dir / f'{args.output_prefix}-train.json'}")
    print(f"   Wrote: {output_dir / f'{args.output_prefix}-val.json'}")
    if missing:
        print(f"   Missing requested ids: {len(missing)}")
        for value in missing[:20]:
            print(f"     - {value}")
        if len(missing) > 20:
            print("     - ...")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Download all Label Studio project images to a local directory.

Designed to run on any system that has access to the project API,
including Perlmutter login nodes or local machines.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

import requests
from dotenv import load_dotenv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download project images from Label Studio")
    parser.add_argument("--project-id", type=int, default=None, help="Label Studio project ID")
    parser.add_argument(
        "--out-dir",
        type=str,
        default="layout-model-training/data/yearbook/images",
        help="Directory where images will be stored",
    )
    parser.add_argument("--page-size", type=int, default=100, help="Task pagination page size")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on number of images")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload files even if they already exist",
    )
    return parser.parse_args()


def get_env_or_fail(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"{name} is required")
    return value


def list_project_tasks(base_url: str, token: str, project_id: int, page_size: int) -> list[dict[str, Any]]:
    all_tasks: list[dict[str, Any]] = []
    page = 1

    while True:
        response = requests.get(
            f"{base_url}/api/projects/{project_id}/tasks",
            params={"page": page, "page_size": page_size},
            headers={"Authorization": f"Token {token}"},
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()

        if isinstance(payload, list):
            tasks_page = payload
            has_next = len(tasks_page) == page_size
        else:
            tasks_page = payload.get("tasks", payload.get("results", []))
            has_next = bool(payload.get("next"))

        if not tasks_page:
            break

        all_tasks.extend(tasks_page)

        if not has_next:
            break
        page += 1

    return all_tasks


def build_image_url(base_url: str, image_ref: str) -> str:
    if image_ref.startswith("s3://"):
        without_scheme = image_ref[len("s3://") :]
        if "/" in without_scheme:
            bucket, key = without_scheme.split("/", 1)
            return f"https://{bucket}.s3.amazonaws.com/{key}"
        return image_ref
    if image_ref.startswith("http://") or image_ref.startswith("https://"):
        return image_ref
    return urljoin(f"{base_url}/", image_ref.lstrip("/"))


def choose_filename(task_id: int, image_url: str) -> str:
    parsed = urlparse(image_url)
    return Path(parsed.path).name or f"task_{task_id}.jpg"


def download_image(url: str, token: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    headers: dict[str, str] = {}
    if "s3.amazonaws.com" not in url:
        headers["Authorization"] = f"Token {token}"

    response = requests.get(url, headers=headers, timeout=120)
    response.raise_for_status()
    output_path.write_bytes(response.content)


def main() -> None:
    load_dotenv()
    args = parse_args()

    token = get_env_or_fail("LABEL_STUDIO_TOKEN")
    base_url = os.getenv("LABEL_STUDIO_BASE_URL", "https://app.humansignal.com").rstrip("/")
    project_id = args.project_id or int(get_env_or_fail("LABEL_STUDIO_PROJECT_ID"))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Fetching tasks from project {project_id}...")
    tasks = list_project_tasks(base_url, token, project_id, args.page_size)
    print(f"Found {len(tasks)} tasks")

    downloaded = 0
    skipped = 0
    failed = 0

    for task in tasks:
        if args.limit is not None and downloaded + skipped >= args.limit:
            break

        task_id = task.get("id")
        image_ref = task.get("data", {}).get("image")
        if not image_ref:
            continue

        image_url = build_image_url(base_url, image_ref)
        filename = choose_filename(task_id, image_url)
        output_path = out_dir / filename

        if output_path.exists() and not args.force:
            skipped += 1
            continue

        try:
            download_image(image_url, token, output_path)
            downloaded += 1
            if downloaded % 25 == 0:
                print(f"  Downloaded {downloaded} images...")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  Failed task {task_id}: {filename} -> {exc}")

    print("\nDone.")
    print(f"Downloaded: {downloaded}")
    print(f"Skipped existing: {skipped}")
    print(f"Failed: {failed}")
    print(f"Output dir: {out_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Import the first few image URLs into a Label Studio project.

Expected input file: one image URL per line.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional

import requests
from dotenv import load_dotenv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import first few image URLs to Label Studio")
    parser.add_argument(
        "--project-id",
        type=int,
        default=None,
        help="Label Studio project ID (overrides LABEL_STUDIO_PROJECT_ID)",
    )
    parser.add_argument(
        "--input-file",
        type=str,
        default="layout-model-training/data/yearbook/images_to_download.txt",
        help="Path to a text file containing one image URL per line",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="How many URLs to import",
    )
    parser.add_argument(
        "--commit",
        action="store_true",
        help="Actually create tasks in Label Studio (default is dry run)",
    )
    return parser.parse_args()


def get_env_or_fail(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"{name} is required")
    return value


def read_urls(path: Path, limit: int) -> list[str]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    urls = [line for line in lines if line and not line.startswith("#")]
    return urls[:limit]


def main() -> None:
    load_dotenv()
    args = parse_args()

    token = get_env_or_fail("LABEL_STUDIO_TOKEN")
    base_url = os.getenv("LABEL_STUDIO_BASE_URL", "https://app.humansignal.com").rstrip("/")

    project_id: Optional[int] = args.project_id
    if project_id is None:
        project_id = int(get_env_or_fail("LABEL_STUDIO_PROJECT_ID"))

    input_path = Path(args.input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    urls = read_urls(input_path, args.limit)
    if not urls:
        print("No URLs found to import.")
        return

    tasks_payload = [{"data": {"image": url}} for url in urls]

    print(f"Project: {project_id}")
    print(f"Base URL: {base_url}")
    print(f"Input file: {input_path}")
    print(f"URLs selected: {len(urls)}")

    if not args.commit:
        print("\nDry run mode (no API write).")
        print("Use --commit to actually import these tasks.")
        for index, url in enumerate(urls, start=1):
            print(f"  {index}. {url}")
        return

    endpoint = f"{base_url}/api/projects/{project_id}/import"
    response = requests.post(
        endpoint,
        headers={
            "Authorization": f"Token {token}",
            "Content-Type": "application/json",
        },
        json=tasks_payload,
        timeout=60,
    )

    if response.status_code not in (200, 201):
        raise RuntimeError(f"Import failed ({response.status_code}): {response.text}")

    print("\nImport request sent successfully.")
    print(response.text)


if __name__ == "__main__":
    main()

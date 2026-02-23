"""Wrapper stage that crops all bbox regions described by bounder JSON output."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from backend.boundingbox import crop_bboxes
from backend.common import BBOX_EXTENSIONS
from backend.utils import save_json


def run_cropper(
    input_dir: Path,
    json_dir: Path,
    output_dir: Path,
    output_json: Optional[Path] = None,
) -> dict:
    """Run crop generation and emit a small summary payload for orchestration."""
    output_dir.mkdir(parents=True, exist_ok=True)
    total = crop_bboxes.crop_all_bboxes(
        input_dir=input_dir,
        json_dir=json_dir,
        output_dir=output_dir,
        extensions=BBOX_EXTENSIONS,
    )
    payload = {
        "input_dir": str(input_dir),
        "json_dir": str(json_dir),
        "crops_dir": str(output_dir),
        "total_crops": int(total),
    }
    if output_json is not None:
        save_json(payload, output_json)
    return payload


def parse_args() -> argparse.Namespace:
    """Parse CLI args for standalone cropper execution."""
    parser = argparse.ArgumentParser(description="Cropper module: crop image regions from bbox JSON files.")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--json_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--output_json", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    """CLI entry point for cropper stage."""
    args = parse_args()
    payload = run_cropper(
        input_dir=Path(args.input_dir).expanduser().resolve(),
        json_dir=Path(args.json_dir).expanduser().resolve(),
        output_dir=Path(args.output_dir).expanduser().resolve(),
        output_json=Path(args.output_json).expanduser().resolve() if args.output_json else None,
    )
    print(f"[INFO] Cropper complete. Total crops: {payload['total_crops']} | Dir: {payload['crops_dir']}")


if __name__ == "__main__":
    main()

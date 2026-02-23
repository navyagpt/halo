#!/usr/bin/env python3
"""
Crop individual bounding boxes from images using exported JSON files.

Example:
    python crop_bboxes.py --input_dir inputs --json_dir outputs/json --output_dir outputs/crops
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2


BBox = Tuple[int, int, int, int]


def list_json_files(json_dir: Path) -> List[Path]:
    """Return all JSON annotation files under json_dir."""
    files = [p for p in json_dir.rglob("*.json") if p.is_file()]
    files.sort()
    return files


def parse_bbox(value: Sequence[int]) -> Optional[BBox]:
    """Parse [x1, y1, x2, y2] into an integer bbox tuple."""
    if len(value) != 4:
        return None
    x1, y1, x2, y2 = (int(v) for v in value)
    return x1, y1, x2, y2


def clamp_bbox(box: BBox, width: int, height: int) -> Optional[BBox]:
    """Clamp bbox to image bounds and reject invalid/empty results."""
    x1, y1, x2, y2 = box
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width - 1))
    y2 = max(0, min(y2, height - 1))
    if x2 < x1 or y2 < y1:
        return None
    return x1, y1, x2, y2


def bbox_sort_key(item: Tuple[str, BBox]) -> Tuple[int, str]:
    """Sort bbox keys like bbox1, bbox2, ... in numeric order."""
    name, _ = item
    suffix = name.lower().replace("bbox", "")
    try:
        return int(suffix), name
    except ValueError:
        return 10**9, name


def find_image_path(
    input_dir: Path,
    json_path: Path,
    json_root: Path,
    image_name: str,
    extensions: Iterable[str],
) -> Optional[Path]:
    """Resolve image path using JSON location and image_name fallback strategies."""
    rel_json = json_path.relative_to(json_root)
    rel_parent = rel_json.parent

    direct_rel = input_dir / rel_parent / image_name
    if direct_rel.exists():
        return direct_rel

    direct_root = input_dir / image_name
    if direct_root.exists():
        return direct_root

    stem = Path(image_name).stem
    for ext in extensions:
        ext_norm = ext if ext.startswith(".") else f".{ext}"
        candidate = input_dir / rel_parent / f"{stem}{ext_norm}"
        if candidate.exists():
            return candidate

    return None


def crop_from_json_file(
    json_path: Path,
    json_root: Path,
    input_dir: Path,
    output_dir: Path,
    extensions: Sequence[str],
) -> int:
    """Crop and save all bbox crops described by one JSON file."""
    with json_path.open("r", encoding="utf-8") as f:
        payload: Dict[str, object] = json.load(f)

    image_name = str(payload.get("image", ""))
    if not image_name:
        print(f"[WARN] Missing image name in JSON: {json_path}")
        return 0

    image_path = find_image_path(input_dir, json_path, json_root, image_name, extensions)
    if image_path is None:
        print(f"[WARN] Image not found for JSON: {json_path}")
        return 0

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        print(f"[WARN] Could not read image: {image_path}")
        return 0

    height, width = image.shape[:2]
    bboxes = payload.get("bboxes", {})
    if not isinstance(bboxes, dict):
        print(f"[WARN] Invalid bboxes format in JSON: {json_path}")
        return 0

    rel_json = json_path.relative_to(json_root)
    image_crop_dir = output_dir / rel_json.with_suffix("")
    image_crop_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    parsed: List[Tuple[str, BBox]] = []
    for name, raw_box in bboxes.items():
        if not isinstance(name, str) or not isinstance(raw_box, list):
            continue
        parsed_box = parse_bbox(raw_box)
        if parsed_box is None:
            continue
        parsed.append((name, parsed_box))

    for name, box in sorted(parsed, key=bbox_sort_key):
        valid_box = clamp_bbox(box, width, height)
        if valid_box is None:
            print(f"[WARN] Invalid bbox {name} in {json_path}")
            continue
        x1, y1, x2, y2 = valid_box
        crop = image[y1 : y2 + 1, x1 : x2 + 1]
        if crop.size == 0:
            print(f"[WARN] Empty crop for bbox {name} in {json_path}")
            continue
        out_path = image_crop_dir / f"{name}.png"
        cv2.imwrite(str(out_path), crop)
        saved += 1

    print(f"[INFO] {json_path.name}: saved {saved} crop(s)")
    return saved


def crop_all_bboxes(
    input_dir: Path,
    json_dir: Path,
    output_dir: Path,
    extensions: Sequence[str],
) -> int:
    """Crop bbox images for all JSON files and return total saved count."""
    json_files = list_json_files(json_dir)
    if not json_files:
        print(f"[WARN] No JSON files found in {json_dir}")
        return 0

    total = 0
    for json_path in json_files:
        total += crop_from_json_file(
            json_path=json_path,
            json_root=json_dir,
            input_dir=input_dir,
            output_dir=output_dir,
            extensions=extensions,
        )
    return total


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Crop each bounding box from images using JSON annotations."
    )
    parser.add_argument("--input_dir", type=str, default="inputs", help="Input image directory.")
    parser.add_argument("--json_dir", type=str, default="outputs/json", help="Directory of bbox JSON files.")
    parser.add_argument("--output_dir", type=str, default="outputs/crops", help="Output crops directory.")
    parser.add_argument(
        "--extensions",
        nargs="*",
        default=["png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp"],
        help="Image extensions used to resolve source files.",
    )
    return parser.parse_args()


def main() -> None:
    """Program entry point."""
    args = parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    json_dir = Path(args.json_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if not json_dir.exists() or not json_dir.is_dir():
        raise FileNotFoundError(f"JSON directory not found: {json_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    total = crop_all_bboxes(
        input_dir=input_dir,
        json_dir=json_dir,
        output_dir=output_dir,
        extensions=args.extensions,
    )
    print(f"[INFO] Done. Total crops saved: {total} | Output: {output_dir}")


if __name__ == "__main__":
    main()

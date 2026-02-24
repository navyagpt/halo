"""Shared utility helpers for argument parsing, IO, and report generation."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List


def normalize_extensions(exts: List[str]) -> set[str]:
    """Normalize extension tokens into a lowercase dotted set."""
    out = set()
    for ext in exts:
        clean = ext.strip().lower()
        if not clean:
            continue
        if not clean.startswith("."):
            clean = f".{clean}"
        out.add(clean)
    return out


def parse_candidate_labels(raw: List[str] | None) -> List[str]:
    """Parse repeated/comma-separated candidate label inputs into a deduplicated list."""
    if not raw:
        return []
    labels: List[str] = []
    for item in raw:
        parts = [p.strip() for p in str(item).split(",")]
        for part in parts:
            if part:
                labels.append(part)
    return list(dict.fromkeys(labels))


def parse_json_dict_arg(raw: str | None, arg_name: str) -> Dict[str, Any]:
    """Decode a JSON object argument with a friendly validation error."""
    if raw is None:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{arg_name} must be valid JSON.") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{arg_name} must decode to a JSON object.")
    return parsed


def collect_images(input_path: Path, recursive: bool, extensions: set[str]) -> List[Path]:
    """Collect supported image files from an input path."""
    if input_path.is_file():
        return [input_path]

    matcher = input_path.rglob if recursive else input_path.glob
    files = [p for p in matcher("*") if p.is_file() and p.suffix.lower() in extensions]
    files.sort()
    return files


def safe_token(value: str) -> str:
    """Convert arbitrary path text to a filename-safe token."""
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    token = token.strip("._")
    return token or "image"


def image_id_for_path(image_path: Path, input_root: Path, index: int) -> str:
    """Build a stable image identifier used across per-image artifacts."""
    if input_root.is_dir():
        rel_no_ext = image_path.relative_to(input_root).with_suffix("")
        rel_token = "__".join(rel_no_ext.parts)
    else:
        rel_token = image_path.stem
    return f"img{index:05d}_{safe_token(rel_token)}"


def ensure_output_dirs(output_dir: Path, save_masks: bool) -> Dict[str, Path]:
    """Create standard output subdirectories and return them as a path map."""
    dirs = {
        "base": output_dir,
        "json": output_dir / "json",
        "vis": output_dir / "vis",
        "crops": output_dir / "crops",
        "reports": output_dir / "reports",
    }
    for key in ("base", "json", "vis", "crops", "reports"):
        dirs[key].mkdir(parents=True, exist_ok=True)

    if save_masks:
        dirs["masks"] = output_dir / "masks"
        dirs["masks"].mkdir(parents=True, exist_ok=True)

    return dirs


def write_flat_csv(output_csv: Path, image_records: List[Dict[str, Any]], class_names: List[str]) -> None:
    """Write flattened detection/classification results for easier downstream consumption."""
    header = [
        "image_id",
        "image_path",
        "bbox_id",
        "x1",
        "y1",
        "x2",
        "y2",
        "centroid_x",
        "centroid_y",
        "crop_path",
        "predicted_label",
        "predicted_class_id",
        "confidence",
    ] + [f"prob_{name}" for name in class_names]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for image in image_records:
            for det in image.get("detections", []):
                x1, y1, x2, y2 = det["bbox_xyxy"]
                centroid_x, centroid_y = det.get("bbox_centroid_xy", ["", ""])
                pred = det.get("prediction")
                row = [
                    image["image_id"],
                    image["image_path"],
                    det["bbox_id"],
                    int(x1),
                    int(y1),
                    int(x2),
                    int(y2),
                    centroid_x,
                    centroid_y,
                    det["crop_path"],
                    pred["label"] if pred else "",
                    pred["class_id"] if pred else "",
                    pred["confidence"] if pred else "",
                ]

                if class_names:
                    probs = pred["probabilities"] if pred else {}
                    row.extend(probs.get(name, "") for name in class_names)
                writer.writerow(row)

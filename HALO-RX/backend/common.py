"""Shared filesystem and JSONL helpers used across pipeline stages."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

BBOX_EXTENSIONS = ["png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp"]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def build_run_paths(output_root: Path, image_path: Path) -> dict[str, Path]:
    """Create the per-run folder layout and return all canonical artifact paths."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{image_path.stem}_{stamp}"
    run_dir = output_root / run_name
    paths = {
        "run_dir": run_dir,
        "inputs_dir": run_dir / "inputs",
        "bbox_dir": run_dir / "bbox_outputs",
        "json_dir": run_dir / "bbox_outputs" / "json",
        "vis_dir": run_dir / "bbox_outputs" / "vis",
        "masks_dir": run_dir / "bbox_outputs" / "masks",
        "crops_dir": run_dir / "crops",
        "prescriber_json": run_dir / "prescriber_output.json",
        "prescription_txt": run_dir / "prescription_candidates.txt",
        "predictions_jsonl": run_dir / "predictions.jsonl",
        "centers_json": run_dir / "centers.json",
        "summary_json": run_dir / "pipeline_result.json",
    }
    for key in ("run_dir", "inputs_dir", "bbox_dir", "crops_dir"):
        paths[key].mkdir(parents=True, exist_ok=True)
    return paths


def list_crop_images(crops_dir: Path) -> list[Path]:
    """Return all crop image files under ``crops_dir`` in deterministic order."""
    return [p for p in sorted(crops_dir.rglob("*")) if p.is_file() and p.suffix.lower() in IMG_EXTS]


def write_jsonl(records: list[dict], path: Path) -> None:
    """Write records as UTF-8 JSONL, one JSON object per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    """Read JSONL payload while skipping blank lines."""
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

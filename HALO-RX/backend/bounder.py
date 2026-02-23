"""Wrapper stage that runs ROI-based bounding-box extraction for one image."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

from backend.boundingbox import main as bbox_main
from backend.common import BBOX_EXTENSIONS
from backend.utils import save_json


def run_bounder(
    image_path: Path,
    inputs_dir: Path,
    bbox_output_dir: Path,
    save_masks: bool,
    no_auto_thresholds: bool,
    v_dark: int,
    s_dark_max: int,
    v_obj: int,
    s_obj: int,
    v_col: int,
    min_area: Optional[int],
    max_aspect: float,
    output_json: Optional[Path] = None,
) -> dict:
    """Stage image into run workspace and invoke ``boundingbox.main`` end-to-end."""
    inputs_dir.mkdir(parents=True, exist_ok=True)
    bbox_output_dir.mkdir(parents=True, exist_ok=True)

    staged_image = inputs_dir / image_path.name
    shutil.copy2(image_path, staged_image)

    bbox_args = SimpleNamespace(
        recursive=False,
        save_masks=save_masks,
        extensions=BBOX_EXTENSIONS,
        v_dark=v_dark,
        s_dark_max=s_dark_max,
        v_obj=v_obj,
        s_obj=s_obj,
        v_col=v_col,
        min_area=min_area,
        max_aspect=max_aspect,
        auto_thresholds=(not no_auto_thresholds),
    )

    json_dir, vis_dir, mask_dir = bbox_main.ensure_dirs(bbox_output_dir)
    bbox_main.process_image(
        image_path=staged_image,
        input_root=inputs_dir,
        args=bbox_args,
        json_root=json_dir,
        vis_root=vis_dir,
        mask_root=mask_dir,
    )

    payload = {
        "staged_image": str(staged_image),
        "bbox_json": str(json_dir / f"{staged_image.stem}.json"),
        "bbox_vis": str(vis_dir / f"{staged_image.stem}_boxed.png"),
        "bbox_masks_dir": str(mask_dir),
    }
    if output_json is not None:
        save_json(payload, output_json)
    return payload


def parse_args() -> argparse.Namespace:
    """Parse CLI args for standalone bounder execution."""
    parser = argparse.ArgumentParser(description="Bounder module: run bounding box detection for one image.")
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--inputs_dir", type=str, required=True)
    parser.add_argument("--bbox_output_dir", type=str, required=True)
    parser.add_argument("--save_masks", action="store_true")
    parser.add_argument("--no_auto_thresholds", action="store_true")
    parser.add_argument("--v_dark", type=int, default=70)
    parser.add_argument("--s_dark_max", type=int, default=120)
    parser.add_argument("--v_obj", type=int, default=135)
    parser.add_argument("--s_obj", type=int, default=60)
    parser.add_argument("--v_col", type=int, default=80)
    parser.add_argument("--min_area", type=int, default=None)
    parser.add_argument("--max_aspect", type=float, default=4.5)
    parser.add_argument("--output_json", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    """CLI entry point for bounder stage."""
    args = parse_args()
    payload = run_bounder(
        image_path=Path(args.image_path).expanduser().resolve(),
        inputs_dir=Path(args.inputs_dir).expanduser().resolve(),
        bbox_output_dir=Path(args.bbox_output_dir).expanduser().resolve(),
        save_masks=bool(args.save_masks),
        no_auto_thresholds=bool(args.no_auto_thresholds),
        v_dark=args.v_dark,
        s_dark_max=args.s_dark_max,
        v_obj=args.v_obj,
        s_obj=args.s_obj,
        v_col=args.v_col,
        min_area=args.min_area,
        max_aspect=args.max_aspect,
        output_json=Path(args.output_json).expanduser().resolve() if args.output_json else None,
    )
    print(f"[INFO] Bounder complete. JSON: {payload['bbox_json']}")


if __name__ == "__main__":
    main()

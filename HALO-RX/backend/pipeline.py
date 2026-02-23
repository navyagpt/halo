"""Top-level orchestrator for prescriber, bounder, cropper, labeler, and center."""

from __future__ import annotations

import argparse
import logging
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from backend.bounder import run_bounder
from backend.center import run_center
from backend.common import build_run_paths, list_crop_images, read_jsonl
from backend.cropper import run_cropper
from backend.labeler import run_labeler
from backend.prescriber import run_prescriber
from backend.utils import save_json

LOGGER = logging.getLogger("backend.pipeline")
PACKAGE_ROOT = Path(__file__).resolve().parent


def _run_bounder_cropper(args: argparse.Namespace, image_path: Path, paths: dict[str, Path]) -> dict:
    """Run geometric stages that only depend on the source image."""
    bounder_payload = run_bounder(
        image_path=image_path,
        inputs_dir=paths["inputs_dir"],
        bbox_output_dir=paths["bbox_dir"],
        save_masks=bool(args.save_masks),
        no_auto_thresholds=bool(args.no_auto_thresholds),
        v_dark=args.v_dark,
        s_dark_max=args.s_dark_max,
        v_obj=args.v_obj,
        s_obj=args.s_obj,
        v_col=args.v_col,
        min_area=args.min_area,
        max_aspect=args.max_aspect,
        output_json=paths["run_dir"] / "bounder_result.json",
    )

    cropper_payload = run_cropper(
        input_dir=paths["inputs_dir"],
        json_dir=paths["json_dir"],
        output_dir=paths["crops_dir"],
        output_json=paths["run_dir"] / "cropper_result.json",
    )
    return {"bounder": bounder_payload, "cropper": cropper_payload}


def _write_prescription_txt(candidates: list[str], path: Path) -> None:
    """Persist candidate list for reproducibility/debugging of inference runs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(candidates) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse CLI args for end-to-end pipeline orchestration."""
    parser = argparse.ArgumentParser(
        description="Run modular end-to-end pipeline: prescriber + bounder + cropper + labeler + center."
    )
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--run_dir", type=str, default=str(PACKAGE_ROOT))
    parser.add_argument("--output_root", type=str, default=str(PACKAGE_ROOT / "runs"))
    parser.add_argument("--prescription_file", type=str, default=None)
    parser.add_argument("--prescription", action="append", default=None)
    parser.add_argument("--target_medicine", type=str, default=None)
    parser.add_argument("--prescriber_interactive", action="store_true")
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        choices=["contrastive", "linear_probe", "partial_unfreeze", "lora_optional"],
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--save_masks", action="store_true")
    parser.add_argument("--no_auto_thresholds", action="store_true")
    parser.add_argument("--v_dark", type=int, default=70)
    parser.add_argument("--s_dark_max", type=int, default=120)
    parser.add_argument("--v_obj", type=int, default=135)
    parser.add_argument("--s_obj", type=int, default=60)
    parser.add_argument("--v_col", type=int, default=80)
    parser.add_argument("--min_area", type=int, default=None)
    parser.add_argument("--max_aspect", type=float, default=4.5)
    return parser.parse_args()


def main() -> None:
    """CLI entry point for the full modular pipeline."""
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    image_path = Path(args.image_path).expanduser().resolve()
    if not image_path.exists() or not image_path.is_file():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    run_dir = Path(args.run_dir).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    prescription_file = Path(args.prescription_file).expanduser().resolve() if args.prescription_file else None

    paths = build_run_paths(output_root=output_root, image_path=image_path)

    # Bounder/cropper and prescriber are independent, so run them concurrently.
    LOGGER.info("Launching bounder+cropper while prescriber collects candidates.")
    with ThreadPoolExecutor(max_workers=2) as executor:
        bbox_crop_future = executor.submit(_run_bounder_cropper, args, image_path, paths)

        prescriber_payload = run_prescriber(
            prescription_file=prescription_file,
            prescriptions=args.prescription,
            target_medicine=args.target_medicine,
            interactive=bool(args.prescriber_interactive),
            output_json=paths["prescriber_json"],
        )
        _write_prescription_txt(prescriber_payload["prescription_candidates"], paths["prescription_txt"])

        bbox_crop_payload = bbox_crop_future.result()

    crops = list_crop_images(paths["crops_dir"])
    if not crops:
        raise RuntimeError(f"No crops found in {paths['crops_dir']}. Check bbox thresholds/ROI settings.")

    LOGGER.info("Running labeler on %d crop(s).", len(crops))
    bbox_json_path = paths["json_dir"] / f"{image_path.stem}.json"
    labeler_payload = run_labeler(
        run_dir=run_dir,
        input_path=paths["crops_dir"],
        prescription_file=prescription_file,
        prescriptions=args.prescription,
        prescriber_json=paths["prescriber_json"],
        mode=args.mode,
        batch_size=args.batch_size,
        top_k=args.top_k,
        threshold=args.threshold,
        output_jsonl=paths["predictions_jsonl"],
        bbox_json=bbox_json_path,
        source_image=image_path.name,
        output_summary_json=paths["run_dir"] / "labeler_result.json",
    )

    target_medicine = prescriber_payload.get("target_medicine")
    center_payload = run_center(
        bbox_json=bbox_json_path,
        predictions_jsonl=paths["predictions_jsonl"],
        target_medicine=target_medicine,
        output_json=paths["centers_json"],
    )

    counts = Counter()
    for row in read_jsonl(paths["predictions_jsonl"]):
        counts.update([row.get("predicted_label", "UNKNOWN")])

    summary = {
        "input_image": str(image_path),
        "run_dir": str(run_dir),
        "mode": labeler_payload["mode"],
        "prescription_candidates_count": len(prescriber_payload["prescription_candidates"]),
        "target_medicine": target_medicine,
        "target_match_count": len(center_payload["target_bbox_centers"]),
        "artifacts": {
            "run_folder": str(paths["run_dir"]),
            "prescriber_output": str(paths["prescriber_json"]),
            "prescription_candidates_txt": str(paths["prescription_txt"]),
            "bbox_json": str(bbox_json_path),
            "bbox_vis": str(paths["vis_dir"] / f"{image_path.stem}_boxed.png"),
            "crops_dir": str(paths["crops_dir"]),
            "predictions_jsonl": str(paths["predictions_jsonl"]),
            "centers_json": str(paths["centers_json"]),
            "bounder_result": str(paths["run_dir"] / "bounder_result.json"),
            "cropper_result": str(paths["run_dir"] / "cropper_result.json"),
            "labeler_result": str(paths["run_dir"] / "labeler_result.json"),
        },
        "predicted_label_counts": dict(counts),
    }
    save_json(summary, paths["summary_json"])

    LOGGER.info("Pipeline complete: %s", paths["run_dir"])
    LOGGER.info("Target centers found: %d", len(center_payload["target_bbox_centers"]))


if __name__ == "__main__":
    main()

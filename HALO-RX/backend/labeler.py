"""Inference stage wrapper that emits per-crop medicine predictions."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

from backend.common import write_jsonl
from backend.infer_core import ensure_text_tokenizer_deps_available
from backend.infer_core import list_images
from backend.infer_core import load_labels
from backend.infer_core import run_contrastive_inference
from backend.infer_core import run_linear_probe_inference
from backend.pipeline_helpers import load_prescription_candidates
from backend.pipeline_helpers import map_candidates_for_linear_probe
from backend.pipeline_helpers import resolve_mode
from backend.utils import load_json, save_json

LOGGER = logging.getLogger("backend.labeler")
PACKAGE_ROOT = Path(__file__).resolve().parent


def _load_candidates(
    prescription_file: Optional[Path],
    prescriptions: Optional[list[str]],
    prescriber_json: Optional[Path],
) -> list[str]:
    """Collect and normalize prescription candidates from CLI and prescriber output."""
    items = list(prescriptions or [])
    if prescriber_json is not None:
        payload = load_json(prescriber_json)
        from_json = payload.get("prescription_candidates", [])
        if isinstance(from_json, list):
            items.extend(str(x) for x in from_json)
    return load_prescription_candidates(prescription_file, items)


def _attach_bbox_metadata(
    records: list[dict], bbox_json: Optional[Path], source_image: Optional[str]
) -> None:
    """Enrich inference records with bbox coordinates and source image provenance."""
    bbox_map = {}
    if bbox_json is not None and bbox_json.exists():
        payload = load_json(bbox_json)
        raw = payload.get("bboxes", {}) if isinstance(payload, dict) else {}
        if isinstance(raw, dict):
            bbox_map = raw

    for rec in records:
        bbox_name = Path(rec["image_path"]).stem
        rec["bbox_name"] = bbox_name
        rec["bbox_coords"] = bbox_map.get(bbox_name)
        if source_image:
            rec["source_image"] = source_image


def run_labeler(
    run_dir: Path,
    input_path: Path,
    prescription_file: Optional[Path],
    prescriptions: Optional[list[str]],
    prescriber_json: Optional[Path],
    mode: Optional[str],
    batch_size: int,
    top_k: int,
    threshold: float,
    output_jsonl: Path,
    bbox_json: Optional[Path] = None,
    source_image: Optional[str] = None,
    output_summary_json: Optional[Path] = None,
) -> dict:
    """Run chosen inference mode over crops and persist JSONL prediction artifacts."""
    resolved_mode = resolve_mode(run_dir, mode)
    ensure_text_tokenizer_deps_available(resolved_mode)

    candidates = _load_candidates(prescription_file, prescriptions, prescriber_json)
    labels = load_labels(run_dir)

    if resolved_mode == "linear_probe":
        mapped, unmatched = map_candidates_for_linear_probe(candidates, labels)
        if unmatched:
            LOGGER.warning("Ignoring %d prescription items not found in label space.", len(unmatched))
        if not mapped:
            raise ValueError("No prescription candidates overlap with labels.json for linear_probe mode.")
        infer_candidates = mapped
    else:
        infer_candidates = candidates

    image_paths = list_images(input_path)
    eff_top_k = min(max(1, top_k), len(infer_candidates))

    if resolved_mode in {"contrastive", "partial_unfreeze", "lora_optional"}:
        records = run_contrastive_inference(
            run_dir=run_dir,
            image_paths=image_paths,
            candidate_labels=infer_candidates,
            top_k=eff_top_k,
            threshold=threshold,
            batch_size=batch_size,
        )
    elif resolved_mode == "linear_probe":
        records = run_linear_probe_inference(
            run_dir=run_dir,
            image_paths=image_paths,
            all_labels=labels,
            candidate_labels=infer_candidates,
            top_k=eff_top_k,
            threshold=threshold,
            batch_size=batch_size,
        )
    else:
        raise ValueError(f"Unsupported mode: {resolved_mode}")

    _attach_bbox_metadata(records, bbox_json=bbox_json, source_image=source_image)
    write_jsonl(records, output_jsonl)

    payload = {
        "mode": resolved_mode,
        "num_images": len(image_paths),
        "num_predictions": len(records),
        "candidate_count": len(infer_candidates),
        "predictions_jsonl": str(output_jsonl),
    }
    if output_summary_json is not None:
        save_json(payload, output_summary_json)
    return payload


def parse_args() -> argparse.Namespace:
    """Parse CLI args for standalone labeler execution."""
    parser = argparse.ArgumentParser(description="Labeler module: infer medicine labels for images/crops.")
    parser.add_argument("--run_dir", type=str, default=str(PACKAGE_ROOT))
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--prescription_file", type=str, default=None)
    parser.add_argument("--prescriber_json", type=str, default=None)
    parser.add_argument("--prescription", action="append", default=None)
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        choices=["contrastive", "linear_probe", "partial_unfreeze", "lora_optional"],
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--output_jsonl", type=str, required=True)
    parser.add_argument("--bbox_json", type=str, default=None)
    parser.add_argument("--source_image", type=str, default=None)
    parser.add_argument("--output_summary_json", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    """CLI entry point for labeler stage."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()
    payload = run_labeler(
        run_dir=Path(args.run_dir).expanduser().resolve(),
        input_path=Path(args.input_path).expanduser().resolve(),
        prescription_file=Path(args.prescription_file).expanduser().resolve() if args.prescription_file else None,
        prescriptions=args.prescription,
        prescriber_json=Path(args.prescriber_json).expanduser().resolve() if args.prescriber_json else None,
        mode=args.mode,
        batch_size=args.batch_size,
        top_k=args.top_k,
        threshold=args.threshold,
        output_jsonl=Path(args.output_jsonl).expanduser().resolve(),
        bbox_json=Path(args.bbox_json).expanduser().resolve() if args.bbox_json else None,
        source_image=args.source_image,
        output_summary_json=Path(args.output_summary_json).expanduser().resolve() if args.output_summary_json else None,
    )
    print(
        "[INFO] Labeler complete. "
        f"Predictions: {payload['num_predictions']} | Output: {payload['predictions_jsonl']}"
    )


if __name__ == "__main__":
    main()

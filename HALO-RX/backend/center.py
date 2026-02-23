"""Compute bbox centers and attach predicted-label metadata."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from backend.common import read_jsonl
from backend.utils import canonicalize_label_text, load_json, save_json


def _bbox_centers_from_json(bbox_json: Path) -> list[dict]:
    """Convert bbox coordinate map into sorted rows with center points."""
    payload = load_json(bbox_json)
    bboxes = payload.get("bboxes", {}) if isinstance(payload, dict) else {}
    if not isinstance(bboxes, dict):
        return []
    rows: list[dict] = []
    for name, coords in bboxes.items():
        if not isinstance(name, str) or not isinstance(coords, list) or len(coords) != 4:
            continue
        x1, y1, x2, y2 = [float(v) for v in coords]
        rows.append(
            {
                "bbox_name": name,
                "bbox_coords": [int(x1), int(y1), int(x2), int(y2)],
                "center": [(x1 + x2) / 2.0, (y1 + y2) / 2.0],
            }
        )
    rows.sort(key=lambda x: x["bbox_name"])
    return rows


def _predictions_by_bbox(predictions_jsonl: Optional[Path]) -> dict[str, dict]:
    """Index prediction records by bbox name for fast metadata joins."""
    if predictions_jsonl is None or not predictions_jsonl.exists():
        return {}
    out: dict[str, dict] = {}
    for rec in read_jsonl(predictions_jsonl):
        name = rec.get("bbox_name")
        if not isinstance(name, str):
            continue
        out[name] = rec
    return out


def run_center(
    bbox_json: Path,
    predictions_jsonl: Optional[Path],
    target_medicine: Optional[str],
    output_json: Path,
) -> dict:
    """Join geometry and predictions, then extract target-medicine center rows."""
    rows = _bbox_centers_from_json(bbox_json)
    pred_map = _predictions_by_bbox(predictions_jsonl)

    enriched: list[dict] = []
    for row in rows:
        rec = dict(row)
        pred = pred_map.get(row["bbox_name"], {})
        if pred:
            rec["predicted_label"] = pred.get("predicted_label")
            rec["confidence"] = pred.get("confidence")
            rec["top_k_labels"] = pred.get("top_k_labels")
            rec["top_k_scores"] = pred.get("top_k_scores")
        enriched.append(rec)

    target_norm = canonicalize_label_text(target_medicine or "").lower()
    if target_norm:
        target_rows = [
            r
            for r in enriched
            if canonicalize_label_text(str(r.get("predicted_label", ""))).lower() == target_norm
        ]
    else:
        target_rows = []

    payload = {
        "bbox_json": str(bbox_json),
        "predictions_jsonl": str(predictions_jsonl) if predictions_jsonl else None,
        "target_medicine": target_medicine or None,
        "all_bbox_centers": enriched,
        "target_bbox_centers": target_rows,
    }
    save_json(payload, output_json)
    return payload


def parse_args() -> argparse.Namespace:
    """Parse CLI args for standalone center computation."""
    parser = argparse.ArgumentParser(description="Center module: compute bbox centers and target medicine centers.")
    parser.add_argument("--bbox_json", type=str, required=True)
    parser.add_argument("--predictions_jsonl", type=str, default=None)
    parser.add_argument("--target_medicine", type=str, default=None)
    parser.add_argument("--output_json", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    """CLI entry point for center stage."""
    args = parse_args()
    payload = run_center(
        bbox_json=Path(args.bbox_json).expanduser().resolve(),
        predictions_jsonl=Path(args.predictions_jsonl).expanduser().resolve() if args.predictions_jsonl else None,
        target_medicine=args.target_medicine,
        output_json=Path(args.output_json).expanduser().resolve(),
    )
    print(
        f"[INFO] Center complete. Total centers: {len(payload['all_bbox_centers'])} | "
        f"Target matches: {len(payload['target_bbox_centers'])}"
    )


if __name__ == "__main__":
    main()

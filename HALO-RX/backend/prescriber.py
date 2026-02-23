"""Prescription-candidate collection stage for the modular pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from backend.pipeline_helpers import load_prescription_candidates
from backend.utils import canonicalize_label_text, save_json


def _prompt_prescriptions() -> list[str]:
    """Interactive fallback when no prescription file/items are provided."""
    print("Enter prescription candidates (one per line).")
    print("Submit an empty line to finish:")
    raw_items: list[str] = []
    while True:
        try:
            line = input("> ")
        except EOFError:
            break
        line = line.strip()
        if not line:
            break
        raw_items.append(line)
    return raw_items


def _prompt_target_medicine() -> Optional[str]:
    """Ask user for an optional target medicine in interactive mode."""
    try:
        value = input("Target medicine to locate (optional, press Enter to skip): ").strip()
    except EOFError:
        return None
    if not value:
        return None
    return canonicalize_label_text(value)


def run_prescriber(
    prescription_file: Optional[Path],
    prescriptions: Optional[list[str]],
    target_medicine: Optional[str],
    interactive: bool,
    output_json: Optional[Path] = None,
) -> dict:
    """Resolve prescription candidates and optional target medicine into a JSON-safe payload."""
    raw_items = list(prescriptions or [])
    if interactive and prescription_file is None and not raw_items:
        raw_items = _prompt_prescriptions()

    candidates = load_prescription_candidates(prescription_file, raw_items)

    resolved_target = canonicalize_label_text(target_medicine or "")
    if not resolved_target and interactive:
        resolved_target = _prompt_target_medicine() or ""

    payload = {
        "prescription_candidates": candidates,
        "target_medicine": resolved_target or None,
    }
    if output_json is not None:
        save_json(payload, output_json)
    return payload


def parse_args() -> argparse.Namespace:
    """Parse CLI args for standalone prescriber stage."""
    parser = argparse.ArgumentParser(
        description="Prescriber module: collect medicine candidates and optional target medicine."
    )
    parser.add_argument("--prescription_file", type=str, default=None)
    parser.add_argument("--prescription", action="append", default=None)
    parser.add_argument("--target_medicine", type=str, default=None)
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--output_json", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    """CLI entry point for prescriber stage."""
    args = parse_args()
    prescription_file = Path(args.prescription_file).expanduser().resolve() if args.prescription_file else None
    payload = run_prescriber(
        prescription_file=prescription_file,
        prescriptions=args.prescription,
        target_medicine=args.target_medicine,
        interactive=bool(args.interactive),
        output_json=Path(args.output_json).expanduser().resolve(),
    )
    print(f"[INFO] Collected {len(payload['prescription_candidates'])} prescription candidate(s).")
    print(f"[INFO] Output: {Path(args.output_json).expanduser().resolve()}")
    if payload["target_medicine"]:
        print(f"[INFO] Target medicine: {payload['target_medicine']}")


if __name__ == "__main__":
    main()

"""Shared helper logic for candidate labels and mode resolution."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from backend.utils import canonicalize_label_text, load_json


def load_prescription_candidates(
    prescription_file: Optional[Path], raw_items: Optional[list[str]]
) -> list[str]:
    """Merge prescription sources into a de-duplicated, canonicalized candidate list."""
    labels: list[str] = []
    seen = set()

    if prescription_file is not None:
        if not prescription_file.exists():
            raise FileNotFoundError(f"Prescription file not found: {prescription_file}")
        for line in prescription_file.read_text(encoding="utf-8").splitlines():
            label = canonicalize_label_text(line)
            if label and label not in seen:
                labels.append(label)
                seen.add(label)

    for item in raw_items or []:
        label = canonicalize_label_text(item)
        if label and label not in seen:
            labels.append(label)
            seen.add(label)

    if not labels:
        raise ValueError("No prescription candidates provided. Use --prescription_file and/or --prescription.")

    return labels


def resolve_mode(run_dir: Path, explicit_mode: Optional[str]) -> str:
    """Choose inference mode from explicit arg first, then ``config.json`` fallback."""
    if explicit_mode:
        return explicit_mode
    cfg_path = run_dir / "config.json"
    if cfg_path.exists():
        return str(load_json(cfg_path).get("mode", "contrastive"))
    return "contrastive"


def map_candidates_for_linear_probe(candidates: list[str], all_labels: list[str]) -> tuple[list[str], list[str]]:
    """Map free-text candidates into the trained label space for linear-probe inference."""
    canon_to_label = {}
    for label in all_labels:
        key = canonicalize_label_text(label).lower()
        canon_to_label.setdefault(key, label)

    mapped: list[str] = []
    unmatched: list[str] = []
    seen = set()
    for med in candidates:
        key = canonicalize_label_text(med).lower()
        if key in canon_to_label:
            resolved = canon_to_label[key]
            if resolved not in seen:
                mapped.append(resolved)
                seen.add(resolved)
        else:
            unmatched.append(med)
    return mapped, unmatched

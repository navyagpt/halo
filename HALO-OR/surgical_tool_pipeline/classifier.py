"""Crop classification helpers backed by MedSigLIP embeddings and a saved classifier."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModel

from . import medsiglip_infer


def _hf_auth_kwargs() -> Dict[str, Any]:
    """Collect optional Hugging Face token kwargs from supported env var names."""
    token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    )
    return {"token": token} if token else {}


def _from_pretrained_with_optional_token(loader: Any, model_id: str) -> Any:
    """Load Hugging Face resources while remaining compatible with legacy token arguments."""
    kwargs = _hf_auth_kwargs()
    try:
        return loader.from_pretrained(model_id, **kwargs)
    except TypeError:
        if "token" not in kwargs:
            raise
        legacy_kwargs = {"use_auth_token": kwargs["token"]}
        return loader.from_pretrained(model_id, **legacy_kwargs)


def load_classifier_metadata(model_dir: Path, model_id_override: str | None) -> Tuple[Any, Dict[int, str], str]:
    """Load saved classifier artifacts and resolve the backbone model id."""
    model_path = model_dir / "model.joblib"
    mapping_path = model_dir / "label_mapping.json"
    config_path = model_dir / "config.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Missing model artifact: {model_path}")
    if not mapping_path.exists():
        raise FileNotFoundError(f"Missing label mapping: {mapping_path}")

    with mapping_path.open("r", encoding="utf-8") as f:
        mapping = json.load(f)
    id2label = {int(k): str(v) for k, v in mapping["id2label"].items()}

    model_id = "google/medsiglip-448"
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as f:
            config = json.load(f)
        model_id = str(config.get("model_id", model_id))

    if model_id_override:
        model_id = model_id_override

    classifier = joblib.load(model_path)
    return classifier, id2label, model_id


def select_label_subset(
    probs: np.ndarray,
    class_ids: List[int],
    id2label: Dict[int, str],
    candidate_labels: List[str],
) -> Tuple[np.ndarray, List[int], List[str]]:
    """Restrict prediction probabilities to a requested label subset."""
    class_labels = [id2label.get(i, str(i)) for i in class_ids]
    if not candidate_labels:
        return probs, class_ids, class_labels

    normalized_to_index = {label.strip().lower(): idx for idx, label in enumerate(class_labels)}
    selected_indices: List[int] = []
    missing: List[str] = []
    for label in candidate_labels:
        idx = normalized_to_index.get(label.strip().lower())
        if idx is None:
            missing.append(label)
        elif idx not in selected_indices:
            selected_indices.append(idx)

    if missing:
        raise ValueError(f"Unknown candidate_labels: {missing}. Available labels: {class_labels}")
    if not selected_indices:
        raise ValueError("No valid candidate labels were selected.")

    subset_probs = probs[:, selected_indices]
    denom = np.sum(subset_probs, axis=1, keepdims=True)
    denom = np.where(denom > 0.0, denom, 1.0)
    subset_probs = subset_probs / denom

    subset_class_ids = [class_ids[i] for i in selected_indices]
    subset_class_labels = [class_labels[i] for i in selected_indices]
    return subset_probs, subset_class_ids, subset_class_labels


def predict_crops(
    crop_paths: List[str],
    classifier: Any,
    id2label: Dict[int, str],
    model_id: str,
    device_name: str,
    batch_size: int,
    num_workers: int,
    use_amp: bool,
    candidate_labels: List[str],
) -> Tuple[Dict[str, Dict[str, Any]], List[str], List[Dict[str, str]]]:
    """Generate class predictions for crop images and return path-keyed results."""
    if not crop_paths:
        return {}, [], []

    device = medsiglip_infer.get_device(device_name)
    amp_enabled = use_amp and device.type == "cuda"

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.allow_tf32 = True

    print(
        f"[classify] device={device} amp={amp_enabled} tf32="
        f"{torch.backends.cuda.matmul.allow_tf32 if device.type == 'cuda' else 'n/a'} "
        f"crops={len(crop_paths)}"
    )

    try:
        processor = _from_pretrained_with_optional_token(AutoImageProcessor, model_id)
        backbone = _from_pretrained_with_optional_token(AutoModel, model_id).to(device).eval()
    except OSError as exc:
        raise RuntimeError(
            "Failed to load MedSigLIP backbone. If using gated repo access, set one of "
            "HF_TOKEN/HUGGINGFACE_HUB_TOKEN/HUGGING_FACE_HUB_TOKEN in the shell, or run "
            "`huggingface-cli login`, then retry."
        ) from exc

    dataset = medsiglip_infer.SafeFileImageDataset(crop_paths)
    X, valid_paths, skipped = medsiglip_infer.run_embedding_extraction(
        dataset=dataset,
        processor=processor,
        model=backbone,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        use_amp=amp_enabled,
    )

    if X.shape[0] == 0:
        return {}, [], skipped

    probs = medsiglip_infer.predict_proba_safe(classifier, X)
    class_ids = [int(x) for x in getattr(classifier, "classes_", sorted(id2label.keys()))]
    probs, class_ids, class_names = select_label_subset(
        probs=probs,
        class_ids=class_ids,
        id2label=id2label,
        candidate_labels=candidate_labels,
    )
    if candidate_labels:
        print(f"[classify] restricted labels={class_names}")

    pred_col_idx = np.argmax(probs, axis=1)
    pred_ids = [class_ids[i] for i in pred_col_idx]

    predictions_by_path: Dict[str, Dict[str, Any]] = {}
    for i, path in enumerate(valid_paths):
        class_probabilities = {class_names[j]: float(probs[i, j]) for j in range(len(class_names))}
        predictions_by_path[str(Path(path).resolve())] = {
            "label": id2label.get(pred_ids[i], str(pred_ids[i])),
            "class_id": int(pred_ids[i]),
            "confidence": float(probs[i, pred_col_idx[i]]),
            "probabilities": class_probabilities,
        }

    return predictions_by_path, class_names, skipped

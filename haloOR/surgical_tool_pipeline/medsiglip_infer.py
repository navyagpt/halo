"""Standalone embedding + classifier inference utilities for image folders."""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import joblib
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def setup_logging() -> logging.Logger:
    """Configure and return logger used by standalone inference mode."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    return logging.getLogger("infer_saved_model")


def parse_args() -> argparse.Namespace:
    """Parse standalone MedSigLIP inference script arguments."""
    parser = argparse.ArgumentParser(
        description="Standalone folder inference script for MedSigLIP embeddings + saved classifier."
    )
    parser.add_argument("--model_dir", type=str, default="outputs/models/best_model")
    parser.add_argument("--input_folder", type=str, required=True)
    parser.add_argument("--output_csv", type=str, default="outputs/reports/preds_from_input.csv")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--no_amp", action="store_true", help="Disable AMP even if CUDA is used.")
    parser.add_argument(
        "--model_id_override",
        type=str,
        default=None,
        help="Optional HF model ID override. Default comes from best_model/config.json",
    )
    parser.add_argument(
        "--candidate_labels",
        nargs="+",
        default=None,
        help=(
            "Optional label subset to predict among. "
            "Accepts space-separated and/or comma-separated labels."
        ),
    )
    return parser.parse_args()


def get_device(device_arg: str) -> torch.device:
    """Resolve runtime torch device from CLI selector."""
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def collect_image_files(folder: Path) -> List[str]:
    """Collect supported image files recursively from a folder."""
    files = [str(p) for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    files.sort()
    return files


class SafeFileImageDataset(Dataset):
    """Dataset that yields per-image load errors instead of raising."""

    def __init__(self, image_paths: Sequence[str]) -> None:
        self.image_paths = list(image_paths)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        path = self.image_paths[index]
        try:
            with Image.open(path) as img:
                rgb = img.convert("RGB")
                rgb.load()
            return {"ok": True, "image": rgb, "path": path}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "path": path, "error": f"{type(exc).__name__}: {exc}"}


def collate_identity(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep DataLoader batches as raw list of dict records."""
    return batch


def extract_image_features(model: Any, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Extract an image embedding tensor from diverse HF model output formats."""
    if hasattr(model, "get_image_features"):
        try:
            features = model.get_image_features(**inputs)
            if isinstance(features, torch.Tensor):
                return features
        except TypeError:
            if "pixel_values" in inputs:
                features = model.get_image_features(pixel_values=inputs["pixel_values"])
                if isinstance(features, torch.Tensor):
                    return features
        except Exception:
            pass

    if hasattr(model, "vision_model") and "pixel_values" in inputs:
        vision_outputs = model.vision_model(pixel_values=inputs["pixel_values"])
        for attr in ("image_embeds", "pooler_output", "last_hidden_state"):
            if hasattr(vision_outputs, attr):
                value = getattr(vision_outputs, attr)
                if isinstance(value, torch.Tensor):
                    return value[:, 0, :] if value.ndim == 3 else value

    outputs = model(**inputs)
    for attr in ("image_embeds", "pooler_output", "last_hidden_state"):
        if hasattr(outputs, attr):
            value = getattr(outputs, attr)
            if isinstance(value, torch.Tensor):
                return value[:, 0, :] if value.ndim == 3 else value

    if isinstance(outputs, (tuple, list)) and outputs and isinstance(outputs[0], torch.Tensor):
        value = outputs[0]
        return value[:, 0, :] if value.ndim == 3 else value

    raise RuntimeError("Could not extract image embeddings from model outputs.")


def run_embedding_extraction(
    dataset: Dataset,
    processor: Any,
    model: Any,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    use_amp: bool,
) -> Tuple[np.ndarray, List[str], List[Dict[str, str]]]:
    """Run batched embedding extraction with robust unreadable-file handling."""
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_identity,
    )

    all_embeddings: List[np.ndarray] = []
    valid_paths: List[str] = []
    skipped: List[Dict[str, str]] = []

    for batch in tqdm(loader, desc="Embedding input images"):
        valid = [item for item in batch if item.get("ok", False)]
        invalid = [item for item in batch if not item.get("ok", False)]

        for bad in invalid:
            skipped.append({"path": str(bad.get("path", "")), "error": str(bad.get("error", "unknown error"))})

        if not valid:
            continue

        images = [item["image"] for item in valid]
        paths = [str(item["path"]) for item in valid]

        inputs = processor(images=images, return_tensors="pt")
        inputs = {
            k: v.to(device, non_blocking=(device.type == "cuda"))
            for k, v in inputs.items()
            if isinstance(v, torch.Tensor)
        }

        with torch.no_grad():
            if use_amp and device.type == "cuda":
                with torch.amp.autocast(device_type="cuda"):
                    features = extract_image_features(model, inputs)
            else:
                features = extract_image_features(model, inputs)

        all_embeddings.append(features.detach().float().cpu().numpy().astype(np.float32, copy=False))
        valid_paths.extend(paths)

    if all_embeddings:
        X = np.concatenate(all_embeddings, axis=0)
    else:
        X = np.empty((0, 0), dtype=np.float32)

    return X, valid_paths, skipped


def save_skipped_log(skipped: Sequence[Dict[str, str]], output_path: Path) -> None:
    """Write a tab-delimited log for files skipped during embedding extraction."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in skipped:
            f.write(f"{row.get('path', '')}\t{row.get('error', '')}\n")


def softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable row-wise softmax."""
    shifted = x - np.max(x, axis=1, keepdims=True)
    e = np.exp(shifted)
    return e / np.sum(e, axis=1, keepdims=True)


def predict_proba_safe(classifier: Any, X: np.ndarray) -> np.ndarray:
    """Return prediction probabilities from classifier or compatible scores."""
    if hasattr(classifier, "predict_proba"):
        return classifier.predict_proba(X)
    if hasattr(classifier, "decision_function"):
        scores = classifier.decision_function(X)
        if scores.ndim == 1:
            probs_pos = 1.0 / (1.0 + np.exp(-scores))
            return np.stack([1.0 - probs_pos, probs_pos], axis=1)
        return softmax(scores)
    raise RuntimeError("Classifier does not support predict_proba or decision_function.")


def parse_candidate_labels(raw: Sequence[str] | None) -> List[str]:
    """Parse repeated/comma-separated label args into a deduplicated list."""
    if not raw:
        return []
    labels: List[str] = []
    for item in raw:
        parts = [p.strip() for p in str(item).split(",")]
        for part in parts:
            if part:
                labels.append(part)
    # Deduplicate while preserving user order.
    return list(dict.fromkeys(labels))


def select_label_subset(
    probs: np.ndarray,
    class_ids: List[int],
    id2label: Dict[int, str],
    candidate_labels: Sequence[str],
) -> Tuple[np.ndarray, List[int], List[str]]:
    """Restrict probabilities to user-requested class labels."""
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
        raise ValueError(
            f"Unknown candidate_labels: {missing}. "
            f"Available labels: {class_labels}"
        )
    if not selected_indices:
        raise ValueError("No valid candidate labels were selected.")

    subset_probs = probs[:, selected_indices]
    denom = np.sum(subset_probs, axis=1, keepdims=True)
    denom = np.where(denom > 0.0, denom, 1.0)
    subset_probs = subset_probs / denom

    subset_class_ids = [class_ids[i] for i in selected_indices]
    subset_class_labels = [class_labels[i] for i in selected_indices]
    return subset_probs, subset_class_ids, subset_class_labels


def main() -> None:
    """Standalone script entrypoint."""
    args = parse_args()
    logger = setup_logging()

    model_dir = Path(args.model_dir)
    input_folder = Path(args.input_folder)
    output_csv = Path(args.output_csv)

    model_path = model_dir / "model.joblib"
    mapping_path = model_dir / "label_mapping.json"
    config_path = model_dir / "config.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Missing model artifact: {model_path}")
    if not mapping_path.exists():
        raise FileNotFoundError(f"Missing label mapping: {mapping_path}")
    if not input_folder.exists():
        raise FileNotFoundError(f"Input folder not found: {input_folder}")

    with mapping_path.open("r", encoding="utf-8") as f:
        mapping = json.load(f)
    id2label = {int(k): v for k, v in mapping["id2label"].items()}

    model_id = "google/medsiglip-448"
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as f:
            config = json.load(f)
        model_id = str(config.get("model_id", model_id))
    if args.model_id_override:
        model_id = args.model_id_override

    image_paths = collect_image_files(input_folder)
    if not image_paths:
        raise RuntimeError(f"No supported image files found under: {input_folder}")

    device = get_device(args.device)
    use_amp = (not args.no_amp) and device.type == "cuda"
    logger.info("Using device: %s | AMP: %s", device, use_amp)
    logger.info("Found %d images in %s", len(image_paths), input_folder)

    logger.info("Loading image processor + MedSigLIP backbone: %s", model_id)
    processor = AutoImageProcessor.from_pretrained(model_id)
    backbone = AutoModel.from_pretrained(model_id).to(device).eval()

    logger.info("Loading saved classifier: %s", model_path)
    classifier = joblib.load(model_path)

    dataset = SafeFileImageDataset(image_paths)
    X, valid_paths, skipped = run_embedding_extraction(
        dataset=dataset,
        processor=processor,
        model=backbone,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_amp=use_amp,
    )

    if skipped:
        skipped_log = output_csv.with_suffix(".skipped.log")
        save_skipped_log(skipped, skipped_log)
        logger.warning("Skipped %d unreadable images. Log: %s", len(skipped), skipped_log)

    if X.shape[0] == 0:
        raise RuntimeError("No valid images were embedded.")

    probs = predict_proba_safe(classifier, X)
    class_ids = [int(x) for x in getattr(classifier, "classes_", sorted(id2label.keys()))]
    candidate_labels = parse_candidate_labels(args.candidate_labels)
    probs, class_ids, class_names = select_label_subset(
        probs=probs,
        class_ids=class_ids,
        id2label=id2label,
        candidate_labels=candidate_labels,
    )
    if candidate_labels:
        logger.info("Restricting predictions to candidate labels: %s", class_names)

    pred_col_idx = np.argmax(probs, axis=1)
    pred_ids = [class_ids[i] for i in pred_col_idx]
    pred_labels = [id2label.get(i, str(i)) for i in pred_ids]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    header = ["filename", "predicted_label"] + [f"prob_{name}" for name in class_names]
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i, path in enumerate(valid_paths):
            writer.writerow([path, pred_labels[i], *[f"{float(p):.6f}" for p in probs[i]]])

    logger.info("Inference complete. Wrote %d predictions to: %s", len(valid_paths), output_csv)


if __name__ == "__main__":
    main()

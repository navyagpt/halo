"""Core inference utilities shared by labeler and direct CLI execution."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Iterable, Optional

import joblib
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModel, AutoProcessor

from backend.models import get_image_features, get_text_features
from backend.utils import canonicalize_label_text, load_json

LOGGER = logging.getLogger(__name__)
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
PACKAGE_ROOT = Path(__file__).resolve().parent


def ensure_text_tokenizer_deps_available(mode: str) -> None:
    """Fail fast if text-tokenization deps are missing for contrastive-family modes."""
    text_modes = {"contrastive", "partial_unfreeze", "lora_optional"}
    if mode not in text_modes:
        return
    try:
        import sentencepiece  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "This inference mode requires MedSigLIP text tokenization and sentencepiece.\n"
            "Install it with: pip install sentencepiece"
        ) from exc
    try:
        import google.protobuf  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "This inference mode requires MedSigLIP text tokenization and protobuf.\n"
            "Install it with: pip install protobuf"
        ) from exc


def parse_args() -> argparse.Namespace:
    """Parse CLI args for standalone inference runs."""
    parser = argparse.ArgumentParser(description="Inference for fine-tuned MedSigLIP pill classifier")
    parser.add_argument("--run_dir", type=str, default=str(PACKAGE_ROOT))
    parser.add_argument("--input_path", type=str, required=True, help="Image file or directory")
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        choices=["contrastive", "linear_probe", "partial_unfreeze", "lora_optional"],
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--candidate_meds_file", type=str, default=None)
    parser.add_argument("--output_jsonl", type=str, default=None)
    return parser.parse_args()


def discover_model_processor_dirs(run_dir: Path) -> tuple[Path, Path]:
    """Locate model and processor artifacts across supported checkpoint layouts."""
    candidates = [
        (run_dir / "final" / "model", run_dir / "final" / "processor"),
        (run_dir / "checkpoints" / "final" / "model", run_dir / "checkpoints" / "final" / "processor"),
        (run_dir / "checkpoints" / "best" / "model", run_dir / "checkpoints" / "best" / "processor"),
        (run_dir / "models", run_dir / "processor"),
        (run_dir / "inference" / "models", run_dir / "inference" / "processor"),
    ]
    for model_dir, proc_dir in candidates:
        if model_dir.exists() and proc_dir.exists():
            return model_dir, proc_dir
    raise FileNotFoundError(
        (
            f"Could not find model/processor artifacts in {run_dir}. Expected one of: "
            "final/, checkpoints/{best,final}, models/processor, or inference/models+inference/processor."
        )
    )


def load_labels(run_dir: Path) -> list[str]:
    """Load class labels from ``labels.json`` supporting two schema variants."""
    labels_obj = load_json(run_dir / "labels.json")
    if "idx_to_label" in labels_obj:
        pairs = sorted(((int(k), v) for k, v in labels_obj["idx_to_label"].items()), key=lambda x: x[0])
        return [v for _, v in pairs]
    if "labels" in labels_obj:
        return list(labels_obj["labels"])
    raise ValueError("labels.json missing idx_to_label or labels.")


def list_images(input_path: Path) -> list[Path]:
    """Resolve input into image paths (single file or directory tree)."""
    if input_path.is_file():
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(f"input_path not found: {input_path}")
    imgs = [p for p in sorted(input_path.rglob("*")) if p.is_file() and p.suffix.lower() in IMG_EXTS]
    if not imgs:
        raise FileNotFoundError(f"No images found in {input_path} with extensions {sorted(IMG_EXTS)}")
    return imgs


def load_candidate_labels(candidate_file: Optional[str], default_labels: list[str]) -> list[str]:
    """Load optional candidate-label file, otherwise fall back to full label list."""
    if not candidate_file:
        return default_labels
    labels = []
    seen = set()
    with Path(candidate_file).open("r", encoding="utf-8") as f:
        for line in f:
            t = canonicalize_label_text(line)
            if t and t not in seen:
                labels.append(t)
                seen.add(t)
    if not labels:
        raise ValueError("candidate_meds_file contained no valid labels.")
    return labels


def batched(items: list, n: int):
    """Yield fixed-size chunks from ``items``."""
    for i in range(0, len(items), n):
        yield items[i : i + n]


def to_probs_from_logits(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Apply temperature-scaled softmax to classifier logits."""
    x = torch.tensor(logits / max(temperature, 1e-3), dtype=torch.float32)
    return torch.softmax(x, dim=-1).numpy()


def expand_probs_to_full_label_space(
    probs_seen: np.ndarray,
    seen_classes: np.ndarray,
    num_labels: int,
) -> np.ndarray:
    """Scatter seen-class probabilities into the full global label space."""
    out = np.zeros((probs_seen.shape[0], num_labels), dtype=np.float32)
    out[:, seen_classes.astype(int)] = probs_seen.astype(np.float32)
    return out


def build_output_record(path: str, labels: list[str], probs: np.ndarray, top_k: int, threshold: float) -> dict:
    """Build one prediction record with top-k outputs and abstain logic."""
    top_k = min(top_k, len(labels))
    order = np.argsort(-probs)[:top_k]
    top_labels = [labels[i] for i in order]
    top_scores = [float(probs[i]) for i in order]
    conf = float(top_scores[0])
    pred = top_labels[0]
    abstain = conf < threshold
    if abstain:
        pred = "UNKNOWN"

    return {
        "image_path": path,
        "top_k_labels": top_labels,
        "top_k_scores": top_scores,
        "predicted_label": pred,
        "confidence": conf,
        "abstain_flag": bool(abstain),
    }


def run_contrastive_inference(
    run_dir: Path,
    image_paths: list[Path],
    candidate_labels: list[str],
    top_k: int,
    threshold: float,
    batch_size: int,
) -> list[dict]:
    """Run image-text similarity inference directly against candidate medicine labels."""
    model_dir, proc_dir = discover_model_processor_dirs(run_dir)
    processor = AutoProcessor.from_pretrained(proc_dir)
    model = AutoModel.from_pretrained(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    resize = transforms.Resize((448, 448), interpolation=transforms.InterpolationMode.BICUBIC)

    with torch.no_grad():
        txt = processor(
            text=candidate_labels,
            return_tensors="pt",
            truncation=True,
            max_length=64,
            padding="max_length",
        )
        txt = {k: v.to(device) for k, v in txt.items()}
        text_feats = get_text_features(model, txt["input_ids"], txt.get("attention_mask"), normalize=True)

    records = []
    with torch.no_grad():
        for chunk in batched(image_paths, batch_size):
            pil_imgs = [resize(Image.open(p).convert("RGB")) for p in chunk]
            enc = processor(images=pil_imgs, return_tensors="pt")
            pixel_values = enc["pixel_values"].to(device)

            img_feats = get_image_features(model, pixel_values, normalize=True)
            sim = img_feats @ text_feats.T
            probs = torch.softmax(sim, dim=-1).cpu().numpy()

            for i, p in enumerate(chunk):
                rec = build_output_record(str(p), candidate_labels, probs[i], top_k=top_k, threshold=threshold)
                records.append(rec)

    return records


def run_linear_probe_inference(
    run_dir: Path,
    image_paths: list[Path],
    all_labels: list[str],
    candidate_labels: list[str],
    top_k: int,
    threshold: float,
    batch_size: int,
) -> list[dict]:
    """Run calibrated linear-probe inference and restrict output to candidate labels."""
    model_dir, proc_dir = discover_model_processor_dirs(run_dir)
    final_dir = run_dir / "final"
    clf_path = final_dir / "classifier.joblib"
    temp_path = final_dir / "calibration.json"

    if not clf_path.exists() or not temp_path.exists():
        raise FileNotFoundError(
            f"Linear-probe artifacts not found. Expected {clf_path} and {temp_path}."
        )

    clf = joblib.load(clf_path)
    # Temperature is learned during validation and stored separately.
    temperature = float(load_json(temp_path).get("temperature", 1.0))
    seen_classes = np.asarray(clf.classes_, dtype=np.int64)

    processor = AutoImageProcessor.from_pretrained(proc_dir)
    model = AutoModel.from_pretrained(model_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    resize = transforms.Resize((448, 448), interpolation=transforms.InterpolationMode.BICUBIC)

    label_to_idx = {l: i for i, l in enumerate(all_labels)}
    restricted_idx = [label_to_idx[l] for l in candidate_labels if l in label_to_idx]
    if not restricted_idx:
        raise ValueError("No candidate labels overlap with trained labels for linear_probe mode.")

    records = []
    with torch.no_grad():
        for chunk in batched(image_paths, batch_size):
            pil_imgs = [resize(Image.open(p).convert("RGB")) for p in chunk]
            enc = processor(images=pil_imgs, return_tensors="pt")
            pixel_values = enc["pixel_values"].to(device)

            emb = get_image_features(model, pixel_values, normalize=True).cpu().numpy()
            logits = clf.decision_function(emb)
            if logits.ndim == 1:
                logits = np.stack([-logits, logits], axis=1)
            probs_seen = to_probs_from_logits(logits, temperature=temperature)
            probs_all = expand_probs_to_full_label_space(
                probs_seen=probs_seen,
                seen_classes=seen_classes,
                num_labels=len(all_labels),
            )

            probs_sub = probs_all[:, restricted_idx]
            probs_sub = probs_sub / np.clip(probs_sub.sum(axis=1, keepdims=True), a_min=1e-8, a_max=None)
            labels_sub = [all_labels[i] for i in restricted_idx]

            for i, p in enumerate(chunk):
                rec = build_output_record(str(p), labels_sub, probs_sub[i], top_k=top_k, threshold=threshold)
                records.append(rec)

    return records


def main() -> None:
    """CLI entry point for standalone inference."""
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    run_dir = Path(args.run_dir)
    cfg_path = run_dir / "config.json"
    if cfg_path.exists():
        cfg = load_json(cfg_path)
        mode = args.mode or cfg.get("mode", "contrastive")
    else:
        mode = args.mode or "contrastive"
    ensure_text_tokenizer_deps_available(mode)

    input_paths = list_images(Path(args.input_path))
    labels = load_labels(run_dir)
    candidate_labels = load_candidate_labels(args.candidate_meds_file, labels)

    if mode in {"contrastive", "partial_unfreeze", "lora_optional"}:
        records = run_contrastive_inference(
            run_dir=run_dir,
            image_paths=input_paths,
            candidate_labels=candidate_labels,
            top_k=args.top_k,
            threshold=args.threshold,
            batch_size=args.batch_size,
        )
    elif mode == "linear_probe":
        records = run_linear_probe_inference(
            run_dir=run_dir,
            image_paths=input_paths,
            all_labels=labels,
            candidate_labels=candidate_labels,
            top_k=args.top_k,
            threshold=args.threshold,
            batch_size=args.batch_size,
        )
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    if args.output_jsonl:
        out = Path(args.output_jsonl)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    else:
        for r in records:
            print(json.dumps(r, ensure_ascii=False))


if __name__ == "__main__":
    main()

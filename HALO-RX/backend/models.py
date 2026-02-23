"""Model helpers for MedSigLIP loading, fine-tuning modes, and feature extraction."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

LOGGER = logging.getLogger(__name__)


@dataclass
class ModeSetupResult:
    """Summary of trainable-vs-total parameter counts after mode setup."""

    trainable_params: int
    total_params: int


def load_medsiglip(model_id: str, gradient_checkpointing: bool = False) -> nn.Module:
    """Load a pretrained MedSigLIP-compatible model."""
    model = AutoModel.from_pretrained(model_id)
    if gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        LOGGER.info("Enabled gradient checkpointing.")
    return model


def clip_symmetric_loss(logits_per_image: torch.Tensor, logits_per_text: torch.Tensor) -> torch.Tensor:
    """Compute the standard symmetric CLIP loss across image/text logits."""
    targets = torch.arange(logits_per_image.size(0), device=logits_per_image.device)
    loss_i = F.cross_entropy(logits_per_image, targets)
    loss_t = F.cross_entropy(logits_per_text, targets)
    return 0.5 * (loss_i + loss_t)


def _set_requires_grad(module: nn.Module, flag: bool) -> None:
    """Set ``requires_grad`` for every parameter under a module."""
    for p in module.parameters():
        p.requires_grad = flag


def freeze_all(model: nn.Module) -> None:
    """Freeze all parameters in the model."""
    _set_requires_grad(model, False)


def _get_vision_layers(model: nn.Module) -> list[nn.Module]:
    """Find vision encoder blocks across known MedSigLIP module layouts."""
    candidates = [
        "vision_model.encoder.layers",
        "vision_model.vision_model.encoder.layers",
    ]
    for key in candidates:
        cur = model
        ok = True
        for part in key.split("."):
            if not hasattr(cur, part):
                ok = False
                break
            cur = getattr(cur, part)
        if ok and isinstance(cur, Iterable):
            return list(cur)
    return []


def _unfreeze_by_name_contains(model: nn.Module, name_fragments: list[str]) -> int:
    """Unfreeze params whose names contain any provided fragments; return param count."""
    n = 0
    for name, p in model.named_parameters():
        if any(frag in name for frag in name_fragments):
            p.requires_grad = True
            n += p.numel()
    return n


def setup_training_mode(
    model: nn.Module,
    mode: str,
    partial_unfreeze_last_n: int = 2,
    freeze_text_tower: bool = False,
) -> ModeSetupResult:
    """Apply trainable/frozen parameter policy for the selected fine-tuning mode."""
    total = sum(p.numel() for p in model.parameters())

    if mode == "linear_probe":
        freeze_all(model)
    elif mode == "contrastive":
        _set_requires_grad(model, True)
        if freeze_text_tower:
            for name, p in model.named_parameters():
                if "text_model" in name:
                    p.requires_grad = False
            LOGGER.info("Contrastive mode: text tower frozen by request.")
    elif mode == "partial_unfreeze":
        freeze_all(model)
        unfrozen_proj = _unfreeze_by_name_contains(
            model,
            ["visual_projection", "text_projection", "logit_scale", "logit_bias", "projection"],
        )
        vision_layers = _get_vision_layers(model)
        if not vision_layers:
            LOGGER.warning("Could not find explicit vision layers; only projection layers are unfrozen.")
        else:
            for block in vision_layers[-partial_unfreeze_last_n:]:
                _set_requires_grad(block, True)
            if hasattr(model, "vision_model"):
                for maybe_name in ["post_layernorm", "layernorm"]:
                    maybe = getattr(model.vision_model, maybe_name, None)
                    if maybe is not None and isinstance(maybe, nn.Module):
                        _set_requires_grad(maybe, True)
        LOGGER.info(
            "Partial unfreeze enabled (projection + last %d image tower blocks).",
            partial_unfreeze_last_n,
        )
        LOGGER.info("Projection params unfrozen: %s", f"{unfrozen_proj:,}")
    elif mode == "lora_optional":
        _set_requires_grad(model, True)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    LOGGER.info("Trainable params: %s / %s", f"{trainable:,}", f"{total:,}")
    return ModeSetupResult(trainable_params=trainable, total_params=total)


def maybe_apply_lora(model: nn.Module, rank: int = 8, alpha: int = 16, dropout: float = 0.05) -> nn.Module:
    """Wrap model with LoRA adapters when optional LoRA mode is requested."""
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except Exception as exc:
        raise RuntimeError(
            "peft is not installed but mode=lora_optional was requested. "
            "Install peft or use a different mode."
        ) from exc

    lora_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def build_optimizer_param_groups(
    model: nn.Module,
    lr_backbone: float,
    lr_head: float,
    weight_decay: float,
) -> list[dict]:
    """Build optimizer groups with separate LR/weight-decay for head vs backbone."""
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    head_keys = ["projection", "visual_projection", "text_projection", "logit_scale", "logit_bias"]

    head_decay, head_nodecay = [], []
    body_decay, body_nodecay = [], []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_head = any(k in name for k in head_keys)
        is_no_decay = any(k in name for k in no_decay)

        if is_head and is_no_decay:
            head_nodecay.append(p)
        elif is_head:
            head_decay.append(p)
        elif is_no_decay:
            body_nodecay.append(p)
        else:
            body_decay.append(p)

    groups = []
    if body_decay:
        groups.append({"params": body_decay, "lr": lr_backbone, "weight_decay": weight_decay})
    if body_nodecay:
        groups.append({"params": body_nodecay, "lr": lr_backbone, "weight_decay": 0.0})
    if head_decay:
        groups.append({"params": head_decay, "lr": lr_head, "weight_decay": weight_decay})
    if head_nodecay:
        groups.append({"params": head_nodecay, "lr": lr_head, "weight_decay": 0.0})

    return groups


def _safe_l2_normalize(x: torch.Tensor) -> torch.Tensor:
    """Numerically stable L2 normalization helper."""
    return F.normalize(x, dim=-1, p=2)


def get_image_features(model: nn.Module, pixel_values: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    """Extract image embeddings from model, supporting two HF output APIs."""
    if hasattr(model, "get_image_features"):
        feats = model.get_image_features(pixel_values=pixel_values)
    else:
        out = model(pixel_values=pixel_values)
        feats = out.image_embeds
    if normalize:
        feats = _safe_l2_normalize(feats)
    return feats


def get_text_features(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    normalize: bool = True,
) -> torch.Tensor:
    """Extract text embeddings from model, supporting two HF output APIs."""
    if hasattr(model, "get_text_features"):
        feats = model.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
    else:
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        feats = out.text_embeds
    if normalize:
        feats = _safe_l2_normalize(feats)
    return feats


class MLPClassifier(nn.Module):
    """Small MLP head used for linear-probe style classification."""

    def __init__(self, in_dim: int, num_classes: int, hidden_dim: int = 512, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TemperatureScaler(nn.Module):
    """Post-hoc temperature calibration module for classification logits."""

    def __init__(self, init_temp: float = 1.0):
        super().__init__()
        self.log_temp = nn.Parameter(torch.log(torch.tensor([init_temp], dtype=torch.float32)))

    @property
    def temperature(self) -> torch.Tensor:
        return torch.exp(self.log_temp)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature.clamp(min=1e-3)

    def fit(self, logits: torch.Tensor, labels: torch.Tensor, max_iter: int = 200) -> float:
        """Optimize temperature with NLL objective and return calibrated value."""
        device = logits.device
        labels = labels.to(device)
        self.to(device)
        nll = nn.CrossEntropyLoss()
        opt = torch.optim.LBFGS([self.log_temp], lr=0.05, max_iter=max_iter)

        def closure():
            opt.zero_grad()
            loss = nll(self.forward(logits), labels)
            loss.backward()
            return loss

        opt.step(closure)
        return float(self.temperature.detach().cpu().item())

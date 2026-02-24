# Modified from the original implementation: https://github.com/NVlabs/vla0
# Authors: Gokul Puthumanaillam, Navya Gupta


from __future__ import annotations

"""Central model registry for train/eval/deploy entrypoints.

The goal of this module is to keep model-specific branching in one place so
the rest of the code remains pipeline-oriented instead of name-based if/else
blocks spread across files.
"""


MODEL_REGISTRY = {
    "qwen": {
        "class_name": "QwenActor",
        "cfg_node": "QWEN",
        "uses_dataset_stats": True,
        "supports_temperature": True,
    },
    "medgemma": {
        "class_name": "MedGemmaActor",
        "cfg_node": "MEDGEMMA",
        "uses_dataset_stats": True,
        "supports_temperature": True,
    },
}

CHECKPOINT_MODELS_WITH_DATASET_STATS = {"qwen", "medgemma", "dp", "qwen_dp"}


def get_model_spec(model_name: str):
    """Return static metadata for a model key."""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unsupported model: {model_name}")
    return MODEL_REGISTRY[model_name]


def build_model(model_name: str, cfg, models_module):
    """Instantiate the model class declared in the registry."""
    spec = get_model_spec(model_name)
    class_name = spec["class_name"]
    cfg_node_name = spec["cfg_node"]
    model_cls = getattr(models_module, class_name)
    if model_cls is None:
        raise ImportError(
            f"Model class {class_name} is unavailable. Check optional dependencies."
        )
    model_cfg = getattr(cfg.MODEL, cfg_node_name)
    return model_cls(**model_cfg)


def model_config_node(cfg):
    """Get the config subtree corresponding to cfg.EXP.MODEL."""
    spec = get_model_spec(cfg.EXP.MODEL)
    return getattr(cfg.MODEL, spec["cfg_node"])


def uses_dataset_stats(model_name: str) -> bool:
    if model_name in MODEL_REGISTRY:
        return bool(MODEL_REGISTRY[model_name]["uses_dataset_stats"])
    return model_name in CHECKPOINT_MODELS_WITH_DATASET_STATS


def checkpoint_includes_dataset_stats(model_name: str) -> bool:
    return model_name in CHECKPOINT_MODELS_WITH_DATASET_STATS


def supports_generation_temperature(model_name: str) -> bool:
    spec = get_model_spec(model_name)
    return bool(spec["supports_temperature"])


def action_horizon(cfg) -> int:
    return int(model_config_node(cfg).horizon)


def action_type(cfg):
    return model_config_node(cfg).action_type

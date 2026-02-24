# Classification Model Build Guide

This document explains how the bundled classifier was selected and how to rebuild a compatible classifier artifact for this pipeline.

## Scope

- This repository contains full inference/runtime code.
- It does not contain the original end-to-end training script used to produce `model.joblib`.
- Rebuild steps below are based on the saved artifacts and runtime expectations in code.

## Artifact Contract

The pipeline expects a model directory with these files:

- `model.joblib`: scikit-learn classifier object
- `label_mapping.json`: class index mappings
- `config.json`: model metadata and best-head summary
- `candidate_results.json`: benchmark results for tried heads

Current packaged model directory:
- `surgical_tool_pipeline/models/best_model`

## Current Packaged Model Metadata

From `surgical_tool_pipeline/models/best_model/config.json`:

- Build timestamp (UTC): `2026-02-18T05:04:07.188196+00:00`
- Embedding backbone: `google/medsiglip-448`
- Embedding normalization: `l2norm`
- Seed: `42`
- Best head: `logreg_C=1.0`
- Validation accuracy: `0.99609375`
- Validation macro-F1: `0.9960935115668681`

From `label_mapping.json`:

- `0 -> forceps`
- `1 -> hemostat`
- `2 -> scalpel`
- `3 -> scissors`

## Candidate Heads Evaluated

From `candidate_results.json`, candidate families included:

- Multinomial logistic regression (`lbfgs`) with C in:
  - `0.01, 0.1, 1.0, 5.0, 10.0, 50.0`
- Calibrated linear SVM with C in:
  - `0.01, 0.1, 1.0, 5.0, 10.0`
- MLP with hidden sizes and alpha in:
  - `(128,), alpha={1e-4, 1e-3}`
  - `(256,), alpha={1e-4, 1e-3}`
  - `(256, 128), alpha={1e-4, 1e-3}`

Selected winner in packaged artifacts:
- `logreg_C=1.0`

## Rebuild Procedure

### 1) Prepare labeled dataset

Use class folders for the same label vocabulary as `label_mapping.json`:

```text
<dataset_root>/
  forceps/
  hemostat/
  scalpel/
  scissors/
```

Each folder should contain crop-like images similar to runtime detector outputs.

### 2) Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-combined.txt
```

If needed for HF access:

```bash
export HF_TOKEN="hf_xxx_your_token_here"
```

### 3) Generate MedSigLIP embeddings

Use the same image backbone:
- `google/medsiglip-448`

Runtime-compatible embedding extraction logic already exists in:
- `surgical_tool_pipeline/medsiglip_infer.py`

Core expectation for compatibility:
- apply L2 normalization to each embedding vector before training candidate heads.

### 4) Train candidate heads

Train and evaluate the same candidate families shown above, with fixed seed `42`.

Recommended selection policy (matches artifact intent):
1. Maximize validation macro-F1.
2. Break ties with validation accuracy.
3. If still tied, choose the simplest head (e.g., logistic regression).

### 5) Save artifacts with the same schema

Write to a directory, e.g. `<new_model_dir>/`:

- `model.joblib`
- `label_mapping.json`
- `candidate_results.json`
- `config.json`

`label_mapping.json` format:

```json
{
  "id2label": {
    "0": "forceps",
    "1": "hemostat",
    "2": "scalpel",
    "3": "scissors"
  },
  "label2id": {
    "forceps": 0,
    "hemostat": 1,
    "scalpel": 2,
    "scissors": 3
  }
}
```

`config.json` should include at least:

- `model_id`
- `embedding_normalization_steps`
- `best_head` (name/type/hyperparameters + val metrics)
- `seed`
- artifact filenames

### 6) Validate with runtime inference

Run standalone classifier inference:

```bash
python -m surgical_tool_pipeline.medsiglip_infer \
  --model_dir <new_model_dir> \
  --input_folder assets/samples/images \
  --output_csv outputs/reports/preds_from_input.csv \
  --device auto
```

Then run full pipeline using the rebuilt model:

```bash
python -m surgical_tool_pipeline.cli \
  --input assets/samples/images/image1.jpg \
  --model_dir <new_model_dir> \
  --output_dir pipeline_outputs \
  --device auto
```

## Reproducibility Checklist

- Keep label names identical to runtime expectations.
- Keep embedding backbone constant unless deliberately migrating.
- Keep normalization step (`l2norm`) consistent.
- Record exact seed and candidate grid.
- Save candidate metrics to `candidate_results.json` for auditability.

## Notes on Model Migration

If you switch backbone model ID:

- update `config.json` in the model directory,
- optionally pass `--model_id_override` at runtime,
- re-evaluate candidate heads because embedding geometry changes.

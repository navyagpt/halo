# HALO-Act

The system predicts robot action trajectories by generating discretized action tokens from multimodal context (images + instruction), then decoding them back to continuous control values.

## Core Design

- **Backbone**: MedGemma causal multimodal language model.
- **Action representation**: continuous actions are discretized into integer bins and serialized as text.
- **Training objective**: causal next-token prediction on assistant action tokens only.
- **Inference**: constrained numeric generation followed by de-quantization to continuous actions.

## Repository Layout

- `rv_train/models/medgemma/model.py`: policy orchestrator (`MedGemmaPolicy` / `MedGemmaActor`).
- `rv_train/pipelines/medgemma/action_codec.py`: action binning and de-binning.
- `rv_train/pipelines/medgemma/vision_adapter.py`: RGB batch validation and image conversion.
- `rv_train/pipelines/medgemma/chat_io.py`: dialog construction and processor input assembly.
- `rv_train/pipelines/medgemma/training_ops.py`: label masking and causal LM loss prep.
- `rv_train/pipelines/medgemma/model_loader.py`: HF model/processor loading and checkpoint restore.
- `rv_train/train.py`: training entrypoint.
- `eval/eval_libero.py`: evaluation entrypoint.
- `rv_train/deploy/service.py`: FastAPI inference service.

## End-to-End Architecture

```text
Dataset sample
  -> vision adapter (history/camera formatting)
  -> action codec (continuous -> binned integer text)
  -> chat I/O (system/user/assistant message build)
  -> MedGemma processor (tokenization + image features)
  -> MedGemma LM forward
  -> label builder (mask prompt/pad, keep action supervision)
  -> causal LM loss

Inference
  -> same encoding path
  -> constrained numeric generation
  -> action codec decode (text -> continuous horizon actions)
```

## Environment Setup

```bash
conda create -y -n medgemma_policy python=3.10
conda activate medgemma_policy
PIP_REQ_EXTRAS=medgemma,libero pip install --no-build-isolation -e ".[medgemma,libero]"
cd libs/RoboVerse
PIP_REQ_EXTRAS=lerobot pip install --no-build-isolation -e ".[lerobot]"
cd ../..
```

## Configuration

Primary experiment config: `configs/medgemma.yaml`

Important fields:

- `EXP.MODEL: "medgemma"`
- `MODEL.MEDGEMMA.medgemma_model_id`
- `MODEL.MEDGEMMA.horizon`
- `MODEL.MEDGEMMA.original_action_dim`
- `MODEL.MEDGEMMA.num_bins_actions`
- `DATALOADER.ROBOVERSE.cfg_path`

## Training

```bash
python -m rv_train.train --exp-config ./configs/medgemma.yaml
```

## Evaluation

```bash
python eval/eval_libero.py \
  --model_path ./runs/medgemma/model_last.pth \
  --task_suite_name libero_goal \
  --task_name put_the_wine_bottle_on_top_of_the_cabinet
```

## Deployment API

```bash
ROBOVERSE_DEPLOY_CHECKPOINT=./runs/medgemma/model_last.pth python rv_train/deploy/service.py
```

API docs are available at `http://<server_ip>:10000/docs`.

## Local Mac Smoke Test

Use this to validate wiring (forward/backward/generation) without full-scale training:

```bash
uv run --with torch python -m rv_train.models.medgemma.smoke_test
```

## Engineering Invariants

- Action normalization/de-normalization must always use dataset stats from `dataset_stats.pkl`.
- Prompt/user/system tokens must not contribute to loss.
- Generated text must decode into exactly one action horizon (pad/trim behavior is explicit in codec).
- Model-selection logic is centralized in `rv_train/model_specs.py`.

## Acknowledgment

This project is modified from the original implementation at: https://github.com/NVlabs/vla0

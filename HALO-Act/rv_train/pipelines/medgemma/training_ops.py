# Modified from the original implementation: https://github.com/NVlabs/vla0
# Authors: Gokul Puthumanaillam, Navya Gupta


from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List

import torch


@dataclass
class LabelMaskConfig:
    action_mask_aug_per: float


class LossLabelBuilder:
    """Builds shifted LM labels that only supervise assistant action tokens."""

    def __init__(self, tokenizer, label_mask_cfg: LabelMaskConfig):
        self.tokenizer = tokenizer
        self.mask_cfg = label_mask_cfg

    def build_labels(
        self,
        model_inputs,
        prompt_token_lens,
        action_texts: List[str],
    ):
        """Mask prompt/pad tokens and optionally mask a subset of action tokens."""
        labels = model_inputs["input_ids"].clone()
        pad_token_id = self.tokenizer.pad_token_id

        for idx, action_text in enumerate(action_texts):
            prompt_len = prompt_token_lens[idx]

            if self.tokenizer.padding_side == "right":
                labels[idx, :prompt_len] = -100
                action_start_idx = prompt_len
            elif self.tokenizer.padding_side == "left":
                num_pad_tokens = 0 if pad_token_id is None else (labels[idx] == pad_token_id).sum().item()
                labels[idx, num_pad_tokens : num_pad_tokens + prompt_len] = -100
                action_start_idx = num_pad_tokens + prompt_len
            else:
                raise ValueError(f"Unknown padding side: {self.tokenizer.padding_side}")

            if random.random() < 0.1:
                mask_ratio = 0.0
            else:
                mask_ratio = random.uniform(0.0, self.mask_cfg.action_mask_aug_per)

            tok_ids = self.tokenizer(
                action_text,
                add_special_tokens=False,
                return_attention_mask=False,
            )["input_ids"]
            total_action_tokens = len(tok_ids)
            num_to_mask = int(total_action_tokens * mask_ratio)

            if num_to_mask <= 0:
                continue

            indices = random.sample(range(total_action_tokens), num_to_mask)
            indices = [action_start_idx + i for i in indices if action_start_idx + i < labels.shape[1]]
            labels[idx, indices] = -100

            replacement_id = self.tokenizer.unk_token_id
            if replacement_id is None:
                replacement_id = self.tokenizer.pad_token_id
            if replacement_id is None:
                replacement_id = self.tokenizer.eos_token_id
            if replacement_id is None:
                replacement_id = 0
            model_inputs["input_ids"][idx, indices] = replacement_id

        if pad_token_id is not None:
            labels[labels == pad_token_id] = -100

        return labels


def compute_causal_lm_loss(logits, labels, loss_fn):
    """Compute next-token loss with standard shift-left LM alignment."""
    logits = logits.float()
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    shift_labels = shift_labels.view(-1).to(shift_logits.device)
    return loss_fn(shift_logits, shift_labels)

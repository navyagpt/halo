# Modified from the original implementation: https://github.com/NVlabs/vla0
# Authors: Gokul Puthumanaillam, Navya Gupta


from __future__ import annotations

import warnings

import torch

try:
    from transformers import LogitsProcessor
except ImportError:

    class LogitsProcessor:  # type: ignore[override]
        pass


class NumericTokenLogitsMask(LogitsProcessor):
    """Constrains generated tokens to digits, spaces, and EOS where possible."""

    def __init__(self, tokenizer):
        self.allowed_tokens = set()
        for digit in range(10):
            tok = tokenizer.encode(str(digit), add_special_tokens=False)
            if len(tok) == 1:
                self.allowed_tokens.add(tok[0])

        space_tok = tokenizer.encode(" ", add_special_tokens=False)
        if len(space_tok) == 1:
            self.allowed_tokens.add(space_tok[0])

        if tokenizer.eos_token_id is not None:
            self.allowed_tokens.add(tokenizer.eos_token_id)

        self.enabled = len(self.allowed_tokens) > 0
        if not self.enabled:
            warnings.warn(
                "Tokenizer does not expose stable single-token digits/spaces. "
                "Numeric generation mask is disabled."
            )

    def __call__(self, input_ids, scores):
        if not self.enabled:
            return scores

        mask = torch.full_like(scores, float("-inf"))
        for token_id in self.allowed_tokens:
            if token_id < scores.shape[1]:
                mask[:, token_id] = 0
        return scores + mask

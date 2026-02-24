# Modified from the original implementation: https://github.com/NVlabs/vla0
# Authors: Gokul Puthumanaillam, Navya Gupta


from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch

import rv_train.constants as C


@dataclass
class ActionCodecConfig:
    action_type: str
    act_dim: int
    horizon: int
    num_bins_actions: int


class ActionTextCodec:
    """Converts continuous actions <-> integer text strings used by the VLM."""

    def __init__(self, cfg: ActionCodecConfig):
        self.cfg = cfg
        self.original_dataset_stats = None
        self.dataset_stats = None
        self._min_act_cache = None
        self._max_act_cache = None

    def set_dataset_stats(self, dataset_stats: dict):
        """Store dataset normalization stats used for binning and debinning actions."""
        self.original_dataset_stats = dataset_stats
        if dataset_stats == {}:
            return
        if self.cfg.action_type != C.ORIGINAL:
            raise NotImplementedError(
                f"Action type {self.cfg.action_type} is not implemented"
            )
        self.dataset_stats = dataset_stats["out_ori_act"]

    def require_stats(self):
        """Guardrail to avoid training/inference without action bounds."""
        if self.dataset_stats is None:
            raise RuntimeError("dataset_stats must be set before calling the model")

    def _get_bounds(self, device: torch.device):
        if (
            self._min_act_cache is None
            or self._max_act_cache is None
            or self._min_act_cache.device != device
            or self._max_act_cache.device != device
        ):
            self._min_act_cache = torch.tensor(self.dataset_stats["min"], device=device)
            self._max_act_cache = torch.tensor(self.dataset_stats["max"], device=device)
        return self._min_act_cache, self._max_act_cache

    def encode(self, actions: torch.Tensor) -> List[str]:
        """Map continuous action tensors into space-separated integer sequences."""
        self.require_stats()
        min_act, max_act = self._get_bounds(actions.device)
        if not (torch.all(min_act <= actions) and torch.all(actions <= max_act)):
            raise AssertionError(f"Action out of range: {actions}")

        scaled = (actions - min_act) / (max_act - min_act)
        binned = torch.round(scaled * self.cfg.num_bins_actions).long()
        binned = binned.reshape(binned.shape[0], -1)
        return [" ".join(map(str, row.tolist())) for row in binned]

    def decode(self, action_texts: List[str]) -> torch.Tensor:
        """Map generated integer text back to continuous action tensors."""
        self.require_stats()
        bs = len(action_texts)
        min_act = torch.tensor(self.dataset_stats["min"])
        max_act = torch.tensor(self.dataset_stats["max"])

        try:
            stripped = [x.strip() for x in action_texts]
            pieces = [[t for t in txt.split(" ") if t != ""] for txt in stripped]
            tensor = torch.tensor(
                [[int(t) for t in row] for row in pieces], dtype=torch.float32
            )

            if bs == 1 and (len(tensor[0]) % self.cfg.act_dim != 0):
                trim = len(tensor[0]) - (len(tensor[0]) % self.cfg.act_dim)
                tensor = tensor[0][:trim][None, :]

            tensor = tensor.reshape(bs, -1, self.cfg.act_dim)
            if tensor.shape[1] < self.cfg.horizon:
                repeat = self.cfg.horizon - tensor.shape[1]
                tensor = torch.cat([tensor, tensor[:, -1:].repeat(1, repeat, 1)], dim=1)
            if tensor.shape[1] > self.cfg.horizon:
                tensor = tensor[:, : self.cfg.horizon]

            return ((tensor / self.cfg.num_bins_actions) * (max_act - min_act)) + min_act
        except Exception as exc:
            print(f"Error parsing action text: {exc}")
            print(action_texts)
            return ((min_act + max_act) / 2).repeat(bs, self.cfg.horizon, 1)

    @property
    def act_dim(self) -> int:
        return self.cfg.act_dim

    @property
    def horizon(self) -> int:
        return self.cfg.horizon

    @property
    def num_bins_actions(self) -> int:
        return self.cfg.num_bins_actions

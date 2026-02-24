# Modified from the original implementation: https://github.com/NVlabs/vla0
# Authors: Gokul Puthumanaillam, Navya Gupta


from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import torch
from PIL import Image


@dataclass
class VisionLayoutConfig:
    rgb_input: bool
    history: int
    num_cam: int
    rgb_img_size: tuple
    tiled_rgb_imgs: bool


class VisionBatchAdapter:
    """Validates vision tensors and converts batch tensors into PIL image lists."""

    def __init__(self, cfg: VisionLayoutConfig):
        self.cfg = cfg

    def validate_inputs(self, pc, rgb_pc, rgb):
        if pc is not None or rgb_pc is not None:
            raise AssertionError("Point cloud inputs are not supported in MedGemma pipeline")

        if self.cfg.rgb_input:
            if rgb is None:
                raise AssertionError("rgb input is required")
            expected_shape = (
                self.cfg.history,
                self.cfg.num_cam,
                *self.cfg.rgb_img_size,
                3,
            )
            if rgb.shape[1:] != expected_shape:
                raise AssertionError(
                    f"Unexpected rgb shape {rgb.shape}, expected (*, {expected_shape})"
                )
            if not ((rgb.min() >= -1e-2) and (1.99 <= rgb.max() <= 255.01)):
                raise AssertionError(
                    f"rgb value range is invalid: min={rgb.min()}, max={rgb.max()}"
                )
        else:
            if rgb is not None:
                raise AssertionError("rgb must be None when rgb_input=False")

    def to_pil_batches(self, rgb: torch.Tensor, batch_size: int) -> List[List[Image.Image]]:
        pil_batches = [[] for _ in range(batch_size)]
        if not self.cfg.rgb_input:
            return pil_batches

        for sample_idx, rgb_sample in enumerate(rgb):
            flat_imgs = []
            for hist_idx in range(self.cfg.history):
                for cam_idx in range(self.cfg.num_cam):
                    flat_imgs.append(rgb_sample[hist_idx][cam_idx])

            if self.cfg.tiled_rgb_imgs:
                flat_imgs = [self.tile_images(flat_imgs)]

            pil_batches[sample_idx] = [
                Image.fromarray(img.cpu().numpy().astype(np.uint8)) for img in flat_imgs
            ]

        return pil_batches

    @staticmethod
    def tile_images(images: List[torch.Tensor]) -> torch.Tensor:
        for img in images:
            if len(img.shape) != 3 or img.shape[2] != 3:
                raise AssertionError(f"Invalid image shape for tiling: {img.shape}")

        heights, widths = zip(*[img.shape[:2] for img in images])
        max_height = max(heights)
        total_width = sum(widths)
        canvas = torch.zeros((max_height, total_width, 3), device=images[0].device)

        current_x = 0
        for img in images:
            canvas[: img.shape[0], current_x : current_x + img.shape[1], :] = img
            current_x += img.shape[1]

        return canvas

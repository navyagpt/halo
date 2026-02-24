# Modified from the original implementation: https://github.com/NVlabs/vla0
# Authors: Gokul Puthumanaillam, Navya Gupta


import os
import time
from contextlib import nullcontext
from typing import List, Optional

import numpy as np
import roboverse
import roboverse.constants as rbc
import torch
from roboverse.unifiers.image_unifier import image_unifier_transform
from roboverse.utils.unifier_utils import remove_keys

from rv_train.train import get_pretrained_model

DEFAULT_CHECKPOINT = "./runs/vla0/model_last.pth"
DEFAULT_SINGLE_CAMERA = "False"


ROBOVERSE_DEPLOY_CHECKPOINT = os.getenv(
    "ROBOVERSE_DEPLOY_CHECKPOINT", DEFAULT_CHECKPOINT
)
ROBOVERSE_SINGLE_CAMERA = (
    os.getenv("ROBOVERSE_SINGLE_CAMERA", "False").lower() == "true"
)


def get_inference_device():
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(checkpoint: str):
    device = get_inference_device()
    model, cfg = get_pretrained_model(
        checkpoint,
        0 if device.type == "cuda" else device.type,
        torch_compile=(device.type == "cuda"),
    )
    model.eval()
    return model, cfg


def transform_from_so100(
    image_rgb: List[np.ndarray],
    state: List[float],
    instr: Optional[str],
    cfg,
    device: torch.device,
):
    """Does the unification from SO100 data to roboverse data"""
    data_sample = {
        "rgb": np.concatenate([x[None, None] for x in image_rgb], 1),
        "instr": instr,
    }
    data_sample = remove_keys(data_sample, rbc.REQUIRED_KEYS_3D_COMPATIBLE)
    if ROBOVERSE_SINGLE_CAMERA:
        sample_cam_list = [rbc.THREE_P1]
    else:
        sample_cam_list = [rbc.THREE_P1, rbc.THREE_P2]
    unified_sample = image_unifier_transform(
        cfg=roboverse.main.get_cfg(
            cfg.DATALOADER.ROBOVERSE.cfg_path, cfg.DATALOADER.ROBOVERSE.cfg_opts
        ),
        sample=data_sample,
        sample_cam_list=sample_cam_list,
        eval=True,
    )
    unified_sample["instr"] = [unified_sample["instr"]]
    for k in [k for k in unified_sample.keys() if k != "instr"]:
        unified_sample[k] = torch.tensor(unified_sample[k][None], device=device, dtype=torch.float)
    print(f"Unified sample: {unified_sample}")
    print(unified_sample["rgb"].shape)
    return unified_sample


def get_so100_action(output_data) -> List[float]:
    """Does the unification from roboverse output to SO100 data"""
    action = output_data["out_ori_act"].squeeze(0).detach().cpu().numpy()
    print(action)
    return action.tolist()


class RoboverseModelManager:
    def __init__(self, checkpoint=ROBOVERSE_DEPLOY_CHECKPOINT):
        self.model, self.cfg = load_model(checkpoint)
        self.device = next(self.model.parameters()).device

    def model_act(self, *args, **kwargs):
        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True)
            if self.device.type == "cuda"
            else nullcontext()
        )
        with autocast_ctx:
            out = self.model(*args, **kwargs, get_loss=False, get_action=True)
        return out


class So100ModelManager(RoboverseModelManager):
    def forward(
        self,
        image_rgb: List[np.ndarray],
        state: List[float],
        instr: Optional[str] = None,
        get_one_step_action: bool = False,
        last_action_txt: str = "",
    ) -> List[float]:
        data_input_batch = transform_from_so100(
            image_rgb, state, instr, self.cfg, self.device
        )
        start_time = time.time()
        output_data = self.model_act(
            **data_input_batch,
            get_one_step_action=get_one_step_action,
            last_action_txt=last_action_txt,
        )
        print(f"Model time taken: {time.time() - start_time}")

        if get_one_step_action:
            out = get_so100_action(output_data), output_data["pred_action_txt"][0]
        else:
            out = get_so100_action(output_data), None

        return out

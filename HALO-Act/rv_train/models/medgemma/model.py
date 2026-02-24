# Modified from the original implementation: https://github.com/NVlabs/vla0
# Authors: Gokul Puthumanaillam, Navya Gupta


from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
from torch import nn

import rv_train.constants as C
from rv_train.pipelines.medgemma import (
    ActionCodecConfig,
    ActionTextCodec,
    LabelMaskConfig,
    LossLabelBuilder,
    MedGemmaChatIO,
    MedGemmaModelLoader,
    NumericTokenLogitsMask,
    VisionBatchAdapter,
    VisionLayoutConfig,
    compute_causal_lm_loss,
    model_device,
)


@dataclass
class MedGemmaRuntimeConfig:
    medgemma_model_id: str
    action_type: str
    original_action_dim: int
    horizon: int
    history: int
    use_lora: bool
    use_qlora: bool
    num_cam: int
    lora_config: str
    lora_rank: int
    rgb_input: bool
    rgb_img_size: tuple
    add_vision_id: bool
    tiled_rgb_imgs: bool
    num_bins_actions: int
    use_flash_attention_2: bool
    action_mask_aug_per: float
    attention_dropout: float


class MedGemmaPolicy(nn.Module):
    """MedGemma VLA policy with a modular encode/train/decode pipeline."""

    def __init__(
        self,
        medgemma_model_id,
        action_type,
        original_action_dim,
        horizon,
        history,
        use_lora=True,
        use_qlora=True,
        num_cam=1,
        lora_config="",
        lora_rank=8,
        rgb_input=False,
        rgb_img_size=(84, 84),
        add_vision_id=False,
        tiled_rgb_imgs=False,
        num_bins_actions=1000,
        use_flash_attention_2=False,
        system_message_version=1,
        action_mask_aug=0,
        action_mask_aug_per=0.1,
        attention_dropout=0.0,
    ):
        super().__init__()

        self.runtime = MedGemmaRuntimeConfig(
            medgemma_model_id=medgemma_model_id,
            action_type=action_type,
            original_action_dim=original_action_dim,
            horizon=horizon,
            history=history,
            use_lora=use_lora,
            use_qlora=use_qlora,
            num_cam=num_cam,
            lora_config=lora_config,
            lora_rank=lora_rank,
            rgb_input=rgb_input,
            rgb_img_size=rgb_img_size,
            add_vision_id=add_vision_id,
            tiled_rgb_imgs=tiled_rgb_imgs,
            num_bins_actions=num_bins_actions,
            use_flash_attention_2=use_flash_attention_2,
            action_mask_aug_per=action_mask_aug_per,
            attention_dropout=attention_dropout,
        )

        self._validate_config()

        # keep checkpoint loading behavior used by training launcher
        self.load_param_before_ddp = True

        self.medgemma_model_id = self.runtime.medgemma_model_id
        self.action_type = self.runtime.action_type
        self.horizon = self.runtime.horizon
        self.use_lora = self.runtime.use_lora
        self.use_qlora = self.runtime.use_qlora
        self.lora_rank = self.runtime.lora_rank
        self.use_flash_attention_2 = self.runtime.use_flash_attention_2
        self.attention_dropout = self.runtime.attention_dropout

        self.act_dim = (
            self.runtime.original_action_dim
            if self.runtime.action_type == C.ORIGINAL
            else 7
        )

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

        self.min_pixel = self.runtime.rgb_img_size[0] * self.runtime.rgb_img_size[1]
        self.max_pixel = self.min_pixel
        if self.runtime.rgb_input and self.runtime.tiled_rgb_imgs:
            self.min_pixel *= self.runtime.history * self.runtime.num_cam
            self.max_pixel *= self.runtime.history * self.runtime.num_cam

        self.model = self.load_medgemma_model(
            medgemma_model_id=self.runtime.medgemma_model_id,
            use_lora=self.runtime.use_lora,
            use_qlora=self.runtime.use_qlora,
            lora_config=self.runtime.lora_config,
            lora_rank=self.runtime.lora_rank,
            use_flash_attention_2=self.runtime.use_flash_attention_2,
            attention_dropout=self.runtime.attention_dropout,
        )
        self.processor = self.load_medgemma_processor(
            medgemma_model_id=self.runtime.medgemma_model_id,
            min_pixel=self.min_pixel,
            max_pixel=self.max_pixel,
            padding_side="left" if self.runtime.use_flash_attention_2 else None,
        )

        self.codec = ActionTextCodec(
            ActionCodecConfig(
                action_type=self.runtime.action_type,
                act_dim=self.act_dim,
                horizon=self.runtime.horizon,
                num_bins_actions=self.runtime.num_bins_actions,
            )
        )
        self.vision = VisionBatchAdapter(
            VisionLayoutConfig(
                rgb_input=self.runtime.rgb_input,
                history=self.runtime.history,
                num_cam=self.runtime.num_cam,
                rgb_img_size=self.runtime.rgb_img_size,
                tiled_rgb_imgs=self.runtime.tiled_rgb_imgs,
            )
        )
        self.chat_io = MedGemmaChatIO(
            processor=self.processor,
            add_vision_id=self.runtime.add_vision_id,
        )
        self.label_builder = LossLabelBuilder(
            tokenizer=self.processor.tokenizer,
            label_mask_cfg=LabelMaskConfig(
                action_mask_aug_per=self.runtime.action_mask_aug_per,
            ),
        )
        self.logits_processor = NumericTokenLogitsMask(self.processor.tokenizer)

        self.system_message = self._build_system_message()
        self.num_tokens = 1024

    def _validate_config(self):
        """Validate runtime flags before allocating model resources."""
        if self.runtime.use_qlora and not self.runtime.use_lora:
            raise AssertionError("use_lora must be True when use_qlora=True")
        if self.runtime.attention_dropout > 0.0 and self.runtime.use_lora:
            raise AssertionError(
                "attention_dropout is only supported when use_lora is False"
            )
        if self.runtime.lora_config not in ["", "default"]:
            raise AssertionError("lora_config must be either '' or 'default'")

    def _build_system_message(self):
        """Construct the fixed instruction that binds output format to action bins."""
        return (
            "Analyze the input image and predict robot actions for the next "
            f"{self.runtime.horizon} timesteps. Each action has {self.act_dim} dimensions. "
            f"Output a single sequence of {self.runtime.horizon * self.act_dim} integers "
            f"(0-{self.runtime.num_bins_actions} each), representing the "
            f"{self.runtime.horizon} timesteps sequentially. Provide only space separated "
            "numbers. Nothing else."
        )

    @staticmethod
    def load_medgemma_model(
        medgemma_model_id,
        use_lora,
        use_qlora,
        lora_config,
        lora_rank,
        use_flash_attention_2,
        attention_dropout,
    ):
        return MedGemmaModelLoader.create_model(
            model_id=medgemma_model_id,
            use_lora=use_lora,
            use_qlora=use_qlora,
            lora_config=lora_config,
            lora_rank=lora_rank,
            use_flash_attention_2=use_flash_attention_2,
            attention_dropout=attention_dropout,
        )

    @staticmethod
    def load_medgemma_processor(
        medgemma_model_id,
        min_pixel,
        max_pixel,
        padding_side,
    ):
        return MedGemmaModelLoader.create_processor(
            model_id=medgemma_model_id,
            min_pixels=min_pixel,
            max_pixels=max_pixel,
            padding_side=padding_side,
        )

    @property
    def original_dataset_stats(self):
        return self.codec.original_dataset_stats

    def set_dataset_stats(self, dataset_stats):
        # The codec owns normalization state; keep one source of truth there.
        self.codec.set_dataset_stats(dataset_stats)

    def _check_mode_arguments(
        self,
        instr,
        get_loss,
        get_action,
        get_one_step_action,
        last_action_txt,
    ):
        """Validate forward mode combinations used by train vs inference."""
        self.codec.require_stats()
        if not isinstance(instr, list):
            raise AssertionError("instr must be a list")
        if get_loss and get_action:
            raise AssertionError("get_loss and get_action cannot both be True")
        if not get_loss and not get_action:
            raise AssertionError("Either get_loss or get_action must be True")
        if get_one_step_action:
            if not get_action:
                raise AssertionError("get_one_step_action requires get_action=True")
            if not isinstance(last_action_txt, str):
                raise AssertionError("last_action_txt must be a string")
            if len(instr) != 1:
                raise AssertionError("one-step generation only supports batch size 1")

    def _build_action_texts(self, out_ori_act, batch_size):
        """Encode continuous action labels into the integer text supervision format."""
        if out_ori_act is None:
            return [""] * batch_size
        return self.codec.encode(out_ori_act)

    def _build_batch_payload(
        self,
        instr,
        rgb,
        out_ori_act,
        drop_assistant,
        add_generation_prompt,
    ):
        """Assemble dialog + processor tensors for either training or generation."""
        batch_size = len(instr)
        image_batches = self.vision.to_pil_batches(rgb=rgb, batch_size=batch_size)
        action_texts = self._build_action_texts(out_ori_act=out_ori_act, batch_size=batch_size)
        dialogs = self.chat_io.build_dialogs(
            system_message=self.system_message,
            image_batches=image_batches,
            instructions=instr,
            action_texts=action_texts,
            drop_assistant=drop_assistant,
        )
        inputs = self.chat_io.build_model_inputs(
            dialogs=dialogs,
            add_generation_prompt=add_generation_prompt,
            model_device=model_device(self.model),
        )
        return dialogs, action_texts, inputs

    def _training_step(self, dialogs, action_texts, model_inputs):
        """Compute causal LM loss on assistant action tokens only."""
        prompt_lens = self.chat_io.prompt_token_lengths(dialogs)
        labels = self.label_builder.build_labels(
            model_inputs=model_inputs,
            prompt_token_lens=prompt_lens,
            action_texts=action_texts,
        )
        outputs = self.model(**model_inputs)
        loss = compute_causal_lm_loss(outputs.logits, labels, self.loss_fn)
        return {"loss": loss}

    def _generation_step(
        self,
        instr,
        action_texts,
        model_inputs,
        generate_temperature,
        get_one_step_action,
        last_action_txt,
    ):
        """Run constrained generation and decode predicted action text back to tensors."""
        tokenizer = self.processor.tokenizer
        sample_args = {"temperature": generate_temperature} if generate_temperature > 0 else {"do_sample": False}

        if get_one_step_action:
            max_new_tokens = self.act_dim * (len(str(self.runtime.num_bins_actions)) + 1)
            if last_action_txt != "":
                last_ids = tokenizer(last_action_txt, return_tensors="pt")["input_ids"].to(
                    model_inputs["input_ids"].device
                )
                model_inputs["input_ids"] = torch.cat(
                    [model_inputs["input_ids"], last_ids], dim=1
                )
                model_inputs["attention_mask"] = torch.cat(
                    [model_inputs["attention_mask"], torch.ones_like(last_ids)],
                    dim=1,
                )
        else:
            max_new_tokens = self.num_tokens

        generated = self.model.generate(
            **model_inputs,
            logits_processor=[self.logits_processor],
            max_new_tokens=max_new_tokens,
            **sample_args,
        )

        input_ids = model_inputs["input_ids"]
        generated_only = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(input_ids, generated)
        ]

        if hasattr(self.processor, "batch_decode"):
            generated_text = self.processor.batch_decode(
                generated_only,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        else:
            generated_text = tokenizer.batch_decode(
                generated_only,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

        if get_one_step_action:
            generated_text = [last_action_txt + generated_text[0]]

        out_actions = self.codec.decode(generated_text)
        if get_one_step_action:
            out_actions = out_actions[:, -1:]

        return {
            "gt_action_text": action_texts,
            "pred_action_txt": generated_text,
            "gt_out_ori_act": out_actions,
            "out_ori_act": out_actions,
        }

    def forward(
        self,
        pc=None,
        rgb_pc=None,
        instr=None,
        rgb=None,
        ori_act=None,
        ee_pos=None,
        ee_rot=None,
        ee_gri=None,
        out_ee_pos=None,
        out_ee_rot=None,
        out_ee_gri=None,
        out_ori_act=None,
        get_loss=True,
        get_action=False,
        generate_temperature=0.1,
        get_one_step_action=False,
        last_action_txt="",
    ):
        self.vision.validate_inputs(pc=pc, rgb_pc=rgb_pc, rgb=rgb)
        self._check_mode_arguments(
            instr=instr,
            get_loss=get_loss,
            get_action=get_action,
            get_one_step_action=get_one_step_action,
            last_action_txt=last_action_txt,
        )

        dialogs, action_texts, model_inputs = self._build_batch_payload(
            instr=instr,
            rgb=rgb,
            out_ori_act=out_ori_act,
            drop_assistant=get_action,
            add_generation_prompt=get_action,
        )

        if get_loss:
            return self._training_step(
                dialogs=dialogs,
                action_texts=action_texts,
                model_inputs=model_inputs,
            )

        return self._generation_step(
            instr=instr,
            action_texts=action_texts,
            model_inputs=model_inputs,
            generate_temperature=generate_temperature,
            get_one_step_action=get_one_step_action,
            last_action_txt=last_action_txt,
        )

    def save_pretrained(self, path):
        self.model.save_pretrained(path)
        self.processor.save_pretrained(path)

    def from_pretrained(self, path, is_trainable=True):
        MedGemmaModelLoader.restore_from_checkpoint(
            actor=self,
            path=path,
            is_trainable=is_trainable,
        )
        self.chat_io = MedGemmaChatIO(
            processor=self.processor,
            add_vision_id=self.runtime.add_vision_id,
        )
        self.label_builder = LossLabelBuilder(
            tokenizer=self.processor.tokenizer,
            label_mask_cfg=LabelMaskConfig(
                action_mask_aug_per=self.runtime.action_mask_aug_per,
            ),
        )
        self.logits_processor = NumericTokenLogitsMask(self.processor.tokenizer)

    def to(self, device):
        super().to(device)
        if isinstance(device, int) or (isinstance(device, str) and device.isnumeric()):
            device = f"cuda:{device}"
        if hasattr(self, "renderer"):
            self.renderer.renderer.device = device
            self.renderer.cameras.to(device)


# Backward-compatible import name used by train.py
MedGemmaActor = MedGemmaPolicy

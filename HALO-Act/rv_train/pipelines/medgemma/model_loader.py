# Modified from the original implementation: https://github.com/NVlabs/vla0
# Authors: Gokul Puthumanaillam, Navya Gupta


from __future__ import annotations

import gc
import warnings

import torch

try:
    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoProcessor,
    )

    try:
        from transformers import AutoModelForImageTextToText
    except ImportError:
        AutoModelForImageTextToText = None
except ImportError:
    AutoConfig = None
    AutoProcessor = None
    AutoModelForCausalLM = None
    AutoModelForImageTextToText = None


def model_device(module):
    try:
        return next(module.parameters()).device
    except StopIteration:
        return torch.device("cpu")


class MedGemmaModelLoader:
    """Owns HF model/processor loading and checkpoint restoration."""

    @staticmethod
    def _require_transformers():
        if AutoProcessor is None or AutoModelForCausalLM is None:
            raise ImportError(
                "transformers is required for MedGemma support. "
                "Install with medgemma extras."
            )

    @staticmethod
    def _build_lora_config(lora_config, use_lora, lora_rank):
        if lora_config == "":
            return None
        if lora_config != "default":
            raise ValueError(f"Invalid lora_config: {lora_config}")
        if not use_lora:
            return None

        from peft import LoraConfig

        return LoraConfig(
            lora_alpha=16,
            lora_dropout=0.05,
            r=lora_rank,
            bias="none",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            task_type="CAUSAL_LM",
        )

    @staticmethod
    def _build_quant_config(use_lora, use_qlora):
        if not (use_lora and use_qlora):
            return None
        if not torch.cuda.is_available():
            warnings.warn(
                "QLoRA requested without CUDA; falling back to non-quantized weights."
            )
            return None

        from transformers import BitsAndBytesConfig

        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_type=torch.bfloat16,
        )

    @staticmethod
    def _extra_model_kwargs(model_id, use_flash_attention_2, attention_dropout):
        kwargs = {}
        if use_flash_attention_2:
            if torch.cuda.is_available():
                kwargs["attn_implementation"] = "flash_attention_2"
            else:
                warnings.warn(
                    "flash_attention_2 requested without CUDA; ignoring option."
                )

        if attention_dropout > 0.0:
            config = AutoConfig.from_pretrained(model_id)
            if hasattr(config, "attention_dropout"):
                config.attention_dropout = attention_dropout
                kwargs["config"] = config
            else:
                warnings.warn("Model config has no attention_dropout; ignoring override.")

        return kwargs

    @classmethod
    def create_model(
        cls,
        model_id,
        use_lora,
        use_qlora,
        lora_config,
        lora_rank,
        use_flash_attention_2,
        attention_dropout,
    ):
        cls._require_transformers()

        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        peft_cfg = cls._build_lora_config(lora_config, use_lora, lora_rank)
        quant_cfg = cls._build_quant_config(use_lora, use_qlora)
        extra_kwargs = cls._extra_model_kwargs(
            model_id, use_flash_attention_2, attention_dropout
        )

        model = None
        load_errors = []
        if AutoModelForImageTextToText is not None:
            try:
                model = AutoModelForImageTextToText.from_pretrained(
                    model_id,
                    quantization_config=quant_cfg,
                    torch_dtype=dtype,
                    **extra_kwargs,
                )
            except Exception as exc:
                load_errors.append(f"AutoModelForImageTextToText failed: {exc}")

        if model is None:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=quant_cfg,
                torch_dtype=dtype,
                **extra_kwargs,
            )
            if load_errors:
                warnings.warn("; ".join(load_errors))

        if use_lora and peft_cfg is not None:
            from peft import get_peft_model

            model = get_peft_model(model, peft_cfg)

        return model

    @classmethod
    def create_processor(cls, model_id, min_pixels, max_pixels, padding_side=None):
        cls._require_transformers()

        kwargs = {}
        if padding_side is not None:
            kwargs["padding_side"] = padding_side

        try:
            processor = AutoProcessor.from_pretrained(
                model_id,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
                **kwargs,
            )
        except TypeError:
            processor = AutoProcessor.from_pretrained(model_id, **kwargs)

        tokenizer = processor.tokenizer
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token

        return processor

    @classmethod
    def restore_from_checkpoint(
        cls,
        actor,
        path,
        is_trainable=True,
    ):
        current_device = model_device(actor)
        del actor.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        if actor.use_lora:
            from peft import PeftModel

            base_model = cls.create_model(
                model_id=actor.medgemma_model_id,
                use_lora=actor.use_lora,
                use_qlora=actor.use_qlora,
                lora_config="",
                lora_rank=actor.lora_rank,
                use_flash_attention_2=actor.use_flash_attention_2,
                attention_dropout=actor.attention_dropout,
            )
            actor.model = PeftModel.from_pretrained(
                base_model,
                path,
                is_trainable=is_trainable,
            )
            print("Loading MedGemma PEFT model from", path)
        else:
            dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            extra_kwargs = cls._extra_model_kwargs(
                path, actor.use_flash_attention_2, actor.attention_dropout
            )

            model = None
            if AutoModelForImageTextToText is not None:
                try:
                    model = AutoModelForImageTextToText.from_pretrained(
                        path,
                        torch_dtype=dtype,
                        **extra_kwargs,
                    )
                except Exception:
                    model = None
            if model is None:
                model = AutoModelForCausalLM.from_pretrained(
                    path,
                    torch_dtype=dtype,
                    **extra_kwargs,
                )
            actor.model = model
            print("Loading MedGemma full model from", path)

        actor.processor = cls.create_processor(
            model_id=path,
            min_pixels=actor.min_pixel,
            max_pixels=actor.max_pixel,
            padding_side="left" if actor.use_flash_attention_2 else None,
        )
        print("Loading MedGemma processor from", path)

        actor.to(current_device)

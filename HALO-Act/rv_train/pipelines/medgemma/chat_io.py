# Modified from the original implementation: https://github.com/NVlabs/vla0
# Authors: Gokul Puthumanaillam, Navya Gupta


from __future__ import annotations

import inspect
from typing import List

import torch


class MedGemmaChatIO:
    """Builds structured dialogs and converts them into processor tensors."""

    def __init__(self, processor, add_vision_id: bool = False):
        self.processor = processor
        self.add_vision_id = add_vision_id

    @staticmethod
    def build_dialog(system_message: str, images, instruction: str, action_text: str):
        return [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_message}],
            },
            {
                "role": "user",
                "content": [{"type": "image", "image": img} for img in images]
                + [{"type": "text", "text": instruction}],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": action_text}],
            },
        ]

    @staticmethod
    def _fallback_template(dialog, add_generation_prompt: bool):
        chunks = []
        for msg in dialog:
            chunks.append(f"{msg['role']}: ")
            for item in msg.get("content", []):
                if item.get("type") == "text":
                    chunks.append(item.get("text", ""))
                elif item.get("type") == "image":
                    chunks.append("<image>")
            chunks.append("\n")
        if add_generation_prompt:
            chunks.append("assistant: ")
        return "".join(chunks)

    def _apply_template(self, dialog, add_generation_prompt: bool):
        """Apply processor-native chat template with a robust fallback path."""
        if not hasattr(self.processor, "apply_chat_template"):
            return self._fallback_template(dialog, add_generation_prompt)

        kwargs = {
            "tokenize": False,
            "add_generation_prompt": add_generation_prompt,
        }
        if self.add_vision_id:
            try:
                signature = inspect.signature(self.processor.apply_chat_template)
                if "add_vision_id" in signature.parameters:
                    kwargs["add_vision_id"] = True
            except (TypeError, ValueError):
                pass

        return self.processor.apply_chat_template(dialog, **kwargs)

    @staticmethod
    def _extract_images(dialog):
        output = []
        for msg in dialog:
            for item in msg.get("content", []):
                if item.get("type") == "image":
                    output.append(item["image"])
        return output

    def _processor_call(self, texts, dialogs):
        """Tokenize text and image content with cross-version processor compatibility."""
        kwargs = {
            "text": texts,
            "return_tensors": "pt",
            "padding": True,
        }
        image_batches = [self._extract_images(dialog) for dialog in dialogs]
        if not any(len(batch) > 0 for batch in image_batches):
            return self.processor(**kwargs)

        kwargs["images"] = image_batches
        try:
            return self.processor(**kwargs)
        except Exception:
            # Compatibility fallback for processors that only accept one image per sample.
            single_image = [batch[0] for batch in image_batches]
            return self.processor(
                text=texts,
                images=single_image,
                return_tensors="pt",
                padding=True,
            )

    def build_model_inputs(self, dialogs, add_generation_prompt: bool, model_device):
        """Create batched model inputs and place tensors on the model device."""
        texts = [
            self._apply_template(dialog, add_generation_prompt=add_generation_prompt)
            for dialog in dialogs
        ]
        model_inputs = self._processor_call(texts=texts, dialogs=dialogs)
        for key, value in model_inputs.items():
            if torch.is_tensor(value):
                model_inputs[key] = value.to(model_device)
        return model_inputs

    def build_prompt_only_inputs(self, dialogs):
        prompt_dialogs = [dialog[:-1] for dialog in dialogs]
        texts = [
            self._apply_template(dialog, add_generation_prompt=True)
            for dialog in prompt_dialogs
        ]
        return self._processor_call(texts=texts, dialogs=prompt_dialogs)

    def prompt_token_lengths(self, dialogs):
        """Compute prompt token spans used to mask non-target labels during training."""
        lengths = []
        for dialog in dialogs:
            prompt_dialog = dialog[:-1]
            prompt_text = self._apply_template(
                prompt_dialog, add_generation_prompt=True
            )
            prompt_inputs = self._processor_call(
                texts=[prompt_text], dialogs=[prompt_dialog]
            )
            lengths.append(prompt_inputs["input_ids"].shape[1])
        return lengths

    def build_dialogs(
        self,
        system_message: str,
        image_batches,
        instructions: List[str],
        action_texts: List[str],
        drop_assistant: bool,
    ):
        dialogs = [
            self.build_dialog(
                system_message=system_message,
                images=image_batches[idx],
                instruction=instructions[idx],
                action_text=action_texts[idx],
            )
            for idx in range(len(instructions))
        ]
        if drop_assistant:
            dialogs = [dialog[:2] for dialog in dialogs]
        return dialogs

# Modified from the original implementation: https://github.com/NVlabs/vla0
# Authors: Gokul Puthumanaillam, Navya Gupta


"""
Mac-safe smoke test for MedGemma integration.

This test monkeypatches MedGemma model/processor loading with small local dummy
components to validate that the training path can execute one forward/backward
step and one action generation call.
"""

from types import SimpleNamespace

import torch
from torch import nn

import rv_train.constants as C
from rv_train.models.medgemma.model import MedGemmaActor


class DummyTokenizer:
    def __init__(self):
        alphabet = (
            ["<pad>", "<eos>", "<unk>"]
            + [" "]
            + list("0123456789")
            + list("abcdefghijklmnopqrstuvwxyz")
            + list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
            + list("\n:,.!?-_/<>")
        )
        self.tok_to_id = {ch: i for i, ch in enumerate(alphabet)}
        self.id_to_tok = {i: ch for ch, i in self.tok_to_id.items()}
        self.pad_token_id = self.tok_to_id["<pad>"]
        self.eos_token_id = self.tok_to_id["<eos>"]
        self.unk_token_id = self.tok_to_id["<unk>"]
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"
        self.padding_side = "right"

    def encode(self, text, add_special_tokens=False):
        ids = [self.tok_to_id.get(ch, self.unk_token_id) for ch in text]
        if add_special_tokens:
            ids.append(self.eos_token_id)
        return ids

    def __call__(self, text, add_special_tokens=False, return_tensors=None, **kwargs):
        if isinstance(text, str):
            text = [text]

        encoded = [self.encode(t, add_special_tokens=add_special_tokens) for t in text]
        max_len = max(len(x) for x in encoded)

        input_ids = []
        attention_mask = []
        for ids in encoded:
            pad_len = max_len - len(ids)
            padded = ids + [self.pad_token_id] * pad_len
            mask = [1] * len(ids) + [0] * pad_len
            input_ids.append(padded)
            attention_mask.append(mask)

        out = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }
        return out

    def decode(self, ids, skip_special_tokens=True):
        if torch.is_tensor(ids):
            ids = ids.tolist()
        chars = []
        for idx in ids:
            if skip_special_tokens and idx in {
                self.pad_token_id,
                self.eos_token_id,
            }:
                continue
            chars.append(self.id_to_tok.get(idx, "?"))
        return "".join(chars)

    def batch_decode(self, batch_ids, skip_special_tokens=True, **kwargs):
        return [self.decode(ids, skip_special_tokens=skip_special_tokens) for ids in batch_ids]


class DummyProcessor:
    def __init__(self):
        self.tokenizer = DummyTokenizer()

    def apply_chat_template(
        self,
        example,
        tokenize=False,
        add_generation_prompt=False,
        **kwargs,
    ):
        chunks = []
        for msg in example:
            chunks.append(f"{msg['role']}: ")
            for c in msg.get("content", []):
                if c.get("type") == "text":
                    chunks.append(c.get("text", ""))
                elif c.get("type") == "image":
                    chunks.append("<image>")
            chunks.append("\n")
        if add_generation_prompt:
            chunks.append("assistant: ")
        return "".join(chunks)

    def __call__(self, text, images=None, return_tensors="pt", padding=True):
        return self.tokenizer(text, return_tensors=return_tensors)

    def batch_decode(self, ids, skip_special_tokens=True, clean_up_tokenization_spaces=False):
        return self.tokenizer.batch_decode(ids, skip_special_tokens=skip_special_tokens)

    def save_pretrained(self, path):
        return


class DummyModel(nn.Module):
    def __init__(self, vocab_size, hidden_size=32):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.proj = nn.Linear(hidden_size, vocab_size)
        self.zero_token_id = 4  # "0"
        self.space_token_id = 3  # " "

    def forward(self, input_ids, attention_mask=None, **kwargs):
        x = self.embed(input_ids)
        logits = self.proj(x)
        return SimpleNamespace(logits=logits)

    def generate(
        self,
        input_ids,
        attention_mask=None,
        logits_processor=None,
        max_new_tokens=16,
        **kwargs,
    ):
        bs = input_ids.shape[0]
        device = input_ids.device
        pattern = [self.zero_token_id, self.space_token_id]
        repeated = pattern * ((max_new_tokens + 1) // 2)
        appended = torch.tensor(repeated[:max_new_tokens], device=device).unsqueeze(0)
        appended = appended.repeat(bs, 1)
        return torch.cat([input_ids, appended], dim=1)

    def save_pretrained(self, path):
        return


def run_smoke_test():
    # Monkeypatch loader methods so test does not require HF downloads.
    MedGemmaActor.load_medgemma_model = staticmethod(
        lambda **kwargs: DummyModel(vocab_size=len(DummyTokenizer().tok_to_id))
    )
    MedGemmaActor.load_medgemma_processor = staticmethod(
        lambda **kwargs: DummyProcessor()
    )

    model = MedGemmaActor(
        medgemma_model_id="google/medgemma-1.5-4b-it",
        action_type=C.ORIGINAL,
        original_action_dim=7,
        horizon=4,
        history=1,
        use_lora=False,
        use_qlora=False,
        num_cam=1,
        lora_config="",
        lora_rank=8,
        rgb_input=True,
        rgb_img_size=(64, 64),
        add_vision_id=False,
        tiled_rgb_imgs=False,
        num_bins_actions=1000,
        use_flash_attention_2=False,
        action_mask_aug_per=0.0,
        attention_dropout=0.0,
    )
    model.set_dataset_stats(
        {
            "out_ori_act": {
                "min": [-1.0] * 7,
                "max": [1.0] * 7,
            }
        }
    )
    # Keep generation short so decode->reshape path is deterministic in smoke test.
    model.num_tokens = model.horizon * model.act_dim * 2

    bs = 2
    instr = ["pick up the bottle", "open the drawer"]
    rgb = torch.randint(0, 256, (bs, 1, 1, 64, 64, 3), dtype=torch.uint8).float()
    out_ori_act = (torch.rand(bs, 4, 7) * 2.0) - 1.0

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer.zero_grad()
    out = model(instr=instr, rgb=rgb, out_ori_act=out_ori_act, get_loss=True)
    loss = out["loss"]
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        pred = model(
            instr=instr,
            rgb=rgb,
            out_ori_act=out_ori_act,
            get_loss=False,
            get_action=True,
            generate_temperature=0.0,
        )

    assert "out_ori_act" in pred
    assert pred["out_ori_act"].shape == (bs, 4, 7), pred["out_ori_act"].shape

    print(f"SMOKE TEST PASSED: loss={loss.item():.6f}")


if __name__ == "__main__":
    run_smoke_test()

import json

import torch

from tinyllava.data.dataset import make_supervised_data_module
from tinyllava.hparams.arguments import DataArguments
from tinyllava.utils.constants import IGNORE_INDEX


class TinyTokenizer:
    pad_token_id = 0
    model_max_length = 64


class TinyProcessor:
    def __init__(self):
        self.tokenizer = TinyTokenizer()

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["return_assistant_tokens_mask"] is True
        input_ids = []
        assistant_masks = []
        for message in messages:
            text = " ".join(
                item["text"]
                for item in message["content"]
                if item["type"] == "text"
            )
            token_ids = list(range(1, max(len(text.split()), 1) + 1))
            input_ids.extend(token_ids)
            assistant_masks.extend(
                [int(message["role"] == "assistant")] * len(token_ids)
            )
        input_ids = torch.tensor([input_ids])
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "assistant_masks": torch.tensor([assistant_masks]),
        }


def test_text_only_dataset_masks_prompts_and_padding(tmp_path):
    dataset_name_or_path = tmp_path / "text_only.json"
    dataset_name_or_path.write_text(
        json.dumps(
            [
                {
                    "id": "one",
                    "conversations": [
                        {"from": "human", "value": "Write a short greeting."},
                        {"from": "gpt", "value": "Hello there."},
                    ],
                },
                {
                    "id": "two",
                    "conversations": [
                        {"from": "human", "value": "Name a primary color."},
                        {"from": "gpt", "value": "Blue."},
                    ],
                },
            ]
        ),
        encoding="utf-8",
    )
    processor = TinyProcessor()
    data_module = make_supervised_data_module(
        processor=processor,
        data_args=DataArguments(
            dataset_name_or_path=str(dataset_name_or_path),
            dataset_adapter="llava_legacy",
            cache_dir=str(tmp_path / "cache"),
        ),
    )

    dataset = data_module["train_dataset"]
    batch = data_module["data_collator"]([dataset[0], dataset[1]])
    assert "pixel_values" not in batch
    assert batch["input_ids"].shape == (2, 6)
    assert (batch["labels"] != IGNORE_INDEX).sum(dim=1).tolist() == [2, 1]
    assert torch.all(batch["labels"][:, :4] == IGNORE_INDEX)
    assert batch["labels"][1, -1] == IGNORE_INDEX

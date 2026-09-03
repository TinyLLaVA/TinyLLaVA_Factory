import json
from types import SimpleNamespace

import torch
from torch import nn

from tinyllava.data.dataset import make_supervised_data_module
from tinyllava.train.modality_trainer import TinyLlavaTrainer
from tinyllava.utils.arguments import DataArguments, TrainingArguments
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


class TinyLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(128, 16)
        self.head = nn.Linear(16, 128)
        self.config = SimpleNamespace()

    def forward(self, input_ids=None, labels=None, **_kwargs):
        logits = self.head(self.embed(input_ids))
        active_labels = labels.clone()
        active_labels[active_labels == IGNORE_INDEX] = 0
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            active_labels.reshape(-1),
        )
        return {"loss": loss, "logits": logits}


def test_text_only_dataset_can_train_one_step(tmp_path):
    data_path = tmp_path / "text_only.json"
    data_path.write_text(
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
            data_path=str(data_path),
            dataset_adapter="llava_legacy",
        ),
    )

    sample = data_module["train_dataset"][0]
    assert "pixel_values" not in sample
    assert torch.any(sample["labels"] != IGNORE_INDEX)

    trainer = TinyLlavaTrainer(
        model=TinyLanguageModel(),
        processing_class=processor,
        args=TrainingArguments(
            output_dir=str(tmp_path / "output"),
            per_device_train_batch_size=2,
            max_steps=1,
            learning_rate=1e-3,
            save_strategy="no",
            eval_strategy="no",
            report_to=[],
            remove_unused_columns=False,
            dataloader_pin_memory=False,
        ),
        **data_module,
    )

    result = trainer.train()

    assert result.global_step == 1

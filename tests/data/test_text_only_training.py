import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from torch import nn

from tinyllava.data.dataset import make_supervised_data_module
import tinyllava.train.train as train_module
from tinyllava.utils.arguments import DataArguments, TrainingArguments
from tinyllava.utils.constants import IGNORE_INDEX


class TinyTokenizer:
    """Minimal tokenizer for template encoding in tests."""

    def __init__(self):
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.model_max_length = 128
        self._vocab = {"<|endoftext|>": self.eos_token_id}

    def _token_to_id(self, token: str) -> int:
        if token not in self._vocab:
            self._vocab[token] = len(self._vocab) + 3
        return self._vocab[token]

    def _tokenize(self, text: str) -> list[int]:
        # Keep Phi template EOS token as an atomic token to preserve label masking logic.
        tokens = text.replace("<|endoftext|>", " <|endoftext|> ").split()
        return [self._token_to_id(tok) for tok in tokens]

    def __call__(self, text: str):
        return SimpleNamespace(input_ids=self._tokenize(text))

    def encode(self, text: str) -> list[int]:
        return self(text).input_ids


class FakeConfig:
    def __init__(self):
        self.use_cache = True
        self.image_aspect_ratio = None

    def load_from_config(self, model_arguments):
        self.model_arguments = model_arguments


class FakeRecipe:
    def __init__(self, training_arguments):
        self.training_arguments = training_arguments
        self.save_called = False

    def add_args(self, model_args):
        return model_args

    def __call__(self, model):
        return model

    def load(self, model, model_args):
        return model

    def save(self, model, trainer):
        self.save_called = True


class FakeArgumentParser:
    _args_payload: tuple[Any, Any, Any] | None = None

    def __init__(self, *args, **kwargs):
        self._args = args

    def parse_args_into_dataclasses(self):
        assert self.__class__._args_payload is not None
        return self.__class__._args_payload


class FakeTinyLlavaModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tokenizer = TinyTokenizer()
        self.embed = nn.Embedding(4096, 16)
        self.lm_head = nn.Linear(16, 4096)
        self.vision_tower = SimpleNamespace(
            _image_processor=SimpleNamespace(
                crop_size={"height": 16, "width": 16},
                size={"shortest_edge": 16},
                image_mean=[0.5, 0.5, 0.5],
            )
        )
        self.connector = nn.Linear(16, 16)

    def load_llm(self, **kwargs):
        self.load_llm_kwargs = kwargs

    def load_vision_tower(self, **kwargs):
        self.load_vision_tower_kwargs = kwargs

    def load_connector(self, **kwargs):
        self.load_connector_kwargs = kwargs

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        del attention_mask, kwargs
        assert input_ids is not None
        hidden = self.embed(input_ids % self.embed.num_embeddings)
        hidden = self.connector(hidden)
        logits = self.lm_head(hidden)

        loss = None
        if labels is not None:
            labels = labels.clone()
            labels[labels == IGNORE_INDEX] = 0
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1) % logits.size(-1),
            )

        return {"loss": loss, "logits": logits}


@pytest.fixture
def text_only_dataset_json(tmp_path: Path):
    data = [
        {
            "id": "txt_001",
            "conversations": [
                {"from": "human", "value": "Write a haiku about rain."},
                {
                    "from": "gpt",
                    "value": "Soft rain taps window. Quiet thoughts flow into dawn.",
                },
            ],
        },
        {
            "id": "txt_002",
            "conversations": [
                {"from": "human", "value": "Explain what a unit test does."},
                {
                    "from": "gpt",
                    "value": "It checks a small piece of code behaves as expected.",
                },
            ],
        },
        {
            "id": "txt_003",
            "conversations": [
                {"from": "human", "value": "Give one Python tip."},
                {
                    "from": "gpt",
                    "value": "Use virtual environments to isolate dependencies.",
                },
            ],
        },
    ]
    path = tmp_path / "text_only_dataset.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_text_only_dataset_can_train_one_step(
    monkeypatch, text_only_dataset_json: Path, tmp_path: Path
):
    tokenizer = TinyTokenizer()
    data_args = DataArguments(
        data_path=str(text_only_dataset_json),
        is_multimodal=False,
        conv_version="phi",
    )
    setattr(data_args, "image_processor", SimpleNamespace())

    data_module = make_supervised_data_module(
        tokenizer=cast(Any, tokenizer), data_args=data_args
    )
    train_dataset = data_module["train_dataset"]

    sample = train_dataset[0]
    assert "image" not in sample
    assert torch.any(sample["labels"] != IGNORE_INDEX)

    model_arguments = train_module.ModelArguments()
    model_arguments.vision_tower = ""
    model_arguments.vision_tower2 = ""
    model_arguments.connector_type = "linear"
    model_arguments.model_name_or_path = "tiny"

    training_args = TrainingArguments(
        output_dir=str(tmp_path / "out"),
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        max_steps=1,
        num_train_epochs=1,
        learning_rate=1e-3,
        logging_steps=1,
        save_strategy="no",
        eval_strategy="no",
        report_to=[],
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        do_eval=False,
    )

    data_args = DataArguments(
        data_path=str(text_only_dataset_json),
        is_multimodal=False,
        conv_version="phi",
    )
    setattr(data_args, "image_processor", SimpleNamespace())

    recipe_instance = FakeRecipe(training_args)

    monkeypatch.setattr(train_module, "TinyLlavaConfig", FakeConfig)
    monkeypatch.setattr(train_module, "TinyLlavaForConditionalGeneration", FakeTinyLlavaModel)
    monkeypatch.setattr(
        train_module,
        "TrainingRecipeFactory",
        lambda name: (lambda args: recipe_instance),
    )
    monkeypatch.setattr(train_module, "logger_setting", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(train_module, "log_trainable_params", lambda *_args, **_kwargs: None)

    class _Parser(FakeArgumentParser):
        _args_payload = (model_arguments, data_args, training_args)

    monkeypatch.setattr(train_module.transformers, "HfArgumentParser", _Parser)

    train_module.train()

    assert recipe_instance.save_called is True

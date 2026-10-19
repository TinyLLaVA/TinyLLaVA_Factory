from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers import CLIPVisionConfig, LlamaConfig, SiglipVisionConfig

from tinyllava.hparams.arguments import TuningArguments
from tinyllava.model import TinyLlavaConfig, TinyLlavaForConditionalGeneration
from tinyllava.train.strategy.base import BaseTrainingStrategy
from tinyllava.train.strategy.peft_strategy import LoraTrainingStrategy


class TinyModelStub(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.language_model = nn.Linear(4, 4)
        self.model.vision_tower = nn.Linear(4, 4)
        self.model.multi_modal_projector = nn.Linear(4, 4)
        self.lm_head = nn.Linear(4, 4)


def _training_args(tune_type_llm: str):
    return SimpleNamespace(
        tune_type_llm=tune_type_llm,
        tune_type_vision_tower="frozen",
        tune_type_connector="frozen",
    )


def test_frozen_llm_also_freezes_output_head():
    model = TinyModelStub()

    BaseTrainingStrategy(SimpleNamespace(), _training_args("frozen"))(model)

    assert not any(
        parameter.requires_grad for parameter in model.model.language_model.parameters()
    )
    assert not any(parameter.requires_grad for parameter in model.lm_head.parameters())


def test_full_llm_also_unfreezes_output_head():
    model = TinyModelStub().requires_grad_(False)

    BaseTrainingStrategy(SimpleNamespace(), _training_args("full"))(model)

    assert all(
        parameter.requires_grad for parameter in model.model.language_model.parameters()
    )
    assert all(parameter.requires_grad for parameter in model.lm_head.parameters())


def tiny_model(vision_cls=CLIPVisionConfig):
    torch.set_num_threads(2)
    return TinyLlavaForConditionalGeneration(
        TinyLlavaConfig(
            text_config=LlamaConfig(
                vocab_size=32,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=2,
                num_attention_heads=2,
                num_key_value_heads=2,
                tie_word_embeddings=False,
            ),
            vision_config=vision_cls(
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=2,
                num_attention_heads=2,
                image_size=16,
                patch_size=8,
            ),
            vision_feature_layer=-1,
            vision_feature_select_strategy="full"
            if vision_cls is SiglipVisionConfig
            else "default",
            image_token_index=31,
        )
    )


@pytest.mark.parametrize("vision_cls", [CLIPVisionConfig, SiglipVisionConfig])
@pytest.mark.parametrize("strategy_cls", [BaseTrainingStrategy, LoraTrainingStrategy])
@pytest.mark.parametrize("reentrant", [False, True])
def test_partial_vision_tuning_updates_only_selected_layers(
    vision_cls, strategy_cls, reentrant
):
    model = strategy_cls(
        SimpleNamespace(gradient_checkpointing=True),
        TuningArguments(
            tune_type_llm="lora" if strategy_cls is LoraTrainingStrategy else "frozen",
            tune_type_vision_tower="partially-tune",
            tune_vision_tower_from_layer=1,
            peft={"r": 2},
        ),
    )(tiny_model(vision_cls))
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": reentrant}
    )
    ids = torch.tensor([[1, 31, 31, 31, 31, 2]])
    model(
        input_ids=ids, pixel_values=torch.randn(1, 3, 16, 16), labels=ids
    ).loss.backward()
    gradients = [
        name
        for name, p in model.named_parameters()
        if "vision_tower" in name and p.grad is not None
    ]
    assert gradients
    assert all("encoder.layers.1." in name for name in gradients)
    assert all(
        not p.requires_grad
        for name, p in model.named_parameters()
        if "original_module" in name
    )


@pytest.mark.parametrize(
    "llm,vision,connector",
    [
        ("lora", "frozen", "full"),
        ("lora", "frozen", "frozen"),
        ("lora", "full", "full"),
        ("full", "lora", "full"),
        ("frozen", "frozen", "lora"),
    ],
)
def test_peft_component_policies_survive_training_and_reload(
    tmp_path, llm, vision, connector
):
    from peft import PeftModel

    base, adapter = tmp_path / "base", tmp_path / "adapter"
    tiny_model().save_pretrained(base)
    model = LoraTrainingStrategy(
        None,
        TuningArguments(
            tune_type_llm=llm,
            tune_type_vision_tower=vision,
            tune_type_connector=connector,
            peft={"r": 2},
        ),
    )(TinyLlavaForConditionalGeneration.from_pretrained(base))
    model.gradient_checkpointing_enable()
    model.train()
    inputs = {
        "input_ids": torch.tensor([[1, 31, 31, 31, 31, 2]]),
        "pixel_values": torch.randn(1, 3, 16, 16),
    }
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=0.01
    )
    model(**inputs, labels=inputs["input_ids"]).loss.backward()
    assert all(
        not p.requires_grad
        for name, p in model.named_parameters()
        if "original_module" in name
    )
    for component, policy in (
        ("language_model", llm),
        ("vision_tower", vision),
        ("multi_modal_projector", connector),
    ):
        parameters = [
            (name, p) for name, p in model.named_parameters() if component in name
        ]
        active = [name for name, p in parameters if p.grad is not None]
        assert bool(active) == (policy != "frozen")
        if policy == "lora":
            assert all("lora_" in name for name in active)
    optimizer.step()
    model.eval()
    model.save_pretrained(adapter)
    restored = PeftModel.from_pretrained(
        TinyLlavaForConditionalGeneration.from_pretrained(base), adapter
    ).eval()
    with torch.no_grad():
        torch.testing.assert_close(restored(**inputs).logits, model(**inputs).logits)

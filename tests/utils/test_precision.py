from types import SimpleNamespace

import torch
from torch import nn

from tinyllava.utils.precision import (
    cast_model_to_training_dtype,
    resolve_precision_flags,
)


class MixedDtypeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fp32 = nn.Linear(4, 4, dtype=torch.float32)
        self.bf16 = nn.Linear(4, 4, dtype=torch.bfloat16)
        self.register_buffer("indices", torch.arange(4, dtype=torch.int64))


def test_cast_model_to_training_dtype_normalizes_floating_state():
    model = MixedDtypeModel()
    training_args = SimpleNamespace(bf16=True, fp16=False)

    result = cast_model_to_training_dtype(model, training_args)

    assert result is model
    assert {parameter.dtype for parameter in model.parameters()} == {
        torch.bfloat16
    }
    assert model.indices.dtype == torch.int64


def test_cast_model_to_training_dtype_only_casts_trainable_quantized_state():
    model = MixedDtypeModel()
    model.bf16.float()
    model.is_loaded_in_8bit = True
    model.fp32.requires_grad_(False)
    training_args = SimpleNamespace(bf16=True, fp16=False)

    cast_model_to_training_dtype(model, training_args)

    assert model.fp32.weight.dtype == torch.float32
    assert model.bf16.weight.dtype == torch.bfloat16


def test_resolve_precision_flags_selects_bf16_before_training_args_init(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)

    assert resolve_precision_flags("auto") == (True, False)

from types import SimpleNamespace

from torch import nn

from tinyllava.train.strategy.base import BaseTrainingStrategy


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

    BaseTrainingStrategy(_training_args("frozen"))(model)

    assert not any(
        parameter.requires_grad
        for parameter in model.model.language_model.parameters()
    )
    assert not any(parameter.requires_grad for parameter in model.lm_head.parameters())


def test_full_llm_also_unfreezes_output_head():
    model = TinyModelStub().requires_grad_(False)

    BaseTrainingStrategy(_training_args("full"))(model)

    assert all(parameter.requires_grad for parameter in model.model.language_model.parameters())
    assert all(parameter.requires_grad for parameter in model.lm_head.parameters())

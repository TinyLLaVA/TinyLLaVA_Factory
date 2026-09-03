import torch

from tinyllava.eval.generation import prepare_generation_inputs


class FakeProcessor:
    def __init__(self):
        self.kwargs = None

    def apply_chat_template(self, messages, **kwargs):
        self.kwargs = kwargs
        return {"input_ids": torch.tensor([[1, 2]]), "attention_mask": torch.tensor([[1, 1]])}


def test_prepare_generation_inputs_passes_padding_through_processor_kwargs():
    processor = FakeProcessor()

    prepare_generation_inputs(
        processor,
        [[{"role": "user", "content": "hello"}]],
        device="cpu",
        padding=True,
    )

    assert "padding" not in processor.kwargs
    assert processor.kwargs["processor_kwargs"] == {"text_kwargs": {"padding": True}}

"""Data collator and label helpers for multimodal supervised fine-tuning."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
import transformers

from tinyllava.utils.constants import IGNORE_INDEX


@dataclass
class DataCollatorForMultimodalSFT:
    """Collate text and multimodal processor outputs for SFT."""

    processor: transformers.ProcessorMixin

    @property
    def tokenizer(self):
        """Return the tokenizer used for text padding metadata."""
        return self.processor.tokenizer

    def __call__(self, instances: Sequence[dict]) -> dict[str, torch.Tensor]:
        """Pad text fields and merge optional multimodal fields."""
        max_length = getattr(self.tokenizer, "model_max_length", None)
        batch: dict[str, Any] = {
            "input_ids": self._pad_1d(
                [instance["input_ids"] for instance in instances],
                padding_value=self.tokenizer.pad_token_id,
                max_length=max_length,
            ),
            "labels": self._pad_1d(
                [instance["labels"] for instance in instances],
                padding_value=IGNORE_INDEX,
                max_length=max_length,
            ),
            "attention_mask": self._pad_1d(
                [instance.get("attention_mask", torch.ones_like(instance["input_ids"])) for instance in instances],
                padding_value=0,
                max_length=max_length,
            ).bool(),
        }

        for key in sorted(set().union(*(instance.keys() for instance in instances))):
            if key in {"input_ids", "labels", "attention_mask"}:
                continue
            values = [instance.get(key) for instance in instances]
            if key == "pixel_values":
                batch[key] = concat_pixel_values([value for value in values if value is not None])
                continue
            if key == "image_sizes":
                batch[key] = concat_optional_tensors([value for value in values if value is not None])
                continue
            if any(value is None for value in values):
                continue
            batch[key] = stack_or_list(values)

        return batch

    @staticmethod
    def _pad_1d(tensors: list[torch.Tensor], padding_value: int, max_length: int | None) -> torch.Tensor:
        """Pad 1D token-like tensors and optionally truncate to model max length."""
        padded = torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True, padding_value=padding_value)
        if max_length is not None:
            padded = padded[:, :max_length]
        return padded


def build_labels(data_dict: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """Build causal LM labels from assistant-token masks.

    Tokens outside assistant generation spans are set to `IGNORE_INDEX`, so they
    do not contribute to the loss.
    """
    input_ids = data_dict["input_ids"]
    assistant_masks = data_dict.get("assistant_masks")
    if assistant_masks is None:
        raise ValueError(
            "Processor did not return `assistant_masks`. Training with HF generation masks requires a fast tokenizer "
            "and a chat template containing `{% generation %}` blocks."
        )

    labels = input_ids.clone()
    labels[assistant_masks.to(dtype=torch.bool) == 0] = IGNORE_INDEX
    return labels


def stack_or_list(values: Sequence[Any]) -> Any:
    """Stack equal-shaped tensors; keep ragged or non-tensor values as a list."""
    if all(isinstance(value, torch.Tensor) for value in values):
        if all(value.shape == values[0].shape for value in values):
            return torch.stack(list(values))
        return list(values)
    return list(values)


def concat_pixel_values(values: Sequence[torch.Tensor]) -> torch.Tensor:
    """Flatten image tensors across samples into `[total_images, C, H, W]`."""
    images = []
    for value in values:
        if value.ndim == 3:
            images.append(value.unsqueeze(0))
        elif value.ndim == 4:
            images.append(value)
        else:
            images.append(value.reshape(-1, *value.shape[-3:]))
    return torch.cat(images, dim=0)


def concat_optional_tensors(values: Sequence[torch.Tensor]) -> torch.Tensor:
    """Concatenate optional per-image metadata tensors such as `image_sizes`."""
    tensors = [value.unsqueeze(0) if value.ndim == 1 else value for value in values]
    return torch.cat(tensors, dim=0)


__all__ = [
    "DataCollatorForMultimodalSFT",
    "build_labels",
    "concat_optional_tensors",
    "concat_pixel_values",
    "stack_or_list",
]

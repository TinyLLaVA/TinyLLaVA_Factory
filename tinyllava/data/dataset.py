"""Processor transform over Hugging Face datasets for multimodal SFT."""

import copy
from collections.abc import Mapping
from typing import Any, cast

import torch
import transformers
from datasets import Dataset as HFDataset
from datasets import load_dataset
from torch.utils.data import Dataset

from .assistant_mask import build_assistant_mask, squeeze_batch
from .collator import DataCollatorForMultimodalSFT, build_labels
from .image_payload import (
    add_image_payloads,
    collect_sample_image_payloads,
    resolve_message_image_payloads,
)
from .message_format import normalize_messages
from ..utils.arguments import DataArguments


class ProcessorSFTDataset(Dataset):
    """Apply multimodal SFT processing lazily over a Hugging Face dataset."""

    def __init__(
        self,
        dataset: HFDataset,
        processor: transformers.ProcessorMixin,
        data_args: DataArguments,
    ):
        super().__init__()
        self.dataset = dataset
        self.processor = processor
        self.data_args = data_args

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i) -> dict[str, torch.Tensor]:
        sample = copy.deepcopy(dict(self.dataset[i]))
        messages = normalize_messages(sample)
        resolve_message_image_payloads(
            messages,
            image_folder=getattr(self.data_args, "image_folder", None),
        )
        add_image_payloads(messages, self._collect_image_payloads(sample))

        encoded = cast(
            Mapping[str, Any],
            self.processor.apply_chat_template(
                messages,
                add_generation_prompt=False,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                return_assistant_tokens_mask=True,
            ),
        )
        data_dict = squeeze_batch(encoded)
        data_dict["assistant_masks"] = build_assistant_mask(
            processor=self.processor,
            messages=messages,
            data_dict=data_dict,
        )
        data_dict["labels"] = build_labels(data_dict)
        data_dict.pop("assistant_masks", None)
        return data_dict

    def _collect_image_payloads(self, sample: Mapping[str, Any]):
        return collect_sample_image_payloads(
            sample,
            image_folder=getattr(self.data_args, "image_folder", None),
        )


def load_training_dataset(data_args: DataArguments) -> HFDataset:
    """Load training rows with Hugging Face datasets."""
    if data_args.data_path is None:
        raise ValueError("`data_path` must be provided for supervised fine-tuning.")

    # TODO: expose the general `datasets.load_dataset` contract here instead of
    # assuming a local legacy JSON file. Future config should support dataset
    # path/name, split, data_files, streaming, parquet, and Hub datasets.
    dataset = load_dataset(
        "json",
        data_files=data_args.data_path,
        split="train",
    )
    return cast(HFDataset, dataset)


def make_supervised_data_module(
    processor: transformers.ProcessorMixin,
    data_args: DataArguments,
) -> dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = ProcessorSFTDataset(
        dataset=load_training_dataset(data_args),
        processor=processor,
        data_args=data_args,
    )
    data_collator = DataCollatorForMultimodalSFT(processor=processor)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


__all__ = ["ProcessorSFTDataset", "load_training_dataset", "make_supervised_data_module"]

"""Processor transform over Hugging Face datasets for multimodal SFT."""

import copy
import hashlib
from collections.abc import Mapping
from functools import cached_property
from pathlib import Path
from typing import Any, cast

import torch
import transformers
from datasets import Dataset as HFDataset
from datasets import load_dataset
from torch.utils.data import Dataset

from .adapters import resolve_dataset_adapter
from .assistant_mask import build_assistant_mask, squeeze_batch
from .collator import DataCollatorForMultimodalSFT, build_labels
from .image_payload import (
    add_image_payloads,
    collect_sample_image_payloads,
    resolve_message_image_payloads,
)
from .message_format import normalize_messages
from .readers import is_json_array, iter_json_array, read_first_json_array_item
from ..utils.arguments import DataArguments
from ..utils.logging import get_logger


logger = get_logger(__name__)


class ProcessorSFTDataset(Dataset):
    """Encode training conversations lazily and supervise assistant tokens.

    Each indexed row is normalized, encoded with the processor's chat template,
    and returned with causal-LM labels. Tokens outside assistant spans receive
    `IGNORE_INDEX`.

    Args:
        dataset: Rows containing conversations and optional image payloads.
        processor: Multimodal processor with an assistant-aware chat template.
        data_args: Data settings, including the root for relative image paths.
    """

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

    @cached_property
    def modality_lengths(self) -> list[int]:
        """Approximate legacy lengths; image samples are positive, text-only negative."""

        lengths = []
        for sample in self.dataset:
            word_count = sum(
                len(str(message.get("content", "")).split())
                for message in sample.get("messages", [])
            )
            word_count = max(word_count, 1)
            lengths.append(word_count if sample.get("image") else -word_count)
        return lengths

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


def _iter_adapted_json_array(path: str, adapter_name: str):
    first_sample = read_first_json_array_item(path)
    adapter = resolve_dataset_adapter(adapter_name, first_sample)
    if adapter is None:
        raise ValueError(
            "Top-level JSON arrays require a registered training dataset "
            "adapter. Set `data.dataset_adapter` explicitly."
        )
    for sample in iter_json_array(path):
        yield adapter.adapt(sample)


def _json_array_fingerprint(path: Path, adapter_name: str, cache_version: str) -> str:
    """Fingerprint a local source without rereading a potentially huge file."""
    stat = path.stat()
    identity = (
        f"json-array-v5\0{adapter_name}\0{cache_version}\0{path.resolve()}\0"
        f"{stat.st_size}\0{stat.st_mtime_ns}"
    )
    return hashlib.sha256(identity.encode()).hexdigest()


def load_training_dataset(data_args: DataArguments) -> HFDataset:
    """Load local JSON training data and normalize rows with a dataset adapter.

    Top-level JSON arrays are read incrementally into an Arrow cache. Other
    supported JSON files use the Hugging Face JSON loader.

    Args:
        data_args: Source path and adapter selection for the training data.

    Returns:
        A Hugging Face dataset of normalized training rows.

    Raises:
        ValueError: The source path is missing or a JSON-array adapter cannot be resolved.
    """
    if data_args.data_path is None:
        raise ValueError("`data_path` must be provided for supervised fine-tuning.")

    data_path = Path(data_args.data_path).expanduser()
    if data_path.is_file() and is_json_array(data_path):
        # datasets 5 loads an array fully and serializes it back to JSONL before
        # creating Arrow. For LLaVA's ~1 GB legacy file this is both slow and
        # memory hungry, and its progress remains at zero during that work.
        logger.info_rank0(
            "Building the Arrow cache incrementally from top-level JSON array %s",
            data_path,
        )
        first_sample = read_first_json_array_item(str(data_path))
        adapter = resolve_dataset_adapter(
            data_args.dataset_adapter,
            first_sample,
        )
        if adapter is None:
            raise ValueError(
                "Could not infer an adapter for this top-level JSON array. "
                "Set `data.dataset_adapter` explicitly."
            )
        logger.info_rank0("Using training dataset adapter %s", adapter.name)
        fingerprint = _json_array_fingerprint(
            data_path,
            adapter.name,
            adapter.cache_version,
        )
        dataset = HFDataset.from_generator(
            _iter_adapted_json_array,
            features=adapter.features,
            gen_kwargs={
                "path": str(data_path),
                "adapter_name": adapter.name,
            },
            fingerprint=fingerprint,
        )
    else:
        # TODO: expose the general `datasets.load_dataset` contract here instead
        # of assuming a local JSON file. Future config should support dataset
        # path/name, split, data_files, streaming, parquet, and Hub datasets.
        dataset = load_dataset(
            "json",
            data_files=data_args.data_path,
            split="train",
        )
        dataset = cast(HFDataset, dataset)
        if len(dataset):
            adapter = resolve_dataset_adapter(
                data_args.dataset_adapter,
                dataset[0],
            )
            if adapter is not None:
                logger.info_rank0("Using training dataset adapter %s", adapter.name)
                dataset = dataset.map(
                    adapter.adapt,
                    remove_columns=dataset.column_names,
                    features=adapter.features,
                    desc=f"Applying {adapter.name} dataset adapter",
                )
    return cast(HFDataset, dataset)


def make_supervised_data_module(
    processor: transformers.ProcessorMixin,
    data_args: DataArguments,
) -> dict:
    """Create the data arguments passed to the Trainer for supervised fine-tuning.

    Args:
        processor: Processor used for conversation encoding and batch padding.
        data_args: Training data path, adapter, and image-root settings.

    Returns:
        A mapping containing `train_dataset`, `data_collator`, and `eval_dataset=None`.
    """
    train_dataset = ProcessorSFTDataset(
        dataset=load_training_dataset(data_args),
        processor=processor,
        data_args=data_args,
    )
    data_collator = DataCollatorForMultimodalSFT(processor=processor)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


__all__ = ["ProcessorSFTDataset", "load_training_dataset", "make_supervised_data_module"]

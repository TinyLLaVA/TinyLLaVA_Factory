"""Processor transform over Hugging Face datasets for multimodal SFT."""

import copy
import inspect
from collections.abc import Mapping
from functools import cached_property
from importlib.metadata import version
from pathlib import Path
from typing import Any, cast

import torch
import transformers
from datasets import Dataset as HFDataset
from datasets import ReadInstruction, concatenate_datasets, load_dataset
from datasets.fingerprint import Hasher
from torch.utils.data import Dataset

from .adapters import resolve_dataset_adapter
from .adapters.base import TrainingDatasetAdapter
from .assistant_mask import build_assistant_mask, squeeze_batch
from .collator import DataCollatorForMultimodalSFT, build_labels
from .image_payload import (
    add_image_payloads,
    collect_sample_image_payloads,
    resolve_message_image_payloads,
)
from .message_format import normalize_messages
from .readers import is_json_array, iter_json_array, read_first_json_array_item
from .readers import json_array as json_array_reader
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
            messages = normalize_messages(sample)
            word_count = sum(
                len(str(block.get("text", "")).split())
                for message in messages
                for block in message["content"]
            )
            word_count = max(word_count, 1)
            has_image = bool(sample.get("images") or sample.get("image")) or any(
                block.get("type") == "image"
                for message in messages
                for block in message["content"]
            )
            lengths.append(word_count if has_image else -word_count)
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


def _iter_adapted_json_array(path: str, adapter: TrainingDatasetAdapter | None):
    for sample in iter_json_array(path):
        yield adapter.adapt(sample) if adapter is not None else sample


def _json_array_fingerprint(path: Path, adapter: TrainingDatasetAdapter | None) -> str:
    """Invalidate caches on source, reader, dependency, or adapter changes.

    Source identity uses file size and nanosecond mtime to avoid an extra full
    scan of large annotations. Implementation hashes replace manual versions.
    """
    stat = path.stat()
    return Hasher.hash(
        (
            str(path.resolve()),
            stat.st_size,
            stat.st_mtime_ns,
            inspect.getsource(json_array_reader),
            inspect.getsource(_iter_adapted_json_array),
            version("ijson"),
            adapter,
            inspect.getsource(type(adapter)) if adapter is not None else None,
        )
    )


def _load_json_array(path: Path, data_args: DataArguments) -> HFDataset:
    first_sample = read_first_json_array_item(str(path))
    adapter = resolve_dataset_adapter(data_args.dataset_adapter, first_sample)
    logger.info_rank0("Building the Arrow cache incrementally from %s", path)
    dataset = HFDataset.from_generator(
        _iter_adapted_json_array,
        features=adapter.features if adapter is not None else None,
        gen_kwargs={"path": str(path), "adapter": adapter},
        cache_dir=data_args.cache_dir,
        fingerprint=_json_array_fingerprint(path, adapter),
    )
    # A single annotation file defines the train split. Use upstream slicing
    # semantics rather than silently ignoring split expressions for arrays.
    instructions = ReadInstruction.from_spec(data_args.split).to_absolute(
        {"train": len(dataset)}
    )
    parts = [
        dataset.select(
            range(item.from_ or 0, item.to if item.to is not None else len(dataset))
        )
        for item in instructions
    ]
    return parts[0] if len(parts) == 1 else concatenate_datasets(parts)


def load_training_dataset(data_args: DataArguments) -> HFDataset:
    """Load a Hub dataset, local directory/file, or HF builder into Arrow.

    HF handles subsets, splits, file lists/globs, and repository revisions.
    Direct local JSON arrays use the incremental reader. Training consumes an
    indexed Dataset; iterable/streaming training is not exposed here.
    """
    source = data_args.dataset_name_or_path
    if not source:
        raise ValueError(
            "`dataset_name_or_path` must be provided for supervised fine-tuning."
        )
    if not isinstance(data_args.split, str) or not data_args.split:
        raise ValueError("`split` must select one dataset using a nonempty string.")

    path = Path(source).expanduser()
    if path.is_file():
        conflicts = [
            key
            for key in ("data_files", "data_dir", "dataset_config_name", "revision")
            if getattr(data_args, key) is not None
        ]
        if conflicts:
            raise ValueError(
                f"A direct dataset file cannot be combined with {conflicts}; use an HF builder instead."
            )
        if path.suffix.lower() in {".json", ".jsonl", ""} and is_json_array(path):
            return _load_json_array(path, data_args)
        # Let HF infer the format, including compression, from this one file.
        source = str(path.resolve().parent)
        data_files = str(path.resolve())
    else:
        if source.startswith(("/", "./", "../", "~")) and not path.exists():
            raise FileNotFoundError(f"Training dataset path does not exist: {path}")
        source = str(path) if path.is_dir() else source
        data_files = data_args.data_files

    dataset = load_dataset(
        source,
        name=data_args.dataset_config_name,
        split=data_args.split,
        data_files=data_files,
        data_dir=data_args.data_dir,
        revision=data_args.revision,
        cache_dir=data_args.cache_dir,
    )
    if not isinstance(dataset, HFDataset):
        raise TypeError(
            "Training requires one indexed Hugging Face Dataset; select a single split."
        )
    if len(dataset):
        adapter = resolve_dataset_adapter(
            data_args.dataset_adapter,
            dataset[0],
            features=dataset.features,
        )
        if adapter is not None:
            logger.info_rank0("Using training dataset adapter %s", adapter.name)
            dataset = dataset.map(
                adapter.adapt,
                remove_columns=dataset.column_names,
                features=adapter.features,
                desc=f"Applying {adapter.name} dataset adapter",
            )
    return dataset


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
    return dict(
        train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator
    )


__all__ = [
    "ProcessorSFTDataset",
    "load_training_dataset",
    "make_supervised_data_module",
]

"""Legacy TinyLLaVA modality-length sampling on the current HF Trainer."""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import torch
from torch.utils.data import Sampler
from transformers import Trainer
from transformers.trainer_utils import has_length


def split_to_even_chunks(
    indices: Sequence[int],
    lengths: Sequence[int],
    num_chunks: int,
) -> list[list[int]]:
    """Balance one length-sorted megabatch across data-parallel workers."""

    if len(indices) % num_chunks:
        return [list(indices[i::num_chunks]) for i in range(num_chunks)]

    chunk_size = len(indices) // num_chunks
    chunks: list[list[int]] = [[] for _ in range(num_chunks)]
    chunk_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest = chunk_lengths.index(min(chunk_lengths))
        chunks[shortest].append(index)
        chunk_lengths[shortest] += lengths[index]
        if len(chunks[shortest]) == chunk_size:
            chunk_lengths[shortest] = float("inf")
    return chunks


def get_length_grouped_indices(
    lengths: Sequence[int],
    batch_size: int,
    world_size: int,
    generator: torch.Generator | None = None,
) -> list[int]:
    indices = torch.randperm(len(lengths), generator=generator).tolist()
    megabatch_size = world_size * batch_size
    megabatches = [
        indices[start : start + megabatch_size]
        for start in range(0, len(indices), megabatch_size)
    ]
    megabatches = [
        sorted(batch, key=lambda index: lengths[index], reverse=True)
        for batch in megabatches
    ]
    balanced = [
        split_to_even_chunks(batch, lengths, world_size) for batch in megabatches
    ]
    return [index for megabatch in balanced for chunk in megabatch for index in chunk]


def get_modality_length_grouped_indices(
    lengths: Sequence[int],
    batch_size: int,
    world_size: int,
    generator: torch.Generator | None = None,
) -> list[int]:
    if any(length == 0 for length in lengths):
        raise ValueError("Modality lengths must be non-zero.")

    multimodal = [(index, length) for index, length in enumerate(lengths) if length > 0]
    language = [(index, -length) for index, length in enumerate(lengths) if length < 0]
    if not multimodal or not language:
        return get_length_grouped_indices(
            [abs(length) for length in lengths], batch_size, world_size, generator
        )

    mm_indices, mm_lengths = zip(*multimodal, strict=True)
    lang_indices, lang_lengths = zip(*language, strict=True)
    mm_order = get_length_grouped_indices(mm_lengths, batch_size, world_size, generator)
    lang_order = get_length_grouped_indices(
        lang_lengths, batch_size, world_size, generator
    )
    mm_shuffle = [mm_indices[index] for index in mm_order]
    lang_shuffle = [lang_indices[index] for index in lang_order]

    megabatch_size = world_size * batch_size
    mm_batches = [
        mm_shuffle[start : start + megabatch_size]
        for start in range(0, len(mm_shuffle), megabatch_size)
    ]
    lang_batches = [
        lang_shuffle[start : start + megabatch_size]
        for start in range(0, len(lang_shuffle), megabatch_size)
    ]
    tail = mm_batches.pop() + lang_batches.pop()
    megabatches = mm_batches + lang_batches
    order = torch.randperm(len(megabatches), generator=generator).tolist()
    megabatches = [megabatches[index] for index in order]
    if len(tail) >= megabatch_size:
        megabatches.insert(0, tail[:megabatch_size])
        tail = tail[megabatch_size:]
    if tail:
        megabatches.append(tail)
    return [index for batch in megabatches for index in batch]


class ModalityLengthGroupedSampler(Sampler[int]):
    """Order samples by modality and approximate length before distributed sharding.

    Args:
        batch_size: Samples per local training batch.
        world_size: Number of balancing chunks. The Trainer supplies process count
            multiplied by gradient-accumulation steps.
        lengths: Nonzero signed lengths: positive for image samples, negative for
            text-only samples. Absolute values estimate processing cost.
        generator: Random generator to use directly, overriding epoch-based seeding.
        seed: Base seed; without a generator, each epoch uses `seed + epoch`.
    """
    def __init__(
        self,
        *,
        batch_size: int,
        world_size: int,
        lengths: Sequence[int],
        generator: torch.Generator | None = None,
        seed: int = 0,
    ) -> None:
        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.seed = seed
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.lengths)

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch used to seed the next iteration.

        Args:
            epoch: Epoch index added to the base seed when no generator is supplied.
        """
        self.epoch = epoch

    def __iter__(self) -> Iterator[int]:
        generator = self.generator
        if generator is None:
            generator = torch.Generator()
            generator.manual_seed(self.seed + self.epoch)
        return iter(
            get_modality_length_grouped_indices(
                self.lengths,
                self.batch_size,
                self.world_size,
                generator,
            )
        )


class TinyLlavaTrainer(Trainer):
    """Hugging Face Trainer with optional modality-and-length grouping.

    Enable `group_by_modality_length` in the training arguments and provide a
    dataset exposing `modality_lengths` to use `ModalityLengthGroupedSampler`.
    Other sampling behavior follows the upstream Trainer.
    """

    def _get_train_sampler(self, train_dataset=None):
        if train_dataset is None:
            train_dataset = self.train_dataset
        if train_dataset is None or not has_length(train_dataset):
            return None
        if not self.args.group_by_modality_length:
            return super()._get_train_sampler(train_dataset)
        lengths = train_dataset.modality_lengths
        return ModalityLengthGroupedSampler(
            batch_size=self.args.train_batch_size,
            world_size=(
                self.args.world_size * self.args.gradient_accumulation_steps
            ),
            lengths=lengths,
            seed=(
                self.args.data_seed
                if self.args.data_seed is not None
                else self.args.seed
            ),
        )


__all__ = [
    "ModalityLengthGroupedSampler",
    "TinyLlavaTrainer",
    "get_length_grouped_indices",
    "get_modality_length_grouped_indices",
    "split_to_even_chunks",
]

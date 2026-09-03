import torch
from datasets import Dataset as HFDataset

from tinyllava.data.dataset import ProcessorSFTDataset
from tinyllava.train.modality_trainer import (
    ModalityLengthGroupedSampler,
    get_modality_length_grouped_indices,
)
from tinyllava.utils.arguments import DataArguments


def test_processor_dataset_exposes_signed_modality_lengths():
    dataset = HFDataset.from_list(
        [
            {
                "image": "image.jpg",
                "messages": [{"role": "user", "content": "one two three"}],
            },
            {
                "image": None,
                "messages": [{"role": "user", "content": "one two"}],
            },
        ]
    )
    wrapped = ProcessorSFTDataset(dataset, processor=None, data_args=DataArguments())

    assert wrapped.modality_lengths == [3, -2]


def test_modality_sampler_preserves_indices_and_separates_full_megabatches():
    lengths = [10, 9, 8, 7, -10, -9, -8, -7]

    indices = get_modality_length_grouped_indices(
        lengths,
        batch_size=2,
        world_size=2,
    )

    assert sorted(indices) == list(range(len(lengths)))
    megabatches = [indices[start : start + 4] for start in range(0, len(indices), 4)]
    assert all(
        all(lengths[index] > 0 for index in batch)
        or all(lengths[index] < 0 for index in batch)
        for batch in megabatches
    )


def test_modality_sampler_does_not_repeat_tail_samples():
    lengths = [10, 9, 8, 7, 6, -10, -9, -8, -7]

    sampler = ModalityLengthGroupedSampler(
        batch_size=2,
        world_size=2,
        lengths=lengths,
        seed=42,
    )
    indices = list(sampler)

    assert len(sampler) == len(lengths)
    assert len(indices) == len(lengths)
    assert sorted(indices) == list(range(len(lengths)))


def test_modality_sampler_is_deterministic_across_rank_rng_states():
    lengths = list(range(1, 65)) + list(range(-1, -65, -1))

    torch.manual_seed(1)
    rank_zero = ModalityLengthGroupedSampler(
        batch_size=2,
        world_size=8,
        lengths=lengths,
        seed=42,
    )
    zero_order = list(rank_zero)

    torch.manual_seed(999)
    rank_one = ModalityLengthGroupedSampler(
        batch_size=2,
        world_size=8,
        lengths=lengths,
        seed=42,
    )
    one_order = list(rank_one)

    assert zero_order == one_order
    rank_one.set_epoch(1)
    assert list(rank_one) != zero_order

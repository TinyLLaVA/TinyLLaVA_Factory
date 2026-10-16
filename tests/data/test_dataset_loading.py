import json

import datasets

from tinyllava.data.dataset import load_training_dataset
from tinyllava.hparams.arguments import DataArguments


def test_load_training_dataset_streams_top_level_array(tmp_path, monkeypatch):
    path = tmp_path / "samples.json"
    samples = [
        {"id": "one", "conversations": [{"from": "human", "value": "Hi"}]},
        {"id": "two", "conversations": [{"from": "gpt", "value": "Hello"}]},
    ]
    path.write_text(json.dumps(samples), encoding="utf-8")
    monkeypatch.setattr(datasets.config, "HF_DATASETS_CACHE", str(tmp_path / "cache"))

    dataset = load_training_dataset(DataArguments(dataset_name_or_path=str(path)))

    assert dataset.to_list() == [
        {
            "id": "one",
            "image": None,
            "messages": [{"role": "user", "content": "Hi"}],
        },
        {
            "id": "two",
            "image": None,
            "messages": [{"role": "assistant", "content": "Hello"}],
        },
    ]


def test_load_training_dataset_fills_columns_missing_from_some_rows(
    tmp_path, monkeypatch
):
    path = tmp_path / "mixed_samples.json"
    samples = [
        {
            "id": "image",
            "image": "image.jpg",
            "conversations": [{"from": "human", "value": "Hi"}],
        },
        {
            "id": "text",
            "model": "gpt-4",
            "conversations": [
                {
                    "from": "gpt",
                    "value": "Hello",
                    "text": "metadata",
                    "markdown": {"type": "code", "index": 0},
                }
            ],
        },
    ]
    path.write_text(json.dumps(samples), encoding="utf-8")
    monkeypatch.setattr(datasets.config, "HF_DATASETS_CACHE", str(tmp_path / "cache"))

    dataset = load_training_dataset(DataArguments(dataset_name_or_path=str(path)))

    assert dataset.column_names == ["id", "image", "messages"]
    assert dataset[1]["image"] is None
    assert "model" not in dataset.column_names
    assert dataset[1]["messages"] == [
        {"role": "assistant", "content": "Hello"}
    ]


def test_llava_adapter_also_applies_to_json_lines(tmp_path, monkeypatch):
    path = tmp_path / "samples.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "text",
                "model": "unused",
                "conversations": [{"from": "gpt", "value": "Hello"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(datasets.config, "HF_DATASETS_CACHE", str(tmp_path / "cache"))

    dataset = load_training_dataset(
        DataArguments(dataset_name_or_path=str(path), dataset_adapter="llava_legacy")
    )

    assert dataset.to_list() == [
        {
            "id": "text",
            "image": None,
            "messages": [{"role": "assistant", "content": "Hello"}],
        }
    ]

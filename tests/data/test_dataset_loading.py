import json

import datasets
import pytest

from tinyllava.data.dataset import load_training_dataset
from tinyllava.data.readers import iter_json_array
import tinyllava.data.readers.json_array as json_array_reader
from tinyllava.utils.arguments import DataArguments


def test_iter_json_array_streams_items_across_read_boundaries(tmp_path, monkeypatch):
    path = tmp_path / "samples.json"
    samples = [
        {"id": "one", "text": "contains [, ], and escaped quote: \""},
        {"id": "two", "text": "你好"},
    ]
    path.write_text(json.dumps(samples, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(json_array_reader, "_READ_CHUNK_SIZE", 7)

    assert list(iter_json_array(str(path))) == samples


def test_load_training_dataset_streams_top_level_array(tmp_path, monkeypatch):
    path = tmp_path / "samples.json"
    samples = [
        {"id": "one", "conversations": [{"from": "human", "value": "Hi"}]},
        {"id": "two", "conversations": [{"from": "gpt", "value": "Hello"}]},
    ]
    path.write_text(json.dumps(samples), encoding="utf-8")
    monkeypatch.setattr(datasets.config, "HF_DATASETS_CACHE", str(tmp_path / "cache"))

    dataset = load_training_dataset(DataArguments(data_path=str(path)))

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

    dataset = load_training_dataset(DataArguments(data_path=str(path)))

    assert dataset.column_names == ["id", "image", "messages"]
    assert dataset[1]["image"] is None
    assert "model" not in dataset.column_names
    assert dataset[1]["messages"] == [
        {"role": "assistant", "content": "Hello"}
    ]


def test_iter_json_array_rejects_non_object_rows(tmp_path):
    path = tmp_path / "samples.json"
    path.write_text('[{"id": "one"}, 2]', encoding="utf-8")

    with pytest.raises(TypeError, match="must contain JSON objects"):
        list(iter_json_array(str(path)))


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
        DataArguments(data_path=str(path), dataset_adapter="llava_legacy")
    )

    assert dataset.to_list() == [
        {
            "id": "text",
            "image": None,
            "messages": [{"role": "assistant", "content": "Hello"}],
        }
    ]

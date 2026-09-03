from pathlib import Path

import pytest

from tinyllava.utils.config import build_eval_arguments, parse_eval_config


CONFIG_DIR = Path(__file__).parents[2] / "configs" / "eval"


def test_parse_eval_config_applies_dotlist_overrides(tmp_path):
    config_file = tmp_path / "eval.yaml"
    config_file.write_text(
        """
model:
  model_name_or_path: output/base
data:
  adapter: mmmu
generation:
  temperature: 0.0
runtime:
  num_chunks: 1
  chunk_idx: 0
output:
  answers_file: answers.jsonl
""",
        encoding="utf-8",
    )

    config = parse_eval_config(
        [
            "--config",
            str(config_file),
            "model.model_name_or_path=output/override",
            "runtime.num_chunks=4",
            "runtime.chunk_idx=2",
        ]
    )

    assert config.model.model_name_or_path == "output/override"
    assert config.data.adapter == "mmmu"
    assert config.generation.temperature == 0.0
    assert config.runtime.num_chunks == 4
    assert config.runtime.chunk_idx == 2
    assert config.output.answers_file == "answers.jsonl"


@pytest.mark.parametrize("config_file", sorted(CONFIG_DIR.glob("*.yaml")))
def test_benchmark_configs_are_valid(config_file, monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL_PATH", "output/test-model")
    monkeypatch.setenv("MODEL_NAME", "test-model")
    monkeypatch.setenv("EVAL_DIR", str(tmp_path / "eval"))

    config = parse_eval_config(["--config", str(config_file)])

    assert config.model.model_name_or_path == "output/test-model"
    assert config.model.model_id == "test-model"
    assert config.output.answers_file.endswith(".jsonl")


def test_eval_config_rejects_invalid_chunk_index():
    with pytest.raises(ValueError, match="runtime.chunk_idx"):
        build_eval_arguments(
            {
                "runtime": {
                    "num_chunks": 2,
                    "chunk_idx": 2,
                }
            }
        )

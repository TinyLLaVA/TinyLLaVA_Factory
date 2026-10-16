import pytest

from tinyllava.hparams.parser import build_eval_arguments, parse_eval_arguments


def test_parse_eval_arguments_applies_dotlist_overrides(tmp_path):
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

    config = parse_eval_arguments(
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

from copy import deepcopy
from unittest.mock import patch

import pytest
from omegaconf import OmegaConf

from tinyllava.eval.batch_generation import main
from tinyllava.eval.workflow import load_eval_config
from tinyllava.hparams.arguments import EvalArguments


@pytest.fixture
def config():
    return {
        "model_name_or_path": "output/base",
        "adapter": "mmmu",
        "question_file": "questions.jsonl",
        "temperature": 0.0,
        "num_chunks": 1,
        "chunk_idx": 0,
        "answers_file": "custom/answers.jsonl",
    }


def test_worker_reads_resolved_config_and_field_overrides(tmp_path, config):
    path = tmp_path / "resolved.yaml"
    OmegaConf.save(OmegaConf.create(config), path)
    with patch("tinyllava.eval.batch_generation.run_generation") as generate:
        main(
            [
                "--config",
                str(path),
                "model_name_or_path=output/override",
                "num_chunks=4",
                "chunk_idx=2",
            ]
        )
    args = generate.call_args.args[0]
    assert args.model_name_or_path == "output/override"
    assert args.question_file == "questions.jsonl"
    assert args.adapter == "mmmu"
    assert args.temperature == 0.0
    assert args.num_chunks == 4
    assert args.chunk_idx == 2
    assert args.answers_file == "custom/answers.jsonl"
    # Worker loading does not add the workflow's model identifier or output paths.
    assert args.model_id is None


def test_argument_construction_preserves_config(config):
    original = deepcopy(config)
    args = EvalArguments(**config)
    assert args.batch_size == 1
    assert config == original


@pytest.mark.parametrize(
    "name,value",
    [
        ("batch_size", 0),
        ("batch_size", 1.5),
        ("num_chunks", 0),
        ("chunk_idx", -1),
        ("chunk_idx", 1),
        ("num_beams", False),
        ("max_new_tokens", 0),
    ],
)
def test_invalid_sizes_fail_before_worker_launch(tmp_path, config, name, value):
    config[name] = value
    path = tmp_path / "eval.yaml"
    OmegaConf.save(OmegaConf.create(config), path)
    with pytest.raises(ValueError, match=name):
        load_eval_config(path)


def test_unknown_field_is_not_discarded_by_workflow(tmp_path, config):
    config["misspelled"] = True
    path = tmp_path / "eval.yaml"
    OmegaConf.save(OmegaConf.create(config), path)
    with pytest.raises(TypeError, match="misspelled"):
        load_eval_config(path)


def test_workflow_resolves_checkpoint_and_answer_paths(tmp_path, config):
    config.update(
        dataset_dir=str(tmp_path),
        model_name_or_path="${dataset_dir}/checkpoint",
        model_id=None,
        answers_file="${output_dir}/predictions.jsonl",
    )
    path = tmp_path / "eval.yaml"
    OmegaConf.save(OmegaConf.create(config), path)
    result = load_eval_config(path)
    assert result["model_name_or_path"] == str(tmp_path / "checkpoint")
    assert result["answers_file"] == "output/checkpoint/eval/eval/predictions.jsonl"
    assert "dataset_dir" not in result
    assert "output_dir" not in result


def test_worker_requires_resolved_input_paths():
    with pytest.raises(TypeError, match="model_name_or_path.*question_file"):
        EvalArguments()


def test_direct_arguments_validate_shards():
    with pytest.raises(ValueError, match="chunk_idx"):
        EvalArguments(
            model_name_or_path="checkpoint",
            question_file="questions.jsonl",
            chunk_idx=1,
        )

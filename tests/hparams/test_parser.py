import subprocess
import sys
from unittest.mock import patch

import pytest
from omegaconf import OmegaConf

from tinyllava.hparams.parser import parse_eval_arguments
from tinyllava.train.train import train
from tinyllava.hparams.training import build_training_arguments as build_arguments


@pytest.mark.parametrize(
    "worker,config",
    [
        (
            train,
            {
                "model": {"pretrained_model_name_or_path": "output/align"},
                "training": {
                    "saving": {"output_dir": "output/custom"},
                    "logging": {"run_name": "custom"},
                },
            },
        ),
        (
            parse_eval_arguments,
            {
                "model": {"model_name_or_path": "output/finetune"},
                "output": {"answers_file": "custom/answers.jsonl"},
            },
        ),
    ],
)
def test_worker_config_is_not_recomposed(tmp_path, monkeypatch, worker, config):
    monkeypatch.setenv("MODEL_PATH", "unrelated/model")
    monkeypatch.setenv("MODEL_NAME", "unrelated")
    monkeypatch.setenv("TINYLLAVA_DATASET_DIR", "/unrelated")
    path = tmp_path / "resolved.yaml"
    OmegaConf.save(OmegaConf.create(config), path)
    target = (
        "tinyllava.train.algorithms.sft.run"
        if worker is train
        else "tinyllava.hparams.parser.build_eval_arguments"
    )
    with patch(target) as consume:
        worker(["--config", str(path)])
    consume.assert_called_once_with(config)


@pytest.mark.parametrize(
    "config",
    [
        {"recipe": "pretrain", "model": "phi"},
        {"stages": {"pretrain": {}}},
    ],
)
def test_sft_rejects_experiment_metadata(config):
    with pytest.raises(ValueError, match="Unknown .* config section"):
        build_arguments(config)


def test_eval_worker_rejects_experiment_metadata(tmp_path):
    path = tmp_path / "experiment.yaml"
    OmegaConf.save(OmegaConf.create({"dataset": "gqa"}), path)
    with pytest.raises(ValueError, match="Unknown .* config section"):
        parse_eval_arguments(["--config", str(path)])


def test_training_worker_does_not_import_trainers_or_parameter_parsers():
    code = """
import sys
import tinyllava.train.train
assert "torch" not in sys.modules
assert "tinyllava.hparams.parser" not in sys.modules
assert "tinyllava.train.algorithms.sft_trainer" not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_parameter_parser_does_not_import_workflows():
    code = """
import sys
import tinyllava.hparams.parser
assert "tinyllava.train.workflow" not in sys.modules
assert "tinyllava.eval.workflow" not in sys.modules
assert "tinyllava.runner" not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr

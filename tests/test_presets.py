"""Validate every shipped experiment and evaluation entry once, offline."""

from pathlib import Path
from unittest.mock import patch

import pytest
from omegaconf import OmegaConf

from tinyllava.eval.workflow import load_eval_config, run as run_eval
from tinyllava.hparams.parser import build_eval_arguments
from tinyllava.train.workflow import training_stages, run as run_train

ROOT = Path(__file__).parents[1]
EXPERIMENTS = sorted((ROOT / "configs/experiments").glob("*.yaml"))
EVALUATIONS = sorted((ROOT / "configs/eval").glob("*.yaml"))


@pytest.mark.parametrize("path", EXPERIMENTS, ids=lambda p: p.stem)
def test_all_experiments_are_resolved_and_linked(path):
    assert "algorithm" in OmegaConf.load(path)
    stages = training_stages(path)
    previous = None
    for name, config, _ in stages:
        assert (
            config["training"]["saving"]["output_dir"] == f"output/{path.stem}/{name}"
        )
        assert config["data"]["dataset_name_or_path"].startswith("datasets/")
        if previous:
            assert config["model"]["pretrained_model_name_or_path"] == previous
            assert (
                not {
                    "connector",
                    "language_model_name_or_path",
                    "vision_model_name_or_path",
                }
                & config["model"].keys()
            )
        previous = config["training"]["saving"]["output_dir"]
    with patch("tinyllava.runner.subprocess") as processes:
        run_train(path, dry_run=True)
    processes.run.assert_not_called()
    processes.Popen.assert_not_called()


@pytest.mark.parametrize("path", EVALUATIONS, ids=lambda p: p.stem)
def test_eval_output_is_outside_datasets(path):
    config = load_eval_config(
        path, ["model.model_name_or_path=output/example/sft"], metadata=True
    )
    assert (
        config["output"]["answers_file"]
        == f"output/example/sft/eval/{config['benchmark']}/answers.jsonl"
    )
    build_eval_arguments(
        {
            key: config[key]
            for key in ("model", "data", "generation", "runtime", "output")
        }
    )
    with patch("tinyllava.runner.subprocess") as processes:
        run_eval(path, ["model.model_name_or_path=output/example/sft"], dry_run=True)
    processes.run.assert_not_called()
    processes.Popen.assert_not_called()

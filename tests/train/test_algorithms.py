import subprocess
import sys
from types import ModuleType
from unittest.mock import patch

import pytest
from omegaconf import OmegaConf

from tinyllava.train.algorithms import get_algorithm
from tinyllava.train.train import train
from tinyllava.train.workflow import train_plan, training_stages


def test_unknown_algorithm_fails_before_launch():
    with pytest.raises(ValueError, match="Unsupported training algorithm"):
        get_algorithm("unknown")


def test_stage_algorithm_override_is_validated(tmp_path):
    path = tmp_path / "experiment.yaml"
    OmegaConf.save(
        OmegaConf.create(
            {
                "stages": {
                    "align": {"algorithm": "sft"},
                    "next": {"algorithm": "unknown"},
                },
            }
        ),
        path,
    )
    with pytest.raises(ValueError, match="unknown"):
        training_stages(path)


def test_algorithm_selection_does_not_import_training_dependencies():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
import tinyllava.train.train
from tinyllava.train.algorithms import get_algorithm
get_algorithm('sft')
from tinyllava.hparams.training import validate_training_config
from tinyllava.hparams.arguments import EvalArguments
validate_training_config({'model': {'connector': {'model_type': 'mlp'}}})
EvalArguments(model_name_or_path='checkpoint', question_file='questions.jsonl')
assert 'torch' not in sys.modules
assert 'transformers' not in sys.modules
assert 'tinyllava.train.algorithms.sft_trainer' not in sys.modules
""",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_training_worker_does_not_recompose_resolved_config(tmp_path):
    config = {
        "model": {"pretrained_model_name_or_path": "output/align"},
        "training": {"output_dir": "output/custom"},
    }
    path = tmp_path / "resolved.yaml"
    OmegaConf.save(OmegaConf.create(config), path)
    with patch("tinyllava.train.algorithms.sft.run") as run:
        train(["--config", str(path)])
    run.assert_called_once_with(config)


def test_launcher_arguments_override_yaml_and_dotlist_values(tmp_path):
    config_path = tmp_path / "train.yaml"
    config_path.write_text(
        "training:\n"
        "  output_dir: output/test\n"
        "  deepspeed: yaml.json\n"
        "  local_rank: 0\n",
        encoding="utf-8",
    )

    with patch("tinyllava.train.algorithms.sft.run") as run:
        train(
            [
                "--config",
                str(config_path),
                "training.deepspeed=dotlist.json",
                "--deepspeed",
                "cli.json",
                "--local-rank",
                "3",
            ]
        )

    config = run.call_args.args[0]
    assert config["training"]["deepspeed"] == "cli.json"
    assert config["training"]["local_rank"] == 3


def test_workflow_accepts_algorithm_owned_sections(tmp_path, monkeypatch):
    from tinyllava.train import algorithms

    algorithm = ModuleType("test_custom_algorithm")
    algorithm.SECTIONS = ("model", "training", "reward")
    seen = []
    algorithm.validate_config = lambda config: seen.append(config["reward"])
    algorithm.prepare_config = lambda config, *, world_size: {
        **config,
        "workers": world_size,
    }
    monkeypatch.setitem(sys.modules, algorithm.__name__, algorithm)
    monkeypatch.setitem(algorithms._ALGORITHMS, "custom", algorithm.__name__)
    path = tmp_path / "custom.yaml"
    OmegaConf.save(
        OmegaConf.create({"algorithm": "custom", "reward": {"scale": 2}}), path
    )
    plan = train_plan(path)
    assert seen == [{"scale": 2}]
    assert plan[0][2]["reward"] == {"scale": 2}
    assert plan[0][2]["workers"] == 1
    assert "data" not in plan[0][2]


def test_stages_select_independent_algorithms_and_inputs(tmp_path, monkeypatch):
    from tinyllava.train import algorithms

    algorithm = ModuleType("test_reward_algorithm")
    algorithm.SECTIONS = ("model", "training", "reward")
    algorithm.validate_config = lambda config: None
    algorithm.prepare_config = lambda config, *, world_size: config
    monkeypatch.setitem(sys.modules, algorithm.__name__, algorithm)
    monkeypatch.setitem(algorithms._ALGORITHMS, "reward", algorithm.__name__)
    path = tmp_path / "mixed.yaml"
    OmegaConf.save(
        OmegaConf.create(
            {
                "model": {"language_model_name_or_path": "base"},
                "stages": {
                    "sft": {
                        "algorithm": "sft",
                        "data": {"dataset_name_or_path": "sft.json"},
                    },
                    "reward": {"algorithm": "reward", "reward": {"scale": 2}},
                },
            }
        ),
        path,
    )

    plan = train_plan(path)
    assert plan[0][2]["algorithm"] == "sft"
    assert plan[0][2]["data"]["dataset_name_or_path"] == "sft.json"
    assert plan[1][2]["algorithm"] == "reward"
    assert plan[1][2]["reward"] == {"scale": 2}
    assert "data" not in plan[1][2]
    assert plan[1][2]["model"]["pretrained_model_name_or_path"] == "output/mixed/sft"


@pytest.mark.parametrize(
    "shared", [{"algorithm": "sft"}, {"data": {"dataset_name_or_path": "shared.json"}}]
)
def test_algorithm_and_data_are_stage_local(tmp_path, shared):
    path = tmp_path / "shared.yaml"
    OmegaConf.save(
        OmegaConf.create({**shared, "stages": {"sft": {"algorithm": "sft"}}}), path
    )
    with pytest.raises(ValueError, match="shared experiment"):
        training_stages(path)

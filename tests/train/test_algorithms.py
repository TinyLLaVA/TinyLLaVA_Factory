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
                "algorithm": "sft",
                "stages": {"align": {}, "next": {"algorithm": "unknown"}},
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
assert 'torch' not in sys.modules
assert 'tinyllava.train.algorithms.sft_trainer' not in sys.modules
""",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_launcher_arguments_override_yaml_and_dotlist_values(tmp_path):
    config_path = tmp_path / "train.yaml"
    config_path.write_text(
        "training:\n"
        "  saving:\n    output_dir: output/test\n"
        "  runtime:\n    deepspeed: yaml.json\n"
        "    local_rank: 0\n",
        encoding="utf-8",
    )

    with patch("tinyllava.train.algorithms.sft.run") as run:
        train(
            [
                "--config",
                str(config_path),
                "training.runtime.deepspeed=dotlist.json",
                "--deepspeed",
                "cli.json",
                "--local-rank",
                "3",
            ]
        )

    config = run.call_args.args[0]
    assert config["training"]["runtime"]["deepspeed"] == "cli.json"
    assert config["training"]["runtime"]["local_rank"] == 3


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

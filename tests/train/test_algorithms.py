import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from omegaconf import OmegaConf

from tinyllava.train.algorithms import get_algorithm
from tinyllava.train.train import train
from tinyllava.train.workflow import training_stages

ROOT = Path(__file__).parents[2]


def test_existing_configs_select_sft():
    config = training_stages(ROOT / "configs/experiments/qwen2_instruct.yaml")[0][1]
    assert config["algorithm"] == "sft"


@pytest.mark.parametrize("algorithm", ["grpo", "ppo", None, {"name": "sft"}])
def test_unimplemented_algorithm_fails_before_launch(algorithm):
    with pytest.raises(ValueError, match="Unsupported training algorithm"):
        get_algorithm(algorithm)


def test_stage_algorithm_override_is_validated(tmp_path):
    path = tmp_path / "experiment.yaml"
    OmegaConf.save(
        OmegaConf.create(
            {
                "algorithm": "sft",
                "stages": {"align": {}, "rl": {"algorithm": "grpo"}},
            }
        ),
        path,
    )
    with pytest.raises(ValueError, match="grpo"):
        training_stages(path)


def test_worker_dispatch_preserves_config_and_launcher_overrides(tmp_path):
    path = tmp_path / "resolved.yaml"
    OmegaConf.save(
        OmegaConf.create(
            {"algorithm": "sft", "training": {"saving": {"output_dir": "out"}}}
        ),
        path,
    )
    with patch("tinyllava.train.algorithms.sft.run") as run:
        train(["--config", str(path), "--local-rank", "2"])
    run.assert_called_once_with(
        {
            "algorithm": "sft",
            "training": {"saving": {"output_dir": "out"}, "runtime": {"local_rank": 2}},
        }
    )


def test_algorithm_selection_does_not_import_training_dependencies():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
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

from pathlib import Path
from types import ModuleType
import sys

import pytest
from omegaconf import OmegaConf

from tinyllava.train.workflow import load_train_config, train_plan

ROOT = Path(__file__).parents[2]


def write_config(tmp_path, config, name="experiment.yaml"):
    path = tmp_path / name
    OmegaConf.save(OmegaConf.create(config), path)
    return path


def test_batch_is_rejected_in_launcher():
    with pytest.raises(ValueError, match="Unknown launcher fields"):
        train_plan(
            ROOT / "configs/experiments/qwen2_instruct.yaml",
            ["launcher.global_batch_size=32"],
        )


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
    path = write_config(tmp_path, {"algorithm": "custom", "reward": {"scale": 2}})
    plan = train_plan(path)
    assert seen == [{"scale": 2}]
    assert plan[0][2]["reward"] == {"scale": 2}
    assert plan[0][2]["workers"] == 1
    assert "data" not in plan[0][2]


def test_experiment_name_does_not_select_model(tmp_path):
    path = write_config(
        tmp_path,
        {
            "experiment_name": "trial",
            "model": {"from": str(ROOT / "configs/models/phi.yaml")},
        },
    )
    resolved = load_train_config(path)
    assert resolved["model"]["language_model_name_or_path"] == "microsoft/phi-2"
    assert resolved["training"]["saving"]["output_dir"] == "output/trial/experiment"
    assert resolved["training"]["logging"]["run_name"] == "trial/experiment"


def test_entry_configs_explicitly_select_algorithm():
    for path in (ROOT / "configs/experiments").glob("*.yaml"):
        assert OmegaConf.load(path).algorithm == "sft"

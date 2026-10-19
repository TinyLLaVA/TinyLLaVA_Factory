from pathlib import Path

import pytest
from omegaconf import OmegaConf

from tinyllava.eval.workflow import load_eval_config
from tinyllava.train.workflow import load_train_config
from tinyllava.utils.config import compose_config


def write_config(path, config):
    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(OmegaConf.create(config), path)
    return path


@pytest.fixture
def presets(tmp_path, monkeypatch):
    from tinyllava.utils import config

    root = tmp_path / "presets"
    write_config(root / "connectors/mlp.yaml", {"depth": 2, "bias": True})
    write_config(root / "connectors/linear.yaml", {"new_field": True})
    write_config(
        root / "models/first.yaml",
        {"id": "first", "connector": "mlp.yaml", "old_field": 1},
    )
    write_config(
        root / "models/second.yaml", {"id": "second", "connector": "linear.yaml"}
    )
    write_config(root / "training/base.yaml", {"lr": 0.1})
    write_config(root / "training/fast.yaml", {"from": "base.yaml", "steps": 4})
    write_config(root / "data/train/shared.yaml", {"split": "train"})
    write_config(root / "data/eval/shared.yaml", {"split": "test"})
    monkeypatch.setattr(config, "CONFIG_ROOT", root)
    return root


@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("suffix", ["", ".yaml"])
def test_preset_lookup_and_scalar_shorthand_have_identical_results(
    tmp_path, presets, compact, suffix
):
    def ref(name):
        name = name.removesuffix(".yaml") + suffix
        return name if compact else {"from": name}

    path = write_config(
        tmp_path / "entry.yaml",
        {
            "model": ref("first.yaml"),
            "stages": {
                "align": {"training": ref("fast.yaml"), "data": ref("shared.yaml")}
            },
        },
    )
    assert compose_config(path) == {
        "model": {
            "id": "first",
            "old_field": 1,
            "connector": {"depth": 2, "bias": True},
        },
        "stages": {
            "align": {"training": {"lr": 0.1, "steps": 4}, "data": {"split": "train"}}
        },
    }


def test_bare_names_use_presets_and_explicit_paths_use_the_declaring_file(
    tmp_path, presets, monkeypatch
):
    write_config(tmp_path / "first.yaml", {"id": "local"})
    path = write_config(tmp_path / "entry.yaml", {"model": "first.yaml"})
    monkeypatch.chdir(tmp_path.parent)
    assert compose_config(path)["model"]["id"] == "first"
    write_config(path, {"model": "./first.yaml"})
    assert compose_config(path)["model"] == {"id": "local"}
    write_config(path, {"model": str(tmp_path / "first.yaml")})
    assert compose_config(path)["model"] == {"id": "local"}


def test_missing_preset_does_not_silently_fall_back_to_local_file(tmp_path, presets):
    write_config(tmp_path / "unknown.yaml", {"id": "local"})
    path = write_config(tmp_path / "entry.yaml", {"model": "unknown.yaml"})
    with pytest.raises(FileNotFoundError, match="presets/models/unknown.yaml"):
        compose_config(path)


def test_training_data_shorthand_and_flat_eval_reference(tmp_path, presets):
    train = write_config(tmp_path / "train.yaml", {"data": "shared"})
    assert compose_config(train) == {"data": {"split": "train"}}
    evaluation = write_config(
        tmp_path / "eval.yaml", {"from": str(presets / "data/eval/shared")}
    )
    assert compose_config(evaluation) == {"split": "test"}


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("suffix", ["", ".yaml"])
def test_cli_shorthand_combines_with_deep_overrides_and_file_switches(
    tmp_path, presets, reverse, suffix
):
    path = write_config(
        tmp_path / "entry.yaml", {"model": {"from": "first.yaml", "explicit": 4}}
    )
    overrides = [
        f"model=second{suffix}",
        f"model.connector=mlp{suffix}",
        "model.connector.depth=9",
    ]
    if reverse:
        overrides.reverse()
    assert compose_config(path, overrides) == {
        "model": {
            "id": "second",
            "explicit": 4,
            "connector": {"depth": 9, "bias": True},
        },
    }


def test_repeated_cli_selection_and_explicit_cli_paths(tmp_path, presets, monkeypatch):
    path = write_config(tmp_path / "entry.yaml", {"model": "first.yaml"})
    monkeypatch.chdir(tmp_path)
    write_config(tmp_path / "model.yml", {"id": "custom"})
    assert compose_config(
        path, ["model=second.yaml", "model=./model.yml", "model.extra=1"]
    ) == {
        "model": {"id": "custom", "extra": 1},
    }


def test_nested_shorthand_overrides_inherited_file_without_reading_old_source(
    tmp_path, presets
):
    write_config(presets / "models/parent.yaml", {"connector": "missing.yaml"})
    path = write_config(
        tmp_path / "entry.yaml",
        {"model": {"from": "parent.yaml", "connector": "mlp.yaml"}},
    )
    assert compose_config(path) == {"model": {"connector": {"depth": 2, "bias": True}}}


def test_file_inheritance_keeps_preset_context_and_relative_origins(tmp_path, presets):
    write_config(presets / "models/base.yaml", {"connector": "./local.yml", "extra": 1})
    write_config(presets / "models/local.yml", {"depth": 7})
    write_config(presets / "models/child.yaml", {"from": "base.yaml", "extra": 2})
    path = write_config(tmp_path / "entry.yaml", {"model": "child.yaml"})
    assert compose_config(path) == {"model": {"connector": {"depth": 7}, "extra": 2}}


def test_source_lists_can_mix_presets_and_explicit_paths(tmp_path, presets):
    write_config(tmp_path / "hardware.yaml", {"lr": 0.2, "workers": 8})
    path = write_config(
        tmp_path / "entry.yaml",
        {
            "training": {"from": ["fast.yaml", "./hardware.yaml"], "steps": 10},
        },
    )
    assert compose_config(path) == {"training": {"lr": 0.2, "steps": 10, "workers": 8}}


def test_scalar_paths_and_stage_names_are_not_reinterpreted(tmp_path, presets):
    path = write_config(
        tmp_path / "entry.yaml",
        {
            "answers_file": "results.yaml",
            "data": {
                "dataset_name_or_path": "dataset.yaml",
                "data_files": ["samples.yaml"],
            },
            "model": {"chat_template_path": "template.yaml"},
            "training": {"output_dir": "run.yaml"},
            "stages": {"model": {"from": "./stage.yaml"}},
        },
    )
    write_config(tmp_path / "stage.yaml", {"training": "base.yaml"})
    config = compose_config(path)
    assert config["answers_file"] == "results.yaml"
    assert config["data"] == {
        "dataset_name_or_path": "dataset.yaml",
        "data_files": ["samples.yaml"],
    }
    assert config["model"]["chat_template_path"] == "template.yaml"
    assert config["training"]["output_dir"] == "run.yaml"
    assert config["stages"]["model"] == {"training": {"lr": 0.1}}


def test_preset_cycles_still_fail_and_duplicate_references_are_independent(
    tmp_path, presets
):
    write_config(presets / "models/loop.yaml", {"from": "loop.yaml"})
    path = write_config(tmp_path / "entry.yaml", {"model": "loop.yaml"})
    with pytest.raises(ValueError, match="Circular config reference:.*loop.yaml"):
        compose_config(path)
    write_config(
        path,
        {
            "stages": {
                "one": {"model": "first.yaml"},
                "two": {"model": {"from": "first.yaml", "connector": {"depth": 4}}},
            }
        },
    )
    config = compose_config(path)
    assert config["stages"]["one"]["model"]["connector"]["depth"] == 2
    assert config["stages"]["two"]["model"]["connector"]["depth"] == 4


def test_train_and_eval_workflows_select_their_data_presets(tmp_path):
    train = write_config(
        tmp_path / "custom_train.yaml",
        {
            "algorithm": "sft",
            "model": "phi",
            "data": "llava_558k",
            "training": "pretrain",
        },
    )
    config = load_train_config(
        train, ["model.model_max_length=4096", "training.learning_rate=0.01"]
    )
    assert config["model"]["language_model_name_or_path"] == "microsoft/phi-2"
    assert config["model"]["model_max_length"] == 4096
    assert config["training"]["learning_rate"] == 0.01
    assert config["data"]["dataset_name_or_path"].startswith("datasets/")
    write_config(
        tmp_path / "eval-model.yaml", {"model_name_or_path": "output/custom/finetune"}
    )
    evaluation = write_config(
        tmp_path / "custom_eval.yaml",
        {
            "benchmark": "gqa",
            "from": [
                str(Path(__file__).parents[2] / "configs/data/eval/gqa.yaml"),
                "./eval-model",
            ],
        },
    )
    config = load_eval_config(evaluation, ["image_folder=/custom/images"])
    assert config["model_name_or_path"] == "output/custom/finetune"
    assert config["adapter"] == "vqa"
    assert config["image_folder"] == "/custom/images"
    assert config["question_file"].startswith("datasets/eval/gqa/")


@pytest.mark.parametrize("compact", [False, True])
def test_extensionless_names_with_dots_and_explicit_paths(tmp_path, presets, compact):
    write_config(presets / "models/custom.v2.yaml", {"id": "preset"})
    write_config(tmp_path / "custom.v2.yaml", {"id": "local"})
    for reference, expected in (
        ("custom.v2", "preset"),
        ("./custom.v2", "local"),
        (str(tmp_path / "custom.v2"), "local"),
    ):
        path = write_config(
            tmp_path / "entry.yaml",
            {
                "model": reference if compact else {"from": reference},
            },
        )
        assert compose_config(path)["model"] == {"id": expected}


def test_extensionless_source_lists_and_inheritance(tmp_path, presets):
    write_config(presets / "training/child.yaml", {"from": "base", "steps": 3})
    write_config(tmp_path / "hardware.yaml", {"workers": 4})
    path = write_config(
        tmp_path / "entry.yaml",
        {
            "training": {"from": ["child", "./hardware"], "lr": 0.2},
        },
    )
    assert compose_config(path) == {"training": {"lr": 0.2, "steps": 3, "workers": 4}}


def test_missing_extensionless_preset_reports_yaml_path_without_local_fallback(
    tmp_path, presets
):
    write_config(tmp_path / "missing.yaml", {"id": "local"})
    path = write_config(tmp_path / "entry.yaml", {"model": "missing"})
    with pytest.raises(FileNotFoundError, match="presets/models/missing.yaml"):
        compose_config(path)


def test_extensionless_cycles_are_detected(tmp_path, presets):
    write_config(presets / "models/loop.yaml", {"from": "loop"})
    path = write_config(tmp_path / "entry.yaml", {"model": "loop"})
    with pytest.raises(ValueError, match="Circular config reference:.*loop.yaml"):
        compose_config(path)

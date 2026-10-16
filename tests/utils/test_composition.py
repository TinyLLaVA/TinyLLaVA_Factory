from pathlib import Path

import pytest
from omegaconf import OmegaConf

from tinyllava.utils.config import compose_config, load_config, resolve_interpolations
from tinyllava.train.workflow import training_stages

ROOT = Path(__file__).parents[2]


def write_config(path, config):
    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(OmegaConf.create(config), path)
    return path


def test_nested_references_keep_their_declaring_directory(tmp_path, monkeypatch):
    write_config(tmp_path / "parts/connector.yaml", {"depth": 2, "bias": True})
    write_config(
        tmp_path / "models/custom.yaml",
        {
            "language_model_name_or_path": "custom/llm",
            "connector": {"from": "../parts/connector.yaml", "depth": 3},
        },
    )
    path = write_config(
        tmp_path / "experiment.yaml",
        {
            "model": {"from": "models/custom.yaml", "connector": {"bias": False}},
            "data": {"image_folder": "images"},
        },
    )
    monkeypatch.chdir(tmp_path.parent)
    config = compose_config(path)
    assert config == {
        "model": {
            "language_model_name_or_path": "custom/llm",
            "connector": {"depth": 3, "bias": False},
        },
        "data": {"image_folder": "images"},
    }
    assert Path.cwd() == tmp_path.parent


def test_multiple_sources_local_and_cli_precedence(tmp_path):
    write_config(
        tmp_path / "first.yaml", {"training": {"batch": 2, "lr": 0.1}, "items": [1, 2]}
    )
    write_config(
        tmp_path / "second.yaml", {"training": {"batch": 4, "epochs": 3}, "items": [3]}
    )
    path = write_config(
        tmp_path / "entry.yaml",
        {
            "from": ["first.yaml", "second.yaml"],
            "training": {"batch": 8},
        },
    )
    assert compose_config(path, ["training.lr=0.2", "new_field=true"]) == {
        "training": {"batch": 8, "lr": 0.2, "epochs": 3},
        "items": [3],
        "new_field": True,
    }


def test_root_inheritance_and_nested_file_switch_merge_before_expansion(tmp_path):
    write_config(
        tmp_path / "base.yaml",
        {
            "model": {"from": "./missing.yaml", "local_setting": 1},
            "training": {"lr": 0.1},
        },
    )
    write_config(tmp_path / "model.yaml", {"new_setting": 2})
    path = write_config(
        tmp_path / "entry.yaml",
        {
            "from": "./base.yaml",
            "model": {"from": "./model.yaml"},
        },
    )
    assert compose_config(path) == {
        "model": {"local_setting": 1, "new_setting": 2},
        "training": {"lr": 0.1},
    }


def test_cli_file_selection_replaces_old_defaults_but_keeps_explicit_fields(
    tmp_path, monkeypatch
):
    write_config(tmp_path / "config/old.yaml", {"obsolete": True, "depth": 2})
    write_config(tmp_path / "new.yaml", {"model_type": "new", "depth": 4})
    write_config(
        tmp_path / "config/model.yaml",
        {"connector": {"from": "./old.yaml", "bias": False}},
    )
    path = write_config(
        tmp_path / "config/entry.yaml", {"model": {"from": "./model.yaml"}}
    )
    monkeypatch.chdir(tmp_path)
    assert compose_config(
        path, ["model.connector.from=./new.yaml", "model.connector.depth=8"]
    ) == {
        "model": {"connector": {"model_type": "new", "depth": 8, "bias": False}},
    }
    # The old reference need not exist when a new one is selected.
    (tmp_path / "config/old.yaml").unlink()
    assert (
        compose_config(path, ["model.connector.from=./new.yaml"])["model"]["connector"][
            "depth"
        ]
        == 4
    )


def test_later_source_can_replace_an_earlier_nested_reference(tmp_path):
    write_config(
        tmp_path / "first.yaml", {"model": {"from": "./missing.yaml", "explicit": 1}}
    )
    write_config(tmp_path / "second.yaml", {"model": {"from": "./selected.yaml"}})
    write_config(tmp_path / "selected.yaml", {"selected": True})
    path = write_config(
        tmp_path / "entry.yaml", {"from": ["first.yaml", "second.yaml"]}
    )
    assert compose_config(path) == {"model": {"selected": True, "explicit": 1}}


def test_references_work_in_arbitrary_sections_and_list_items(tmp_path):
    write_config(tmp_path / "reward.yaml", {"scale": 2})
    path = write_config(
        tmp_path / "entry.yaml",
        {
            "reward": {"from": "./reward.yaml"},
            "evaluators": [{"from": "./reward.yaml", "scale": 3}],
        },
    )
    assert compose_config(path) == {
        "reward": {"scale": 2},
        "evaluators": [{"scale": 3}],
    }


def test_nested_values_remain_unresolved_until_runtime_defaults_exist(tmp_path):
    write_config(tmp_path / "data.yaml", {"dataset": "${dataset_dir}/train.json"})
    path = write_config(tmp_path / "entry.yaml", {"data": {"from": "./data.yaml"}})
    config = compose_config(path)
    assert config["data"]["dataset"] == "${dataset_dir}/train.json"
    config["dataset_dir"] = "/runtime/data"
    assert (
        resolve_interpolations(config)["data"]["dataset"] == "/runtime/data/train.json"
    )


@pytest.mark.parametrize(
    "reference", [None, [], 1, True, {"file": "a.yaml"}, ["a.yaml", 1], "", "${file}"]
)
def test_invalid_references_report_the_config_node(tmp_path, reference):
    path = write_config(tmp_path / "entry.yaml", {"model": {"from": reference}})
    with pytest.raises(
        ValueError, match=r"entry.yaml.model.from must be a literal YAML path"
    ):
        compose_config(path)


@pytest.mark.parametrize("content", ["[]", "null", "scalar"])
def test_reference_requires_a_mapping(tmp_path, content):
    (tmp_path / "invalid.yaml").write_text(content)
    path = write_config(tmp_path / "entry.yaml", {"model": {"from": "./invalid.yaml"}})
    with pytest.raises(ValueError, match="Config must be a mapping.*invalid.yaml"):
        compose_config(path)


def test_missing_reference_reports_the_missing_file(tmp_path):
    write_config(tmp_path / "base.yaml", {"from": "./missing.yaml"})
    path = write_config(tmp_path / "entry.yaml", {"from": "./base.yaml"})
    with pytest.raises(
        FileNotFoundError, match="Config reference not found:.*missing.yaml"
    ):
        compose_config(path)


@pytest.mark.parametrize("nested", [False, True])
def test_self_and_mutual_cycles_report_file_chain(tmp_path, nested):
    a = {"from": "./b.yaml"}
    b = {"from": "./a.yaml"}
    if nested:
        a, b = {"model": a}, {"connector": b}
    path = write_config(tmp_path / "a.yaml", a)
    write_config(tmp_path / "b.yaml", b)
    with pytest.raises(
        ValueError,
        match="Circular config reference:.*a.yaml.*b.yaml|Circular config reference:.*b.yaml.*a.yaml",
    ):
        compose_config(path)
    path = write_config(tmp_path / "self.yaml", {"from": "./self.yaml"})
    with pytest.raises(ValueError, match="Circular config reference:.*self.yaml"):
        compose_config(path)


def test_repeated_diamond_and_finite_nested_reuse_are_allowed(tmp_path):
    write_config(tmp_path / "base.yaml", {"value": 1})
    write_config(tmp_path / "left.yaml", {"from": "./base.yaml", "left": True})
    write_config(tmp_path / "right.yaml", {"from": "./base.yaml", "right": True})
    path = write_config(
        tmp_path / "entry.yaml",
        {
            "from": ["left.yaml", "right.yaml"],
            "one": {"from": "./base.yaml", "value": 2},
            "two": {"from": "./base.yaml"},
        },
    )
    assert compose_config(path) == {
        "value": 1,
        "left": True,
        "right": True,
        "one": {"value": 2},
        "two": {"value": 1},
    }
    path = write_config(
        tmp_path / "entry.yaml",
        {"from": "./base.yaml", "nested": {"from": "./base.yaml"}},
    )
    assert compose_config(path) == {"value": 1, "nested": {"value": 1}}


def test_composition_does_not_mutate_files_or_cache_across_calls(tmp_path):
    source = write_config(tmp_path / "model.yaml", {"value": 1})
    path = write_config(tmp_path / "entry.yaml", {"model": {"from": "./model.yaml"}})
    original = path.read_text()
    assert compose_config(path, ["model.value=2"])["model"]["value"] == 2
    assert compose_config(path)["model"]["value"] == 1
    write_config(source, {"value": 3})
    assert compose_config(path)["model"]["value"] == 3
    assert path.read_text() == original


def test_repo_model_and_connector_can_be_selected_independently():
    config = training_stages(
        ROOT / "configs/experiments/qwen2_instruct.yaml",
        [
            f"model.from={ROOT}/configs/models/phi.yaml",
            f"model.connector.from={ROOT}/configs/connectors/mof.yaml",
            "model.connector.clip_config.depth=4",
        ],
    )[0][1]
    assert config["model"]["language_model_name_or_path"] == "microsoft/phi-2"
    connector = config["model"]["connector"]
    assert connector["clip_config"]["depth"] == 4
    assert connector["clip_config"]["act"] == "gelu"
    assert connector["dinov2_config"]["depth"] == 2
    assert "depth" not in connector


def test_resolved_worker_round_trip_needs_no_composition(tmp_path):
    config = training_stages(ROOT / "configs/experiments/qwen2_instruct.yaml")[0][1]
    path = write_config(tmp_path / "worker.yaml", config)
    assert load_config(path) == config
    assert "from:" not in path.read_text()

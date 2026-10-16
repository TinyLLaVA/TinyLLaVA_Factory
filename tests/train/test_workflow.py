import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from omegaconf import OmegaConf

from tinyllava.eval.workflow import eval_plan
from tinyllava.train.workflow import (
    load_train_config,
    training_stages,
    train_plan,
    run as run_train,
)

ROOT = Path(__file__).parents[2]


@pytest.fixture
def single_stage_config(tmp_path):
    path = tmp_path / "pretrain.yaml"
    OmegaConf.save(
        OmegaConf.create(
            {
                "algorithm": "sft",
                "model": {"from": str(ROOT / "configs/models/qwen2_instruct.yaml")},
                "data": {"from": str(ROOT / "configs/data/train/llava_558k.yaml")},
                "training": {"from": str(ROOT / "configs/training/pretrain.yaml")},
            }
        ),
        path,
    )
    return path


def test_arbitrary_stage_names_and_previous_output_override(tmp_path):
    config = tmp_path / "experiment.yaml"
    config.write_text(f"""model:
  from: {ROOT}/configs/models/phi.yaml
stages:
  align:
    data:
      from: {ROOT}/configs/data/train/llava_558k.yaml
    training:
      from: {ROOT}/configs/training/pretrain.yaml
      saving:
        output_dir: output/custom-align
  instruction:
    training:
      from: {ROOT}/configs/training/finetune.yaml
    data:
      from: {ROOT}/configs/data/train/llava_665k.yaml
""")
    stages = training_stages(config)
    assert stages[0][1]["training"]["tuning"]["tune_type_llm"] == "frozen"
    assert stages[1][1]["training"]["tuning"]["tune_type_llm"] == "full"
    assert (
        stages[1][1]["model"]["pretrained_model_name_or_path"] == "output/custom-align"
    )
    assert (
        stages[1][1]["training"]["saving"]["output_dir"]
        == "output/experiment/instruction"
    )


def test_selectors_and_nested_overrides_resolve_before_interpolation(
    single_stage_config,
):
    config = load_train_config(
        single_stage_config,
        [
            "model.from=configs/models/phi.yaml",
            "model.model_max_length=4096",
            "data.from=configs/data/train/llava_665k.yaml",
            "dataset_dir=/local/data",
            "training.learning_rate=0.01",
            "experiment_name=custom",
            "stage=align",
        ],
    )
    assert config["model"]["language_model_name_or_path"] == "microsoft/phi-2"
    assert config["model"]["model_max_length"] == 4096
    assert (
        config["data"]["dataset_name_or_path"]
        == "/local/data/text_files/llava_v1_5_mix665k.json"
    )
    assert config["training"]["saving"]["output_dir"] == "output/custom/align"
    assert config["training"]["learning_rate"] == 0.01
    config = load_train_config(single_stage_config, ["model.model_max_length=4096"])
    assert config["model"]["model_max_length"] == 4096
    assert config["model"]["language_model_name_or_path"] == "Qwen/Qwen2-0.5B-Instruct"


def test_shared_model_override_keeps_stage_overrides():
    stages = training_stages(
        ROOT / "configs/experiments/qwen2_base_legacy.yaml",
        ["model.from=configs/models/phi.yaml"],
    )
    assert stages[0][1]["model"]["language_model_name_or_path"] == "microsoft/phi-2"
    assert stages[0][1]["model"]["chat_template_path"].endswith("pretrain_legacy.jinja")


def test_legacy_batch_size_and_selected_stage():
    plan = train_plan(
        ROOT / "configs/experiments/qwen2_base_legacy.yaml",
        ["launcher.devices=[0,1]"],
    )
    assert plan[0][2]["training"]["gradient_accumulation_steps"] == 8
    assert plan[1][2]["training"]["gradient_accumulation_steps"] == 16
    with patch("tinyllava.train.workflow._run_training_step") as run:
        run_train(
            ROOT / "configs/experiments/qwen2_base_legacy.yaml",
            ["launcher.devices=[0,1]"],
            steps=["finetune"],
        )
    assert run.call_count == 1
    assert (
        run.call_args.args[1]["model"]["pretrained_model_name_or_path"]
        == "output/qwen2_base_legacy/pretrain"
    )


def test_invalid_stage_dependency_and_output_collision(tmp_path):
    path = tmp_path / "bad.yaml"
    OmegaConf.save(
        OmegaConf.create(
            {
                "model": {"from": str(ROOT / "configs/models/phi.yaml")},
                "stages": {"one": {"from_stage": "later"}},
            }
        ),
        path,
    )
    with pytest.raises(ValueError, match="unknown or later"):
        training_stages(path)
    OmegaConf.save(
        OmegaConf.create(
            {
                "model": {"from": str(ROOT / "configs/models/phi.yaml")},
                "training": {"saving": {"output_dir": "same"}},
                "stages": {"one": {}, "two": {}},
            }
        ),
        path,
    )
    with pytest.raises(ValueError, match="distinct output"):
        training_stages(path)


@pytest.mark.parametrize("stages", [None, [], "pretrain"])
def test_invalid_stages_report_configuration_error(tmp_path, stages):
    path = tmp_path / "invalid.yaml"
    OmegaConf.save(OmegaConf.create({"stages": stages}), path)
    with pytest.raises(ValueError, match="stages must be a nonempty mapping"):
        training_stages(path)


def test_train_plan_rejects_invalid_global_batch(single_stage_config):
    with pytest.raises(ValueError, match="not divisible"):
        train_plan(
            single_stage_config,
            [
                "launcher.distributed=true",
                "launcher.devices=[0,1]",
                "training.global_batch_size=3",
                "training.per_device_train_batch_size=2",
            ],
        )


def test_eval_plan_validates_shard_devices():
    with pytest.raises(ValueError, match="distinct GPU IDs"):
        eval_plan(ROOT / "configs/eval/gqa.yaml", ["devices=[0,0]", "shard=true"])


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ([], "output/checkpoints/first"),
        (["stages.second.from_stage=first"], "output/checkpoints/first"),
        (
            ["stages.second.model.pretrained_model_name_or_path=external/model"],
            "external/model",
        ),
        (["stages.second.checkpoint=explicit/model"], "explicit/model"),
        (["stages.second.checkpoint=null"], None),
    ],
)
def test_stage_checkpoint_selection(tmp_path, overrides, expected):
    path = tmp_path / "checkpoints.yaml"
    path.write_text(
        f"model:\n  from: {ROOT}/configs/models/phi.yaml\nstages:\n  first: {{}}\n  second: {{}}\n"
    )
    model = training_stages(path, overrides)[1][1]["model"]
    assert model.get("pretrained_model_name_or_path") == expected
    assert ("language_model_name_or_path" in model) == (expected is None)


def test_standalone_from_stage_and_explicit_checkpoint_precedence(tmp_path):
    path = tmp_path / "stage.yaml"
    path.write_text(
        "experiment_name: example\nfrom_stage: align\n"
        "model:\n  pretrained_model_name_or_path: external/model\n"
    )
    model = load_train_config(path)["model"]
    assert model["pretrained_model_name_or_path"] == "output/example/align"
    model = load_train_config(path, ["checkpoint=explicit/model"])["model"]
    assert model["pretrained_model_name_or_path"] == "explicit/model"


def test_distributed_launch_uses_selected_devices_and_resolved_batch_size():
    def check_command(command, *, cwd, env, check):
        assert command[:3] == [sys.executable, "-m", "torch.distributed.run"]
        assert "--nproc_per_node=2" in command
        assert "--master_port=29999" in command
        assert command[command.index("--module") + 1] == "tinyllava.train.train"
        assert env["CUDA_VISIBLE_DEVICES"] == "2,5"
        assert check is True
        config = OmegaConf.load(command[-1])
        assert config.training.gradient_accumulation_steps == 4
        assert config.model.language_model_name_or_path

    with patch("tinyllava.runner.subprocess.run", side_effect=check_command) as launch:
        run_train(
            ROOT / "configs/experiments/qwen2_instruct.yaml",
            [
                "launcher.distributed=true",
                "launcher.devices=[2,5]",
                "launcher.master_port=29999",
                "stages.pretrain.training.global_batch_size=16",
                "stages.pretrain.training.per_device_train_batch_size=2",
            ],
            steps=["pretrain"],
        )
    launch.assert_called_once()
    assert not Path(launch.call_args.args[0][-1]).exists()


def write_config(tmp_path, config, name="experiment.yaml"):
    path = tmp_path / name
    OmegaConf.save(OmegaConf.create(config), path)
    return path


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

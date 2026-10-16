from pathlib import Path
from unittest.mock import patch

import pytest
from omegaconf import OmegaConf

from tinyllava.eval.workflow import load_eval_config
from tinyllava.train.workflow import load_train_config, training_stages
from tinyllava.eval.workflow import eval_plan, run as run_eval
from tinyllava.train.workflow import train_plan, run as run_train

ROOT = Path(__file__).parents[1]
EXPERIMENTS = sorted((ROOT / "configs/experiments").glob("*.yaml"))
EVALUATIONS = sorted((ROOT / "configs/eval").glob("*.yaml"))


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


@pytest.mark.parametrize("path", EXPERIMENTS, ids=lambda p: p.stem)
def test_all_experiments_are_resolved_and_linked(path):
    stages = training_stages(path)
    previous = None
    for name, config, _ in stages:
        assert (
            config["training"]["saving"]["output_dir"] == f"output/{path.stem}/{name}"
        )
        assert config["data"]["dataset_name_or_path"].startswith("datasets/")
        if previous:
            assert config["model"]["pretrained_model_name_or_path"] == previous
            assert "connector" not in config["model"]
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
    with patch("tinyllava.runner.subprocess") as processes:
        run_eval(path, ["model.model_name_or_path=output/example/sft"], dry_run=True)
    processes.run.assert_not_called()
    processes.Popen.assert_not_called()


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


@pytest.mark.parametrize("name", ["phi_lora", "phi_qlora"])
def test_adapter_experiments_inherit_model_preferences(name):
    base = training_stages(ROOT / "configs/experiments/phi.yaml")
    stages = training_stages(ROOT / f"configs/experiments/{name}.yaml")
    for (_, config, _), (_, expected, _) in zip(stages, base, strict=True):
        assert (
            config["model"]["model_max_length"] == expected["model"]["model_max_length"]
        )
        assert (
            config["model"]["attn_implementation"]
            == expected["model"]["attn_implementation"]
        )
    assert stages[-1][1]["training"]["tuning"]["peft"]["r"] == 128


@pytest.mark.parametrize(
    "preset", sorted((ROOT / "configs/training").glob("*.yaml")), ids=lambda p: p.stem
)
def test_training_presets_are_scoped_and_do_not_select_model_options(
    single_stage_config, preset
):
    config = load_train_config(
        single_stage_config,
        [
            f"training.from={preset}",
            "model.model_max_length=4096",
            "model.attn_implementation=sdpa",
        ],
    )
    assert config["model"]["model_max_length"] == 4096
    assert config["model"]["attn_implementation"] == "sdpa"
    assert "training" not in config["training"]
    assert config["training"]["learning_rate"] > 0


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


def test_explicit_checkpoint_and_fresh_stage(tmp_path):
    path = tmp_path / "test.yaml"
    OmegaConf.save(
        OmegaConf.create(
            {
                "model": {"from": str(ROOT / "configs/models/phi.yaml")},
                "stages": {
                    "one": {},
                    "two": {"checkpoint": "external/checkpoint"},
                    "fresh": {"checkpoint": None},
                },
            }
        ),
        path,
    )
    stages = training_stages(path)
    assert (
        stages[1][1]["model"]["pretrained_model_name_or_path"] == "external/checkpoint"
    )
    assert "pretrained_model_name_or_path" not in stages[2][1]["model"]
    assert "language_model_name_or_path" in stages[2][1]["model"]


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

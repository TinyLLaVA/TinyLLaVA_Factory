from pathlib import Path
from unittest.mock import patch

import pytest
from omegaconf import OmegaConf

from tinyllava.configuration import load_eval_config, load_train_config, training_stages
from tinyllava.run import eval_plan, run_plan, train_plan

ROOT = Path(__file__).parents[1]
EXPERIMENTS = sorted((ROOT / "configs/experiments").glob("*.yaml"))
EVALUATIONS = sorted((ROOT / "configs/eval").glob("*.yaml"))


@pytest.mark.parametrize("path", EXPERIMENTS, ids=lambda p: p.stem)
def test_all_experiments_are_resolved_and_linked(path):
    stages = training_stages(path)
    previous = None
    for name, config, _ in stages:
        assert config["training"]["output_dir"] == f"output/{path.stem}/{name}"
        assert config["data"]["dataset_name_or_path"].startswith("datasets/")
        if previous:
            assert config["model"]["pretrained_model_name_or_path"] == previous
            assert "connector_config" not in config["model"]
        previous = config["training"]["output_dir"]
    with patch("subprocess.run") as run, patch("subprocess.Popen") as popen:
        run_plan(train_plan(path), dry_run=True)
    run.assert_not_called()
    popen.assert_not_called()


@pytest.mark.parametrize("path", EVALUATIONS, ids=lambda p: p.stem)
def test_eval_output_is_outside_datasets(path):
    config = load_eval_config(path, ["model=output/example/sft"], metadata=True)
    assert (
        config["output"]["answers_file"]
        == f"output/example/sft/eval/{config['benchmark']}/answers.jsonl"
    )
    with patch("subprocess.run") as run, patch("subprocess.Popen") as popen:
        run_plan(eval_plan(path, ["model=output/example/sft"]), dry_run=True)
    run.assert_not_called()
    popen.assert_not_called()


def test_arbitrary_stage_names_and_previous_output_override(tmp_path):
    config = tmp_path / "experiment.yaml"
    config.write_text("""model: phi
stages:
  align:
    recipe: pretrain
    dataset: llava_558k
    training:
      output_dir: output/custom-align
  instruction:
    recipe: finetune
    dataset: llava_665k
""")
    stages = training_stages(config)
    assert stages[0][1]["training"]["tune_type_llm"] == "frozen"
    assert stages[1][1]["training"]["tune_type_llm"] == "full"
    assert (
        stages[1][1]["model"]["pretrained_model_name_or_path"] == "output/custom-align"
    )
    assert stages[1][1]["training"]["output_dir"] == "output/experiment/instruction"


def test_selectors_and_nested_overrides_resolve_before_interpolation():
    config = load_train_config(
        ROOT / "configs/train/pretrain.yaml",
        [
            "model=phi",
            "model.model_max_length=4096",
            "dataset=llava_665k",
            "dataset_dir=/local/data",
            "training.learning_rate=0.01",
            "name=custom",
            "stage=align",
        ],
    )
    assert config["model"]["language_model_name_or_path"] == "microsoft/phi-2"
    assert config["model"]["model_max_length"] == 4096
    assert (
        config["data"]["dataset_name_or_path"]
        == "/local/data/text_files/llava_v1_5_mix665k.json"
    )
    assert config["training"]["output_dir"] == "output/custom/align"
    assert config["training"]["learning_rate"] == 0.01
    config = load_train_config(
        ROOT / "configs/train/pretrain.yaml", ["model.model_max_length=4096"]
    )
    assert config["model"]["model_max_length"] == 4096
    assert config["model"]["language_model_name_or_path"] == "Qwen/Qwen2-0.5B-Instruct"


def test_shared_model_override_keeps_stage_overrides():
    stages = training_stages(
        ROOT / "configs/experiments/qwen2_base_legacy.yaml", ["model=phi"]
    )
    assert stages[0][1]["model"]["language_model_name_or_path"] == "microsoft/phi-2"
    assert stages[0][1]["model"]["chat_template_path"].endswith("pretrain_legacy.jinja")


def test_legacy_batch_size_and_selected_stage():
    plan = train_plan(
        ROOT / "configs/experiments/qwen2_base_legacy.yaml", ["launcher.devices=0,1"]
    )
    assert plan[0][2]["training"]["gradient_accumulation_steps"] == 8
    assert plan[1][2]["training"]["gradient_accumulation_steps"] == 16
    with patch("tinyllava.run.run_step") as run:
        run_plan(plan, steps=["finetune"])
    assert run.call_count == 1
    assert (
        run.call_args.args[1]["model"]["pretrained_model_name_or_path"]
        == "output/qwen2_base_legacy/pretrain"
    )


def test_recipe_model_overrides_apply():
    config = load_train_config(ROOT / "configs/train/lora_finetune.yaml")
    assert config["model"]["model_max_length"] == 3072
    assert config["model"]["attn_implementation"] == "flash_attention_2"
    assert config["peft"]["r"] == 128


def test_invalid_stage_dependency_and_output_collision(tmp_path):
    path = tmp_path / "bad.yaml"
    OmegaConf.save(
        OmegaConf.create({"model": "phi", "stages": {"one": {"from": "later"}}}), path
    )
    with pytest.raises(ValueError, match="unknown or later"):
        training_stages(path)
    OmegaConf.save(
        OmegaConf.create(
            {
                "model": "phi",
                "training": {"output_dir": "same"},
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
                "model": "phi",
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

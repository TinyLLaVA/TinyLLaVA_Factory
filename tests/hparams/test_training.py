from dataclasses import fields
from pathlib import Path

import pytest

from tinyllava.utils.config import load_config

from tinyllava.hparams.training import (
    CORE_FIELDS,
    GROUP_FIELDS,
    build_training_arguments as build_arguments,
    prepare_training_config as prepare_config,
)


def test_training_arguments_separate_tuning_from_trainer():
    args = build_arguments(
        {
            "training": {
                "saving": {"output_dir": "/tmp/test-sft"},
                "runtime": {"use_cpu": True},
                "logging": {"report_to": []},
                "tuning": {
                    "strategy": "lora",
                    "tune_type_llm": "lora",
                    "peft": {"r": 8},
                },
            }
        }
    )
    assert args.tuning.strategy == "lora"
    assert args.tuning.peft == {"r": 8}
    assert args.model.language_model_name_or_path is None


def test_batch_resolution_is_shared_and_does_not_mutate_input():
    config = {
        "training": {
            "global_batch_size": 32,
            "per_device_train_batch_size": 4,
            "gradient_accumulation_steps": 2,
        }
    }
    resolved = prepare_config(config, world_size=2)
    assert resolved["training"]["gradient_accumulation_steps"] == 4
    assert "global_batch_size" not in resolved["training"]
    assert config["training"]["gradient_accumulation_steps"] == 2
    assert config["training"]["global_batch_size"] == 32


@pytest.mark.parametrize("warmup_steps,expected_steps", [(0.1, 20), (7, 7)])
def test_nested_training_groups_reach_hf_arguments(warmup_steps, expected_steps):
    args = build_arguments(
        {
            "training": {
                "learning_rate": 0.003,
                "saving": {"output_dir": "/tmp/nested-test", "save_steps": 123},
                "logging": {"logging_steps": 7, "report_to": [], "run_name": "nested"},
                "runtime": {"use_cpu": True},
                "optimizer": {"weight_decay": 0.02},
                "scheduler": {"warmup_steps": warmup_steps},
                "dataloader": {
                    "dataloader_num_workers": 2,
                    "train_sampling_strategy": "group_by_length",
                },
                "tuning": {"tune_type_llm": "full"},
            }
        }
    )
    assert args.training.save_steps == 123
    assert args.training.logging_steps == 7
    assert args.training.weight_decay == 0.02
    assert args.training.warmup_steps == warmup_steps
    assert args.training.get_warmup_steps(200) == expected_steps
    assert args.training.dataloader_num_workers == 2
    assert args.training.train_sampling_strategy == "group_by_length"
    assert args.training.learning_rate == 0.003
    assert args.tuning.tune_type_llm == "full"


def test_training_fields_match_trainer_arguments():
    from tinyllava.hparams.arguments import TrainingArguments

    trainer_fields = {field.name for field in fields(TrainingArguments)}
    config_fields = (CORE_FIELDS - {"global_batch_size"}).union(*GROUP_FIELDS.values())
    assert config_fields <= trainer_fields


@pytest.mark.parametrize(
    "preset",
    sorted((Path(__file__).parents[2] / "configs/training").glob("*.yaml")),
    ids=lambda path: path.stem,
)
def test_training_presets_build_with_fractional_warmup(preset, tmp_path):
    training = load_config(preset)
    training["runtime"].update(use_cpu=True, precision="fp32")
    training["logging"]["report_to"] = []
    training["saving"]["output_dir"] = str(tmp_path)
    args = build_arguments({"training": training})
    assert args.training.warmup_steps == 0.03
    assert args.training.get_warmup_steps(1000) == 30


@pytest.mark.parametrize(
    "training",
    [
        {"save_steps": 1},
        {"saving": {"learning_rate": 0.1}},
        {"logging": None},
        {"tuning": "lora"},
    ],
)
def test_training_group_ownership_is_validated(training):
    with pytest.raises(ValueError):
        build_arguments({"training": training})


def test_precision_is_resolved_before_deepspeed_auto_values(tmp_path):
    deepspeed_path = tmp_path / "zero2.json"
    deepspeed_path.write_text(
        '{"bf16":{"enabled":"auto"},"fp16":{"enabled":"auto"},'
        '"zero_optimization":{"stage":2}}',
        encoding="utf-8",
    )

    arguments = build_arguments(
        {
            "training": {
                "saving": {"output_dir": str(tmp_path / "output")},
                "runtime": {
                    "precision": "bf16",
                    "deepspeed": str(deepspeed_path),
                    "use_cpu": True,
                },
            }
        }
    )

    config = arguments.training.hf_deepspeed_config.config
    assert config["bf16"]["enabled"] is True
    assert config["fp16"]["enabled"] is False


def test_parse_data_source_options_from_yaml(tmp_path):
    config_path = tmp_path / "sources.yaml"
    config_path.write_text(
        "data:\n"
        "  dataset_name_or_path: parquet\n"
        "  data_files:\n"
        "    train: [part-1.parquet, part-2.parquet]\n"
        "  split: train[:10%]\n"
        "  cache_dir: /tmp/data-cache\n"
        "training:\n"
        "  saving:\n    output_dir: /tmp/train-output\n"
        "  runtime:\n    use_cpu: true\n"
        "  logging:\n    report_to: []\n"
    )
    arguments = build_arguments(load_config(config_path))
    assert arguments.data.dataset_name_or_path == "parquet"
    assert arguments.data.data_files == {"train": ["part-1.parquet", "part-2.parquet"]}
    assert arguments.data.split == "train[:10%]"
    assert arguments.data.cache_dir == "/tmp/data-cache"

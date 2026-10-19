from pathlib import Path

import pytest

from tinyllava.hparams.training import (
    build_training_arguments,
    prepare_training_config,
    validate_training_config,
)
from tinyllava.utils.config import load_config


def test_training_arguments_separate_tuning_from_trainer():
    args = build_training_arguments(
        {
            "training": {
                "output_dir": "/tmp/test-sft",
                "use_cpu": True,
                "report_to": [],
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
    resolved = prepare_training_config(config, world_size=2)
    assert resolved["training"]["gradient_accumulation_steps"] == 4
    assert "global_batch_size" not in resolved["training"]
    assert config["training"]["gradient_accumulation_steps"] == 2
    assert config["training"]["global_batch_size"] == 32


@pytest.mark.parametrize("warmup_steps,expected_steps", [(0.1, 20), (7, 7)])
def test_flat_fields_and_native_mappings_reach_hf_arguments(
    warmup_steps, expected_steps
):
    args = build_training_arguments(
        {
            "training": {
                "learning_rate": 0.003,
                "output_dir": "/tmp/flat-test",
                "save_steps": 123,
                "logging_steps": 7,
                "report_to": [],
                "run_name": "flat",
                "use_cpu": True,
                "weight_decay": 0.02,
                "warmup_steps": warmup_steps,
                "lr_scheduler_kwargs": {"num_cycles": 0.5},
                "dataloader_num_workers": 2,
                "train_sampling_strategy": "group_by_length",
                "skip_memory_metrics": False,
                "tuning": {"tune_type_llm": "full"},
            }
        }
    )
    assert args.training.save_steps == 123
    assert args.training.logging_steps == 7
    assert args.training.weight_decay == 0.02
    assert args.training.warmup_steps == warmup_steps
    assert args.training.get_warmup_steps(200) == expected_steps
    assert args.training.lr_scheduler_kwargs == {"num_cycles": 0.5}
    assert args.training.dataloader_num_workers == 2
    assert args.training.train_sampling_strategy == "group_by_length"
    assert args.training.skip_memory_metrics is False
    assert args.training.learning_rate == 0.003
    assert args.tuning.tune_type_llm == "full"


@pytest.mark.parametrize(
    "preset",
    sorted((Path(__file__).parents[2] / "configs/training").glob("*.yaml")),
    ids=lambda path: path.stem,
)
def test_training_presets_build_with_fractional_warmup(preset, tmp_path):
    training = load_config(preset)
    training.update(
        use_cpu=True, precision="fp32", report_to=[], output_dir=str(tmp_path)
    )
    args = build_training_arguments({"training": training})
    assert args.training.warmup_steps == 0.03
    assert args.training.get_warmup_steps(1000) == 30


@pytest.mark.parametrize("tuning", ["lora", {"peft": None}])
def test_invalid_tuning_is_rejected_before_launch(tuning):
    with pytest.raises(ValueError, match="mapping"):
        validate_training_config({"training": {"tuning": tuning}})


@pytest.mark.parametrize("field", ["misspelled", "bf16", "fp16"])
def test_trainer_rejects_unknown_or_derived_constructor_fields(field):
    with pytest.raises(TypeError, match=field):
        build_training_arguments({"training": {field: True}})


@pytest.mark.parametrize("section", ["model", "data"])
def test_unknown_parameters_fail_during_workflow_validation(section):
    with pytest.raises(TypeError, match="misspelled"):
        validate_training_config({section: {"misspelled": True}})


def test_precision_is_resolved_before_deepspeed_auto_values(tmp_path):
    deepspeed_path = tmp_path / "zero2.json"
    deepspeed_path.write_text(
        '{"bf16":{"enabled":"auto"},"fp16":{"enabled":"auto"},"zero_optimization":{"stage":2}}',
        encoding="utf-8",
    )
    arguments = build_training_arguments(
        {
            "training": {
                "output_dir": str(tmp_path / "output"),
                "precision": "bf16",
                "deepspeed": str(deepspeed_path),
                "use_cpu": True,
            }
        }
    )
    config = arguments.training.hf_deepspeed_config.config
    assert config["bf16"]["enabled"] is True
    assert config["fp16"]["enabled"] is False


def test_parse_data_source_options_from_yaml(tmp_path):
    config_path = tmp_path / "sources.yaml"
    config_path.write_text(
        "data:\n  dataset_name_or_path: parquet\n  data_files:\n    train: [part-1.parquet, part-2.parquet]\n"
        "  split: train[:10%]\n  cache_dir: /tmp/data-cache\n"
        "training:\n  output_dir: /tmp/train-output\n  use_cpu: true\n  report_to: []\n"
    )
    arguments = build_training_arguments(load_config(config_path))
    assert arguments.data.dataset_name_or_path == "parquet"
    assert arguments.data.data_files == {"train": ["part-1.parquet", "part-2.parquet"]}
    assert arguments.data.split == "train[:10%]"
    assert arguments.data.cache_dir == "/tmp/data-cache"

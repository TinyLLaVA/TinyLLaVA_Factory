"""Shared Trainer configuration: groups, validation, arguments, and batch sizing."""

from copy import deepcopy
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING

from .tuning import TuningArguments

if TYPE_CHECKING:
    from .arguments import ModelArguments, DataArguments, TrainingArguments


@dataclass
class TrainingInputs:
    model: "ModelArguments"
    data: "DataArguments"
    training: "TrainingArguments"
    tuning: TuningArguments


SECTIONS = ("model", "data", "training")


CORE_FIELDS = {
    "learning_rate",
    "num_train_epochs",
    "max_steps",
    "per_device_train_batch_size",
    "gradient_accumulation_steps",
    "global_batch_size",
}
GROUP_FIELDS = {
    "saving": {
        "output_dir",
        "overwrite_output_dir",
        "save_strategy",
        "save_steps",
        "save_total_limit",
        "save_safetensors",
        "save_on_each_node",
        "save_only_model",
        "resume_from_checkpoint",
        "restore_callback_states_from_checkpoint",
        "push_to_hub",
        "hub_model_id",
        "hub_strategy",
        "hub_token",
        "hub_private_repo",
        "hub_always_push",
    },
    "logging": {
        "run_name",
        "report_to",
        "logging_dir",
        "logging_strategy",
        "logging_steps",
        "logging_first_step",
        "logging_nan_inf_filter",
        "log_level",
        "log_level_replica",
        "log_on_each_node",
        "disable_tqdm",
        "include_tokens_per_second",
        "include_num_input_tokens_seen",
    },
    "optimizer": {
        "optim",
        "optim_args",
        "optim_target_modules",
        "weight_decay",
        "adam_beta1",
        "adam_beta2",
        "adam_epsilon",
        "max_grad_norm",
    },
    "scheduler": {
        "lr_scheduler_type",
        "lr_scheduler_kwargs",
        "warmup_ratio",
        "warmup_steps",
    },
    "runtime": {
        "precision",
        "bf16",
        "fp16",
        "bf16_full_eval",
        "fp16_full_eval",
        "tf32",
        "half_precision_backend",
        "use_cpu",
        "use_mps_device",
        "seed",
        "data_seed",
        "full_determinism",
        "deepspeed",
        "local_rank",
        "ddp_backend",
        "ddp_timeout",
        "ddp_find_unused_parameters",
        "ddp_bucket_cap_mb",
        "ddp_broadcast_buffers",
        "fsdp",
        "fsdp_config",
        "fsdp_transformer_layer_cls_to_wrap",
        "gradient_checkpointing",
        "gradient_checkpointing_kwargs",
        "torch_compile",
        "torch_compile_backend",
        "torch_compile_mode",
        "torch_empty_cache_steps",
        "accelerator_config",
    },
    "dataloader": {
        "dataloader_num_workers",
        "dataloader_pin_memory",
        "dataloader_persistent_workers",
        "dataloader_prefetch_factor",
        "dataloader_drop_last",
        "remove_unused_columns",
        "group_by_length",
        "length_column_name",
        "group_by_modality_length",
        "ignore_data_skip",
    },
    "evaluation": {
        "eval_strategy",
        "eval_steps",
        "eval_delay",
        "eval_accumulation_steps",
        "per_device_eval_batch_size",
        "prediction_loss_only",
        "load_best_model_at_end",
        "metric_for_best_model",
        "greater_is_better",
        "do_eval",
        "eval_on_start",
        "eval_do_concat_batches",
        "batch_eval_metrics",
        "include_for_metrics",
        "label_names",
    },
}


def flatten_training(training: dict) -> dict:
    """Validate ownership before flattening; tuning remains a separate object."""
    if not isinstance(training, dict):
        raise ValueError("training must be a mapping")
    unknown = set(training) - CORE_FIELDS - set(GROUP_FIELDS) - {"tuning"}
    if unknown:
        destinations = {
            field: group for group, fields in GROUP_FIELDS.items() for field in fields
        }
        hints = [
            f"training.{destinations[field]}.{field}"
            for field in sorted(unknown)
            if field in destinations
        ]
        raise ValueError(
            f"Unknown or misplaced training fields: {sorted(unknown)}. Use: {hints}"
        )
    result = {key: value for key, value in training.items() if key in CORE_FIELDS}
    for group, allowed in GROUP_FIELDS.items():
        values = training.get(group, {})
        if not isinstance(values, dict):
            raise ValueError(f"training.{group} must be a mapping")
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(f"Unknown training.{group} fields: {sorted(unknown)}")
        result.update(values)
    return result


def validate_training_config(config: dict, *, sections=SECTIONS) -> None:
    unknown = set(config) - set(sections) - {"algorithm"}
    if unknown:
        raise ValueError(f"Unknown train config section(s): {sorted(unknown)}")
    for section in sections:
        if section in config and not isinstance(config[section], dict):
            raise ValueError(f"{section} must be a mapping")
    model = config.get("model", {})
    if "connector" in model:
        connector = model["connector"]
        if not isinstance(connector, dict) or "model_type" not in connector:
            raise ValueError(
                "model.connector must be a resolved mapping with model_type"
            )
    training = config.get("training", {})
    flatten_training(training)
    tuning = training.get("tuning", {})
    if not isinstance(tuning, dict):
        raise ValueError("training.tuning must be a mapping")
    unknown = set(tuning) - {field.name for field in fields(TuningArguments)}
    if unknown:
        raise ValueError(f"Unknown tuning fields: {sorted(unknown)}")
    if "peft" in tuning and not isinstance(tuning["peft"], dict):
        raise ValueError("tuning.peft must be a mapping")


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def prepare_training_config(config: dict, *, world_size: int) -> dict:
    """Resolve effective batch size for gradient-accumulating Trainer loops using the launcher's worker count."""
    validate_training_config(config)
    config = deepcopy(config)
    training = config.setdefault("training", {})
    if "global_batch_size" in training:
        batch = _positive_int(training.pop("global_batch_size"), "global_batch_size")
        micro = _positive_int(
            training.get("per_device_train_batch_size", 8),
            "per_device_train_batch_size",
        )
        count = _positive_int(world_size, "world_size")
        if batch % (count * micro):
            raise ValueError(
                f"Global batch {batch} is not divisible by {count} workers x micro batch {micro}"
            )
        training["gradient_accumulation_steps"] = batch // (count * micro)
    return config


def build_training_arguments(config: dict) -> TrainingInputs:
    """Construct shared Trainer inputs in the worker, after workflow preparation."""
    from .arguments import (
        ModelArguments,
        DataArguments,
        TrainingArguments,
    )

    validate_training_config(config)
    if "global_batch_size" in config.get("training", {}):
        raise ValueError(
            "global_batch_size must be resolved by the training workflow before worker launch"
        )

    return TrainingInputs(
        model=ModelArguments(**config.get("model", {})),
        data=DataArguments(**config.get("data", {})),
        training=TrainingArguments(**flatten_training(config.get("training", {}))),
        tuning=TuningArguments(**config.get("training", {}).get("tuning", {})),
    )

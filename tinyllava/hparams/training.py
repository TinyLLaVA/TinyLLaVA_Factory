"""Build training arguments from a resolved stage configuration.

`model` and `data` construct project parameter objects. Fields under `training`
map directly to HF Trainer settings, with `tuning` collected separately.
The workflow resolves `global_batch_size` using the number of launched workers.
HF validates its fields when the worker constructs TrainingArguments; workflow
checks cover project structure and parameters without importing HF.
"""

from copy import deepcopy

from tinyllava.utils.config import positive_int, validate_fields

from .arguments import DataArguments, ModelArguments, TrainingArgBundle, TuningArguments

SECTIONS = ("model", "data", "training")


def validate_training_config(config: dict) -> None:
    """Check resolved training sections without importing HF or loading a model."""
    validate_fields(config, SECTIONS, {"algorithm"}, "train")
    ModelArguments(**config.get("model", {}))
    DataArguments(**config.get("data", {}))
    training = config.get("training", {})
    tuning = training.get("tuning", {})
    if not isinstance(tuning, dict):
        raise ValueError("training.tuning must be a mapping")
    TuningArguments(**tuning)


def prepare_training_config(config: dict, *, world_size: int) -> dict:
    """Resolve effective batch size for gradient-accumulating Trainer loops using the launcher's worker count."""
    validate_training_config(config)
    config = deepcopy(config)
    training = config.setdefault("training", {})
    if "global_batch_size" in training:
        batch = positive_int(training.pop("global_batch_size"), "global_batch_size")
        micro = positive_int(
            training.get("per_device_train_batch_size", 8),
            "per_device_train_batch_size",
        )
        count = positive_int(world_size, "world_size")
        if batch % (count * micro):
            raise ValueError(
                f"Global batch {batch} is not divisible by {count} workers x micro batch {micro}"
            )
        training["gradient_accumulation_steps"] = batch // (count * micro)
    return config


def build_training_arguments(config: dict) -> TrainingArgBundle:
    """Construct shared Trainer inputs in the worker, after workflow preparation."""
    from .hf import TrainingArguments

    validate_training_config(config)
    if "global_batch_size" in config.get("training", {}):
        raise ValueError(
            "global_batch_size must be resolved by the training workflow before worker launch"
        )

    training = dict(config.get("training", {}))
    tuning = TuningArguments(**training.pop("tuning", {}))
    return TrainingArgBundle(
        model=ModelArguments(**config.get("model", {})),
        data=DataArguments(**config.get("data", {})),
        training=TrainingArguments(**training),
        tuning=tuning,
    )

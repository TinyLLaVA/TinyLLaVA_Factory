"""Parameter update policy schema, independent of algorithms and trainers."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TuningArguments:
    """Parameter update policies, independent of the training algorithm."""

    strategy: str = field(
        default="common",
        metadata={
            "help": "Training strategy plugin name, for example 'common', 'lora', or 'lora_int8'."
        },
    )
    tune_type_llm: str = field(
        default="frozen",
        metadata={
            "help": (
                "LLM tuning policy: 'frozen', 'full', or 'lora'. "
                "tuning.strategy selects the concrete plugin."
            )
        },
    )
    tune_type_vision_tower: str = field(
        default="frozen",
        metadata={
            "help": (
                "Vision tower tuning strategy: 'frozen' (no update), 'full' "
                "(fine-tune all layers), 'partially-tune' (tune last N layers)."
            )
        },
    )
    tune_vision_tower_from_layer: int | None = field(
        default=10,
        metadata={
            "help": (
                "If tune_type_vision_tower='partially-tune', fine-tune from this "
                "layer onwards. Higher number = deeper layers."
            )
        },
    )
    tune_type_connector: str = field(
        default="full",
        metadata={
            "help": "Connector/projector tuning strategy: 'frozen', 'full', or 'lora'."
        },
    )
    peft: dict[str, Any] = field(
        default_factory=dict,
        metadata={
            "help": (
                "PEFT config domain passed to peft.LoraConfig for LoRA strategies. "
                "Use upstream field names such as r, lora_alpha, lora_dropout, and bias."
            )
        },
    )

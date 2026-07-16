from dataclasses import dataclass, field
from typing import Any, Literal

import transformers

from tinyllava.utils.precision import resolve_precision_flags


@dataclass
class ModelArguments:
    """Component paths and model assembly options."""

    pretrained_model_name_or_path: str | None = field(
        default=None,
        metadata={
            "help": (
                "Complete TinyLLaVA checkpoint path. When set, load the composite "
                "model instead of assembling language and vision components."
            )
        },
    )
    language_model_name_or_path: str | None = field(
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        metadata={"help": "Language model name or path."},
    )
    tokenizer_name_or_path: str | None = field(
        default=None,
        metadata={
            "help": "Tokenizer name or path. Defaults to language_model_name_or_path."
        },
    )
    vision_model_name_or_path: str | None = field(
        default=None,
        metadata={"help": "Vision tower name or path."},
    )
    connector_config: str | None = field(
        default=None,
        metadata={"help": "YAML config for the multimodal connector."},
    )
    cache_dir: str | None = field(
        default=None,
        metadata={"help": "Hugging Face cache directory."},
    )
    attn_implementation: str | None = field(
        default=None,
        metadata={"help": "Attention implementation passed to the language model."},
    )
    model_max_length: int = field(
        default=512,
        metadata={"help": "Tokenizer model_max_length."},
    )
    tokenizer_use_fast: bool = field(
        default=False,
        metadata={"help": "Whether to load a fast tokenizer."},
    )
    tokenizer_padding_side: str = field(
        default="right",
        metadata={"help": "Tokenizer padding side."},
    )
    vision_feature_layer: int | list[int] = field(
        default=-2,
        metadata={"help": "Vision layer(s) used by the connector."},
    )
    vision_feature_select_strategy: Literal["default", "full"] = field(
        default="default",
        metadata={"help": "LLaVA feature selection strategy."},
    )


@dataclass
class DataArguments:
    """Training dataset inputs."""

    data_path: str | None = field(
        default=None,
        metadata={"help": "Training data file or dataset path."},
    )
    image_folder: str | None = field(
        default=None,
        metadata={"help": "Root directory for relative image paths."},
    )
    dataset_adapter: str = field(
        default="auto",
        metadata={
            "help": (
                "Training sample adapter name. 'auto' detects known legacy "
                "formats; use 'llava_legacy' for LLaVA conversation JSON."
            )
        },
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    """HF TrainingArguments plus TinyLLaVA strategy options."""

    precision: Literal["auto", "fp32", "fp16", "bf16"] = field(
        default="auto",
        metadata={
            "help": (
                "TinyLLaVA precision policy. 'auto' selects BF16 > FP16 > FP32 "
                "from hardware capability; explicit values force the corresponding dtype."
            )
        },
    )
    training_strategy: str = field(
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
                "The training_strategy selects the concrete plugin."
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
        metadata={"help": "Connector/projector tuning strategy: 'frozen', 'full', or 'lora'."},
    )
    peft_config: dict[str, Any] = field(
        default_factory=dict,
        metadata={
            "help": (
                "PEFT config domain passed to peft.LoraConfig for LoRA strategies. "
                "Use upstream field names such as r, lora_alpha, lora_dropout, and bias."
            )
        },
    )

    def __post_init__(self) -> None:
        # HF resolves DeepSpeed values set to "auto" in its own __post_init__.
        # Resolve TinyLLaVA's policy first so both configurations use one dtype.
        self.bf16, self.fp16 = resolve_precision_flags(self.precision)
        super().__post_init__()

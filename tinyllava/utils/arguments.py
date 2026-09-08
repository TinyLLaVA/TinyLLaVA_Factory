from dataclasses import dataclass, field
from typing import Any, Literal

import transformers

from tinyllava.utils.precision import resolve_precision_flags


@dataclass
class ModelArguments:
    """Select component sources and multimodal preprocessing for training.

    Attributes:
        pretrained_model_name_or_path: Complete TinyLLaVA checkpoint. When set,
            weights and component configs are loaded from this source.
        language_model_name_or_path: Language-model source for component assembly.
        tokenizer_name_or_path: Tokenizer override; otherwise use the composite
            checkpoint or, during assembly, the language-model source.
        vision_model_name_or_path: Vision-tower source for component assembly.
        connector_config: Connector YAML path used during component assembly.
        cache_dir: Hugging Face download cache directory.
        attn_implementation: Attention backend passed to the language model.
        model_max_length: Tokenizer length limit used by the SFT collator.
        tokenizer_use_fast: Request a fast tokenizer for token-to-character alignment.
        tokenizer_padding_side: Padding side configured on the tokenizer.
        chat_template: Inline Jinja template override, exclusive with `chat_template_path`.
        chat_template_path: UTF-8 Jinja template file overriding the tokenizer template.
        vision_feature_layer: Vision hidden-state layer indices used by the connector.
        vision_feature_select_strategy: `default` drops the first vision token;
            `full` retains all tokens. Patch-only backbones require `full`.
    """

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
    chat_template: str | None = field(
        default=None,
        metadata={"help": "Inline Hugging Face/Jinja chat template override."},
    )
    chat_template_path: str | None = field(
        default=None,
        metadata={"help": "Path to a Hugging Face/Jinja chat template override."},
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
    """Locate training annotations and resolve their sample format.

    Attributes:
        data_path: Local JSON or JSONL training annotations.
        image_folder: Root directory for relative image paths.
        dataset_adapter: Registered adapter name. `auto` detects known formats;
            `llava_legacy` selects LLaVA conversation records explicitly.
    """

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
class EvalModelArguments:
    """Select the checkpoint and prediction-export identifier.

    Attributes:
        model_name_or_path: Local TinyLLaVA checkpoint or Hugging Face repository ID.
        model_id: Optional model identifier written to benchmark answer files.
    """

    model_name_or_path: str = field(
        default="output/tinyllava",
        metadata={"help": "TinyLLaVA checkpoint path or Hugging Face repo id."},
    )
    model_id: str | None = field(
        default=None,
        metadata={"help": "Optional model id written to benchmark answer files."},
    )


@dataclass
class EvalDataArguments:
    """Locate benchmark samples and select their loader.

    Attributes:
        adapter: Registered evaluation loader name.
        question_file: Benchmark annotation file in the selected loader's format.
        image_folder: Root directory for relative benchmark image paths.
        single_pred_prompt: Append the single-choice instruction for ScienceQA.
    """

    adapter: str = field(
        default="vqa",
        metadata={"help": "Evaluation dataset loader adapter name."},
    )
    question_file: str = field(
        default="tables/question.jsonl",
        metadata={"help": "Benchmark question/annotation file."},
    )
    image_folder: str = field(
        default="",
        metadata={"help": "Root directory for benchmark images."},
    )
    single_pred_prompt: bool = field(
        default=False,
        metadata={"help": "Whether to append single-choice prompting for ScienceQA."},
    )


@dataclass
class EvalGenerationArguments:
    """Configure answer decoding and the evaluation chat template.

    Attributes:
        temperature: Sampling temperature; nonpositive values disable sampling.
        top_p: Nucleus-sampling probability cutoff.
        num_beams: Number of decoding beams; must be positive.
        max_new_tokens: Maximum generated tokens per answer; must be positive.
        chat_template: Inline Jinja override, exclusive with `chat_template_path`.
        chat_template_path: UTF-8 Jinja file overriding the checkpoint template.
    """

    temperature: float = 0.2
    top_p: float | None = None
    num_beams: int = 1
    max_new_tokens: int = 128
    chat_template: str | None = None
    chat_template_path: str | None = None


@dataclass
class EvalRuntimeArguments:
    """Control evaluation placement, batching, and dataset sharding.

    Attributes:
        device: Torch device string, or `None` for the entry point's default.
        batch_size: Conversations per generation batch; must be positive.
        num_chunks: Total dataset shards; must be positive.
        chunk_idx: Zero-based shard index in `[0, num_chunks)`.
    """

    device: str | None = None
    batch_size: int = 1
    num_chunks: int = 1
    chunk_idx: int = 0


@dataclass
class EvalOutputArguments:
    """Choose where generic benchmark predictions are written.

    Attributes:
        answers_file: Output JSONL path; use a separate file for each evaluation shard.
    """

    answers_file: str = "answer.jsonl"


@dataclass
class EvalArguments:
    """Group the five sections of an evaluation configuration.

    Attributes:
        model: Checkpoint and model identifier.
        data: Benchmark files and adapter selection.
        generation: Decoding parameters and chat-template overrides.
        runtime: Device, batch size, and sharding settings.
        output: Prediction output path.
    """

    model: EvalModelArguments
    data: EvalDataArguments
    generation: EvalGenerationArguments
    runtime: EvalRuntimeArguments
    output: EvalOutputArguments


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    """Extend Hugging Face TrainingArguments with component-tuning policies.

    Attributes:
        precision: `auto` selects BF16, FP16, then FP32 according to hardware support.
            An explicit value sets the corresponding Trainer precision flags.
        training_strategy: Registered strategy, such as `common`, `lora`, or `lora_int8`.
        tune_type_llm: Language-model and LM-head policy supported by the strategy.
        tune_type_vision_tower: Vision policy: `frozen`, `full`, or `partially-tune`.
        tune_vision_tower_from_layer: First trainable encoder layer for partial tuning.
        tune_type_connector: Connector policy supported by the selected strategy.
        group_by_modality_length: Group image and text-only samples by signed length.
        peft_config: Upstream `LoraConfig` fields used by LoRA strategies.
    """

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
    group_by_modality_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Group multimodal and text-only samples by approximate length, "
                "matching the legacy TinyLLaVA fine-tuning sampler."
            )
        },
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

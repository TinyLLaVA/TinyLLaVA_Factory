"""Project parameter objects for training and benchmark generation.

Training uses ModelArguments, DataArguments, and TuningArguments, collected in
TrainingArgBundle alongside the HF Trainer settings. Evaluation uses EvalArguments.

This module can be imported by workflows without Torch or Transformers. The
TrainingArguments subclass in hparams.hf is loaded when a training worker builds
its Trainer settings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from tinyllava.utils.config import positive_int

if TYPE_CHECKING:
    from .hf import TrainingArguments


@dataclass
class ModelArguments:
    """Select component sources and multimodal preprocessing for training.

    Configuration location: `model` in the resolved training stage.
    """

    # Sources and loading.
    pretrained_model_name_or_path: str | None = None
    """Complete TinyLLaVA or PEFT checkpoint. When set,
    weights and component configs are loaded from this source.
    """
    language_model_name_or_path: str | None = None
    """Language-model source for component assembly."""
    vision_model_name_or_path: str | None = None
    """Vision-tower source for component assembly."""
    connector: dict[str, Any] | None = None
    """Resolved connector configuration mapping used during assembly."""
    cache_dir: str | None = None
    """Hugging Face download cache directory."""
    attn_implementation: str | None = None
    """Attention backend passed to the language model."""

    # Tokenization.
    tokenizer_name_or_path: str | None = None
    """Tokenizer override; otherwise use the composite
    checkpoint or, during assembly, the language-model source.
    """
    model_max_length: int = 512
    """Tokenizer length limit used by the SFT collator."""
    tokenizer_use_fast: bool = False
    """Request a fast tokenizer for token-to-character alignment."""
    tokenizer_padding_side: str = "right"
    """Padding side configured on the tokenizer."""

    # Conversation formatting.
    chat_template: str | None = None
    """Inline Jinja template override, exclusive with `chat_template_path`."""
    chat_template_path: str | None = None
    """UTF-8 Jinja template file overriding the tokenizer template."""

    # Vision features.
    vision_feature_layer: int | list[int] | None = None
    """Vision hidden-state layer indices used by the connector.
    `None` preserves the checkpoint setting, or the model default for assembly.
    """
    vision_feature_select_strategy: Literal["default", "full"] | None = None
    """`default` drops the first vision token;
    `full` retains all tokens. When omitted, inherit the checkpoint setting
    or infer it from the vision backbone when assembling a new model.
    """

    def __post_init__(self):
        if self.connector is not None and (
            not isinstance(self.connector, dict) or "model_type" not in self.connector
        ):
            raise ValueError(
                "model.connector must be a resolved mapping with model_type"
            )


@dataclass
class DataArguments:
    """Locate training annotations and resolve their sample format.

    Configuration location: `data` in the training stage.
    """

    # Source and selection.
    dataset_name_or_path: str | None = None
    """Hub repository, local dataset directory/file, or HF builder."""
    dataset_config_name: str | None = None
    """Dataset subset/configuration name (HF `name`)."""
    revision: str | None = None
    """Hub branch, tag, or commit."""
    split: str = "train"
    """Training split, including HF slice expressions."""

    # Files and cache.
    data_files: str | list[str] | dict[str, str | list[str]] | None = None
    """File paths, glob patterns, or a mapping from splits to files."""
    data_dir: str | None = None
    """Data subdirectory passed to the HF loader."""
    cache_dir: str | None = None
    """Directory for downloaded data and Arrow caches."""

    # Sample interpretation.
    image_folder: str | None = None
    """Root directory for relative image paths."""
    dataset_adapter: str = "auto"
    """Registered adapter name. `auto` detects known formats;
    `llava_legacy` selects LLaVA conversation records explicitly.
    """


@dataclass
class TuningArguments:
    """Parameter update policies, independent of the training algorithm.

    Configuration location: `training.tuning` in the training stage.
    """

    # Strategy.
    strategy: str = "common"
    """Training strategy plugin, such as `common`, `lora`, or `lora_int8`."""

    # Component updates.
    tune_type_llm: str = "frozen"
    """Language-model update policy: `frozen`, `full`, or `lora`."""
    tune_type_vision_tower: str = "frozen"
    """Vision update policy, also supporting `partially-tune`."""
    tune_vision_tower_from_layer: int | None = 10
    """First trainable layer for partial vision tuning."""
    tune_type_connector: str = "full"
    """Connector update policy: `frozen`, `full`, or `lora`."""

    # Adapter options.
    peft: dict[str, Any] = field(default_factory=dict)
    """Options passed to `peft.LoraConfig`, using upstream field names."""

    def __post_init__(self):
        if not isinstance(self.peft, dict):
            raise ValueError("training.tuning.peft must be a mapping")


@dataclass
class TrainingArgBundle:
    """Parameter objects collected for a Trainer-based training algorithm.

    Built by `hparams.training.build_training_arguments` from resolved YAML.
    Tuning is separated from Trainer settings because update strategies consume
    it independently.
    """

    model: ModelArguments
    """Component and preprocessing settings from `model`."""
    data: DataArguments
    """Training dataset settings from `data`."""
    training: TrainingArguments
    """HF Trainer settings from `training`, excluding `tuning`."""
    tuning: TuningArguments
    """Component update policy from `training.tuning`."""


@dataclass(kw_only=True)
class EvalArguments:
    """Parameters consumed by a benchmark generation worker.

    Configuration location: the evaluation YAML root.
    Workflow paths and environment defaults are resolved before construction.
    Task-specific prompts belong to the benchmark loader.
    """

    # Model.
    model_name_or_path: str
    """TinyLLaVA checkpoint directory or Hub repository."""
    model_id: str | None = None
    """Model identifier written to answers; defaults to the checkpoint name."""

    # Data.
    adapter: str = "vqa"
    """Registered benchmark loader name."""
    question_file: str
    """Benchmark annotations in the selected adapter's format."""
    image_folder: str = ""
    """Root directory for relative image paths."""

    # Generation.
    temperature: float = 0.2
    """Sampling temperature; nonpositive values disable sampling."""
    top_p: float | None = None
    """Nucleus-sampling probability cutoff."""
    num_beams: int = 1
    """Number of decoding beams."""
    max_new_tokens: int = 128
    """Maximum generated tokens per answer."""
    chat_template: str | None = None
    """Inline Jinja template override."""
    chat_template_path: str | None = None
    """Jinja template file, exclusive with `chat_template`."""

    # Runtime.
    device: str | None = None
    """Torch device, or `None` to select CUDA when available, otherwise CPU."""
    batch_size: int = 1
    """Conversations per generation batch."""
    num_chunks: int = 1
    """Number of evaluation shards."""
    chunk_idx: int = 0
    """Zero-based shard index."""

    # Output.
    answers_file: str = "answer.jsonl"
    """Output JSONL path."""

    def __post_init__(self):
        for name in ("batch_size", "num_chunks"):
            positive_int(getattr(self, name), name)
        for name in ("max_new_tokens", "num_beams"):
            positive_int(getattr(self, name), name)
        if (
            isinstance(self.chunk_idx, bool)
            or not isinstance(self.chunk_idx, int)
            or not 0 <= self.chunk_idx < self.num_chunks
        ):
            raise ValueError("chunk_idx must be an integer in [0, num_chunks)")

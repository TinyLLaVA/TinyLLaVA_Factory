from dataclasses import dataclass, field
from typing import TYPE_CHECKING
import transformers


if TYPE_CHECKING:
    import transformers


@dataclass
class VisionTowerArguments:
    """Vision tower model configuration parameters."""
    vision_tower: str | None = field(
        default="",
        metadata={"help": "Vision tower model name or path (e.g., openai/clip-vit-large-patch14-336). Empty string means no vision tower."}
    )
    vision_tower2: str | None = field(
        default="",
        metadata={"help": "Secondary vision tower model name (for multi-tower architectures). Empty string means not used."}
    )
    vision_tower_pretrained: str | None = field(
        default=None,
        metadata={"help": "Path to pretrained vision tower weights. If None, uses weights from HuggingFace model hub."}
    )
    mm_vision_select_layer: int | None = field(
        default=-1,
        metadata={"help": "Which layer of the vision tower to select features from. -1 denotes the last layer."}
    )
    mm_patch_merge_type: str | None = field(
        default="flat",
        metadata={"help": "Patch merging strategy: 'flat' (concatenate all patches) or 'spatial' (preserve spatial structure)."}
    )
    mm_vision_select_feature: str | None = field(
        default="patch",
        metadata={"help": "Which features to select: 'patch' (patch tokens), 'cls_patch' (CLS + patches), or 'cls' (only CLS token)."}
    )


@dataclass
class ConnectorArguments:
    """Multimodal connector/projector configuration parameters."""
    mm_projector_type: str = field(
        default="linear",
        metadata={"help": "Type of connector to project vision features to language model space: 'linear', 'mlp', 'ldp' (learned dynamic projection), or other architectures."}
    )
    mm_projector_depth: int = field(
        default=1,
        metadata={"help": "Number of connector layers. 1 means single linear layer, >1 means multi-layer MLP."}
    )
    connector_type: str = field(
        default="linear",
        metadata={"help": "[DEPRECATED: Use mm_projector_type] Kept for backward compatibility."}
    )

    def __post_init__(self):
        # Keep backward compatibility with old connector_type while introducing mm_projector_type.
        if self.mm_projector_type != self.connector_type:
            if self.mm_projector_type == "linear":
                self.mm_projector_type = self.connector_type
            else:
                self.connector_type = self.mm_projector_type


@dataclass
class TinyLLaVAFinetuningArguments:
    """Fine-tuning methodology and LoRA hyperparameter configuration."""
    finetuning_type: str = field(
        default="lora",
        metadata={"help": "Fine-tuning strategy: 'lora' (parameter-efficient LoRA), 'full' (full model fine-tuning)."}
    )
    lora_rank: int = field(
        default=8,
        metadata={"help": "LoRA rank (dimension of low-rank decomposition). Larger values capture more expressiveness but use more memory."}
    )
    lora_alpha: int | None = field(
        default=None,
        metadata={"help": "LoRA scaling factor. If None, automatically set to lora_rank * 2. Controls the learning rate of LoRA parameters."}
    )
    lora_target: str | list[str] | None = field(
        default=None,
        metadata={"help": "Target module names for LoRA (comma-separated string or list). e.g., 'q_proj,v_proj' or ['q_proj', 'v_proj']. If None, applies to all linear layers."}
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "LoRA dropout rate (0.0-1.0). Regularizes LoRA modules to prevent overfitting."}
    )

    def __post_init__(self):
        # Auto-compute lora_alpha if not specified
        self.lora_alpha = self.lora_alpha or self.lora_rank * 2

        # Convert comma-separated string to list
        if isinstance(self.lora_target, str):
            self.lora_target = [m.strip() for m in self.lora_target.split(",") if m.strip()]

        # Validate finetuning_type
        if self.finetuning_type not in ["lora", "full"]:
            raise ValueError("Invalid finetuning_type, must be one of: lora, full")


@dataclass
class ModelArguments(VisionTowerArguments, ConnectorArguments):
    """Model configuration parameters combining vision, connector, and LLM components."""
    cache_dir: str | None = field(
        default=None,
        metadata={"help": "Directory to cache downloaded models and tokenizers (e.g., '/path/to/cache'). If None, uses HuggingFace's default cache."}
    )

    model_name_or_path: str | None = field(
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    tokenizer_name_or_path: str | None = field(
        default=None,
        metadata={"help": "Path to tokenizer or tokenizer identifier. If None, uses the same as model_name_or_path."}
    )
    attn_implementation: str | None = field(
        default=None,
        metadata={"help": "Attention implementation: 'eager' (PyTorch default), 'sdpa' (scaled dot-product attention), 'flash_attention_2', or None (auto-detect)."}
    )
    resampler_hidden_size: int | None = field(
        default=768,
        metadata={"help": "Hidden dimension size for the resampler module if used. Controls the bottleneck width."}
    )
    num_queries: int | None = field(
        default=128,
        metadata={"help": "Number of query tokens in the resampler. Determines the number of tokens output by the resampler."}
    )
    num_resampler_layers: int | None = field(
        default=3,
        metadata={"help": "Number of transformer layers in the resampler. Larger values increase expressiveness but add computation."}
    )
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum input sequence length (tokens). Sequences will be right padded and possibly truncated to this length."}
    )
    tokenizer_use_fast: bool = field(
        default=False,
        metadata={"help": "Whether to use fast tokenizer implementation (Rust-based) for faster tokenization."}
    )
    tokenizer_padding_side: str = field(
        default="right",
        metadata={"help": "Padding side for tokenizer: 'left' or 'right'. Most models use 'right'."}
    )


@dataclass
class DataArguments:
    """Data loading and preprocessing configuration parameters."""
    data_path: str | None = field(
        default=None,
        metadata={"help": "Path to the training data file (JSON, JSONL, CSV, or other formats). Can be a single file or directory of files."}
    )
    lazy_preprocess: bool = field(
        default=False,
        metadata={"help": "Whether to use lazy preprocessing. If True, data is processed on-the-fly during training (saves memory but slower)."}
    )
    is_multimodal: bool = field(
        default=True,
        metadata={"help": "Whether the dataset contains multimodal data (images + text). If False, treats as text-only."}
    )
    image_folder: str | None = field(
        default=None,
        metadata={"help": "Root directory containing image files referenced in the training data."}
    )
    image_aspect_ratio: str = field(
        default="square",
        metadata={"help": "Image aspect ratio handling: 'square' (resize to square), 'pad' (pad to match ratio), or 'scale' (scale keeping ratio)."}
    )
    conv_version: str = field(
        default="pretrain",
        metadata={"help": "Conversation template version: 'pretrain' (pre-training format), 'finetune' (fine-tuning format), 'eval' (evaluation format), etc."}
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    """Training hyperparameters extending HuggingFace TrainingArguments with TinyLLaVA-specific options."""
    training_recipe: str = field(
        default="common",
        metadata={"help": "Training recipe/strategy: 'common' (standard training), 'lora' (LoRA-specific), 'dpo' (direct preference optimization), etc."}
    )
    tune_type_llm: str = field(
        default="frozen",
        metadata={"help": "LLM tuning strategy: 'frozen' (no update), 'full' (full fine-tune), 'lora' (LoRA), 'qlora_int4' (QLoRA 4-bit), 'qlora_int8' (QLoRA 8-bit)."}
    )
    tune_type_vision_tower: str = field(
        default="frozen",
        metadata={"help": "Vision tower tuning strategy: 'frozen' (no update), 'full' (fine-tune all layers), 'partially-tune' (tune last N layers)."}
    )
    tune_vision_tower_from_layer: int | None = field(
        default=10,
        metadata={"help": "If tune_type_vision_tower='partially-tune', fine-tune from this layer onwards. Higher number = deeper layers."}
    )
    tune_type_connector: str = field(
        default="full",
        metadata={"help": "Connector/projector tuning strategy: 'frozen' (no update), 'full' (fine-tune all layers)."}
    )
    tune_embed_tokens: int | None = field(
        default=False,
        metadata={"help": "Whether to fine-tune embedding tokens. True means update all embeddings, False means frozen."}
    )

    optim: str = field(
        default="adamw_torch",
        metadata={"help": "Optimizer to use: 'adamw_torch', 'adamw_8bit', 'adafactor', 'adamw_bnb_8bit', or others."}
    )
    remove_unused_columns: bool = field(
        default=False,
        metadata={"help": "Remove columns not required by the model. Set to False to keep auxiliary columns."}
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress quantization statistics through double quantization (for QLoRA)."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type for QLoRA: 'fp4' (4-bit float) or 'nf4' (4-bit normalized float)."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "Quantization bits: 4, 8, 16. Controls precision/memory trade-off."}
    )
    lora_r: int = field(
        default=64,
        metadata={"help": "LoRA rank (r parameter in LoRA). Larger values increase model capacity but use more memory."}
    )
    lora_alpha: int = field(
        default=16,
        metadata={"help": "LoRA alpha scaling factor. Controls the learning rate of LoRA modules relative to base model."}
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "Dropout probability in LoRA modules (0.0-1.0). Regularizes to prevent overfitting."}
    )
    lora_weight_path: str = field(
        default="",
        metadata={"help": "Path to pretrained LoRA weights to load. Empty string means don't load."}
    )
    lora_bias: str = field(
        default="none",
        metadata={"help": "LoRA bias configuration: 'all' (add bias to all), 'none' (no bias), 'lora_only' (only to LoRA)."}
    )
    mm_projector_lr: float | None = field(
        default=None,
        metadata={"help": "Learning rate for multimodal projector. If None, uses default learning rate."}
    )
    group_by_modality_length: bool = field(
        default=False,
        metadata={"help": "Group samples by modality (image vs text) to improve batch efficiency."}
    )
    vision_tower_lr: float | None = field(
        default=None,
        metadata={"help": "Learning rate for vision tower. If None, uses default learning rate."}
    )
    pretrained_model_path: str | None = field(
        default=None,
        metadata={"help": "Path to pretrained model checkpoint for initialization. Takes priority over model_name_or_path."}
    )

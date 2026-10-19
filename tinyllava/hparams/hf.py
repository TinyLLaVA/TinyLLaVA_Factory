"""HF TrainingArguments extension, imported when building training worker inputs.

Inheriting HF TrainingArguments loads Transformers and Torch. Keeping that
subclass here lets workflows validate configuration using hparams.arguments
and hparams.training before loading training dependencies. Project-owned
parameter objects, including TrainingArgBundle, live in hparams.arguments.
"""

from dataclasses import dataclass, field
from typing import Literal

from transformers import TrainingArguments as HFTrainingArguments

from tinyllava.utils.precision import resolve_precision_flags


@dataclass
class TrainingArguments(HFTrainingArguments):
    """HF Trainer settings with project-specific precision and sampling policies.

    Configuration location: `training` in the resolved training stage.
    Inherited fields keep HF's defaults and validation. Precision and modality
    sampling extend the upstream schema; `bf16` and `fp16` are derived flags.
    """

    precision: Literal["auto", "fp32", "fp16", "bf16"] = "auto"
    """Training dtype policy; `auto` chooses BF16, FP16, or FP32
    according to the available hardware. This determines `bf16` and `fp16`.
    """

    group_by_modality_length: bool = False
    """Group multimodal and text-only samples by length."""

    # Precision is configured through the policy, not independent HF flags.
    bf16: bool = field(default=False, init=False)
    fp16: bool = field(default=False, init=False)

    def __post_init__(self):
        # Resolve precision before HF fills DeepSpeed's "auto" dtype values.
        self.bf16, self.fp16 = resolve_precision_flags(self.precision)
        super().__post_init__()

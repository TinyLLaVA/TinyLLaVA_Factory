from typing import Any

import torch

from tinyllava.utils.logging import get_logger


logger = get_logger(__name__)


def resolve_training_precision(
    training_args: Any,
    model_config: Any,
) -> None:
    """Resolve TinyLLaVA's dtype policy into HF mixed-precision flags.

    `precision` is the user-facing policy. It defaults to `auto`, which chooses
    BF16 > FP16 > FP32 from hardware capability. Explicit values such as `fp32`
    force the corresponding dtype. TF32 is intentionally left to HF's separate
    `tf32` argument because it controls CUDA math kernels rather than the
    training dtype. Model config dtype is used only for diagnostics and future
    compatibility rules.
    """

    precision = getattr(training_args, "precision", "auto")
    _apply_precision(training_args, precision)
    _log_resolved_precision(training_args, precision)
    _warn_if_model_dtype_differs(training_args, model_config)


def training_torch_dtype(training_args: Any) -> torch.dtype:
    if training_args.bf16:
        return torch.bfloat16
    if training_args.fp16:
        return torch.float16
    return torch.float32


def cast_model_to_training_dtype(
    model: torch.nn.Module,
    training_args: Any,
) -> torch.nn.Module:
    """Cast floating-point model state to the resolved training dtype."""
    dtype = training_torch_dtype(training_args)
    if _is_quantized_model(model):
        for parameter in model.parameters():
            if parameter.requires_grad and parameter.is_floating_point():
                parameter.data = parameter.data.to(dtype=dtype)
    else:
        model.to(dtype=dtype)
    logger.info_rank0("Cast model parameters and buffers to dtype=%s.", dtype)
    return model


def _is_quantized_model(model: torch.nn.Module) -> bool:
    return any(
        bool(getattr(model, attribute, False))
        for attribute in ("is_quantized", "is_loaded_in_4bit", "is_loaded_in_8bit")
    )


def resolve_precision_flags(precision: str) -> tuple[bool, bool]:
    """Return ``(bf16, fp16)`` before HF processes DeepSpeed auto values."""
    match precision:
        case "auto":
            if not torch.cuda.is_available():
                return False, False
            bf16 = torch.cuda.is_bf16_supported()
            return bf16, not bf16
        case "fp32":
            return False, False
        case "fp16":
            return False, True
        case "bf16":
            return True, False
        case _:
            raise ValueError("precision must be one of: auto, fp32, fp16, bf16")


def _apply_precision(training_args: Any, precision: str) -> None:
    bf16, fp16 = resolve_precision_flags(precision)
    _set_hf_mixed_precision_flags(training_args, bf16=bf16, fp16=fp16)


def _set_hf_mixed_precision_flags(
    training_args: Any,
    *,
    bf16: bool,
    fp16: bool,
) -> None:
    training_args.bf16 = bf16
    training_args.fp16 = fp16


def _log_resolved_precision(training_args: Any, precision: str) -> None:
    logger.info_rank0(
        "Resolved precision policy %r to dtype=%s, bf16=%s, fp16=%s, tf32=%s.",
        precision,
        training_torch_dtype(training_args),
        training_args.bf16,
        training_args.fp16,
        getattr(training_args, "tf32", None),
    )


def _warn_if_model_dtype_differs(
    training_args: Any,
    model_config: Any,
) -> None:
    model_dtype = _model_torch_dtype(model_config)
    if model_dtype is None:
        return

    selected_dtype = training_torch_dtype(training_args)
    if model_dtype != selected_dtype:
        logger.warning_rank0(
            "Model config torch_dtype is %s, while training precision resolves to %s. "
            "This may be intentional; known model precision compatibility rules can "
            "be added here later.",
            model_dtype,
            selected_dtype,
        )


def _model_torch_dtype(model_config: Any) -> torch.dtype | None:
    candidates = [
        getattr(model_config, "torch_dtype", None),
        getattr(getattr(model_config, "text_config", None), "torch_dtype", None),
    ]
    for dtype in candidates:
        normalized = _normalize_torch_dtype(dtype)
        if normalized is not None:
            return normalized
    return None


def _normalize_torch_dtype(dtype: Any) -> torch.dtype | None:
    if isinstance(dtype, torch.dtype):
        return dtype
    if isinstance(dtype, str):
        return {
            "float32": torch.float32,
            "torch.float32": torch.float32,
            "float": torch.float32,
            "float16": torch.float16,
            "torch.float16": torch.float16,
            "half": torch.float16,
            "bfloat16": torch.bfloat16,
            "torch.bfloat16": torch.bfloat16,
        }.get(dtype)
    return None


__all__ = [
    "cast_model_to_training_dtype",
    "resolve_precision_flags",
    "resolve_training_precision",
    "training_torch_dtype",
]

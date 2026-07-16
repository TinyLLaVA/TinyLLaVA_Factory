from typing import Any

from tinyllava.utils.logging import get_logger


logger = get_logger(__name__)


def configure_zero3_gradient_checkpointing(training_args: Any) -> None:
    """Select checkpointing compatible with ZeRO-3 parameter partitioning."""
    hf_deepspeed_config = getattr(training_args, "hf_deepspeed_config", None)
    if (
        hf_deepspeed_config is None
        or not hf_deepspeed_config.is_zero3()
        or not getattr(training_args, "gradient_checkpointing", False)
        or getattr(training_args, "gradient_checkpointing_kwargs", None) is not None
    ):
        return

    # Non-reentrant checkpointing records metadata for tensors saved during the
    # forward. ZeRO-3 releases gathered parameters back to shape-[0] placeholders
    # before recomputation, which makes that metadata check fail. Reentrant
    # checkpointing recomputes the layer without retaining those tensor records.
    training_args.gradient_checkpointing_kwargs = {"use_reentrant": True}
    logger.warning_rank0(
        "Using reentrant gradient checkpointing for DeepSpeed ZeRO-3 "
        "compatibility. Set training.gradient_checkpointing_kwargs explicitly "
        "to override this behavior."
    )


__all__ = ["configure_zero3_gradient_checkpointing"]

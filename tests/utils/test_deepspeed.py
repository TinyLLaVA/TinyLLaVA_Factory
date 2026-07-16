from unittest.mock import Mock

from tinyllava.utils.deepspeed import configure_zero3_gradient_checkpointing


def test_configure_zero3_gradient_checkpointing_uses_reentrant_mode():
    training_args = Mock(
        hf_deepspeed_config=Mock(is_zero3=Mock(return_value=True)),
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs=None,
    )

    configure_zero3_gradient_checkpointing(training_args)

    assert training_args.gradient_checkpointing_kwargs == {"use_reentrant": True}


def test_configure_zero3_gradient_checkpointing_preserves_explicit_kwargs():
    explicit_kwargs = {"use_reentrant": False, "determinism_check": "none"}
    training_args = Mock(
        hf_deepspeed_config=Mock(is_zero3=Mock(return_value=True)),
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs=explicit_kwargs,
    )

    configure_zero3_gradient_checkpointing(training_args)

    assert training_args.gradient_checkpointing_kwargs is explicit_kwargs

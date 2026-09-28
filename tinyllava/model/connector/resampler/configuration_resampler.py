"""Perceiver resampler configuration."""

from __future__ import annotations


from huggingface_hub.dataclasses import strict

from ..configuration_base import BaseConnectorConfig


@strict
class ResamplerConnectorConfig(BaseConnectorConfig):
    model_type = "resampler__tlf_connector"
    num_queries: int = 32
    num_hidden_layers: int = 2
    hidden_size: int = 768
    num_attention_heads: int = 8
    head_dim: int = 64
    intermediate_size: int = 3072

    def get_output_sequence_length(self, input_length: int) -> int:
        return self.num_queries

    def __post_init__(self, **kwargs):
        for name in (
            "num_queries",
            "num_hidden_layers",
            "hidden_size",
            "num_attention_heads",
            "head_dim",
            "intermediate_size",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        super().__post_init__(**kwargs)


__all__ = ["ResamplerConnectorConfig"]

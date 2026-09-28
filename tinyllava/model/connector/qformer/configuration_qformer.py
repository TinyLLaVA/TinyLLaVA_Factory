"""Configuration for a query-only BLIP-2 Q-Former connector."""

from __future__ import annotations


from huggingface_hub.dataclasses import strict

from ..configuration_base import BaseConnectorConfig


@strict
class QFormerConnectorConfig(BaseConnectorConfig):
    model_type = "qformer__tlf_connector"
    num_queries: int = 32
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    cross_attention_frequency: int = 2
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    initializer_range: float = 0.02

    def get_output_sequence_length(self, input_length: int) -> int:
        return self.num_queries

    def __post_init__(self, **kwargs):
        for name in (
            "num_queries",
            "hidden_size",
            "num_hidden_layers",
            "num_attention_heads",
            "intermediate_size",
            "cross_attention_frequency",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        super().__post_init__(**kwargs)


__all__ = ["QFormerConnectorConfig"]

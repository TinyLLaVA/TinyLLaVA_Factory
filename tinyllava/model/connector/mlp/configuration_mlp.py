"""MLP connector model configuration"""

from huggingface_hub.dataclasses import strict

from ..configuration_base import BaseConnectorConfig


@strict
class MLPConnectorConfig(BaseConnectorConfig):
    model_type = "mlp__tlf_connector"

    act: str = "gelu"
    bias: bool = True
    depth: int = 2
    hidden_size: int | None = None

    def __post_init__(self, **kwargs):
        if self.depth < 1:
            raise ValueError("depth must be at least 1")
        super().__post_init__(**kwargs)


__all__ = ["MLPConnectorConfig"]

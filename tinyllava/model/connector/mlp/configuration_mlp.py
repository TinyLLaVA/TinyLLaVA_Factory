"""MLP connector model configuration"""

from huggingface_hub.dataclasses import strict

from ..configuration_base import _BaseConnectorConfig


@strict
class MLPConnectorConfig(_BaseConnectorConfig):
    model_type = "mlp__tlf_connector"

    act: str = "gelu"
    bias: bool = True
    depth: int = 2


__all__ = ["MLPConnectorConfig"]

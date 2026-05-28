import logging

from transformers import PreTrainedConfig


logger = logging.getLogger(__name__)


class _BaseConnectorConfig(PreTrainedConfig):
    vision_hidden_size: int = 1024
    text_hidden_size: int = 4096
    vision_feature_layer: int | list[int] = -2


__all__ = ["_BaseConnectorConfig"]

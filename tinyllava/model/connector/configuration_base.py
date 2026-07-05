import logging

from transformers import PreTrainedConfig


logger = logging.getLogger(__name__)


class BaseConnectorConfig(PreTrainedConfig):
    """Base class for connector configs."""


__all__ = ["BaseConnectorConfig"]

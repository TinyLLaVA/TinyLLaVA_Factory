import logging

from transformers import PreTrainedConfig

logger = logging.getLogger(__name__)


class BaseConnectorConfig(PreTrainedConfig):
    """Describe connector parameters and its output sequence length."""

    def get_output_sequence_length(self, input_length: int) -> int:
        """Map selected vision tokens to projected tokens without model weights."""
        return input_length


__all__ = ["BaseConnectorConfig"]

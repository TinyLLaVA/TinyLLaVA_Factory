"""Base classes for connector models."""

from transformers import PreTrainedModel

from .configuration_base import BaseConnectorConfig


class BaseConnectorModel(PreTrainedModel):
    config_class = BaseConnectorConfig

    def __init__(
        self,
        config: BaseConnectorConfig,
        *,
        vision_hidden_size: int,
        text_hidden_size: int,
        vision_feature_layer: int | list[int],
    ):
        super().__init__(config)
        self.vision_hidden_size = _require_positive_int(
            vision_hidden_size,
            "vision_hidden_size",
        )
        self.text_hidden_size = _require_positive_int(
            text_hidden_size,
            "text_hidden_size",
        )
        self.vision_feature_layer = _require_feature_layer(vision_feature_layer)

    @property
    def num_vision_feature_layers(self) -> int:
        if isinstance(self.vision_feature_layer, int):
            return 1
        return len(self.vision_feature_layer)


def _require_positive_int(value: int, name: str) -> int:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")
    return value


def _require_feature_layer(value: int | list[int]) -> int | list[int]:
    if isinstance(value, int):
        return value
    if isinstance(value, list) and value and all(isinstance(item, int) for item in value):
        return value
    raise ValueError(
        "vision_feature_layer must be an integer or a non-empty list of integers, "
        f"got {value!r}."
    )


__all__ = ["BaseConnectorModel"]

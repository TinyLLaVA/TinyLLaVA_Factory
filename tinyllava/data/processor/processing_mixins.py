"""Reusable component metadata and length rules for HF processors."""

from typing import Any

from transformers import PreTrainedConfig

from tinyllava.model.connector.auto.configuration_auto import CONNECTOR_CONFIG_MAPPING
from tinyllava.model.connector.configuration_base import BaseConnectorConfig
from tinyllava.model.vision_tower.auto.configuration_auto import VISION_TOWER_CONFIG_MAPPING


class ComponentProcessorMixin:
    """Compose with a concrete HF processor to persist component configuration.

    Place this mixin before the HF processor in the inheritance list. Length
    helpers are opt-in: the concrete processor owns placeholder formatting and
    decides where its native feature selection and merging fit in the pipeline.
    Declare sub-processors and serializable settings explicitly in the concrete
    constructor so HF can discover and restore them.
    """

    def __init__(
        self,
        *args,
        connector_config: BaseConnectorConfig | dict[str, Any] | None = None,
        vision_config: PreTrainedConfig | dict[str, Any] | None = None,
        **kwargs,
    ):
        if isinstance(connector_config, dict):
            values = dict(connector_config)
            model_type = values.pop("model_type")
            connector_config = CONNECTOR_CONFIG_MAPPING[model_type](**values)
        self.connector_config = connector_config
        if isinstance(vision_config, dict):
            values = dict(vision_config)
            model_type = values.pop("model_type")
            vision_config = VISION_TOWER_CONFIG_MAPPING[model_type](**values)
        self.vision_config = vision_config
        super().__init__(*args, **kwargs)

    def get_vision_output_sequence_length(
        self, image_inputs: dict[str, Any], image_idx: int, **kwargs
    ) -> int | None:
        """Return a custom selected-feature count, or None for native handling.

        Forward processed inputs and selection options to the vision config.
        The concrete processor supplies the options its vision rule requires.
        """
        vision_length = getattr(self.vision_config, "get_output_sequence_length", None)
        if vision_length is None:
            return None
        return vision_length(image_inputs, image_idx, **kwargs)

    def get_output_sequence_length(self, input_length: int) -> int:
        """Apply the configured connector rule, or preserve length if absent."""
        if self.connector_config is None:
            return input_length
        return self.connector_config.get_output_sequence_length(input_length)

    def to_dict(self) -> dict[str, Any]:
        values = super().to_dict()
        if self.connector_config is not None:
            values["connector_config"] = self.connector_config.to_dict()
        if self.vision_config is not None:
            values["vision_config"] = self.vision_config.to_dict()
        return values


__all__ = ["ComponentProcessorMixin"]

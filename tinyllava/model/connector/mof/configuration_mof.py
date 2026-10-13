"""Independent configurable projections for Interleaved Mixture-of-Features."""

from huggingface_hub.dataclasses import strict

from ..auto.configuration_auto import CONNECTOR_CONFIG_MAPPING
from ..configuration_base import BaseConnectorConfig


@strict
class MofConnectorConfig(BaseConnectorConfig):
    model_type = "mof__tlf_connector"
    sub_configs = {
        "clip_config": BaseConnectorConfig,
        "dinov2_config": BaseConnectorConfig,
    }

    vision_hidden_sizes: list[int] | tuple[int, ...] = (1024, 1024)
    clip_config: BaseConnectorConfig | dict | None = None
    dinov2_config: BaseConnectorConfig | dict | None = None

    def get_output_sequence_length(self, input_length: int) -> int:
        clip_length = self.clip_config.get_output_sequence_length(input_length)
        dinov2_length = self.dinov2_config.get_output_sequence_length(input_length)
        if clip_length != dinov2_length:
            raise ValueError("Interleaved MoF requires equal branch output lengths.")
        return 2 * clip_length

    def __post_init__(self, **kwargs):
        self.vision_hidden_sizes = tuple(self.vision_hidden_sizes)
        if len(self.vision_hidden_sizes) != 2 or any(
            isinstance(size, bool) or not isinstance(size, int) or size <= 0
            for size in self.vision_hidden_sizes
        ):
            raise ValueError("vision_hidden_sizes must contain two positive integers")
        for name in ("clip_config", "dinov2_config"):
            config = getattr(self, name)
            if config is None:
                config = CONNECTOR_CONFIG_MAPPING["mlp__tlf_connector"]()
            elif isinstance(config, dict):
                values = dict(config)
                model_type = values.pop("model_type")
                config = CONNECTOR_CONFIG_MAPPING[model_type](**values)
            setattr(self, name, config)
        super().__post_init__(**kwargs)


__all__ = ["MofConnectorConfig"]

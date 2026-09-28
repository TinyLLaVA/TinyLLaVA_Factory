"""Multimodal token expansion driven by the connector's sequence contract."""

from __future__ import annotations

from typing import Any

from transformers import AutoProcessor, LlavaProcessor
from transformers.image_processing_utils import BaseImageProcessor
from transformers.processing_utils import MultiModalData
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from tinyllava.model.configuration_tinyllava import TinyLlavaConfig
from tinyllava.model.connector.auto.configuration_auto import CONNECTOR_CONFIG_MAPPING
from tinyllava.model.connector.configuration_base import BaseConnectorConfig


class TinyLlavaProcessor(LlavaProcessor):
    """Keep prompt placeholders aligned with the configured connector output.

    The connector config supplies a pure sequence-length function. Persisting
    that config lets a standalone processor reproduce token counts without
    constructing a connector or importing any model implementation.
    """

    def __init__(
        self,
        image_processor: BaseImageProcessor | None = None,
        tokenizer: PreTrainedTokenizerBase | None = None,
        connector_config: BaseConnectorConfig | dict[str, Any] | None = None,
        patch_size: int | None = None,
        vision_feature_select_strategy: str | None = None,
        chat_template: str | None = None,
        image_token: str = "<image>",
        num_additional_image_tokens: int = 0,
        **kwargs,
    ):
        if connector_config is None:
            connector_config = CONNECTOR_CONFIG_MAPPING["mlp__tlf_connector"]()
        elif isinstance(connector_config, dict):
            values = dict(connector_config)
            model_type = values.pop("model_type")
            connector_config = CONNECTOR_CONFIG_MAPPING[model_type](**values)
        self.connector_config = connector_config
        super().__init__(
            image_processor=image_processor,
            tokenizer=tokenizer,
            patch_size=patch_size,
            vision_feature_select_strategy=vision_feature_select_strategy,
            chat_template=chat_template,
            image_token=image_token,
            num_additional_image_tokens=num_additional_image_tokens,
            **kwargs,
        )

    @classmethod
    def from_config(cls, config: TinyLlavaConfig, **kwargs) -> TinyLlavaProcessor:
        return cls(connector_config=config.connector_config, **kwargs)

    def to_dict(self) -> dict[str, Any]:
        values = super().to_dict()
        values["connector_config"] = self.connector_config.to_dict()
        return values

    def replace_image_token(self, image_inputs: dict[str, Any], image_idx: int) -> str:
        input_tokens = super().replace_image_token(image_inputs, image_idx)
        input_length = input_tokens.count(self.image_token)
        output_length = self.connector_config.get_output_sequence_length(input_length)
        return self.image_token * output_length

    def _get_num_multimodal_tokens(
        self, image_sizes: list[tuple[int, int]] | None = None, **kwargs
    ) -> MultiModalData:
        data = super()._get_num_multimodal_tokens(image_sizes=image_sizes, **kwargs)
        if data.num_image_tokens is not None:
            data.num_image_tokens = [
                self.connector_config.get_output_sequence_length(count)
                for count in data.num_image_tokens
            ]
        return data


AutoProcessor.register(TinyLlavaConfig, TinyLlavaProcessor, exist_ok=True)

__all__ = ["TinyLlavaProcessor"]

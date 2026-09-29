"""Shared processor construction, connector metadata, and token lengths."""

from __future__ import annotations

from typing import Any

from transformers import LlavaProcessor
from transformers.image_processing_utils import BaseImageProcessor
from transformers.processing_utils import MultiModalData, ProcessorMixin
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from tinyllava.data.chat_template import inject_tinyllava_anchors
from tinyllava.model.configuration_tinyllava import TinyLlavaConfig
from tinyllava.model.connector.auto.configuration_auto import CONNECTOR_CONFIG_MAPPING
from tinyllava.model.connector.configuration_base import BaseConnectorConfig
from tinyllava.utils.constants import DEFAULT_IMAGE_TOKEN

from .auto import AutoProcessor


class BaseProcessor(LlavaProcessor):
    """Extend LLaVA preprocessing with connector lengths and model construction."""

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

    def get_output_sequence_length(self, input_length: int) -> int:
        return self.connector_config.get_output_sequence_length(input_length)

    def to_dict(self) -> dict[str, Any]:
        values = super().to_dict()
        values["connector_config"] = self.connector_config.to_dict()
        return values

    def replace_image_token(self, image_inputs: dict[str, Any], image_idx: int) -> str:
        input_tokens = super().replace_image_token(image_inputs, image_idx)
        input_length = input_tokens.count(self.image_token)
        output_length = self.get_output_sequence_length(input_length)
        return self.image_token * output_length

    def _get_num_multimodal_tokens(
        self, image_sizes: list[tuple[int, int]] | None = None, **kwargs
    ) -> MultiModalData:
        data = super()._get_num_multimodal_tokens(image_sizes=image_sizes, **kwargs)
        if data.num_image_tokens is not None:
            data.num_image_tokens = [
                self.get_output_sequence_length(count)
                for count in data.num_image_tokens
            ]
        return data

    @classmethod
    def from_model(
        cls,
        *,
        tokenizer: Any,
        image_processor: Any,
        model: Any | None = None,
        chat_template: str | None = None,
    ) -> ProcessorMixin:
        """Build the registered multimodal processor and synchronize its image token.

        The tokenizer is updated in place with the image token and template anchors.
        When a model is supplied, its image-token ID and embedding size are synchronized.

        Args:
            tokenizer: Tokenizer used to encode conversations.
            image_processor: Image preprocessor matching the vision tower.
            model: Model providing vision settings and processor registration. Without
                a model, use the default TinyLLaVA configuration.
            chat_template: Jinja template override; otherwise use the tokenizer template.

        Returns:
            A registered processor combining the tokenizer and image processor.

        Raises:
            ValueError: The image token cannot be encoded as one token, or the vision
                feature selection conflicts with the backbone's additional tokens.
        """
        cls.ensure_image_token(tokenizer, model=model)

        template = chat_template or getattr(tokenizer, "chat_template", None)
        if template is not None:
            template = inject_tinyllava_anchors(template).chat_template
            tokenizer.chat_template = template

        config = getattr(model, "config", None)
        vision_config = getattr(config, "vision_config", None)
        vision_feature_select_strategy = getattr(
            config, "vision_feature_select_strategy", "default"
        )
        num_additional_image_tokens = cls._get_num_additional_image_tokens(
            vision_config
        )
        cls._validate_image_feature_configuration(
            num_additional_image_tokens=num_additional_image_tokens,
            vision_feature_select_strategy=vision_feature_select_strategy,
        )

        config = config if config is not None else TinyLlavaConfig()
        return AutoProcessor.from_config(
            config,
            connector_config=config.connector_config,
            image_processor=image_processor,
            tokenizer=tokenizer,
            patch_size=getattr(vision_config, "patch_size", 14),
            vision_feature_select_strategy=vision_feature_select_strategy,
            num_additional_image_tokens=num_additional_image_tokens,
            chat_template=template,
            image_token=DEFAULT_IMAGE_TOKEN,
        )

    @staticmethod
    def ensure_image_token(tokenizer: Any, model: Any | None = None) -> int:
        image_token_id = BaseProcessor._encode_single_token(
            tokenizer, DEFAULT_IMAGE_TOKEN
        )
        if image_token_id is None and hasattr(tokenizer, "add_special_tokens"):
            tokenizer.add_special_tokens(
                {"additional_special_tokens": [DEFAULT_IMAGE_TOKEN]}
            )
            image_token_id = BaseProcessor._encode_single_token(
                tokenizer, DEFAULT_IMAGE_TOKEN
            )

        if image_token_id is None:
            raise ValueError(
                f"Tokenizer cannot encode TinyLLaVA image token {DEFAULT_IMAGE_TOKEN!r}."
            )

        tokenizer.image_token = DEFAULT_IMAGE_TOKEN
        if model is not None and hasattr(model, "resize_token_embeddings"):
            try:
                model.resize_token_embeddings(len(tokenizer))
            except TypeError:
                pass
        config = getattr(model, "config", None)
        if config is not None:
            config.image_token_id = image_token_id
        return image_token_id

    @staticmethod
    def _encode_single_token(tokenizer: Any, token: str) -> int | None:
        try:
            token_ids = tokenizer.encode(token, add_special_tokens=False)
        except TypeError:
            token_ids = tokenizer.encode(token)
        except Exception:
            return None

        if isinstance(token_ids, int):
            return token_ids
        if len(token_ids) == 1:
            return int(token_ids[0])
        return None

    @staticmethod
    def _get_num_additional_image_tokens(vision_config: Any) -> int:
        value = getattr(vision_config, "num_additional_image_tokens", None)
        if value is not None:
            return value

        model_type = getattr(vision_config, "model_type", "")
        if "siglip" in model_type:
            return 0
        return 1

    @staticmethod
    def _validate_image_feature_configuration(
        *,
        num_additional_image_tokens: int,
        vision_feature_select_strategy: str,
    ) -> None:
        if vision_feature_select_strategy not in {"default", "full"}:
            raise ValueError(
                "vision_feature_select_strategy must be either 'default' or 'full', "
                f"got {vision_feature_select_strategy!r}."
            )
        if (
            isinstance(num_additional_image_tokens, bool)
            or not isinstance(num_additional_image_tokens, int)
            or num_additional_image_tokens < 0
        ):
            raise ValueError(
                "num_additional_image_tokens must be a non-negative integer, "
                f"got {num_additional_image_tokens!r}."
            )
        if (
            vision_feature_select_strategy == "default"
            and num_additional_image_tokens == 0
        ):
            raise ValueError(
                "Invalid image feature configuration: vision_feature_select_strategy='default' "
                "drops the first vision feature, but num_additional_image_tokens=0 means the "
                "vision tower has no CLS/additional token to drop. Use "
                "vision_feature_select_strategy='full' for patch-only backbones such as SigLIP."
            )


__all__ = ["BaseProcessor"]

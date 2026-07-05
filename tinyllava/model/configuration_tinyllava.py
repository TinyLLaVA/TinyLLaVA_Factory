# Copyright 2023 Microsoft Research & University of Wisconsin-Madison, the HuggingFace Inc. team and TinyLLaVA group. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""TinyLLaVA model configuration"""

from typing import Literal

from huggingface_hub.dataclasses import strict

from transformers import (
    PreTrainedConfig,
    CONFIG_MAPPING,
    AutoConfig,
)

from .llm import LANGUAGE_CONFIG_MAPPING
from .vision_tower import VISION_TOWER_CONFIG_MAPPING
from .connector import CONNECTOR_CONFIG_MAPPING


@strict
class TinyLlavaConfig(PreTrainedConfig):
    model_type = "tinyllava"
    attribute_map = {
        "image_token_id": "image_token_index",
    }
    sub_configs = {
        "text_config": AutoConfig,
        "vision_config": AutoConfig,
        "connector_config": AutoConfig,
    }

    vision_config: dict | PreTrainedConfig | None = None
    connector_config: dict | PreTrainedConfig | None = None
    text_config: dict | PreTrainedConfig | None = None
    image_token_index: int = 32000
    image_seq_length: int = 576
    projector_hidden_act: str = "gelu"
    vision_feature_select_strategy: Literal["default", "full"] = "default"
    vision_feature_layer: int | list[int] = -2
    multimodal_projector_bias: bool = True
    tie_word_embeddings: bool = False

    def __post_init__(self, **kwargs):
        if isinstance(self.vision_config, dict):
            self.vision_config["model_type"] = self.vision_config.get("model_type", "clip_vision_model")
            self.vision_config = VISION_TOWER_CONFIG_MAPPING[self.vision_config["model_type"]](**self.vision_config)
        elif self.vision_config is None:
            self.vision_config = VISION_TOWER_CONFIG_MAPPING["clip_vision_model"](
                intermediate_size=4096,
                hidden_size=1024,
                patch_size=14,
                image_size=336,
                num_hidden_layers=24,
                num_attention_heads=16,
                vocab_size=32000,
                projection_dim=768,
            )

        if isinstance(self.text_config, dict):
            self.text_config["model_type"] = self.text_config.get("model_type", "llama")
            self.text_config = LANGUAGE_CONFIG_MAPPING[self.text_config["model_type"]](**self.text_config)
        elif self.text_config is None:
            self.text_config = LANGUAGE_CONFIG_MAPPING["llama"]()

        # The default value is `False` but this config is used with many model types
        # Attr `tie_word_embeddings` was saved in text config for those models, so we
        # need an ugly workaround and forward-pass the attr from text config
        if not self.tie_word_embeddings and self.text_config.tie_word_embeddings:
            self.tie_word_embeddings = self.text_config.tie_word_embeddings

        if isinstance(self.connector_config, dict):
            self.connector_config["model_type"] = self.connector_config.get("model_type", "mlp__tlf_connector")
            self.connector_config = CONNECTOR_CONFIG_MAPPING[self.connector_config["model_type"]](
                **self.connector_config,
            )
        elif self.connector_config is None:
            self.connector_config = CONNECTOR_CONFIG_MAPPING["mlp__tlf_connector"](
                bias=self.multimodal_projector_bias,
                act=self.projector_hidden_act,
            )

        super().__post_init__(**kwargs)


CONFIG_MAPPING.register("tinyllava", TinyLlavaConfig)


__all__ = ["CONFIG_MAPPING", "TinyLlavaConfig"]

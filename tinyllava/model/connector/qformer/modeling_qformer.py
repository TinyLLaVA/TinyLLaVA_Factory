"""Query-only connector using Transformers' maintained BLIP-2 Q-Former.

Source: Li et al., "BLIP-2: Bootstrapping Language-Image Pre-training with
Frozen Image Encoders and Large Language Models" (2023),
https://arxiv.org/abs/2301.12597.
The connector uses learned queries over image features, followed by a language
projection; BLIP-2's two-stage pre-training objectives are outside this module.
"""

from __future__ import annotations

import torch
from torch import nn
from transformers import Blip2QFormerConfig, Blip2QFormerModel

from ..modeling_base import BaseConnectorModel
from .configuration_qformer import QFormerConnectorConfig


class QFormerConnector(BaseConnectorModel):
    config_class = QFormerConnectorConfig

    def __init__(
        self,
        config: QFormerConnectorConfig,
        *,
        vision_hidden_size: int,
        text_hidden_size: int,
        vision_feature_layer: int | list[int],
    ):
        super().__init__(
            config,
            vision_hidden_size=vision_hidden_size,
            text_hidden_size=text_hidden_size,
            vision_feature_layer=vision_feature_layer,
        )
        qformer_config = Blip2QFormerConfig(
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            intermediate_size=config.intermediate_size,
            cross_attention_frequency=config.cross_attention_frequency,
            hidden_dropout_prob=config.hidden_dropout_prob,
            attention_probs_dropout_prob=config.attention_probs_dropout_prob,
            initializer_range=config.initializer_range,
            encoder_hidden_size=self.vision_hidden_size
            * self.num_vision_feature_layers,
        )
        self.qformer = Blip2QFormerModel(qformer_config)
        self.query_tokens = nn.Parameter(
            torch.empty(1, config.num_queries, config.hidden_size)
        )
        nn.init.normal_(self.query_tokens, std=config.initializer_range)
        self.output_projection = nn.Linear(config.hidden_size, self.text_hidden_size)

    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        queries = self.query_tokens.expand(vision_features.shape[0], -1, -1)
        output = self.qformer(
            query_embeds=queries, encoder_hidden_states=vision_features
        )
        return self.output_projection(output.last_hidden_state)


__all__ = ["QFormerConnector"]

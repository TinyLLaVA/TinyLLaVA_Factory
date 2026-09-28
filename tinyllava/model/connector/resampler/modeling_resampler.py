"""Image connector using the Perceiver Resampler architecture from Flamingo.

Source: Alayrac et al., "Flamingo: a Visual Language Model for Few-Shot
Learning" (2022), https://arxiv.org/abs/2204.14198.
This connector operates on image tokens and projects resampled latents to
language-model width; it does not implement Flamingo's full fusion architecture.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from ..modeling_base import BaseConnectorModel
from .configuration_resampler import ResamplerConnectorConfig


class PerceiverAttention(nn.Module):
    def __init__(self, config: ResamplerConnectorConfig):
        super().__init__()
        self.heads = config.num_attention_heads
        self.head_dim = config.head_dim
        inner = self.heads * self.head_dim
        self.norm_media = nn.LayerNorm(config.hidden_size)
        self.norm_latents = nn.LayerNorm(config.hidden_size)
        self.to_q = nn.Linear(config.hidden_size, inner, bias=False)
        self.to_kv = nn.Linear(config.hidden_size, inner * 2, bias=False)
        self.to_out = nn.Linear(inner, config.hidden_size, bias=False)

    def forward(self, features: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
        features = self.norm_media(features)
        latents = self.norm_latents(latents)
        query = self.to_q(latents)
        key, value = self.to_kv(torch.cat((features, latents), dim=1)).chunk(2, dim=-1)
        query, key, value = [
            x.reshape(x.shape[0], -1, self.heads, self.head_dim).transpose(1, 2)
            for x in (query, key, value)
        ]
        result = F.scaled_dot_product_attention(query, key, value)
        return self.to_out(result.transpose(1, 2).flatten(2))


class ResamplerLayer(nn.Module):
    """Update latent queries with attention and a residual feed-forward block."""

    def __init__(self, config: ResamplerConnectorConfig):
        super().__init__()
        self.attention = PerceiverAttention(config)
        self.feed_forward = nn.Sequential(
            nn.LayerNorm(config.hidden_size),
            nn.Linear(config.hidden_size, config.intermediate_size, bias=False),
            nn.GELU(),
            nn.Linear(config.intermediate_size, config.hidden_size, bias=False),
        )

    def forward(self, features: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
        latents = latents + self.attention(features, latents)
        return latents + self.feed_forward(latents)


class ResamplerConnector(BaseConnectorModel):
    config_class = ResamplerConnectorConfig

    def __init__(
        self,
        config: ResamplerConnectorConfig,
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
        self.latents = nn.Parameter(torch.randn(config.num_queries, config.hidden_size))
        self.input_projection = nn.Linear(
            self.vision_hidden_size * self.num_vision_feature_layers, config.hidden_size
        )
        self.layers = nn.ModuleList(
            [ResamplerLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = nn.LayerNorm(config.hidden_size)
        self.output_projection = nn.Linear(config.hidden_size, self.text_hidden_size)

    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        features = self.input_projection(vision_features)
        latents = self.latents.unsqueeze(0).expand(features.shape[0], -1, -1)
        for layer in self.layers:
            latents = layer(features, latents)
        return self.output_projection(self.norm(latents))


__all__ = ["ResamplerConnector"]

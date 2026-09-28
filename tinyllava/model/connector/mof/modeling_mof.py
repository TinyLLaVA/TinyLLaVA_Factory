"""Project each encoder independently, then interleave corresponding tokens."""

from __future__ import annotations

import torch

from ..mlp.configuration_mlp import MLPConnectorConfig
from ..mlp.modeling_mlp import MLPConnector
from ..modeling_base import BaseConnectorModel
from .configuration_mof import MofConnectorConfig


class MofConnector(BaseConnectorModel):
    config_class = MofConnectorConfig

    def __init__(
        self,
        config: MofConnectorConfig,
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
        self.vision_hidden_sizes = config.vision_hidden_sizes
        if sum(self.vision_hidden_sizes) != self.vision_hidden_size:
            raise ValueError("vision_hidden_sizes must sum to vision_hidden_size")
        branch_config = MLPConnectorConfig(
            depth=config.depth,
            act=config.act,
            bias=config.bias,
            hidden_size=config.hidden_size,
        )
        self.clip = MLPConnector(
            branch_config,
            vision_hidden_size=self.vision_hidden_sizes[0],
            text_hidden_size=text_hidden_size,
            vision_feature_layer=vision_feature_layer,
        )
        self.dinov2 = MLPConnector(
            branch_config,
            vision_hidden_size=self.vision_hidden_sizes[1],
            text_hidden_size=text_hidden_size,
            vision_feature_layer=vision_feature_layer,
        )

    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        # Multiple selected layers are packed as [clip_l1, dino_l1, clip_l2, ...].
        layers = vision_features.split(self.vision_hidden_size, dim=-1)
        branches = [layer.split(self.vision_hidden_sizes, dim=-1) for layer in layers]
        clip = self.clip(torch.cat([branch[0] for branch in branches], dim=-1))
        dinov2 = self.dinov2(torch.cat([branch[1] for branch in branches], dim=-1))
        return torch.stack((clip, dinov2), dim=2).flatten(1, 2)


__all__ = ["MofConnector"]

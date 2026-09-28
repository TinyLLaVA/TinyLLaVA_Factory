"""Parameter-free connector for matching feature widths."""

from __future__ import annotations

import torch

from ..modeling_base import BaseConnectorModel
from .configuration_identity import IdentityConnectorConfig


class IdentityConnector(BaseConnectorModel):
    config_class = IdentityConnectorConfig

    def __init__(
        self,
        config: IdentityConnectorConfig,
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
        if (
            self.vision_hidden_size * self.num_vision_feature_layers
            != self.text_hidden_size
        ):
            raise ValueError(
                "Identity connector requires matching vision and text feature widths."
            )

    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        return vision_features


__all__ = ["IdentityConnector"]

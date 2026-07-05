"""MLP connector model."""

import warnings

from torch import nn
from transformers.activations import ACT2FN

from ..modeling_base import BaseConnectorModel
from .configuration_mlp import MLPConnectorConfig


class MLPConnector(BaseConnectorModel):
    config_class = MLPConnectorConfig

    def __init__(
        self,
        config: MLPConnectorConfig,
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

        hidden_size = config.hidden_size or self.text_hidden_size
        if config.depth >= 3 and config.hidden_size is None:
            warnings.warn(
                f"MLP connector depth is {config.depth} but hidden_size is not set; "
                f"defaulting intermediate layers to text_hidden_size={self.text_hidden_size}.",
                UserWarning,
                stacklevel=2,
            )

        self.layers = nn.ModuleList()
        self.activation = ACT2FN[config.act]

        input_size = self.vision_hidden_size * self.num_vision_feature_layers
        if config.depth == 1:
            self.layers.append(nn.Linear(input_size, self.text_hidden_size, bias=config.bias))
        else:
            self.layers.append(nn.Linear(input_size, hidden_size, bias=config.bias))
            for _ in range(1, config.depth - 1):
                self.layers.append(nn.Linear(hidden_size, hidden_size, bias=config.bias))
            self.layers.append(nn.Linear(hidden_size, self.text_hidden_size, bias=config.bias))

    def forward(self, vision_features):
        vision_features = self.layers[0](vision_features)
        for layer in self.layers[1:]:
            vision_features = self.activation(vision_features)
            vision_features = layer(vision_features)
        return vision_features


__all__ = ["MLPConnector"]

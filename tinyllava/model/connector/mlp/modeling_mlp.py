"""MLP connector model."""

from torch import nn
from transformers import PreTrainedModel
from transformers.activations import ACT2FN

from .configuration_mlp import MLPConnectorConfig


class MLPConnector(PreTrainedModel):
    def __init__(self, config: MLPConnectorConfig):
        super().__init__(config)

        if config.depth < 1:
            raise ValueError("config.depth must be at least 1")
        num_feature_layers = 1 if isinstance(config.vision_feature_layer, int) else len(config.vision_feature_layer)

        self.layers = nn.ModuleList()
        self.activation = ACT2FN[config.act]

        self.layers.append(nn.Linear(config.vision_hidden_size * num_feature_layers, config.text_hidden_size, bias=config.bias))
        for _ in range(1, config.depth):
            self.layers.append(nn.Linear(config.text_hidden_size, config.text_hidden_size, bias=config.bias))

    def forward(self, vision_features):
        vision_features = self.layers[0](vision_features)
        for layer in self.layers[1:]:
            vision_features = self.activation(vision_features)
            vision_features = layer(vision_features)
        return vision_features


__all__ = ["MLPConnector"]

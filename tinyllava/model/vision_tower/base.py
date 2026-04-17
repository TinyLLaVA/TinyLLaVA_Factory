import os
import warnings
import torch
import torch.nn as nn
from transformers import PreTrainedModel, BaseImageProcessor


# TODO: (incompatible change) cancel the improper inheritance from nn.Module
class VisionTower(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self._vision_tower_cls: type[PreTrainedModel] | None = None
        self._vision_tower: nn.Module | None = None
        self._image_processor: BaseImageProcessor | None = None
        self.config = cfg

    def load_model(self, vision_tower_name, **kwargs):
        if self._vision_tower is PreTrainedModel:
            warnings.warn(
                "vision_tower has already been loaded; skip repeated load_model call.",
                UserWarning,
                stacklevel=2,
            )
            return
        self._load_model(vision_tower_name, **kwargs)
        # TODO: should refer to the config for whether to freeze the vision tower
        self.vision_tower.requires_grad_(False)

    def _load_model(self, vision_tower_name, pretrained_vision_tower_path=None, **kwargs):
        if self._vision_tower_cls is not None:
            load_name = pretrained_vision_tower_path or vision_tower_name
            self._vision_tower = self._vision_tower_cls.from_pretrained(load_name, **kwargs)
        # TODO: any case could touch this branch? should be tested and possibly removed in the future
        elif self._vision_tower is not None and isinstance(self._vision_tower, nn.Module):
            if pretrained_vision_tower_path is not None:
                vision_tower_weights = torch.load(
                    os.path.join(pretrained_vision_tower_path, "pytorch_model.bin"),
                    map_location="cpu",
                )
                self._vision_tower.load_state_dict(vision_tower_weights)
        else:
            raise ValueError("vision_tower is not initialized and _vision_tower_cls is not set")

        print("Loading vision tower from ", pretrained_vision_tower_path or vision_tower_name)

    def forward(self, x, **kwargs):
        image_features = self.vision_tower(x, output_hidden_states=True)
        image_features = image_features.hidden_states[
            kwargs.get("vision_feature_layer", -2)
        ]

        # TODO: fix #203
        if kwargs.get("vision_feature_select_strategy", "patch") == "patch":
            image_features = image_features[:, 1:]
        elif kwargs.get("vision_feature_select_strategy", "patch") == "cls_patch":
            image_features = image_features
        else:
            raise ValueError(
                f"Unexpected select feature: {kwargs.get('vision_feature_select_strategy')}"
            )

        return image_features

    @property
    def vision_tower(self) -> nn.Module:
        if self._vision_tower is None:
            raise RuntimeError("vision_tower is not initialized")
        return self._vision_tower

    @property
    def image_processor(self) -> BaseImageProcessor:
        if self._image_processor is None:
            raise RuntimeError("image_processor is not initialized")
        return self._image_processor

    @image_processor.setter
    def image_processor(self, image_processor: BaseImageProcessor):
        self._image_processor = image_processor

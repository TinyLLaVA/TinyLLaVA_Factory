from transformers import SiglipVisionModel, SiglipVisionConfig, SiglipImageProcessor

from . import register_vision_tower
from .base import VisionTower


@register_vision_tower('siglip')
class SIGLIPVisionTower(VisionTower):
    def __init__(self, cfg):
        super().__init__(cfg)
        self._vision_tower = SiglipVisionModel(cfg)
        self._image_processor = SiglipImageProcessor.from_pretrained(cfg.model_name_or_path)

    def forward(self, x, **kwargs):
        """
        Forward pass for SigLIP vision tower.

        Unlike CLIP-based models, SigLIP does NOT have a [CLS] token at index 0.
        All tokens are image patch embeddings, so we should NOT remove the first token.

        See: https://github.com/TinyLLaVA/TinyLLaVA_Factory/issues/203
        """
        image_features = self._vision_tower(x, output_hidden_states=True)
        image_features = image_features.hidden_states[kwargs.get('vision_feature_layer', -2)]

        # SigLIP has no [CLS] token - all tokens are patch embeddings
        # Do NOT slice off the first token regardless of vision_feature_select_strategy
        # For 'patch' strategy: return all patch tokens (all tokens for SigLIP)
        # For 'cls_patch' strategy: return all tokens (no CLS to prepend)

        return image_features

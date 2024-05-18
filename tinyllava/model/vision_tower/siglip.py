from transformers import SiglipVisionModel, SiglipVisionConfig, SiglipImageProcessor

from . import register_vision_tower
from .base import VisionTower


@register_vision_tower('siglip')      
class SIGLIPVisionTower(VisionTower):
    def __init__(self, cfg):
        super().__init__(cfg)
        self._vision_tower = SiglipVisionModel(cfg)
        self._image_processor = SiglipImageProcessor.from_pretrained(cfg.model_name_or_path)
        
        
#    def forward(self, x, **kwargs):
#        image_features = self._vision_tower(x, output_hidden_states=True)
#        image_features = image_features.hidden_states[kwargs.get('vision_feature_layer', -2)]


#        return image_features

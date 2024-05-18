from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig

from . import register_vision_tower
from .base import VisionTower


@register_vision_tower('clip')      
class CLIPVisionTower(VisionTower):
    def __init__(self, cfg):
        super().__init__(cfg)
        self._vision_tower = CLIPVisionModel(cfg)
        self._image_processor = CLIPImageProcessor.from_pretrained(cfg.model_name_or_path)
  


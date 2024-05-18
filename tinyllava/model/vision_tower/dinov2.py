from transformers import Dinov2Model, AutoImageProcessor

from . import register_vision_tower
from .base import VisionTower


@register_vision_tower('dinov2')      
class DINOv2VisionTower(VisionTower):
    def __init__(self, cfg):
        super().__init__(cfg)
        self._vision_tower = Dinov2Model(cfg)
        self._image_processor = AutoImageProcessor.from_pretrained(cfg.model_name_or_path)
  

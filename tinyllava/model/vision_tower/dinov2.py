from transformers import Dinov2Model, AutoImageProcessor

from . import register_vision_tower
from .base import VisionTower


@register_vision_tower("dinov2")
class DINOv2VisionTower(VisionTower):
    _vision_tower_cls = Dinov2Model
    _image_processor_cls = AutoImageProcessor

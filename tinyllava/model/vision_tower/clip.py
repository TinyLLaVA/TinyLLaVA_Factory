from transformers import CLIPVisionModel, CLIPImageProcessor

from . import register_vision_tower
from .base import VisionTower


@register_vision_tower("clip")
class CLIPVisionTower(VisionTower):
    _vision_tower_cls = CLIPVisionModel
    _image_processor_cls = CLIPImageProcessor

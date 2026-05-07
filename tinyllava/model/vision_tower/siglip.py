from transformers import SiglipVisionModel, SiglipImageProcessor

from . import register_vision_tower
from .base import VisionTower


@register_vision_tower("siglip")
class SIGLIPVisionTower(VisionTower):
    _vision_tower_cls = SiglipVisionModel
    _image_processor_cls = SiglipImageProcessor

import os
import torch
import torch.nn as nn
from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig, Dinov2Model, AutoConfig

from . import register_vision_tower
from .base import VisionTower





class MoF(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.clip = CLIPVisionModel(cfg)

        cfg_dinov2 = AutoConfig.from_pretrained(cfg.model_name_or_path2)
        self.dinov2 = Dinov2Model(cfg_dinov2)


#     def enable_input_require_grads(self):
#         def make_inputs_require_grad(module, input, output):
#             output.requires_grads()

#         if hasattr(self.clip, 'enable_input_require_grads'):
#             self.clip.enable_input_require_grads()
#         else:
#             self.clip.get_input_embeddings(make_inputs_require_grad)

#         if hasattr(self.dinov2, 'enable_input_require_grads'):
#             self.dinov2.enable_input_require_grads()
#         else:
#             self.dinov2.get_input_embeddings(make_inputs_require_grad)


    def forward(self, x, **kwargs):
 
        image_features_clip = self.clip(x, output_hidden_states=True)
        image_features_clip = image_features_clip.hidden_states[kwargs.get('vision_feature_layer', -2)]

        image_features_dinov2 = self.dinov2(x, output_hidden_states=True)
        image_features_dinov2 = image_features_dinov2.hidden_states[kwargs.get('vision_feature_layer', -2)]

        if kwargs.get('vision_feature_select_strategy', 'patch') == 'patch':
            image_features_clip = image_features_clip[:, 1:]
            image_features_dinov2 = image_features_dinov2[:, 1:]
        elif kwargs.get('vision_feature_select_strategy', 'patch') == 'cls_patch':
            image_features_clip = image_features_clip
            image_features_dinov2 = image_features_dinov2
        else:
            raise ValueError(f"Unexpected select feature: {kwargs.get('vision_feature_select_strategy')}")


        image_features = image_features_clip, image_features_dinov2

        return image_features






@register_vision_tower('mof')      
class MoFVisionTower(VisionTower):
    def __init__(self, cfg):
        super().__init__(cfg)

        self._vision_tower = MoF(cfg)

        self._image_processor = CLIPImageProcessor.from_pretrained(cfg.model_name_or_path)
  

    def _load_model(self, vision_tower_name, **kwargs):
        pretrained_vision_tower_path = kwargs.pop('pretrained_vision_tower_path', None)
        if pretrained_vision_tower_path is None:
            model_name_or_path_dinov2 = kwargs.pop('model_name_or_path2')
            self._vision_tower.clip = self._vision_tower.clip.from_pretrained(vision_tower_name, **kwargs)
            self._vision_tower.dinov2 = self._vision_tower.dinov2.from_pretrained(model_name_or_path_dinov2, **kwargs)
            print("Loading vision tower1 from ", vision_tower_name)
            print("Loading vision tower2 from ", model_name_or_path_dinov2)
        else: # nn.Module
            if pretrained_vision_tower_path is not None:
                vision_tower_weights = torch.load(os.path.join(pretrained_vision_tower_path, 'pytorch_model.bin'), map_location='cpu')
                def get_w(weights, keyword):
                    return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}
                self._vision_tower.load_state_dict(vision_tower_weights)
            print("Loading vision tower from ", pretrained_vision_tower_path)


    def forward(self, x, **kwargs):
        device = x.data.device
        self.to(device)
        return self._vision_tower(x, **kwargs)



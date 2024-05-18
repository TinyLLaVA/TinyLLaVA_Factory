import torch
import torch.nn as nn

from . import register_connector
from .base import Connector


    
       
class MoFMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        modules_clip = [nn.Linear(config.vision_hidden_size, config.hidden_size), 
                    nn.GELU(),
                    nn.Linear(config.hidden_size, config.hidden_size)
                    ]

        modules_dinov2 = [nn.Linear(config.vision_hidden_size, config.hidden_size), 
                    nn.GELU(),
                    nn.Linear(config.hidden_size, config.hidden_size)
                    ]

        self.clip = nn.Sequential(*modules_clip)
        self.dinov2 = nn.Sequential(*modules_dinov2)



    def forward(self, x):

        image_features_clip = self.clip(x[0])
        image_features_dinov2 = self.dinov2(x[1])

        bs = image_features_clip.size(0)
        total_len = image_features_clip.size(1)+image_features_dinov2.size(1)
        dim = image_features_clip.size(-1)

        merged_features = torch.empty(bs, total_len, dim).to(device=x[0].device, dtype=x[0].dtype)
        merged_features[:,0::2] = image_features_clip
        merged_features[:,1::2] = image_features_dinov2

        return merged_features
    
    

    
@register_connector('mof_mlp')    
class MoFMLPConnector(Connector):
    def __init__(self, config):
        super().__init__()

        self._connector = MoFMLP(config)

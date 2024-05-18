import re

import torch.nn as nn

from . import register_connector
from .base import Connector


ACT_TYPE = {
    'relu': nn.ReLU,
    'gelu': nn.GELU
}


    
    
@register_connector('mlp')    
class MLPConnector(Connector):
    def __init__(self, config):
        super().__init__()
        
        mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', config.connector_type)
        act_type = config.connector_type.split('_')[-1]
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.vision_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(ACT_TYPE[act_type]())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
            
        self._connector = nn.Sequential(*modules)

   
        
#     @property
#     def config(self):
#         return {"connector_type": 'mlp',
#                 "in_hidden_size": self.in_hidden_size, 
#                 "out_hidden_size": self.out_hidden_size
#                }
    

import torch.nn as nn

from . import register_connector
from .base import Connector



    
    
@register_connector('linear')    
class LinearConnector(Connector):
    def __init__(self, config):
        super().__init__()
        self._connector =  nn.Linear(config.vision_hidden_size, config.hidden_size)

        
    # @property
    # def config(self):
    #     return {"connector_type": 'linear',
    #             "in_hidden_size": self.in_hidden_size, 
    #             "out_hidden_size": self.out_hidden_size
    #            }

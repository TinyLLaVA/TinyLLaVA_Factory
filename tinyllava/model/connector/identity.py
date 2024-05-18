import torch.nn as nn

from . import register_connector
from .base import Connector


    
@register_connector('identity')    
class IdentityConnector(Connector):
    def __init__(self, config=None):
        super().__init__()
        self._connector = nn.Identity()
        
        
    

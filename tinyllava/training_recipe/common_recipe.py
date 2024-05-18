import os

import torch

from .base import BaseTrainingRecipe
from . import register_training_recipe
from ..utils import log
from ..utils import get_state_maybe_zero_3
from ..model import TinyLlavaConfig, TinyLlavaForConditionalGeneration


@register_training_recipe('common')
class CommonTrainingRecipe(BaseTrainingRecipe):
    ... 

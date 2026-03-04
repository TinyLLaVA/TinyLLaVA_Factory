from .base import BaseTrainingRecipe
from . import register_training_recipe


@register_training_recipe("common")
class CommonTrainingRecipe(BaseTrainingRecipe): ...

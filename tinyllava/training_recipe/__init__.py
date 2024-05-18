import os

from ..utils import import_modules


RECIPE_FACTORY = {}

def TrainingRecipeFactory(training_recipe):
    recipe = None
    for name in RECIPE_FACTORY.keys():
        if name.lower() == training_recipe.lower():
            recipe = RECIPE_FACTORY[name]
    assert recipe, f"{training_recipe} is not registered"
    return recipe


def register_training_recipe(name):
    def register_training_recipe_cls(cls):
        if name in RECIPE_FACTORY:
            return RECIPE_FACTORY[name]
        RECIPE_FACTORY[name] = cls
        return cls
    return register_training_recipe_cls


models_dir = os.path.dirname(__file__)
import_modules(models_dir, "tinyllava.training_recipe")

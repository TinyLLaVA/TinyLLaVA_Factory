import importlib
import os

def import_modules(models_dir, namespace):
    for file in os.listdir(models_dir):
        path = os.path.join(models_dir, file)
        if (
            not file.startswith("_")
            and not file.startswith(".")
            and file.endswith(".py")
        ):
            model_name = file[: file.find(".py")] if file.endswith(".py") else file
            importlib.import_module(namespace + "." + model_name)
import os

from ...utils import import_modules


LLM_FACTORY = {}

def LLMFactory(model_name_or_path):
    model, tokenizer_and_post_load = None, None
    for name in LLM_FACTORY.keys():
        if name in model_name_or_path.lower():
            model, tokenizer_and_post_load = LLM_FACTORY[name]()
    assert model, f"{model_name_or_path} is not registered"
    return model, tokenizer_and_post_load


def register_llm(name):
    def register_llm_cls(cls):
        if name in LLM_FACTORY:
            return LLM_FACTORY[name]
        LLM_FACTORY[name] = cls
        return cls
    return register_llm_cls


# automatically import any Python files in the models/ directory
models_dir = os.path.dirname(__file__)
import_modules(models_dir, "tinyllava.model.llm")

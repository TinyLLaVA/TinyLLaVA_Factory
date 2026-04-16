import os
from typing import Callable, TypeVar
from transformers import PreTrainedModel, AutoTokenizer
from transformers.generation.utils import GenerationMixin
from ...utils import import_modules

P = TypeVar("P")

PostLoadCallable = Callable[[P], P]


class PreTrainedModelWithGenerationMixin(PreTrainedModel, GenerationMixin):
    pass


ModelAndTokenizer = tuple[
    type[PreTrainedModelWithGenerationMixin],
    tuple[type[AutoTokenizer], PostLoadCallable],
]

ReturnLlmCallable = Callable[[], ModelAndTokenizer]

LLM_FACTORY: dict[str, ReturnLlmCallable] = {}


def LLMFactory(model_name_or_path: str) -> ModelAndTokenizer:
    model, tokenizer_and_post_load = None, None
    for name in LLM_FACTORY.keys():
        if name in model_name_or_path.lower():
            if model is not None:
                raise ValueError(
                    f"Multiple LLMs found for {model_name_or_path}, "
                    "please specify the model name more precisely"
                )
            model, tokenizer_and_post_load = LLM_FACTORY[name]()
    if not model or not tokenizer_and_post_load:
        raise ValueError(f"{model_name_or_path} is not registered")
    return model, tokenizer_and_post_load


def register_llm(
    name: str,
) -> Callable[[ReturnLlmCallable], ReturnLlmCallable]:
    def register_llm_cls(fn: ReturnLlmCallable) -> ReturnLlmCallable:
        if name in LLM_FACTORY:
            raise ValueError(f"{name} is already registered")
        LLM_FACTORY[name] = fn
        return fn

    return register_llm_cls


# automatically import any Python files in the models/ directory
models_dir = os.path.dirname(__file__)
import_modules(models_dir, "tinyllava.model.llm")

"""PEFT-based training strategies."""

from typing import cast

import torch
from peft import LoraConfig, get_peft_model
from transformers import BitsAndBytesConfig

from tinyllava.model.modeling_tinyllava import TinyLlavaForConditionalGeneration
from tinyllava.train.strategy.base import BaseTrainingStrategy
from tinyllava.utils.logging import log
from tinyllava.utils.precision import training_torch_dtype


class LoraTrainingStrategy(BaseTrainingStrategy):
    supported_llm_tune_types = ("frozen", "full", "lora")
    supported_vision_tower_tune_types = ("frozen", "full", "partially-tune", "lora")
    supported_connector_tune_types = ("frozen", "full", "lora")

    lora_tune_type = "lora"
    always_add_adapter = False

    def prepare_model(
        self,
        model: TinyLlavaForConditionalGeneration,
    ) -> TinyLlavaForConditionalGeneration:
        lora_kwargs = dict(getattr(self.training_arguments, "peft_config", {}))
        if "target_modules" not in lora_kwargs:
            lora_kwargs["target_modules"] = _default_lora_target_modules(
                model,
                self.lora_skip_modules(),
            )
        lora_kwargs.setdefault("task_type", "CAUSAL_LM")
        lora_config = LoraConfig(
            **lora_kwargs,
        )
        if self.always_add_adapter or getattr(model, "peft_config", None) is None:
            log("Adding LoRA adapters...")
            model = cast(TinyLlavaForConditionalGeneration, get_peft_model(model, lora_config))
        return model

    def lora_skip_modules(self) -> list[str]:
        module_tune_types = {
            "multi_modal_projector": self.training_arguments.tune_type_connector,
            "vision_tower": self.training_arguments.tune_type_vision_tower,
            "language_model": self.training_arguments.tune_type_llm,
        }
        return [
            module_name
            for module_name, tune_type in module_tune_types.items()
            if tune_type != self.lora_tune_type
        ]


class LoraInt8TrainingStrategy(LoraTrainingStrategy):
    lora_tune_type = "lora"
    always_add_adapter = True

    def language_model_loading_kwargs(self) -> dict:
        torch_dtype = training_torch_dtype(self.training_arguments)
        return {
            "torch_dtype": torch_dtype,
            "low_cpu_mem_usage": True,
            "quantization_config": BitsAndBytesConfig(load_in_8bit=True),
        }


def _default_lora_target_modules(
    model: TinyLlavaForConditionalGeneration,
    skip_keywords: list[str],
) -> list[str]:
    target_modules: set[str] = set()
    for name, module in model.named_modules():
        if any(skip_keyword in name for skip_keyword in skip_keywords):
            continue
        if _is_output_head(name):
            continue
        if isinstance(module, torch.nn.Linear):
            target_modules.add(name)
    return sorted(target_modules)


def _is_output_head(module_name: str) -> bool:
    parts = module_name.split(".")
    return any(part in {"lm_head", "output_layer", "head"} for part in parts)

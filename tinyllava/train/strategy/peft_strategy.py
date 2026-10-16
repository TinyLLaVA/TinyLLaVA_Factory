"""PEFT-based training strategies."""

from typing import cast

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from peft.utils.other import ModulesToSaveWrapper
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
        lora_kwargs = dict(self.tuning.peft)
        if "target_modules" not in lora_kwargs:
            lora_kwargs["target_modules"] = _default_lora_target_modules(
                model,
                self.lora_skip_modules(),
            )
        # PEFT checkpoints must also retain components trained without adapters.
        full_modules = {
            "language_model": self.tuning.tune_type_llm,
            "lm_head": self.tuning.tune_type_llm,
            "vision_tower": self.tuning.tune_type_vision_tower,
            "multi_modal_projector": self.tuning.tune_type_connector,
        }
        lora_kwargs["modules_to_save"] = sorted(
            set(lora_kwargs.get("modules_to_save") or [])
            | {
                name
                for name, policy in full_modules.items()
                if policy in {"full", "partially-tune"}
            }
        )
        lora_kwargs.setdefault("task_type", "CAUSAL_LM")
        lora_config = LoraConfig(
            **lora_kwargs,
        )
        if self.always_add_adapter or getattr(model, "peft_config", None) is None:
            log("Adding LoRA adapters...")
            model = cast(
                TinyLlavaForConditionalGeneration, get_peft_model(model, lora_config)
            )
        return model

    def apply_tuning_policy(self, model: PeftModel) -> PeftModel:
        super().apply_tuning_policy(model.get_base_model())
        # Only the active saved copy participates in forward; keep its base frozen.
        for module in model.modules():
            if isinstance(module, ModulesToSaveWrapper):
                module.original_module.requires_grad_(False)
        return model

    def lora_skip_modules(self) -> list[str]:
        module_tune_types = {
            "multi_modal_projector": self.tuning.tune_type_connector,
            "vision_tower": self.tuning.tune_type_vision_tower,
            "language_model": self.tuning.tune_type_llm,
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
            "quantization_config": BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_skip_modules=[
                    "lm_head",
                    "model.vision_tower",
                    "model.multi_modal_projector",
                ],
            ),
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

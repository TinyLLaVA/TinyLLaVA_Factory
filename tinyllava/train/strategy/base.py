from transformers import Trainer

from tinyllava.model.modeling_tinyllava import TinyLlavaForConditionalGeneration
from tinyllava.utils.arguments import TrainingArguments
from tinyllava.utils.logging import log
from tinyllava.utils.precision import training_torch_dtype


class BaseTrainingStrategy:
    supported_llm_tune_types = ("frozen", "full")
    supported_vision_tower_tune_types = ("frozen", "full", "partially-tune")
    supported_connector_tune_types = ("frozen", "full")

    def __init__(self, training_arguments: TrainingArguments):
        self.training_arguments = training_arguments

    def language_model_loading_kwargs(self) -> dict:
        return {"torch_dtype": training_torch_dtype(self.training_arguments)}

    def __call__(
        self,
        model: TinyLlavaForConditionalGeneration,
    ) -> TinyLlavaForConditionalGeneration:
        model = self.prepare_model(model)
        model = self.apply_tuning_policy(model)
        return model

    def apply_tuning_policy(
        self,
        model: TinyLlavaForConditionalGeneration,
    ) -> TinyLlavaForConditionalGeneration:
        model = self._set_llm_tuning(model)
        model = self._set_vision_tower_tuning(model)
        model = self._set_connector_tuning(model)
        return model

    def _set_llm_tuning(
        self,
        model: TinyLlavaForConditionalGeneration,
    ) -> TinyLlavaForConditionalGeneration:
        tune_type = self.training_arguments.tune_type_llm.lower()
        self._require_tune_type(
            tune_type,
            component="LLM",
            supported=self.supported_llm_tune_types,
        )
        if tune_type == "full":
            model.model.language_model.requires_grad_(True)
        elif tune_type == "frozen":
            model.model.language_model.requires_grad_(False)
        return model

    def _set_vision_tower_tuning(
        self,
        model: TinyLlavaForConditionalGeneration,
    ) -> TinyLlavaForConditionalGeneration:
        tune_type = self.training_arguments.tune_type_vision_tower.lower()
        self._require_tune_type(
            tune_type,
            component="vision tower",
            supported=self.supported_vision_tower_tune_types,
        )
        if tune_type == "full":
            model.model.vision_tower.requires_grad_(True)
        elif tune_type == "frozen":
            model.model.vision_tower.requires_grad_(False)
        elif tune_type == "partially-tune":
            from_layer = self.training_arguments.tune_vision_tower_from_layer
            if from_layer is None:
                raise ValueError(
                    "tune_vision_tower_from_layer must be set when "
                    "tune_type_vision_tower='partially-tune'."
            )
            if from_layer > -1:
                log(f"Tune the vision tower from layer {from_layer}!")
                for name, parameter in model.model.vision_tower.named_parameters():
                    if "vision_model.encoder.layers." in name:
                        layer_id = int(
                            name.split("vision_model.encoder.layers.")[-1].split(".")[0]
                        )
                        parameter.requires_grad = layer_id >= from_layer
                    else:
                        parameter.requires_grad = False
        return model

    def _set_connector_tuning(
        self,
        model: TinyLlavaForConditionalGeneration,
    ) -> TinyLlavaForConditionalGeneration:
        tune_type = self.training_arguments.tune_type_connector.lower()
        self._require_tune_type(
            tune_type,
            component="connector",
            supported=self.supported_connector_tune_types,
        )
        if tune_type == "full":
            model.model.multi_modal_projector.requires_grad_(True)
        elif tune_type == "frozen":
            model.model.multi_modal_projector.requires_grad_(False)
        return model

    def prepare_model(
        self,
        model: TinyLlavaForConditionalGeneration,
    ) -> TinyLlavaForConditionalGeneration:
        return model

    def _require_tune_type(
        self,
        tune_type: str,
        *,
        component: str,
        supported: tuple[str, ...],
    ) -> None:
        if tune_type not in supported:
            supported_values = ", ".join(supported)
            raise ValueError(
                f"{component} tune_type {tune_type!r} is not supported by "
                f"{type(self).__name__}. Supported values: {supported_values}."
            )

    def save(
        self,
        model: TinyLlavaForConditionalGeneration,
        trainer: Trainer,
    ) -> None:
        trainer.save_state()
        trainer.save_model(self.training_arguments.output_dir)

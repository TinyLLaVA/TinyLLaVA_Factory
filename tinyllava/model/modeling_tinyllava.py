# Copyright 2023 the HuggingFace Inc. team and TinyLLaVA. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch TinyLlava model."""

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from transformers import (
    AutoModelForCausalLM,
    PreTrainedModel,
    GenerationMixin,
    Cache,
)
from transformers.processing_utils import Unpack
from transformers.modeling_outputs import BaseModelOutputWithPast, BaseModelOutputWithPooling
from transformers.utils.generic import (
    ModelOutput,
    TransformersKwargs,
    can_return_tuple,
    merge_with_config_defaults,
)
from transformers.utils.import_utils import torch_compilable_check

from .llm import AutoLanguageModel
from .vision_tower import AutoVisionTowerModel
from .connector import AutoConnectorModel
from .configuration_tinyllava import TinyLlavaConfig


@dataclass
class TinyLlavaModelOutputWithPast(BaseModelOutputWithPast):
    r"""
    past_key_values (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
        It is a [`~cache_utils.Cache`] instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

        Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
        `past_key_values` input) to speed up sequential decoding.
    image_hidden_states (`torch.FloatTensor`, *optional*):
        A `torch.FloatTensor` of size `(batch_size, num_images, sequence_length, hidden_size)`.
        image_hidden_states of the model produced by the vision encoder and after projecting the last hidden state.
    """

    image_hidden_states: torch.FloatTensor | None = None


@dataclass
class TinyLlavaCausalLMOutputWithPast(ModelOutput):
    r"""
    loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
        Language modeling loss (for next-token prediction).
    logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
        Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
    past_key_values (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
        It is a [`~cache_utils.Cache`] instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

        Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
        `past_key_values` input) to speed up sequential decoding.
    image_hidden_states (`torch.FloatTensor`, *optional*):
        A `torch.FloatTensor` of size `(batch_size, num_images, sequence_length, hidden_size)`.
        image_hidden_states of the model produced by the vision encoder and after projecting the last hidden state.
    """

    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    past_key_values: Cache | None = None
    hidden_states: tuple[torch.FloatTensor] | None = None
    attentions: tuple[torch.FloatTensor] | None = None
    image_hidden_states: torch.FloatTensor | None = None


class TinyLlavaPreTrainedModel(PreTrainedModel):
    config_class = TinyLlavaConfig
    base_model_prefix = "model"
    input_modalities = ["image", "text"]
    supports_gradient_checkpointing = True
    _skip_keys_device_placement = ["past_key_values"]

    _supports_flash_attn = True
    _supports_sdpa = True

    _can_compile_fullgraph = True
    _supports_flex_attn = True
    _supports_attention_backend = True


def build_connector(config: TinyLlavaConfig) -> PreTrainedModel:
    return AutoConnectorModel.from_config(
        config.connector_config,
        vision_hidden_size=config.vision_config.hidden_size,
        text_hidden_size=config.text_config.hidden_size,
        vision_feature_layer=config.vision_feature_layer,
    )


class TinyLlavaModel(TinyLlavaPreTrainedModel):
    def __init__(
        self,
        config: TinyLlavaConfig,
        *,
        language_model: PreTrainedModel | None = None,
        vision_tower: PreTrainedModel | None = None,
        multi_modal_projector: PreTrainedModel | None = None,
    ):
        super().__init__(config)

        self.language_model = (
            language_model
            if language_model is not None
            else AutoLanguageModel.from_config(config.text_config)
        )
        self.vision_tower = (
            vision_tower
            if vision_tower is not None
            else AutoVisionTowerModel.from_config(config.vision_config)
        )
        self.multi_modal_projector = (
            multi_modal_projector
            if multi_modal_projector is not None
            else build_connector(config)
        )

        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    @merge_with_config_defaults
    @can_return_tuple
    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        vision_feature_layer: int | list[int] | None = None,
        vision_feature_select_strategy: str | None = None,
        output_hidden_states: bool | None = None,
        image_sizes: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        # this is not memory efficient at all (output_hidden_states=True) will save all the hidden states.
        image_outputs = self.vision_tower(
            pixel_values,
            output_hidden_states=True,  # Ignore arg on purpose
            return_dict=True,
            **kwargs,
        )

        # If we have one vision feature layer, return the corresponding hidden states,
        # otherwise, select the hidden states of each feature layer and concatenate them
        if isinstance(vision_feature_layer, int):
            selected_image_feature = image_outputs.hidden_states[vision_feature_layer]
            if vision_feature_select_strategy == "default":
                selected_image_feature = selected_image_feature[:, 1:]
        else:
            hs_pool = [image_outputs.hidden_states[layer_idx] for layer_idx in vision_feature_layer]
            # For default; crop CLS from each hidden state in the hidden state pool
            if vision_feature_select_strategy == "default":
                hs_pool = [hs[:, 1:] for hs in hs_pool]
            selected_image_feature = torch.cat(hs_pool, dim=-1)

        image_features = self.multi_modal_projector(selected_image_feature)

        # If image_sizes is provided, we need to split the image features accordingly,
        # but only if the image_sizes is not None (the default in this and related architectures)
        if image_sizes is not None:
            split_sizes = (
                (torch.as_tensor(image_sizes, device=image_features.device) // self.vision_tower.patch_size)
                .prod(dim=-1)
                .tolist()
            )
            image_features = torch.split(image_features.squeeze(0), split_sizes)
        else:
            image_features = list(image_features)
        image_outputs.pooler_output = image_features

        return image_outputs

    def get_placeholder_mask(
        self, input_ids: torch.LongTensor, inputs_embeds: torch.FloatTensor, image_features: torch.FloatTensor
    ):
        """
        Obtains multimodal placeholder mask from `input_ids` or `inputs_embeds`, and checks that the placeholder token count is
        equal to the length of multimodal features. If the lengths are different, an error is raised.
        """
        if input_ids is None:
            special_image_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_image_mask = special_image_mask.all(-1)
        else:
            special_image_mask = input_ids == self.config.image_token_id

        n_image_tokens = special_image_mask.sum()
        n_image_features = image_features.shape[0] * image_features.shape[1]
        special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        torch_compilable_check(
            inputs_embeds[special_image_mask].numel() == image_features.numel(),
            f"Image features and image tokens do not match, tokens: {n_image_tokens}, features: {n_image_features}",
        )
        return special_image_mask

    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        vision_feature_layer: int | list[int] | None = None,
        vision_feature_select_strategy: str | None = None,
        image_sizes: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | TinyLlavaModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if pixel_values is not None:
            image_features = self.get_image_features(
                pixel_values=pixel_values,
                vision_feature_layer=vision_feature_layer,
                vision_feature_select_strategy=vision_feature_select_strategy,
                image_sizes=image_sizes,
                return_dict=True,
            ).pooler_output
            image_features = torch.cat(image_features, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            special_image_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_features
            )
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

        return TinyLlavaModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_features if pixel_values is not None else None,
        )


class TinyLlavaForConditionalGeneration(TinyLlavaPreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.language_model.embed_tokens.weight"}

    def __init__(
        self,
        config: TinyLlavaConfig,
        *,
        language_model: PreTrainedModel | None = None,
        vision_tower: PreTrainedModel | None = None,
        multi_modal_projector: PreTrainedModel | None = None,
        lm_head: nn.Module | None = None,
    ):
        super().__init__(config)
        self.model = TinyLlavaModel(
            config,
            language_model=language_model,
            vision_tower=vision_tower,
            multi_modal_projector=multi_modal_projector,
        )
        self.lm_head = (
            lm_head
            if lm_head is not None
            else nn.Linear(
                config.text_config.hidden_size,
                config.text_config.vocab_size,
                bias=False,
            )
        )
        self.post_init()

    @classmethod
    def from_pretrained_components(
        cls,
        config: TinyLlavaConfig,
        *,
        language_model_name_or_path: str,
        vision_model_name_or_path: str,
        language_model_loading_kwargs: dict[str, Any] | None = None,
        vision_model_loading_kwargs: dict[str, Any] | None = None,
    ) -> "TinyLlavaForConditionalGeneration":
        """Assemble a new TinyLLaVA model from pretrained base components."""
        language_model_loading_kwargs = dict(language_model_loading_kwargs or {})
        vision_model_loading_kwargs = dict(vision_model_loading_kwargs or {})

        causal_lm = AutoModelForCausalLM.from_pretrained(
            language_model_name_or_path,
            config=config.text_config,
            **language_model_loading_kwargs,
        )
        language_model = causal_lm.base_model
        lm_head = causal_lm.get_output_embeddings()
        if language_model is causal_lm or lm_head is None:
            raise ValueError(
                f"{type(causal_lm).__name__} must expose a base model and output "
                "embeddings to initialize TinyLLaVA."
            )

        vision_tower = AutoVisionTowerModel.from_pretrained(
            vision_model_name_or_path,
            config=config.vision_config,
            **vision_model_loading_kwargs,
        )
        model = cls(
            config,
            language_model=language_model,
            vision_tower=vision_tower,
            lm_head=lm_head,
        )
        model.generation_config = causal_lm.generation_config
        return model

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head

    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        vision_feature_layer: int | list[int] | None = None,
        vision_feature_select_strategy: str | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        return self.model.get_image_features(
            pixel_values=pixel_values,
            vision_feature_layer=vision_feature_layer,
            vision_feature_select_strategy=vision_feature_select_strategy,
            **kwargs,
        )

    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        vision_feature_layer: int | list[int] | None = None,
        vision_feature_select_strategy: str | None = None,
        labels: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        image_sizes: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | TinyLlavaCausalLMOutputWithPast:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Example:

        ```python
        >>> from PIL import Image
        >>> import httpx
        >>> from io import BytesIO
        >>> from transformers import AutoProcessor, LlavaForConditionalGeneration

        >>> model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf")
        >>> processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

        >>> prompt = "USER: <image>\nWhat's the content of the image? ASSISTANT:"
        >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        >>> with httpx.stream("GET", url) as response:
        ...     image = Image.open(BytesIO(response.read()))

        >>> inputs = processor(images=image, text=prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(**inputs, max_new_tokens=15)
        >>> processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "USER:  \nWhat's the content of the image? ASSISTANT: The image features a busy city street with a stop sign prominently displayed"
        ```"""
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            vision_feature_layer=vision_feature_layer,
            vision_feature_select_strategy=vision_feature_select_strategy,
            image_sizes=image_sizes,
            **kwargs,
        )

        hidden_states = outputs[0]
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size, **kwargs
            )

        return TinyLlavaCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=outputs.image_hidden_states,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        inputs_embeds=None,
        pixel_values=None,
        attention_mask=None,
        logits_to_keep=None,
        is_first_iteration=False,
        **kwargs,
    ):
        # Overwritten -- in specific circumstances we don't want to forward image inputs to the model

        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            logits_to_keep=logits_to_keep,
            is_first_iteration=is_first_iteration,
            **kwargs,
        )

        if is_first_iteration or not kwargs.get("use_cache", True):
            # Pixel values are used only in the first iteration if available
            # In subsequent iterations, they are already merged with text and cached
            # NOTE: first iteration doesn't have to be prefill, it can be the first
            # iteration with a question and cached system prompt (continue generate from cache)
            model_inputs["pixel_values"] = pixel_values

        return model_inputs


__all__ = [
    "TinyLlavaForConditionalGeneration",
    "TinyLlavaPreTrainedModel",
    "TinyLlavaModel",
    "build_connector",
]

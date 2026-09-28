"""CLIP + DINOv2 branches for Interleaved Mixture-of-Features.

Source: Tong et al., Eyes Wide Shut? (CVPR 2024), arXiv:2401.06209.
The connector applies separate projections and interleaves the resulting tokens.
"""

from __future__ import annotations

from dataclasses import dataclass
from os import PathLike
from typing import Any

import torch
from torch import nn
from transformers import AutoModel, CLIPVisionModel, Dinov2Model, PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPooling

from .configuration_mof import MofVisionConfig


@dataclass
class MofVisionOutput(BaseModelOutputWithPooling):
    clip_attentions: tuple[torch.Tensor, ...] | None = None
    dinov2_attentions: tuple[torch.Tensor, ...] | None = None


class MofVisionModel(PreTrainedModel):
    """Return HF model outputs with CLIP/DINOv2 channels packed at each layer.

    Construction from config performs no downloads. Use
    `from_pretrained_components` to assemble pretrained branches, then
    `save_pretrained` / `AutoModel.from_pretrained` for a standalone vision tower.
    """

    config_class = MofVisionConfig
    base_model_prefix = "vision_tower"
    main_input_name = "pixel_values"
    input_modalities = ["image"]
    supports_gradient_checkpointing = True
    _supports_sdpa = True
    _supports_flash_attn = True
    _no_split_modules = ["CLIPEncoderLayer", "Dinov2Layer"]

    def __init__(
        self,
        config: MofVisionConfig,
        *,
        clip: PreTrainedModel | None = None,
        dinov2: PreTrainedModel | None = None,
    ):
        super().__init__(config)
        self.clip = (
            clip if clip is not None else AutoModel.from_config(config.clip_config)
        )
        self.dinov2 = (
            dinov2
            if dinov2 is not None
            else AutoModel.from_config(config.dinov2_config)
        )
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.clip.get_input_embeddings()

    @classmethod
    def from_pretrained_components(
        cls,
        *,
        clip_model_name_or_path: str | PathLike[str],
        dinov2_model_name_or_path: str | PathLike[str],
        clip_loading_kwargs: dict[str, Any] | None = None,
        dinov2_loading_kwargs: dict[str, Any] | None = None,
    ) -> MofVisionModel:
        clip = CLIPVisionModel.from_pretrained(
            clip_model_name_or_path, **(clip_loading_kwargs or {})
        )
        dinov2 = Dinov2Model.from_pretrained(
            dinov2_model_name_or_path, **(dinov2_loading_kwargs or {})
        )
        config = MofVisionConfig(clip_config=clip.config, dinov2_config=dinov2.config)
        return cls(config, clip=clip, dinov2=dinov2)

    def forward(
        self,
        pixel_values: torch.Tensor,
        output_hidden_states: bool | None = None,
        output_attentions: bool | None = None,
        return_dict: bool | None = None,
        **kwargs,
    ) -> MofVisionOutput | tuple[Any, ...]:
        output_hidden_states = (
            self.config.output_hidden_states
            if output_hidden_states is None
            else output_hidden_states
        )
        output_attentions = (
            self.config.output_attentions
            if output_attentions is None
            else output_attentions
        )
        return_dict = self.config.return_dict if return_dict is None else return_dict
        branch_kwargs = dict(
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=True,
            **kwargs,
        )
        clip = self.clip(pixel_values, **branch_kwargs)
        dinov2 = self.dinov2(pixel_values, **branch_kwargs)
        hidden_states = None
        if output_hidden_states:
            hidden_states = tuple(
                torch.cat((clip_state, dino_state), dim=-1)
                for clip_state, dino_state in zip(
                    clip.hidden_states, dinov2.hidden_states, strict=True
                )
            )
        output = MofVisionOutput(
            last_hidden_state=torch.cat(
                (clip.last_hidden_state, dinov2.last_hidden_state), dim=-1
            ),
            hidden_states=hidden_states,
            clip_attentions=clip.attentions,
            dinov2_attentions=dinov2.attentions,
        )
        return output if return_dict else output.to_tuple()


AutoModel.register(MofVisionConfig, MofVisionModel)

__all__ = ["MofVisionModel", "MofVisionOutput"]

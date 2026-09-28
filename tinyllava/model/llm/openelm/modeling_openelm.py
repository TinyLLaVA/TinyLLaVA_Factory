from __future__ import annotations

from typing import Any

# Copyright (C) 2024 Apple Inc. All Rights Reserved.
# Adapted from the OpenELM implementation previously maintained in this repository.
# Changes: HF config aliases, lazy registration, current cache and generation APIs.
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from transformers import AutoModel, AutoModelForCausalLM, PreTrainedModel
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.masking_utils import create_causal_mask
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)

from .configuration_openelm import OpenELMConfig, make_divisible


class OpenELMRMSNorm(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-6):
        """
        Initialize the OpenELMRMSNorm normalization layer.
        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.
        Attributes:
            eps (float): A small value added to the denominator for numerical stability.
            weight (nn.Parameter): Learnable scaling parameter.
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_features))
        self.num_features = num_features

    def _norm(self, x: Tensor) -> Tensor:
        """
        Apply the OpenELMRMSNorm normalization to the input tensor.
        Args:
            x (torch.Tensor): The input tensor.
        Returns:
            torch.Tensor: The normalized tensor.
        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the OpenELMRMSNorm layer.
        Args:
            x (torch.Tensor): The input tensor.
        Returns:
            torch.Tensor: The output tensor after applying OpenELMRMSNorm.
        """
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

    def extra_repr(self) -> str:
        return (
            super().extra_repr() + f"num_features={self.num_features}, eps={self.eps}"
        )


class OpenELMPreTrainedModel(PreTrainedModel):
    config_class = OpenELMConfig
    base_model_prefix = "transformer"
    supports_gradient_checkpointing = True
    _supports_sdpa = True
    _no_split_modules = ["OpenELMDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=self.config.initializer_range)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, OpenELMRotaryEmbedding):
            frequencies = 1.0 / (
                module.freq_constant
                ** (
                    torch.arange(
                        0, module.model_dim, 2, device=module.inv_freq.device
                    ).float()
                    / module.model_dim
                )
            )
            module.inv_freq.copy_(frequencies)
        elif isinstance(module, OpenELMRMSNorm):
            nn.init.ones_(module.weight)


class OpenELMRotaryEmbedding(nn.Module):
    def __init__(self, model_dim: int, max_seq_length: int, freq_constant: int):
        super().__init__()
        self.model_dim = model_dim
        self.freq_constant = freq_constant
        self.register_buffer(
            "inv_freq",
            1.0
            / (freq_constant ** (torch.arange(0, model_dim, 2).float() / model_dim)),
            persistent=False,
        )

    def forward(
        self, query: Tensor, key: Tensor, position_ids: Tensor
    ) -> tuple[Tensor, Tensor]:
        with torch.autocast(device_type=query.device.type, enabled=False):
            angles = position_ids.float().unsqueeze(-1) * self.inv_freq.float()
            angles = torch.cat((angles, angles), dim=-1).unsqueeze(1)
            cos, sin = angles.cos(), angles.sin()

            def rotate(x: Tensor) -> Tensor:
                first, second = x.float().chunk(2, dim=-1)
                return (x.float() * cos + torch.cat((-second, first), dim=-1) * sin).to(
                    x.dtype
                )

            return rotate(query), rotate(key)


class OpenELMMultiHeadCausalAttention(nn.Module):
    def __init__(self, config: OpenELMConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        head_dim = config.head_dim
        q_heads = config.num_query_heads[layer_idx]
        k_heads = config.num_kv_heads[layer_idx]
        v_heads = config.num_kv_heads[layer_idx]

        self.qkv_proj = nn.Linear(
            in_features=config.model_dim,
            out_features=(q_heads + k_heads + v_heads) * head_dim,
            bias=False,
        )

        self.pos_embedding = OpenELMRotaryEmbedding(
            model_dim=config.head_dim,
            max_seq_length=config.rope_max_length,
            freq_constant=config.rope_freq_constant,
        )

        if config.normalize_qk_projections:
            self.q_norm = OpenELMRMSNorm(
                num_features=config.head_dim,
            )
            self.k_norm = OpenELMRMSNorm(
                num_features=config.head_dim,
            )
        else:
            self.q_norm = None
            self.k_norm = None

        self.out_proj = nn.Linear(
            in_features=q_heads * head_dim,
            out_features=config.model_dim,
            bias=False,
        )

        self.head_dim = config.head_dim
        self.num_q_heads = q_heads
        self.num_k_heads = k_heads
        self.num_v_heads = v_heads
        self.transformer_dim = config.model_dim
        self.num_groups = self.num_q_heads // self.num_k_heads

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None,
        position_ids: Tensor,
        past_key_values: Cache | None = None,
        output_attentions: bool = False,
    ) -> tuple[Tensor, Tensor | None]:
        batch, length, _ = hidden_states.shape
        qkv = (
            self.qkv_proj(hidden_states)
            .reshape(batch, length, -1, self.head_dim)
            .transpose(1, 2)
        )
        query, key, value = qkv.split(
            (self.num_q_heads, self.num_k_heads, self.num_v_heads), dim=1
        )
        if self.q_norm is not None:
            query = self.q_norm(query)
            key = self.k_norm(key)
        query, key = self.pos_embedding(query, key, position_ids)
        if past_key_values is not None:
            key, value = past_key_values.update(key, value, self.layer_idx)
        key = key.repeat_interleave(self.num_groups, dim=1)
        value = value.repeat_interleave(self.num_groups, dim=1)
        weights = None
        if output_attentions:
            scores = query @ key.transpose(-1, -2) * self.head_dim**-0.5
            if attention_mask is not None:
                if attention_mask.dtype == torch.bool:
                    scores = scores.masked_fill(
                        ~attention_mask, torch.finfo(scores.dtype).min
                    )
                else:
                    scores = scores + attention_mask
            elif length > 1:
                mask = torch.ones(
                    length, key.shape[-2], device=query.device, dtype=torch.bool
                ).tril(diagonal=key.shape[-2] - length)
                scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
            weights = scores.float().softmax(dim=-1).to(query.dtype)
            result = weights @ value
        else:
            result = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attention_mask,
                is_causal=attention_mask is None and length > 1,
            )
        return self.out_proj(result.transpose(1, 2).reshape(batch, length, -1)), weights


class OpenELMFeedForwardNetwork(nn.Module):
    def __init__(self, config: OpenELMConfig, layer_idx: int):
        super().__init__()
        ffn_multiplier = config.ffn_multipliers[layer_idx]
        intermediate_dim = int(
            make_divisible(
                ffn_multiplier * config.model_dim,
                divisor=config.ffn_dim_divisor,
            )
        )
        if config.ffn_with_glu:
            # FFN with Gated linear unit, as described in https://arxiv.org/abs/2002.05202v1.
            self.proj_1 = nn.Linear(
                in_features=config.model_dim,
                out_features=2 * intermediate_dim,
                bias=False,
            )
            self.proj_2 = nn.Linear(
                in_features=intermediate_dim,
                out_features=config.model_dim,
                bias=False,
            )
            self.ffn_with_glu = True
        else:
            # Standard FFN, as described in https://arxiv.org/abs/1706.03762
            self.proj_1 = nn.Linear(
                in_features=config.model_dim,
                out_features=intermediate_dim,
                bias=False,
            )
            self.proj_2 = nn.Linear(
                in_features=intermediate_dim,
                out_features=config.model_dim,
                bias=False,
            )
            self.ffn_with_glu = False

        self.act = ACT2FN[config.activation_fn_name]

    def extra_repr(self) -> str:
        return super().extra_repr() + f"(ffn_with_glu) : {self.ffn_with_glu}"

    def forward(self, x: Tensor) -> Tensor:
        """Forward function of FFN layer.
        Args:
            x: Input tensor of the shape [batch size, sequence length, model dimension].
        Returns:
            A tensor of the same shape as the input.
        """
        if self.ffn_with_glu:
            y_12 = self.proj_1(x)
            y_1, y_2 = y_12.chunk(2, dim=-1)
            y = self.act(y_1) * y_2
            return self.proj_2(y)
        else:
            return self.proj_2(self.act(self.proj_1(x)))


class OpenELMDecoderLayer(nn.Module):
    def __init__(self, config: OpenELMConfig, layer_idx: int):
        super().__init__()
        self.attn = OpenELMMultiHeadCausalAttention(config, layer_idx)
        self.ffn = OpenELMFeedForwardNetwork(config, layer_idx)
        self.ffn_norm = OpenELMRMSNorm(config.model_dim)
        self.attn_norm = OpenELMRMSNorm(config.model_dim)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None,
        position_ids: Tensor,
        past_key_values: Cache | None = None,
        output_attentions: bool = False,
    ) -> tuple[Tensor, Tensor | None]:
        attention, weights = self.attn(
            self.attn_norm(hidden_states),
            attention_mask,
            position_ids,
            past_key_values,
            output_attentions,
        )
        hidden_states = hidden_states + attention
        return hidden_states + self.ffn(self.ffn_norm(hidden_states)), weights


class OpenELMModel(OpenELMPreTrainedModel):
    def __init__(self, config: OpenELMConfig):
        super().__init__(config)
        self.token_embeddings = nn.Embedding(config.vocab_size, config.model_dim)
        self.layers = nn.ModuleList(
            [
                OpenELMDecoderLayer(config, i)
                for i in range(config.num_transformer_layers)
            ]
        )
        self.norm = OpenELMRMSNorm(config.model_dim)
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.token_embeddings

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.token_embeddings = value

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.Tensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        **kwargs,
    ) -> BaseModelOutputWithPast | tuple[Any, ...]:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids and inputs_embeds")
        use_cache = self.config.use_cache if use_cache is None else use_cache
        output_attentions = (
            self.config.output_attentions
            if output_attentions is None
            else output_attentions
        )
        output_hidden_states = (
            self.config.output_hidden_states
            if output_hidden_states is None
            else output_hidden_states
        )
        return_dict = self.config.return_dict if return_dict is None else return_dict
        if self.gradient_checkpointing and self.training:
            use_cache = False
            past_key_values = None
        if inputs_embeds is None:
            inputs_embeds = self.token_embeddings(input_ids)
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)
        cache = past_key_values if use_cache else None
        past_length = cache.get_seq_length() if cache is not None else 0
        if position_ids is None:
            position_ids = torch.arange(
                past_length,
                past_length + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            ).unsqueeze(0)
        mask = create_causal_mask(
            self.config, inputs_embeds, attention_mask, cache, position_ids=position_ids
        )
        hidden_states = inputs_embeds
        states = () if output_hidden_states else None
        attentions = () if output_attentions else None
        for layer in self.layers:
            if output_hidden_states:
                states += (hidden_states,)
            if self.gradient_checkpointing and self.training:
                hidden_states, weights = self._gradient_checkpointing_func(
                    layer.__call__,
                    hidden_states,
                    mask,
                    position_ids,
                    None,
                    output_attentions,
                )
            else:
                hidden_states, weights = layer(
                    hidden_states, mask, position_ids, cache, output_attentions
                )
            if output_attentions:
                attentions += (weights,)
        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            states += (hidden_states,)
        output = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=cache,
            hidden_states=states,
            attentions=attentions,
        )
        return output if return_dict else output.to_tuple()


class OpenELMForCausalLM(OpenELMPreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "transformer.token_embeddings.weight"}

    def __init__(self, config: OpenELMConfig):
        super().__init__(config)
        self.transformer = OpenELMModel(config)
        self.lm_head = nn.Linear(config.model_dim, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.transformer.token_embeddings

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.transformer.token_embeddings = value

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head

    def set_output_embeddings(self, value: nn.Linear) -> None:
        self.lm_head = value

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs,
    ) -> CausalLMOutputWithPast | tuple[Any, ...]:
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            **kwargs,
        )
        indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
        logits = self.lm_head(outputs.last_hidden_state[:, indices, :])
        loss = (
            None
            if labels is None
            else self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.vocab_size,
                **kwargs,
            )
        )
        output = CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        return_dict = self.config.return_dict if return_dict is None else return_dict
        return output if return_dict else output.to_tuple()


AutoModel.register(OpenELMConfig, OpenELMModel)
AutoModelForCausalLM.register(OpenELMConfig, OpenELMForCausalLM)

__all__ = ["OpenELMPreTrainedModel", "OpenELMModel", "OpenELMForCausalLM"]

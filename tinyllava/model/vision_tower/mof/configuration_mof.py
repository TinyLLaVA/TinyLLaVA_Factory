"""Configuration for the CLIP + DINOv2 vision branches of I-MoF."""

from __future__ import annotations


from huggingface_hub.dataclasses import strict
from transformers import AutoConfig, CLIPVisionConfig, Dinov2Config, PreTrainedConfig


@strict
class MofVisionConfig(PreTrainedConfig):
    """Describe aligned vision branches; their features are packed by channel.

    Both branches consume the same CLIP-preprocessed image, as in the upstream
    TinyLLaVA implementation. Matching patch sizes and depths preserve spatial
    and hidden-layer correspondence. Projection and token interleaving belong
    to the MoF connector.
    """

    model_type = "mof"
    sub_configs = {"clip_config": CLIPVisionConfig, "dinov2_config": Dinov2Config}
    num_additional_image_tokens = 1

    clip_config: dict | CLIPVisionConfig | None = None
    dinov2_config: dict | Dinov2Config | None = None

    def __post_init__(self, **kwargs):
        if isinstance(self.clip_config, dict):
            self.clip_config = CLIPVisionConfig(**self.clip_config)
        elif self.clip_config is None:
            self.clip_config = CLIPVisionConfig(
                hidden_size=1024,
                intermediate_size=4096,
                num_hidden_layers=24,
                num_attention_heads=16,
                image_size=336,
                patch_size=14,
            )
        if isinstance(self.dinov2_config, dict):
            self.dinov2_config = Dinov2Config(**self.dinov2_config)
        elif self.dinov2_config is None:
            self.dinov2_config = Dinov2Config(
                hidden_size=1024,
                num_hidden_layers=24,
                num_attention_heads=16,
                image_size=336,
                patch_size=14,
            )
        if not isinstance(self.clip_config, CLIPVisionConfig) or not isinstance(
            self.dinov2_config, Dinov2Config
        ):
            raise ValueError("MoF requires CLIPVisionConfig and Dinov2Config branches.")
        for name in ("patch_size", "num_hidden_layers", "num_channels"):
            if getattr(self.clip_config, name) != getattr(self.dinov2_config, name):
                raise ValueError(f"MoF branches must have matching {name}.")
        super().__post_init__(**kwargs)

    @property
    def hidden_sizes(self) -> tuple[int, int]:
        return (self.clip_config.hidden_size, self.dinov2_config.hidden_size)

    @property
    def hidden_size(self) -> int:
        return sum(self.hidden_sizes)

    @property
    def patch_size(self) -> int:
        return self.clip_config.patch_size

    @property
    def image_size(self) -> int:
        return self.clip_config.image_size


AutoConfig.register(MofVisionConfig.model_type, MofVisionConfig)

__all__ = ["MofVisionConfig"]

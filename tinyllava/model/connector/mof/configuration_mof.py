"""Independent MLP projections for Interleaved Mixture-of-Features."""

from huggingface_hub.dataclasses import strict

from ..mlp.configuration_mlp import MLPConnectorConfig


@strict
class MofConnectorConfig(MLPConnectorConfig):
    model_type = "mof__tlf_connector"

    vision_hidden_sizes: list[int] | tuple[int, ...] = (1024, 1024)

    def get_output_sequence_length(self, input_length: int) -> int:
        return 2 * input_length

    def __post_init__(self, **kwargs):
        self.vision_hidden_sizes = tuple(self.vision_hidden_sizes)
        if len(self.vision_hidden_sizes) != 2 or any(
            isinstance(size, bool) or not isinstance(size, int) or size <= 0
            for size in self.vision_hidden_sizes
        ):
            raise ValueError("vision_hidden_sizes must contain two positive integers")
        super().__post_init__(**kwargs)


__all__ = ["MofConnectorConfig"]

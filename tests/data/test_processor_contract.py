from types import SimpleNamespace

from PIL import Image
from transformers import CLIPImageProcessor

from tests.data.test_auto_vision_processing import tokenizer
from tinyllava.data.processor import TinyLlavaProcessor
from tinyllava.model import TinyLlavaConfig
from tinyllava.model.connector.configuration_base import BaseConnectorConfig


def test_new_connector_length_rule_needs_no_processor_registration() -> None:
    class TripleConnectorConfig(BaseConnectorConfig):
        def get_output_sequence_length(self, input_length: int) -> int:
            return 3 * input_length

    config = TinyLlavaConfig(connector_config=TripleConnectorConfig())
    config.vision_config.patch_size = 8
    processor = TinyLlavaProcessor.from_model(
        tokenizer=tokenizer(),
        image_processor=CLIPImageProcessor(
            size={"shortest_edge": 16},
            crop_size={"height": 16, "width": 16},
        ),
        model=SimpleNamespace(config=config),
    )
    assert type(processor) is TinyLlavaProcessor
    result = processor(
        text="<image>", images=Image.new("RGB", (16, 16)), return_tensors="pt"
    )
    assert result.input_ids.eq(processor.image_token_id).sum().item() == 12
    assert processor._get_num_multimodal_tokens(
        image_sizes=[(16, 16)]
    ).num_image_tokens == [12]



def test_component_mixin_preserves_qwen_native_expansion(tmp_path) -> None:
    import torch
    from transformers import (
        CLIPVisionConfig,
        LlavaProcessor,
        PreTrainedConfig,
        PreTrainedTokenizerBase,
        Qwen2VLImageProcessor,
        Qwen3VLProcessor,
        Qwen3VLVideoProcessor,
    )

    from tinyllava.data.processor import ComponentProcessorMixin
    from tinyllava.model.connector.resampler import ResamplerConnectorConfig

    class ComponentQwenProcessor(ComponentProcessorMixin, Qwen3VLProcessor):
        def __init__(
            self,
            image_processor: Qwen2VLImageProcessor | None = None,
            tokenizer: PreTrainedTokenizerBase | None = None,
            video_processor: Qwen3VLVideoProcessor | None = None,
            chat_template: str | None = None,
            connector_config: BaseConnectorConfig | dict | None = None,
            vision_config: PreTrainedConfig | dict | None = None,
            **kwargs,
        ):
            super().__init__(
                image_processor=image_processor,
                tokenizer=tokenizer,
                video_processor=video_processor,
                chat_template=chat_template,
                connector_config=connector_config,
                vision_config=vision_config,
                **kwargs,
            )

    processor = ComponentQwenProcessor(
        image_processor=Qwen2VLImageProcessor(merge_size=2),
        tokenizer=tokenizer(),
        video_processor=Qwen3VLVideoProcessor(),
        connector_config=ResamplerConnectorConfig(num_queries=7),
        vision_config=CLIPVisionConfig(patch_size=8),
    )
    image_inputs = {"image_grid_thw": torch.tensor([[1, 4, 6]])}
    # The native processor already accounts for spatial merging. Merely mixing
    # in component rules must not apply the connector to that result again.
    assert processor.replace_image_token(image_inputs, 0) == processor.image_token * 6
    assert processor.get_output_sequence_length(24) == 7
    assert processor.get_vision_output_sequence_length(image_inputs, 0) is None
    assert not isinstance(processor, LlavaProcessor)
    assert processor.replace_video_token.__func__ is Qwen3VLProcessor.replace_video_token

    processor.save_pretrained(tmp_path)
    restored = ComponentQwenProcessor.from_pretrained(tmp_path)
    assert restored.vision_config.patch_size == 8
    assert restored.get_output_sequence_length(256) == 7
    assert restored.replace_image_token(image_inputs, 0) == restored.image_token * 6

    native = ComponentQwenProcessor(
        image_processor=processor.image_processor,
        tokenizer=processor.tokenizer,
        video_processor=processor.video_processor,
    )
    assert native.connector_config is None
    assert native.get_output_sequence_length(24) == 24

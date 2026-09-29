from types import SimpleNamespace

from PIL import Image
from transformers import CLIPImageProcessor

from tests.data.test_auto_vision_processing import tokenizer
from tinyllava.data.processor import BaseProcessor
from tinyllava.data.processor.tinyllava import TinyLlavaProcessor
from tinyllava.model import TinyLlavaConfig
from tinyllava.model.connector.configuration_base import BaseConnectorConfig


def test_new_connector_length_rule_needs_no_processor_registration() -> None:
    class TripleConnectorConfig(BaseConnectorConfig):
        def get_output_sequence_length(self, input_length: int) -> int:
            return 3 * input_length

    config = TinyLlavaConfig(connector_config=TripleConnectorConfig())
    config.vision_config.patch_size = 8
    processor = BaseProcessor.from_model(
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


def test_base_processor_subclass_shares_llava_length_and_config():
    from tinyllava.data.processor import BaseProcessor
    from tinyllava.model.connector.resampler import ResamplerConnectorConfig

    class CustomProcessor(BaseProcessor):
        pass

    processor = CustomProcessor(
        image_processor=CLIPImageProcessor(),
        tokenizer=tokenizer(),
        connector_config=ResamplerConnectorConfig(num_queries=7),
    )
    assert processor.get_output_sequence_length(64) == 7
    saved = processor.to_dict()
    restored = CustomProcessor(
        image_processor=processor.image_processor,
        tokenizer=processor.tokenizer,
        connector_config=saved["connector_config"],
    )
    assert restored.get_output_sequence_length(256) == 7
    assert not isinstance(restored, TinyLlavaProcessor)

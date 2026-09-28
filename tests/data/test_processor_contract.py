from types import SimpleNamespace

from PIL import Image
from transformers import CLIPImageProcessor

from tests.data.test_auto_vision_processing import tokenizer
from tinyllava.data.processor import create_tinyllava_processor
from tinyllava.data.processor.tinyllava import TinyLlavaProcessor
from tinyllava.model import TinyLlavaConfig
from tinyllava.model.connector.configuration_base import BaseConnectorConfig


def test_new_connector_length_rule_needs_no_processor_registration() -> None:
    class TripleConnectorConfig(BaseConnectorConfig):
        def get_output_sequence_length(self, input_length: int) -> int:
            return 3 * input_length

    config = TinyLlavaConfig(connector_config=TripleConnectorConfig())
    config.vision_config.patch_size = 8
    processor = create_tinyllava_processor(
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

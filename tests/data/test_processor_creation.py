from types import SimpleNamespace

import pytest
from transformers import SiglipImageProcessor, SiglipVisionConfig

from tests.data.test_auto_vision_processing import tokenizer
from tinyllava.data.processor import TinyLlavaProcessor
from tinyllava.model import TinyLlavaConfig


def test_explicit_incompatible_selection_is_not_silently_replaced():
    config = TinyLlavaConfig(
        vision_config=SiglipVisionConfig(), vision_feature_select_strategy="default"
    )
    with pytest.raises(ValueError, match="vision_feature_select_strategy='full'"):
        TinyLlavaProcessor.from_model(
            model=SimpleNamespace(config=config),
            tokenizer=tokenizer(),
            image_processor=SiglipImageProcessor(),
        )

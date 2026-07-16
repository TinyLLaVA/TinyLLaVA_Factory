from types import SimpleNamespace

import pytest

from tinyllava.data.processor.creation import (
    _get_num_additional_image_tokens,
    _validate_image_feature_configuration,
)


def test_siglip_uses_patch_only_image_token_count():
    vision_config = SimpleNamespace(model_type="siglip_vision_model")

    assert _get_num_additional_image_tokens(vision_config) == 0


def test_patch_only_backbone_requires_full_feature_selection():
    with pytest.raises(ValueError, match="vision_feature_select_strategy='full'"):
        _validate_image_feature_configuration(
            num_additional_image_tokens=0,
            vision_feature_select_strategy="default",
        )

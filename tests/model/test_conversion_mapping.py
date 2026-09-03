from transformers.conversion_mapping import get_checkpoint_conversion_mapping

import tinyllava.model  # noqa: F401 - registers TinyLLaVA checkpoint conversions.


def test_tinyllava_registers_legacy_checkpoint_key_mappings():
    mapping = get_checkpoint_conversion_mapping("tinyllava")

    assert mapping is not None
    pairs = {
        (item.source_patterns[0], item.target_patterns[0])
        for item in mapping
    }
    assert (
        r"^vision_tower\._vision_tower\.",
        "model.vision_tower.",
    ) in pairs
    assert (
        r"^connector\._connector\.0\.",
        "model.multi_modal_projector.layers.0.",
    ) in pairs
    assert (
        r"^connector\._connector\.2\.",
        "model.multi_modal_projector.layers.1.",
    ) in pairs

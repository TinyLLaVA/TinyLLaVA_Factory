from transformers.core_model_loading import WeightRenaming
from transformers.conversion_mapping import (
    get_checkpoint_conversion_mapping,
    register_checkpoint_conversion_mapping,
)


def register_tinyllava_checkpoint_conversion_mapping() -> None:
    """Register TinyLLaVA checkpoint key migrations with Transformers."""
    mapping = get_checkpoint_conversion_mapping("tinyllava") or []
    mapping.extend([
        WeightRenaming(
            r"^language_model\.lm_head\.",
            "lm_head.",
        ),
        WeightRenaming(
            r"^language_model\.model\.",
            "model.language_model.",
        ),
        WeightRenaming(
            r"^vision_tower\._vision_tower\.",
            "model.vision_tower.",
        ),
        WeightRenaming(
            r"^connector\._connector\.0\.",
            "model.multi_modal_projector.layers.0.",
        ),
        WeightRenaming(
            r"^connector\._connector\.2\.",
            "model.multi_modal_projector.layers.1.",
        ),
        WeightRenaming(
            r"^model\.multi_modal_projector\.linear_1\.",
            "model.multi_modal_projector.layers.0.",
        ),
        WeightRenaming(
            r"^model\.multi_modal_projector\.linear_2\.",
            "model.multi_modal_projector.layers.1.",
        ),
    ])
    register_checkpoint_conversion_mapping(
        "tinyllava",
        mapping,
        overwrite=True,
    )


__all__ = ["register_tinyllava_checkpoint_conversion_mapping"]

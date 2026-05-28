from transformers.core_model_loading import WeightRenaming
from transformers.conversion_mapping import (
    get_checkpoint_conversion_mapping,
    register_checkpoint_conversion_mapping,
)


def append_llava_checkpoint_conversion_mapping():
    llava_mapping = get_checkpoint_conversion_mapping("llava") or []
    llava_mapping.extend([
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
        "llava",
        llava_mapping,
        overwrite=True
    )

from collections.abc import Mapping, Sequence
from typing import Any

import torch
import transformers


IMAGE_MARKER = "<image>"


def build_assistant_mask(
    processor: transformers.ProcessorMixin,
    messages: list[dict[str, Any]],
    data_dict: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    """Build assistant labels mask after multimodal placeholder expansion.

    HF chat templates compute `{% generation %}` spans on the rendered prompt
    before multimodal processors expand a single image placeholder into many
    image tokens.  In that case the tokenizer-level assistant mask can become
    misaligned with the final `input_ids`.

    This mirrors the upstream fix direction discussed in:
    https://github.com/huggingface/transformers/issues/44521

    Steps:
      1. Recompute the assistant mask with the tokenizer only, before image
         placeholder expansion.
      2. Replace each unexpanded image placeholder mask entry with N zeros,
         where N is the number of image tokens emitted by the processor.
      3. Keep assistant text and assistant terminators, such as `<|im_end|>`,
         exactly as marked by the chat template generation blocks.
    """
    if not hasattr(processor.tokenizer, "apply_chat_template"):
        assistant_masks = data_dict.get("assistant_masks")
        if assistant_masks is None:
            raise ValueError(
                "Processor did not return `assistant_masks`, and tokenizer cannot recompute them."
            )
        return assistant_masks

    text_only = processor.tokenizer.apply_chat_template(
        messages,
        chat_template=processor.chat_template,
        add_generation_prompt=False,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        return_assistant_tokens_mask=True,
    )
    text_only = squeeze_batch(text_only)
    input_ids = text_only["input_ids"]
    assistant_masks = text_only.get("assistant_masks")
    if assistant_masks is None:
        raise ValueError(
            "Tokenizer did not return `assistant_masks`. Chat template must contain `{% generation %}` blocks."
        )

    image_token_id = processor.tokenizer.convert_tokens_to_ids(
        getattr(processor, "image_token", IMAGE_MARKER)
    )
    replacement_counts = image_replacement_counts(processor, data_dict)

    return expand_assistant_mask_over_image_tokens(
        input_ids=input_ids,
        assistant_mask=assistant_masks,
        image_token_id=image_token_id,
        replacement_counts=replacement_counts,
        target_length=int(data_dict["input_ids"].numel()),
        device=data_dict["input_ids"].device,
    )


def expand_assistant_mask_over_image_tokens(
    input_ids: torch.Tensor,
    assistant_mask: torch.Tensor,
    image_token_id: int,
    replacement_counts: Sequence[int],
    target_length: int | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Map a tokenizer-only assistant mask onto expanded multimodal input ids.

    Example:
      input_ids before processor expansion:
        [USER, <image>, TEXT, ASSISTANT_TEXT, <|im_end|>]

      assistant mask before expansion:
        [0,    0,       0,    1,              1]

      if `<image>` expands to three image tokens, the final mask becomes:
        [0,    0, 0, 0, 0,    1,              1]

    Image tokens are always unsupervised, even if the placeholder appears inside
    a generation block.
    """
    expanded_mask: list[int] = []
    image_index = 0

    for token_id, mask in zip(input_ids.tolist(), assistant_mask.tolist()):
        if token_id == image_token_id:
            if image_index >= len(replacement_counts):
                raise ValueError("Chat template contains more image tokens than processed images.")
            expanded_mask.extend([0] * replacement_counts[image_index])
            image_index += 1
        else:
            expanded_mask.append(int(mask))

    if image_index != len(replacement_counts):
        raise ValueError("Processed images contain more images than rendered image tokens.")

    if target_length is not None and len(expanded_mask) != target_length:
        raise ValueError(
            "Assistant mask length does not match processed input length: "
            f"{len(expanded_mask)} != {target_length}."
        )

    return torch.tensor(expanded_mask, dtype=torch.long, device=device or input_ids.device)


def image_replacement_counts(
    processor: transformers.ProcessorMixin,
    data_dict: Mapping[str, torch.Tensor],
) -> list[int]:
    pixel_values = data_dict.get("pixel_values")
    if pixel_values is None:
        return []

    if pixel_values.ndim == 3:
        image_tensors = pixel_values.unsqueeze(0)
    elif pixel_values.ndim == 4:
        image_tensors = pixel_values
    else:
        image_tensors = pixel_values.reshape(-1, *pixel_values.shape[-3:])

    patch_size = processor.patch_size
    num_additional_image_tokens = getattr(processor, "num_additional_image_tokens", 0)
    select_strategy = getattr(processor, "vision_feature_select_strategy", None)
    counts = []
    for image_tensor in image_tensors:
        height, width = image_tensor.shape[-2:]
        count = (height // patch_size) * (width // patch_size)
        count += num_additional_image_tokens
        if select_strategy == "default":
            count -= 1
        counts.append(count)
    return counts


def squeeze_batch(encoded: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    data_dict = {}
    for key, value in encoded.items():
        if isinstance(value, torch.Tensor):
            data_dict[key] = value.squeeze(0) if value.ndim > 0 and value.shape[0] == 1 else value
        else:
            data_dict[key] = torch.as_tensor(value[0] if isinstance(value, list) and len(value) == 1 else value)
    return data_dict


__all__ = [
    "build_assistant_mask",
    "expand_assistant_mask_over_image_tokens",
    "image_replacement_counts",
    "squeeze_batch",
]

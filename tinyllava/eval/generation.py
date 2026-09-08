from collections.abc import Sequence
from typing import Any

import torch
from PIL import Image
from transformers.image_utils import load_image

from tinyllava.data.image_payload import add_image_payloads
from tinyllava.data.message_format import normalize_content
from tinyllava.data.processor.creation import create_tinyllava_processor
from tinyllava.utils.constants import DEFAULT_IMAGE_TOKEN


def build_processor_from_model(model: Any, tokenizer: Any | None = None, image_processor: Any | None = None):
    tokenizer = tokenizer or model.tokenizer
    if image_processor is None:
        processor = getattr(model, "processor", None)
        image_processor = getattr(processor, "image_processor", None)
    if image_processor is None:
        raise ValueError("`image_processor` must be provided unless `model.processor.image_processor` exists.")
    return create_tinyllava_processor(
        tokenizer=tokenizer,
        image_processor=image_processor,
        model=model,
    )


def make_user_message(text: str, images: Sequence[Image.Image] = ()) -> dict[str, Any]:
    """Build a user message with ordered image blocks followed by text.

    Args:
        text: Prompt text. Inline `<image>` markers are removed before encoding.
        images: Images to attach in the order expected by the prompt.

    Returns:
        A Hugging Face multimodal message with `role` and `content` keys.

    Raises:
        ValueError: The prompt contains an image marker but no images are supplied.
    """
    if DEFAULT_IMAGE_TOKEN in text and not images:
        raise ValueError(
            "Eval prompt contains a legacy `<image>` marker, but no image payload was provided."
        )
    content = [{"type": "image"} for _ in images]
    content.extend(normalize_content(strip_legacy_image_markers(text)))
    message = {"role": "user", "content": content}
    add_image_payloads([message], images)
    return message


def strip_legacy_image_markers(text: str) -> str:
    """Remove inline LLaVA image markers from eval prompts.

    Evaluation messages carry images as HF multimodal content blocks. Keeping a
    legacy inline marker in text would create an extra image block when the text
    is normalized, so the marker is treated only as a dataset-era placeholder.
    """

    return text.replace(DEFAULT_IMAGE_TOKEN, "").strip()


def normalize_image_files(image_files: str | Sequence[str] | None) -> list[str]:
    if image_files is None:
        return []
    if isinstance(image_files, str):
        return [image_files]
    return list(image_files)


def load_images(image_files: Sequence[str]) -> list[Image.Image]:
    return [load_image(image_file).convert("RGB") for image_file in image_files]


def prepare_generation_inputs(
    processor: Any,
    messages: list[dict[str, Any]] | list[list[dict[str, Any]]],
    *,
    device: torch.device | str,
    chat_template: str | None = None,
    padding: bool = False,
) -> dict[str, torch.Tensor]:
    """Encode conversations with a generation prompt and move tensors to a device.

    Args:
        processor: Multimodal processor with a configured chat template.
        messages: One conversation or a batch of conversations in HF message format.
        device: Target device for model input tensors.
        chat_template: Optional Jinja template override.
        padding: Pad conversations to a common token length for batch generation.

    Returns:
        Token IDs, attention masks, and processor-specific image inputs on `device`.
    """
    template_kwargs = {}
    if padding:
        template_kwargs["processor_kwargs"] = {"text_kwargs": {"padding": True}}
    inputs = processor.apply_chat_template(
        messages,
        chat_template=chat_template,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        **template_kwargs,
    )
    return move_to_device(dict(inputs), device=device)


def move_to_device(inputs: dict[str, Any], *, device: torch.device | str) -> dict[str, Any]:
    moved = {}
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def generate_response(
    *,
    model: Any,
    processor: Any,
    messages: list[dict[str, Any]],
    chat_template: str | None = None,
    temperature: float = 0.2,
    top_p: float | None = None,
    num_beams: int = 1,
    max_new_tokens: int = 512,
    streamer: Any | None = None,
    **generate_kwargs,
) -> str:
    """Generate one answer from a multimodal conversation.

    Args:
        model: Loaded model supporting `generate`; set it to evaluation mode first.
        processor: Processor matching the model's tokenizer and vision settings.
        messages: Conversation history ending with the user's request.
        chat_template: Optional Jinja template override.
        temperature: Sampling temperature; nonpositive values disable sampling.
        top_p: Nucleus-sampling probability cutoff.
        num_beams: Beam count passed to generation.
        max_new_tokens: Maximum number of tokens to generate beyond the prompt.
        streamer: Optional Transformers streamer receiving generated tokens.
        **generate_kwargs (Any): Additional `model.generate` options that do not duplicate
            options supplied by this helper, such as `use_cache` or `pad_token_id`.

    Returns:
        Decoded answer with prompt tokens, special tokens, and outer whitespace removed.
    """
    return generate_responses(
        model=model,
        processor=processor,
        messages_batch=[messages],
        chat_template=chat_template,
        temperature=temperature,
        top_p=top_p,
        num_beams=num_beams,
        max_new_tokens=max_new_tokens,
        streamer=streamer,
        **generate_kwargs,
    )[0]


def generate_responses(
    *,
    model: Any,
    processor: Any,
    messages_batch: list[list[dict[str, Any]]],
    chat_template: str | None = None,
    temperature: float = 0.2,
    top_p: float | None = None,
    num_beams: int = 1,
    max_new_tokens: int = 512,
    streamer: Any | None = None,
    **generate_kwargs,
) -> list[str]:
    """Generate one decoded answer per conversation in a batch.

    Use a processor configured for left padding with decoder-only models. Prompt
    tokens are removed using the padded input width before answers are decoded.

    Args:
        model: Loaded model supporting `generate`; set it to evaluation mode first.
        processor: Processor matching the model and configured for batch padding.
        messages_batch: Nonempty batch of conversations in HF message format.
        chat_template: Optional Jinja template override.
        temperature: Sampling temperature; nonpositive values disable sampling.
        top_p: Nucleus-sampling probability cutoff.
        num_beams: Beam count passed to generation.
        max_new_tokens: Maximum number of new tokens per answer.
        streamer: Optional Transformers streamer; its batch-size limits apply.
        **generate_kwargs (Any): Additional `model.generate` options. Keep
            `num_return_sequences=1` and tensor output; do not repeat options
            already supplied by this helper.

    Returns:
        Answer strings in the same order as `messages_batch`, with special tokens
        and outer whitespace removed.
    """
    device = getattr(model, "device", None)
    if device is None:
        device = next(model.parameters()).device

    inputs = prepare_generation_inputs(
        processor,
        messages_batch,
        device=device,
        chat_template=chat_template,
        padding=len(messages_batch) > 1,
    )
    prompt_width = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            do_sample=temperature > 0,
            temperature=temperature,
            top_p=top_p,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            streamer=streamer,
            use_cache=True,
            pad_token_id=processor.tokenizer.pad_token_id,
            **generate_kwargs,
        )

    responses = []
    for row_idx in range(len(messages_batch)):
        response_ids = output_ids[row_idx, prompt_width:]
        response = processor.tokenizer.decode(
            response_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        responses.append(response.strip())
    return responses


__all__ = [
    "build_processor_from_model",
    "generate_response",
    "generate_responses",
    "load_images",
    "make_user_message",
    "normalize_image_files",
    "prepare_generation_inputs",
    "strip_legacy_image_markers",
]

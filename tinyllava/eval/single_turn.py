import argparse
from collections.abc import Sequence

import torch

from tinyllava.eval.generation import (
    build_processor_from_model,
    generate_response,
    load_images,
    make_user_message,
    normalize_image_files,
)
from tinyllava.model.modeling_tinyllava import TinyLlavaForConditionalGeneration
from tinyllava.utils.model_loading import load_tinyllava_checkpoint_bundle


def run_single_turn(
    model_path: str | None = None,
    model: TinyLlavaForConditionalGeneration | None = None,
    image_files: str | Sequence[str] | None = None,
    query: str = "",
    chat_template: str | None = None,
    device: str | None = None,
    temperature: float = 0.2,
    top_p: float | None = None,
    num_beams: int = 1,
    max_new_tokens: int = 512,
    **kwargs,
):
    if not (model is None) ^ (model_path is None):
        raise ValueError("Exactly one of model or model_path must be provided")
    if model is None:
        bundle = load_tinyllava_checkpoint_bundle(model_path, device=device)
        model = bundle.model
        tokenizer = bundle.tokenizer
        image_processor = bundle.image_processor
    else:
        if device is not None:
            model = model.to(device)
        tokenizer = model.tokenizer
        image_processor = None

    processor = build_processor_from_model(
        model,
        tokenizer=tokenizer,
        image_processor=image_processor,
    )
    image_file_list = normalize_image_files(image_files)
    images = load_images(image_file_list)
    messages = [make_user_message(query, images)]
    outputs = generate_response(
        model=model,
        processor=processor,
        messages=messages,
        chat_template=chat_template,
        temperature=temperature,
        top_p=top_p,
        num_beams=num_beams,
        max_new_tokens=max_new_tokens,
        **kwargs,
    )
    print(outputs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--image-file", type=str, action="append", default=None)
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--chat-template", type=str, default=None)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    args = parser.parse_args()

    run_single_turn(
        model_path=args.model_path,
        image_files=args.image_file,
        query=args.query,
        chat_template=args.chat_template,
        device=args.device,
        temperature=args.temperature,
        top_p=args.top_p,
        num_beams=args.num_beams,
        max_new_tokens=args.max_new_tokens,
    )

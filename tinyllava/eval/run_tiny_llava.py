import argparse
import requests
from PIL import Image
from io import BytesIO
from collections.abc import Sequence

import torch

from tinyllava.utils.eval_utils import disable_torch_init, KeywordsStoppingCriteria
from tinyllava.utils.constants import DEFAULT_IMAGE_TOKEN
from tinyllava.utils.message import Message
from tinyllava.data.image_preprocess import ImagePreprocess
from tinyllava.data.text_preprocess import TextPreprocess
from tinyllava.model.modeling_tinyllava import TinyLlavaForConditionalGeneration
from tinyllava.model.load_model import load_pretrained_model


def normalize_image_files(
    image_files: str | Sequence[str] | None,
) -> list[str]:
    if image_files is None:
        return []
    if isinstance(image_files, str):
        return [image_files.strip()] if image_files.strip() else []
    return [f.strip() for f in image_files if isinstance(f, str) and f.strip()]


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out


def eval_model(
    model_path: str | None = None,
    model: TinyLlavaForConditionalGeneration | None = None,
    image_files: str | Sequence[str] | None = None,
    query: str = "",
    template: str | None = None,
    device: str | None = None,
    temperature: float = 0.2,
    top_p: float | None = None,
    num_beams: int = 1,
    max_new_tokens: int = 512,
    **kwargs,
):
    disable_torch_init()

    if not (model is None) ^ (model_path is None):
        raise ValueError("Exactly one of model or model_path must be provided")
    if model is None:
        load_kwargs = {}
        if device is not None:
            load_kwargs["device"] = device
        model, tokenizer, image_processor, context_len = load_pretrained_model(
            model_path, **load_kwargs
        )
    else:
        if device is not None:
            model = model.to(device)
        tokenizer = model.tokenizer
        image_processor = model.vision_tower._image_processor

    target_device = device or model.device

    image_file_list = normalize_image_files(image_files)
    include_image = len(image_file_list) > 0

    if include_image:
        query = DEFAULT_IMAGE_TOKEN + "\n" + query

    text_processor = TextPreprocess(tokenizer, template)
    data_args = model.config
    image_processor = ImagePreprocess(image_processor, data_args)

    msg = Message()
    msg.add_message(query)

    result = text_processor(msg.messages, mode="eval")
    input_ids = result["input_ids"]
    input_ids = input_ids.unsqueeze(0).to(target_device)

    images_tensor = None
    if include_image:
        images = load_images(image_file_list)[0]
        images_tensor = image_processor(images)
        images_tensor = images_tensor.unsqueeze(0).half().to(target_device)

    stop_str = text_processor.template.separator.apply()[1]
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images_tensor,
            do_sample=True if temperature > 0 else False,
            temperature=temperature,
            top_p=top_p,
            num_beams=num_beams,
            pad_token_id=tokenizer.pad_token_id,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
            **kwargs,
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[: -len(stop_str)]
    outputs = outputs.strip()
    print(outputs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--image-file", type=str, action="append", default=None)
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--template", type=str, default=None)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    args = parser.parse_args()

    eval_model(
        model_path=args.model_path,
        image_files=args.image_file,
        query=args.query,
        template=args.template,
        device=args.device,
        temperature=args.temperature,
        top_p=args.top_p,
        num_beams=args.num_beams,
        max_new_tokens=args.max_new_tokens,
    )

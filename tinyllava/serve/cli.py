import argparse
import requests
from PIL import Image
from io import BytesIO
from collections.abc import Sequence

import torch
from transformers import TextStreamer

from tinyllava.utils.eval_utils import disable_torch_init
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


def main(
    model_path: str | None = None,
    model: TinyLlavaForConditionalGeneration | None = None,
    image_files: str | Sequence[str] | None = None,
    template: str | None = None,
    device: str | None = None,
    temperature: float = 0.2,
    top_p: float | None = None,
    max_new_tokens: int = 512,
    debug: bool = False,
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

    text_processor = TextPreprocess(tokenizer, template)
    data_args = model.config
    image_processor = ImagePreprocess(image_processor, data_args)

    if getattr(text_processor.template, "role", None) is None:
        roles = ["USER", "ASSISTANT"]
    else:
        roles = text_processor.template.role.apply()

    msg = Message()

    images_tensor = None
    if include_image:
        images = load_images(image_file_list)[0]
        images_tensor = image_processor(images)
        images_tensor = images_tensor.unsqueeze(0).half().to(target_device)

    while True:
        try:
            inp = input(f"{roles[0]}: ")
        except EOFError:
            inp = ""
        if not inp:
            print("exit...")
            break

        print(f"{roles[1]}: ", end="")

        if include_image:
            # first message
            inp = DEFAULT_IMAGE_TOKEN + "\n" + inp
            msg.add_message(inp)
            include_image = False
        else:
            # later messages
            msg.add_message(inp)
        result = text_processor(msg.messages, mode="eval")
        prompt = result["prompt"]
        input_ids = result["input_ids"].unsqueeze(0).to(target_device)

        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=images_tensor,
                do_sample=True if temperature > 0 else False,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                streamer=streamer,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
                **kwargs,
            )

        outputs = tokenizer.decode(output_ids[0, input_ids.shape[1] :]).strip()
        msg.messages[-1]["value"] = outputs

        if debug:
            print("\n", {"prompt": prompt, "outputs": outputs}, "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path", type=str, default="tinyllava/TinyLLaVA-Phi-2-SigLIP-3.1B"
    )
    parser.add_argument("--image-file", type=str, action="append", default=None)
    parser.add_argument("--template", type=str, default="phi")
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    main(
        model_path=args.model_path,
        image_files=args.image_file,
        template=args.template,
        device=args.device,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        debug=args.debug,
    )

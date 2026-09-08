"""Gradio demo backed by the TinyLLaVA v2 Hugging Face pipeline."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from tinyllava.data.chat_template.loading import resolve_chat_template
from tinyllava.eval.generation import generate_response, make_user_message
from tinyllava.utils.model_loading import load_tinyllava_checkpoint_bundle


DEFAULT_MODEL_PATH = "tinyllava/TinyLLaVA-Phi-2-SigLIP-3.1B"


def add_turn(
    *,
    model: Any,
    processor: Any,
    messages: Sequence[dict[str, Any]] | None,
    history: Sequence[dict[str, Any]] | None,
    text: str,
    image: Any | None,
    chat_template: str | None,
    temperature: float,
    top_p: float | None,
    max_new_tokens: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Generate one response and append it to HF and Gradio histories."""

    text = text.strip()
    if not text and image is None:
        return list(messages or []), list(history or [])

    current_messages = list(messages or [])
    user_message = make_user_message(text, [] if image is None else [image])
    response = generate_response(
        model=model,
        processor=processor,
        messages=[*current_messages, user_message],
        chat_template=chat_template,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
    )
    current_messages.extend(
        [
            user_message,
            {
                "role": "assistant",
                "content": [{"type": "text", "text": response}],
            },
        ]
    )
    current_history = list(history or [])
    display_text = text if image is None else f"[image] {text}".strip()
    current_history.extend(
        [
            {"role": "user", "content": display_text},
            {"role": "assistant", "content": response},
        ]
    )
    return current_messages, current_history


def build_demo(
    *,
    model: Any,
    processor: Any,
    chat_template: str | None = None,
):
    """Build the Gradio interface without importing Gradio at package import time."""

    import gradio as gr

    def respond(messages, history, text, image, temperature, top_p, max_new_tokens):
        messages, history = add_turn(
            model=model,
            processor=processor,
            messages=messages,
            history=history,
            text=text,
            image=image,
            chat_template=chat_template,
            temperature=float(temperature),
            top_p=float(top_p),
            max_new_tokens=int(max_new_tokens),
        )
        return messages, history, "", None

    def clear_history():
        return [], [], "", None

    example_dir = Path(__file__).parent / "examples"
    with gr.Blocks(title="TinyLLaVA") as demo:
        gr.Markdown("# TinyLLaVA")
        messages = gr.State([])
        chatbot = gr.Chatbot(label="Chat")
        with gr.Row():
            image = gr.Image(type="pil", label="Image")
            text = gr.Textbox(label="Prompt")
        with gr.Accordion("Generation", open=False):
            temperature = gr.Slider(0.0, 1.0, value=0.2, step=0.1)
            top_p = gr.Slider(0.05, 1.0, value=0.9, step=0.05)
            max_new_tokens = gr.Slider(1, 1024, value=256, step=1)
        with gr.Row():
            submit = gr.Button("Send", variant="primary")
            clear = gr.Button("Clear")
        gr.Examples(
            examples=[
                [
                    str(example_dir / "extreme_ironing.jpg"),
                    "What is unusual about this image?",
                ],
                [
                    str(example_dir / "waterview.jpg"),
                    "What should I be cautious about when visiting here?",
                ],
            ],
            inputs=[image, text],
        )

        inputs = [
            messages,
            chatbot,
            text,
            image,
            temperature,
            top_p,
            max_new_tokens,
        ]
        outputs = [messages, chatbot, text, image]
        submit.click(respond, inputs, outputs)
        text.submit(respond, inputs, outputs)
        clear.click(clear_history, None, outputs, queue=False)
    return demo


def parse_args(args: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(description="Launch the TinyLLaVA web demo.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--chat-template-path")
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--share", action="store_true")
    return parser.parse_args(args)


def main(args: Sequence[str] | None = None) -> None:
    parsed = parse_args(args)
    bundle = load_tinyllava_checkpoint_bundle(
        parsed.model_path,
        device=parsed.device,
    )
    bundle.model.eval()
    chat_template = resolve_chat_template(
        chat_template_path=parsed.chat_template_path,
    )
    demo = build_demo(
        model=bundle.model,
        processor=bundle.processor,
        chat_template=chat_template,
    )
    demo.queue()
    demo.launch(
        server_name=parsed.host,
        server_port=parsed.port,
        share=parsed.share,
    )


if __name__ == "__main__":
    main()


__all__ = ["add_turn", "build_demo", "main", "parse_args"]

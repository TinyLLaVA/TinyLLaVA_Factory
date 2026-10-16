"""Terminal conversations using the shared multimodal generation API."""

from __future__ import annotations

from collections.abc import Sequence

from tinyllava.data.chat_template.loading import resolve_chat_template
from tinyllava.eval.generation import generate_response, load_images, make_user_message
from tinyllava.utils.model_loading import load_tinyllava_checkpoint_bundle


def run_chat(
    *,
    model_path: str,
    query: str | None = None,
    image_files: Sequence[str] | None = None,
    device: str | None = None,
    chat_template_path: str | None = None,
    temperature: float = 0.2,
    top_p: float | None = None,
    num_beams: int = 1,
    max_new_tokens: int = 512,
    interactive: bool = True,
) -> None:
    """Load once, retain multimodal history, and optionally read further turns.

    Initial images are attached to the first prompt and remain in the history.
    In interactive mode, /reset clears history, /exit exits, and /image PATH
    attaches an image to the next prompt. EOF and Ctrl-C also end the session.
    """
    if device is None:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    bundle = load_tinyllava_checkpoint_bundle(model_path, device=device)
    bundle.model.eval()
    template = resolve_chat_template(chat_template_path=chat_template_path)
    messages = []
    pending_images = list(image_files or [])
    if interactive:
        print("Commands: /image PATH, /reset, /exit", flush=True)
    while True:
        if query is None:
            if not interactive:
                return
            try:
                query = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return
            if query == "/exit":
                return
            if query == "/reset":
                messages.clear()
                pending_images.clear()
                query = None
                continue
            if query.startswith("/image "):
                pending_images.append(query[len("/image ") :].strip())
                query = None
                continue
            if not query:
                query = None
                continue
        user_message = make_user_message(query, load_images(pending_images))
        response = generate_response(
            model=bundle.model,
            processor=bundle.processor,
            messages=[*messages, user_message],
            chat_template=template,
            temperature=temperature,
            top_p=top_p,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
        )
        messages.extend(
            [
                user_message,
                {"role": "assistant", "content": [{"type": "text", "text": response}]},
            ]
        )
        print(response, flush=True)
        if not interactive:
            return
        pending_images.clear()
        query = None

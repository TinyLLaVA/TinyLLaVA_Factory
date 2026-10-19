# Inference

Use a composite checkpoint with its saved tokenizer and image processor.

```bash
python -m tinyllava.cli single-turn \
  --model-path output/my-finetune \
  --image-file tinyllava/serve/examples/waterview.jpg \
  --query "Describe the image." \
  --temperature 0 --max-new-tokens 128
```

The CLI currently spells its checkpoint option `--model-path`; evaluation YAML
uses `model_name_or_path`. Repeat `--image-file` for multiple images when
the selected model/processor supports that input.

## CLI options { #options }

| Option | Use |
| --- | --- |
| `--model-path` | Trained checkpoint directory or Hub ID. |
| `--query` | User prompt; required for `single-turn`, optional for `chat`. |
| `--image-file` | Image path; repeat for multiple images with a compatible model. |
| `--device` | Execution device; defaults to CUDA when available, otherwise CPU. |
| `--chat-template-path` | Jinja conversation template file. |
| `--temperature` | Sampling temperature; `0` disables sampling. |
| `--top-p` | Nucleus sampling cutoff. |
| `--num-beams` | Decoding beam count; default `1`. |
| `--max-new-tokens` | Response token limit; default `512` for these CLI commands. |

## Multi-turn terminal chat

Run `chat` to retain conversation history across prompts. Attach images with `--image-file` on startup or `/image PATH` before a later prompt. `/reset` clears history and pending images; `/exit` ends the session.

```bash
python -m tinyllava.cli chat --model-path output/my-finetune
```

## Python

```python
from PIL import Image
from tinyllava.utils.model_loading import load_tinyllava_checkpoint_bundle
from tinyllava.eval.generation import generate_response, make_user_message

bundle = load_tinyllava_checkpoint_bundle("output/my-finetune", device="cuda")
bundle.model.eval()
with Image.open("tinyllava/serve/examples/waterview.jpg") as image:
    message = make_user_message("Describe the image.", [image.convert("RGB")])
    answer = generate_response(
        model=bundle.model,
        processor=bundle.processor,
        messages=[message],
        temperature=0,
        max_new_tokens=128,
    )
print(answer)
```

For batched calls use `generate_responses` with `messages_batch`. The helper
handles left padding and returns one decoded response per input conversation.

## Web demo

```bash
python -m tinyllava.serve.app --help
python -m tinyllava.serve.app --model-path output/my-finetune
```

The UI uses the same checkpoint bundle and generation helpers. Install compatible
Gradio and Gradio Client versions for the Web UI.

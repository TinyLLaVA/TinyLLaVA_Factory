# Chat templates

`llava.jinja` is copied without modification from the
[HF LLaVA 1.5 7B template](https://huggingface.co/llava-hf/llava-1.5-7b-hf/resolve/b234b804b114d9e37bb655e11cbbb5f5e971b7a9/chat_template.jinja).

It expects normalized multimodal message content, renders images before text,
and includes assistant generation spans. It adds no default system prompt and
no EOS token; assistant text ends with a space, matching the upstream file.
The Phi-2 and OpenELM presets explicitly select this template.

`pretrain_legacy.jinja` and `qwen2_base_legacy.jinja` retain the legacy experiment
formats and are selected explicitly by the corresponding configurations.

## Preset audit (2026-09-29)

| Preset | Tokenizer | Template source |
| --- | --- | --- |
| Phi-2 | `microsoft/phi-2` | Explicit `llava.jinja`; tokenizer has no template |
| OpenELM | `meta-llama/Llama-2-7b-hf` | Explicit `llava.jinja`; tokenizer has no template |
| Qwen2 Base | `Qwen/Qwen2-0.5B` | Tokenizer template; legacy experiment stages retain their overrides |
| Qwen2 Instruct | `Qwen/Qwen2-0.5B-Instruct` | Tokenizer template |
| StableLM | `stabilityai/stablelm-2-zephyr-1_6b` | Tokenizer template |
| TinyLlama | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | Tokenizer template |
| Gemma | `google/gemma-2b-it` | Built-in template documented in the official model card |

The four public tokenizer templates were also checked with the project's
multimodal/generation-span injection: image markers, user questions, and assistant
answers survive rendering. Gemma's restricted tokenizer config could not be
retrieved in this environment; its built-in template is documented in the
[official model card](https://huggingface.co/google/gemma-2b-it#chat-template).

There is no automatic generic fallback. Explicit model/stage template settings
take priority over the tokenizer template. If neither exists, applying the chat
template fails. Keep the same saved template for checkpoint inference.

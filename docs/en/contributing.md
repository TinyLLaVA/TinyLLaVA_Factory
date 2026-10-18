# Documentation development

The site uses Zensical and mkdocstrings. English pages live in `docs/en`,
Chinese pages in `docs/zh`; their navigation is defined in `zensical.toml`
and `zensical.zh.toml`.

## Build and preview

Use the project environment with Python 3.11 or newer. From the repository root:

```bash
python -m pip install -e ".[docs]"
python docs/build.py
python -m http.server 8000 --bind 127.0.0.1 --directory site
```

The `docs` extra adds the documentation tools to the project environment.
Regular installation uses `pip install -e .`. With uv, use
`uv sync --locked --extra docs --inexact`, then `uv run --no-sync python docs/build.py`.

Open `http://localhost:8000/` for English or `http://localhost:8000/zh/`
for Chinese. Forward port 8000 when working over SSH. Rebuild after editing.
For single-language live reload, use `zensical serve` or
`zensical serve -f zensical.zh.toml`.

The build script builds both languages with Zensical's `--strict` checks.
Generated output is stored in `site/`.

## Edit pages and API references

Keep corresponding guides and navigation entries in both languages up to date.
The language menu opens each language's homepage.

API pages use `:::` directives referencing symbols in `tinyllava/`. Select
public members explicitly and keep the same symbol lists in both languages.

mkdocstrings uses Griffe to read source and allows dynamic inspection when source
analysis is unavailable (`allow_inspection = true`). For runtime-generated APIs,
use a targeted Griffe extension or `force_inspection` after checking the rendered
members and signatures. See the
[inspection options](https://mkdocstrings.github.io/python/usage/configuration/general/#allow_inspection).

### Docstring style

Use [Google-style docstrings](https://mkdocstrings.github.io/griffe/reference/docstrings/#google-style): start with a short description of the operation, then document `Args`, `Returns`, `Raises`, and `Examples` as needed. Describe units, tensor shapes, masking rules, side effects, and defaults that affect usage. Keep types and default values in the Python signature; add types in the docstring only when an annotation is unavailable. Document dataclass fields under `Attributes`, with names matching the source exactly.

Separate each section with a blank line and indent its entries by four spaces. Use Markdown links and fenced Python examples. Examples should use TinyLLaVA's public interfaces and identify any required model or dataset assets.

The reference uses mkdocstrings' built-in symbol badges, parameter headings, separate signatures, and list-style sections. Keep the Python handler options identical in both language configurations. See the [official site's configuration](https://github.com/mkdocstrings/mkdocstrings/blob/main/zensical.toml).

Write each Chinese prose paragraph on one source line to avoid spaces introduced by Markdown soft line breaks. Preserve established API terms such as `checkpoint`, `tokenizer`, `processor`, `collator`, and `chat template`. Keep code blocks, tables, and separate list items on their own lines.

## Tests and CI

Install the test extra and run the suites:

```bash
python -m pip install -e ".[test]"
python -m pytest
python -m pytest -m slow
```

The default suite excludes `slow`. The slow test creates a tiny random Llama/CLIP
model and exactly two images, red and blue, with the same color question. It runs
the training CLI, reloads the saved checkpoint, and requires both generated
one-word answers to match. It runs offline on CPU for every preset in
`configs/training/`, with seeds 0 and 42. The presets retain their tuning policies,
including the current int8 strategy in `qlora_finetune`; the test overrides batch
size, learning rate, steps, and runtime settings for this small task. Gradient
checkpointing stays enabled as specified by the presets.

The image ends the prompt so connector-only pretraining can learn with a frozen
random language model. This is a bounded overfit test, covering training, adapter
saving, and inference for these presets and seeds.

The fast suite also runs tiny versions of all seven model presets through
backward, checkpoint reload, and generation, and checks PEFT component policies,
int8 component loading, multi-image connectors, and evaluation sharding. A second
slow test assembles Qwen2/SigLIP from local components, checks seeded training
reproducibility, resumes a LoRA checkpoint, chains two LoRA stages, and continues
with full fine-tuning.

The `CI` workflow uses CPU PyTorch wheels and runs fast tests with the minimum
supported Transformers version and the latest release allowed by
`pyproject.toml`. It builds the bilingual documentation once, in the latest
version job. `uv.lock` remains available for reproducing a development
environment; it does not constrain CI dependency resolution.
Enable `slow` in a manual run to include these training tests. Its `github-pages`
artifact contains the built site and is retained for 14 days.

## Publish with GitHub Actions

To publish:

1. Keep `.github/workflows/ci.yml` synchronized on `main` and `develop` to
   enable manual runs.
2. Set **Settings → Pages → Source** to **GitHub Actions**.
3. Allow `develop` in the `github-pages` environment's deployment rules.
4. Open **Actions → CI → Run workflow**, select `develop`, and
   enable `deploy`.

Publishing uses the same validated artifact, built with the base URL returned
by GitHub Pages. To test a deployment URL locally:

```bash
python docs/build.py --base-url https://example.org/project
```

Run `docs/build.py` without `--base-url` to restore the localhost preview paths.

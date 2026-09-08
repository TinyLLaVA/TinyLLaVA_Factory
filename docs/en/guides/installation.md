# Installation

Run commands from the repository root on the `develop` branch.
The runtime requires Python 3.10 or newer; exact package constraints live in
`pyproject.toml`, with the resolved development environment in `uv.lock`.

```bash
git clone --branch develop https://github.com/TinyLLaVA/TinyLLaVA_Factory.git
cd TinyLLaVA_Factory
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Select a PyTorch build appropriate for your CUDA installation before running
GPU jobs. DeepSpeed, bitsandbytes, and optional model implementations can impose
additional platform requirements.

## Attention and precision

Use `training.precision` (`auto`, `bf16`, `fp16`, or `fp32`) and
`model.attn_implementation` in training YAML. FlashAttention is optional: install
it only if you select that backend and have a compatible CUDA/PyTorch toolchain.
The Qwen2 legacy fine-tuning recipe selects SDPA.

## Check the entry points

```bash
python -m tinyllava.eval.single_turn --help
python -m tinyllava.serve.app --help
```

The Web UI additionally needs a working Gradio/Gradio Client installation.

To build the documentation, follow
[Contributing to docs](../contributing.md).

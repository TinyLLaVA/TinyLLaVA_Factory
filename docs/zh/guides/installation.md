# 安装

使用 `develop` 分支。运行环境需要 Python 3.10 或更高版本，依赖约束见 `pyproject.toml`，开发环境锁定版本见 `uv.lock`。

```bash
git clone --branch develop https://github.com/TinyLLaVA/TinyLLaVA_Factory.git
cd TinyLLaVA_Factory
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

GPU 训练前请选择与 CUDA 匹配的 PyTorch。DeepSpeed、bitsandbytes 和可选模型有各自的平台与依赖要求。

## 注意力与精度

训练 YAML 用 `training.precision` 选择 `auto`、`bf16`、`fp16` 或 `fp32`，用 `model.attn_implementation` 选择注意力后端。FlashAttention 是可选依赖，仅在选择该后端且工具链兼容时安装；Qwen2 legacy 微调配置使用 SDPA。

```bash
python -m tinyllava.eval.single_turn --help
python -m tinyllava.serve.app --help
```

Web UI 还需要可用的 Gradio / Gradio Client 组合。只构建文档请参考[维护文档](../contributing.md)。

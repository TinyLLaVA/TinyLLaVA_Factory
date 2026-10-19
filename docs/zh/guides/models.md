# 模型与 checkpoint

TinyLLaVA 由语言模型、视觉编码器和连接器组成。初次训练时选择组件预设，
继续训练、评测或对话时加载已有 checkpoint。

## 组件预设

在实验中设置 `model: phi`，或用 `model=qwen2_instruct` 覆写。内置预设选择以下组件：

| 预设 | 语言模型 | 视觉编码器 |
| --- | --- | --- |
| `phi` | `microsoft/phi-2` | SigLIP SO400M，384 px |
| `qwen2_base` | `Qwen/Qwen2-0.5B` | SigLIP SO400M，384 px |
| `qwen2_instruct` | `Qwen/Qwen2-0.5B-Instruct` | SigLIP SO400M，384 px |
| `tinyllama` | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | SigLIP SO400M，384 px |
| `stablelm` | `stabilityai/stablelm-2-zephyr-1_6b` | CLIP ViT-L/14，336 px |
| `gemma` | `google/gemma-2b-it` | SigLIP SO400M，384 px |
| `openelm` | `apple/OpenELM-270M-Instruct` | SigLIP SO400M，384 px |

这些预设都使用 `mlp2x_gelu`。OpenELM 显式使用 Llama-2 tokenizer，其他预设未覆写时
使用语言模型自带的 tokenizer。

引用预设后，可以修改组件或预处理设置：

```yaml
model:
  from: phi
  model_max_length: 4096
  attn_implementation: sdpa
  connector:
    from: mlp2x_gelu
    depth: 3
```

`model_max_length` 控制训练截断长度，`attn_implementation` 选择语言模型和运行环境支持的注意力后端。
tokenizer 与模板的覆写选项见[模型参数](parameters.md#model)。

## 连接器与视觉特征

| 连接器预设 | 用途 | 主要选项 |
| --- | --- | --- |
| `mlp2x_gelu` | 将视觉特征投影到语言模型的嵌入空间 | `depth`、`act`、`bias` |
| `identity` | 透传维度兼容的特征 | — |
| `qformer` | 提取固定数量的视觉查询 token | `num_queries` |
| `resampler` | 将视觉特征重采样为固定数量的 token | `num_queries` |
| `mof` | 组合 MoF 的两路视觉特征 | `vision_hidden_sizes`、`clip_config`、`dinov2_config` |

`model.vision_feature_layer` 选择隐藏层。未指定 `vision_feature_select_strategy` 时，
SigLIP 保留全部 patch，CLIP、DINO 和 MoF 去掉首个 CLS token；加载 checkpoint 时保留其保存的策略。
显式设置 `full` 保留全部 token，设置 `default` 去掉首个 token。
自定义视觉塔与连接器组合见[扩展指南](extending.md)。

## 加载 checkpoint

| 任务 | checkpoint 设置 |
| --- | --- |
| 初始化训练阶段 | `stages.finetune.checkpoint=/path/to/checkpoint` |
| 评测 | `model_name_or_path=/path/to/checkpoint` |
| 单轮推理或多轮对话 | `--model-path /path/to/checkpoint` |

后续训练阶段默认加载上一阶段输出。`from_stage` 可选择更早的阶段，`checkpoint: null`
表示重新组装组件。checkpoint 提供保存的组件配置和权重。
`training.resume_from_checkpoint` 还会恢复训练状态，详见[训练指南](training.md)。

Python 加载与生成示例见[推理指南](inference.md)，类与方法签名见[模型 API](../reference/model.md)。

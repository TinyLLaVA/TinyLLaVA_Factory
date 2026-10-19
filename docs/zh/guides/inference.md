# 推理

使用保存了 tokenizer 和 image processor 的组合 checkpoint。

```bash
python -m tinyllava.cli single-turn \
  --model-path output/my-finetune \
  --image-file tinyllava/serve/examples/waterview.jpg \
  --query "Describe the image." \
  --temperature 0 --max-new-tokens 128
```

CLI checkpoint 选项目前为 `--model-path`，评测 YAML 则为 `model_name_or_path`。模型与 processor 支持多图时可重复传入 `--image-file`。

## CLI 选项 { #options }

| 选项 | 用途 |
| --- | --- |
| `--model-path` | 已训练 checkpoint 的目录或 Hub ID。 |
| `--query` | 用户提问，`single-turn` 必填，`chat` 可选。 |
| `--image-file` | 图片路径；模型支持多图时可重复传入。 |
| `--device` | 运行设备，默认优先 CUDA，否则 CPU。 |
| `--chat-template-path` | Jinja 对话模板文件。 |
| `--temperature` | 采样温度，`0` 关闭采样。 |
| `--top-p` | 核采样的概率阈值。 |
| `--num-beams` | 解码束数，默认 `1`。 |
| `--max-new-tokens` | 回答 token 上限，这两个 CLI 命令默认 `512`。 |

## 终端多轮对话

`chat` 在轮次之间保留对话历史。启动时用 `--image-file` 添加图像，对话中用 `/image PATH` 为下一轮添加图像。`/reset` 清空历史和待发送图像，`/exit` 退出。

```bash
python -m tinyllava.cli chat --model-path output/my-finetune
```

## Python 接口

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

批量调用使用 `generate_responses` 和 `messages_batch`，每段输入对话返回一个答案。

```bash
python -m tinyllava.serve.app --model-path output/my-finetune
```

Web UI 复用同一套加载和生成接口，需要额外的 Gradio 依赖。参见[评测 API](../reference/evaluation.md)。

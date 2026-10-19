# 配置方式

用 `--config` 选择 YAML 入口，再用 `key=value` 覆写具体值。
字段含义见[参数速查](parameters.md)，组件与 checkpoint 的选择见[模型指南](models.md)。

## 训练阶段

```yaml
# configs/experiments/custom.yaml
model: phi

stages:
  pretrain:
    algorithm: sft
    data: llava_558k
    training: pretrain

  finetune:
    algorithm: sft
    data: llava_665k
    training:
      from: lora_finetune
      learning_rate: 0.0001
```

每个 stage 选择自己的 `algorithm` 和 `data`。根部可以共享 `model`、`training`、
`launcher`；阶段值覆盖共享值，包括引用的预设中的值。
单阶段入口省略 `stages`，将算法、数据、模型和训练设置直接放在根部。

`save_steps`、`logging_steps` 等 Trainer 参数直接放在 `training` 下。
组件更新策略位于 `training.tuning`，LoRA 参数位于 `training.tuning.peft`。

## 复用预设

`from` 将 YAML 文件加载到当前节点，本地字段覆盖文件中的值：

```yaml
model:
  from: phi
  model_max_length: 4096
  connector:
    from: mlp2x_gelu
    depth: 3
```

只选择文件时，`model: phi` 等价于 `model: {from: phi}`。不带路径的名称在以下目录查找：

| 节点 | 预设目录 |
| --- | --- |
| `model` | `configs/models/` |
| `connector` | `configs/connectors/` |
| `training` | `configs/training/` |
| `data` | `configs/data/train/` |

阶段内和仓库外的入口文件同样适用。省略扩展名时补 `.yaml`，也支持显式 `.yaml` 和 `.yml`。
用 `./custom.yaml` 选择当前 YAML 旁的文件。带目录的相对路径以声明它的 YAML 为基准，
也支持绝对路径和 `~`；命令行中的显式相对路径以工作目录为基准。
checkpoint、标注、图片和模板路径保持为普通字段值。

通过列表组合文件，例如 `training: {from: [pretrain, ./hardware]}`。
来源文件从左到右合并，再由本地字段、命令行覆写。字典递归合并，列表和标量整体替换。
切换引用时保留显式同级覆写，并替换旧预设的默认值。根部和列表内的字典也支持 `from`。

## 评测入口

评测字段直接位于根部，通过根部引用复用数据定义：

```yaml
# configs/eval/custom.yaml
from: ../data/eval/gqa.yaml
benchmark: gqa

model_name_or_path: output/phi/finetune
temperature: 0.0
batch_size: 4
output_dir: output/custom-eval
```

`output_dir` 指定结果目录，`answers_file` 指定预测文件；评测流程中后者默认是
`${output_dir}/answers.jsonl`。数据准备、分片和计分见[评测指南](evaluation.md)。

## 覆写与预览

```bash
python -m tinyllava.cli train --config configs/experiments/phi.yaml \
  model=phi model.model_max_length=4096 \
  stages.finetune.training.learning_rate=1e-5 --dry-run

python -m tinyllava.cli eval --config configs/eval/gqa.yaml \
  model_name_or_path=output/phi/finetune batch_size=4 --dry-run
```

`--dry-run` 打印解析后的设置与命令。用 `--steps finetune` 选择训练阶段，
或用 `--steps generate` 生成评测答案。选择训练阶段时仍保留 checkpoint 依赖，详见[训练指南](training.md)。

普通值支持 OmegaConf 插值，例如 `${dataset_dir}/images` 和
`${oc.env:EVAL_CHAT_TEMPLATE_PATH,null}`；`from` 使用字面路径。
Python 配置接口见[API 参考](../reference/configuration.md)。

# 评测

生成答案与计算分数是两步。`tinyllava.eval.batch_generation` 生成预测，`configs/eval/` 执行各任务的转换或本地计分；VQAv2 和 MM-Vet 需要额外评测流程。数据准备参考 [LLaVA 评测说明](https://github.com/haotian-liu/LLaVA/blob/main/docs/Evaluation.md)。

```bash
export MODEL_PATH="$PWD/output/phi/finetune"
export MODEL_NAME="my-finetune"
export EVAL_DIR="${PWD}/datasets/eval"
python -m tinyllava.run --config configs/eval/textvqa.yaml
```

支持的入口包括 `vqav2.yaml`、`gqa.yaml`、`scienceqa.yaml`、`textvqa.yaml`、`pope.yaml`、`mme.yaml`、`mmvet.yaml` 和 `mmmu.yaml`。ScienceQA 配置名是 `scienceqa.yaml`，MM-Vet 数据目录名是 `mm-vet`。

仅生成答案或覆盖 batch：

```bash
python -m tinyllava.eval.batch_generation \
  --config configs/eval/textvqa.yaml \
  runtime.batch_size=4 runtime.device=cuda:0
```

评测 YAML 包含 `model`、`data`、`generation`、`runtime`、`output`。多 GPU 评测需为各分片分别启动进程，设置独立的 `runtime.chunk_idx`、设备和 `output.answers_file`，并使用相同的 `runtime.num_chunks`。全部完成后合并答案文件。

## legacy 配置与分数对齐

```bash
export MODEL_PATH="$PWD/output/qwen2_base_legacy/finetune"
export MODEL_NAME="qwen2-base-legacy"
python -m tinyllava.run --config configs/eval/scienceqa_qwen2_base_legacy.yaml
```

该入口选择 legacy 提示词；ScienceQA 配置允许最多 1024 个生成 token，与通用配置不同。比较论文时要对齐 SQA-image 子集、MME perception 指标、POPE 聚合口径，并保留 checkpoint ID、预测文件、样本数和计分命令。导出的答案需交给相应评测器计分。

GQA 和 VQAv2 流程按 `CUDA_VISIBLE_DEVICES`（或 `devices` 覆盖值）分片，全部成功后再合并。使用 `--steps generate` 仅生成答案，按配置中的阶段名称选择转换或计分步骤。`--dry-run` 只预览配置和命令，不加载模型。流程的参数覆盖写作 `runtime.batch_size=4`，直接生成时仍用 `runtime.batch_size=4`。

默认数据根目录为 `datasets/eval/`。对 `output/` 下的 checkpoint 评测时，结果自动存入 `output/<实验名>/<阶段名>/eval/<评测集>/`，可用 `output_dir` 修改。MME 的转换器在独立输出工作目录执行，数据目录只用于输入。各评测集的输入路径统一定义于 `configs/data/eval/`。

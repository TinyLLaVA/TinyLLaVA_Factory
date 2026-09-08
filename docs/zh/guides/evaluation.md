# 评测

生成答案与计算分数是两步。`tinyllava.eval.batch_generation` 生成预测，`scripts/eval/` 执行各任务的转换或本地计分；VQAv2 和 MM-Vet 需要额外评测流程。数据准备参考 [LLaVA 评测说明](https://github.com/haotian-liu/LLaVA/blob/main/docs/Evaluation.md)。

```bash
export MODEL_PATH="$PWD/output/my-finetune"
export MODEL_NAME="my-finetune"
export EVAL_DIR="/path/to/dataset/eval"
bash scripts/eval/textvqa.sh
```

支持的入口包括 `vqav2.sh`、`gqa.sh`、`sqa.sh`、`textvqa.sh`、`pope.sh`、`mme.sh`、`mmvet.sh` 和 `mmmu.sh`。注意 ScienceQA 脚本名是 `sqa.sh`，MM-Vet 数据目录名是 `mm-vet`。

仅生成答案或覆盖 batch：

```bash
python -m tinyllava.eval.batch_generation \
  --config configs/eval/textvqa.yaml \
  runtime.batch_size=4 runtime.device=cuda:0
```

评测 YAML 包含 `model`、`data`、`generation`、`runtime`、`output`。多 GPU 评测需为各分片分别启动进程，设置独立的 `runtime.chunk_idx`、设备和 `output.answers_file`，并使用相同的 `runtime.num_chunks`。全部完成后合并答案文件。

## legacy 配置与分数对齐

```bash
export MODEL_PATH="$PWD/output/tinyllava-qwen2-base-legacy-finetune"
export MODEL_NAME="qwen2-base-legacy"
bash scripts/eval/eval_qwen2_base_legacy.sh scienceqa
```

该入口选择 legacy 提示词；ScienceQA 配置允许最多 1024 个生成 token，与通用配置不同。比较论文时要对齐 SQA-image 子集、MME perception 指标、POPE 聚合口径，并保留 checkpoint ID、预测文件、样本数和计分命令。导出的答案需交给相应评测器计分。

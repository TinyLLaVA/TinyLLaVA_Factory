# Evaluation

Generation and scoring are separate steps. `tinyllava.eval.batch_generation`
writes predictions; the workflows in `configs/eval/` also perform benchmark-specific
conversion or local scoring. VQAv2 and MM-Vet exports require a separate evaluator.

Prepare the LLaVA evaluation assets following the
[upstream instructions](https://github.com/haotian-liu/LLaVA/blob/main/docs/Evaluation.md).
For the custom MMMU layout, use the
[TinyLLaVA MMMU bundle](https://drive.google.com/file/d/1TJszQ23X-7TeMYDA7hVKpoHy9yo-lsc5/view).

| Benchmark | Required files under `EVAL_DIR` | Workflow YAML |
| --- | --- | --- |
| VQAv2 | `vqav2/test2015/`, `llava_vqav2_mscoco_test2015.jsonl` in `vqav2/` | `vqav2.yaml` |
| GQA | `gqa/images/`, `llava_gqa_testdev_balanced.jsonl`, official scoring assets in `gqa/` | `gqa.yaml` |
| ScienceQA | `scienceqa/images/test/`, `llava_test_CQM-A.json`, `problems.json`, `pid_splits.json` in `scienceqa/` | `scienceqa.yaml` |
| TextVQA | `textvqa/train_images/`, `llava_textvqa_val_v051_ocr.jsonl`, `TextVQA_0.5.1_val.json` in `textvqa/` | `textvqa.yaml` |
| POPE | `pope/val2014/`, `pope/coco/`, `pope/llava_pope_test.jsonl` | `pope.yaml` |
| MME | `MME/MME_Benchmark_release_version/`, `MME/eval_tool/`, `MME/llava_mme.jsonl`, conversion script | `mme.yaml` |
| MM-Vet | `mm-vet/images/`, `mm-vet/llava-mm-vet.jsonl` | `mmvet.yaml` |
| MMMU | `MMMU/all_images/`, `MMMU/anns_for_eval.json`, `MMMU/eval/` | `mmmu.yaml` |

## Run a benchmark

```bash
export MODEL_PATH="$PWD/output/phi/finetune"
export MODEL_NAME="my-finetune"
export EVAL_DIR="${PWD}/datasets/eval"
python -m tinyllava.run --config configs/eval/textvqa.yaml
```

To run only generation, or change the batch size:

```bash
python -m tinyllava.eval.batch_generation \
  --config configs/eval/textvqa.yaml \
  runtime.batch_size=4 runtime.device=cuda:0
```

Evaluation YAML has `model`, `data`, `generation`, `runtime`, and `output`
sections. For multi-GPU evaluation, launch one process per shard with its own
`runtime.chunk_idx`, device, and `output.answers_file`; set the same
`runtime.num_chunks` for all processes. Merge the answer files after every shard finishes.

## Legacy Qwen2 evaluation

```bash
export MODEL_PATH="$PWD/output/qwen2_base_legacy/finetune"
export MODEL_NAME="qwen2-base-legacy"
python -m tinyllava.run --config configs/eval/scienceqa_qwen2_base_legacy.yaml
```

The workflow selects the legacy prompt, including a dedicated ScienceQA
configuration with a 1024-token cap. Generic ScienceQA uses different generation
settings. Keep training and evaluation templates aligned.

Compare SQA-image against image-subset accuracy, MME perception against the
perception score, and POPE using the same metric/category aggregation. Retain
predictions, sample counts, scoring commands, and checkpoint IDs. Submit exported
answers to the benchmark evaluator to obtain the final score.

GQA and VQAv2 workflows shard across `CUDA_VISIBLE_DEVICES` (or the `devices` override), then merge only after all workers succeed. Use `--steps generate` for generation only, or `--steps convert,score` to rerun downstream steps where available. `--dry-run` previews the workflow without loading models. Workflow overrides use `runtime.batch_size=4`; direct generation keeps `runtime.batch_size=4`.

The default input root is `datasets/eval/`. Predictions and conversions are stored under `output/<experiment>/<stage>/eval/<benchmark>/` when evaluating a checkpoint under `output/`. Override `output_dir` for another result directory. MME runs its converter in an isolated output workspace; the dataset tree is used only as input. Benchmark input definitions are shared through `configs/data/eval/`.

"""Run a training experiment or evaluation config: python -m tinyllava.run --config PATH."""

from __future__ import annotations

import argparse
import shutil
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from tinyllava.configuration import (
    EVAL_SECTIONS,
    load_eval_config,
    read_mapping,
    training_stages,
)
from tinyllava.runner import prepare_step, run_step


def train_plan(
    path: str | Path, overrides: Sequence[str] = ()
) -> list[tuple[str, dict[str, Any], dict[str, Any] | None]]:
    result = []
    for name, config, launcher in training_stages(path, overrides):
        step = {"kind": "train", **launcher}
        config = prepare_step({**step, "resolved": config})
        result.append((name, step, config))
    return result


def eval_plan(
    path: str | Path, overrides: Sequence[str] = ()
) -> list[tuple[str, dict[str, Any], dict[str, Any] | None]]:
    config = load_eval_config(path, overrides, metadata=True)
    benchmark = config["benchmark"]
    data = Path(config["eval_dir"]).expanduser().resolve()
    output = Path(config["output_dir"]).expanduser().resolve()
    answers = Path(config["output"]["answers_file"]).expanduser().resolve()
    config["output"]["answers_file"] = str(answers)
    generation = {k: v for k, v in config.items() if k in EVAL_SECTIONS}
    result = [
        (
            "generate",
            {
                "kind": "generate",
                "shard": config.get("shard", False),
                "devices": config.get("devices", "auto"),
            },
            generation,
        )
    ]
    base = [
        "{python}",
        "-m",
        "tinyllava.eval.tasks.auto.evaluation_auto",
        benchmark,
        "--prediction-file",
        str(answers),
    ]
    converted = str(output / "predictions.json")

    def command(name: str, args: list[str], cwd: str | Path | None = None) -> None:
        result.append(
            (
                name,
                {
                    "kind": "command",
                    "args": args,
                    "cwd": str(cwd) if cwd else None,
                    "mkdir": [str(output)],
                },
                None,
            )
        )

    if benchmark == "gqa":
        command("convert", base + ["--output-file", converted])
        command(
            "score",
            [
                "{python}",
                str(data / "gqa/eval/eval.py"),
                "--tier",
                "testdev_balanced",
                "--predictions",
                converted,
            ],
            cwd=data / "gqa",
        )
    elif benchmark == "vqav2":
        command(
            "convert",
            base
            + [
                "--split-file",
                str(data / "vqav2/llava_vqav2_mscoco_test2015.jsonl"),
                "--output-file",
                converted,
            ],
        )
    elif benchmark == "scienceqa":
        command(
            "score",
            base
            + [
                "--base-dir",
                str(data / "scienceqa"),
                "--output-file",
                str(output / "predictions.jsonl"),
                "--output-result-file",
                str(output / "metrics.json"),
            ],
        )
    elif benchmark == "textvqa":
        command(
            "score",
            base + ["--annotation-file", str(data / "textvqa/TextVQA_0.5.1_val.json")],
        )
    elif benchmark == "pope":
        command(
            "score",
            base
            + [
                "--annotation-dir",
                str(data / "pope/coco"),
                "--question-file",
                str(data / "pope/llava_pope_test.jsonl"),
            ],
        )
    elif benchmark == "mmvet":
        command("convert", base + ["--output-file", converted])
    elif benchmark == "mmmu":
        command("convert", base + ["--output-file", converted])
        command(
            "score",
            [
                "{python}",
                str(data / "MMMU/eval/main_eval_only.py"),
                "--output_path",
                converted,
            ],
            cwd=data / "MMMU/eval",
        )
    elif benchmark == "mme":
        # The released converter uses fixed relative paths. Give it an isolated
        # workspace under output/, with read-only inputs linked from the dataset.
        workspace = output / "mme"
        result.append(
            (
                "prepare",
                {
                    "kind": "mme_workspace",
                    "workspace": str(workspace),
                    "data": str(data / "MME/MME_Benchmark_release_version"),
                    "answers": str(answers),
                },
                None,
            )
        )
        command(
            "convert",
            [
                "{python}",
                str(data / "MME/convert_answer_to_mme.py"),
                "--experiment",
                "model",
            ],
            cwd=workspace,
        )
        command(
            "score",
            [
                "{python}",
                str(data / "MME/eval_tool/calculation.py"),
                "--results_dir",
                str(workspace / "eval_tool/answers/model"),
            ],
        )
    else:
        raise ValueError(f"No scoring workflow for benchmark {benchmark!r}")
    return result


def run_plan(
    plan: list[tuple[str, dict[str, Any], dict[str, Any] | None]],
    *,
    steps: Sequence[str] | None = None,
    dry_run: bool = False,
) -> None:
    names = [name for name, _, _ in plan]
    selected = names if steps is None else steps
    unknown = set(selected) - set(names)
    if unknown or not selected:
        raise ValueError(
            f"Unknown or empty stage selection: {selected}; available: {names}"
        )
    by_name = {name: (step, config) for name, step, config in plan}
    for name in selected:
        step, config = by_name[name]
        print(f"Stage: {name}", flush=True)
        if step["kind"] == "mme_workspace":
            print(f"MME workspace: {step['workspace']}", flush=True)
            if not dry_run:
                workspace = Path(step["workspace"])
                (workspace / "answers").mkdir(parents=True, exist_ok=True)
                link = workspace / "MME_Benchmark_release_version"
                if not link.exists():
                    link.symlink_to(step["data"], target_is_directory=True)
                shutil.copyfile(step["answers"], workspace / "answers/model.jsonl")
        else:
            with tempfile.TemporaryDirectory(prefix="tinyllava-run-") as directory:
                run_step(step, config, directory, dry_run=dry_run)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--steps", help="Comma-separated stage names; default runs YAML order"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    spec = read_mapping(args.config)
    is_eval = "generation" in spec or "runtime" in spec or "benchmark" in spec
    plan = (
        eval_plan(args.config, args.overrides)
        if is_eval
        else train_plan(args.config, args.overrides)
    )
    run_plan(
        plan, steps=args.steps.split(",") if args.steps else None, dry_run=args.dry_run
    )


if __name__ == "__main__":
    main()

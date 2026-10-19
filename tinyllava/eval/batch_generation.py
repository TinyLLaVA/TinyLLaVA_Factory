"""Batch generation loop for evaluation datasets."""

from __future__ import annotations

import argparse
import json
import math
import os
from collections.abc import Sequence
from typing import Any

import shortuuid
import torch
from tqdm import tqdm

from tinyllava.data.chat_template.loading import resolve_chat_template
from tinyllava.eval.generation import generate_responses, load_images, make_user_message
from tinyllava.eval.tasks.auto.loader_auto import AutoDatasetLoader
from tinyllava.eval.tasks.loader_base import DatasetLoader
from tinyllava.hparams.arguments import EvalArguments
from tinyllava.utils.config import load_config
from tinyllava.utils.model_loading import load_tinyllava_checkpoint_bundle


def split_list(items: list[Any], n: int) -> list[list[Any]]:
    chunk_size = math.ceil(len(items) / n)
    return [items[i * chunk_size : (i + 1) * chunk_size] for i in range(n)]


def get_chunk(items: list[Any], n: int, k: int) -> list[Any]:
    return split_list(items, n)[k]


def iter_batches(items: list[Any], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def use_left_padding_for_generation(processor: Any) -> None:
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None:
        tokenizer.padding_side = "left"


def run_generation(
    config: EvalArguments,
    adapter: DatasetLoader | str | None = None,
) -> None:
    if adapter is None:
        adapter = config.adapter
    if isinstance(adapter, str):
        adapter = AutoDatasetLoader.from_name(adapter)

    model_path = os.path.expanduser(config.model_name_or_path)
    device = config.device or ("cuda" if torch.cuda.is_available() else "cpu")
    bundle = load_tinyllava_checkpoint_bundle(model_path, device=device)
    use_left_padding_for_generation(bundle.processor)

    samples = adapter.load_samples(config.question_file)
    samples = get_chunk(
        samples,
        config.num_chunks,
        config.chunk_idx,
    )
    answers_file = os.path.expanduser(config.answers_file)
    answer_dir = os.path.dirname(answers_file)
    if answer_dir:
        os.makedirs(answer_dir, exist_ok=True)

    model_id = config.model_id or os.path.basename(model_path)
    chat_template = resolve_chat_template(
        chat_template=config.chat_template,
        chat_template_path=config.chat_template_path,
    )
    with open(answers_file, "w", encoding="utf-8") as ans_file:
        for sample_batch in tqdm(
            iter_batches(samples, config.batch_size),
            total=math.ceil(len(samples) / config.batch_size),
        ):
            examples = [
                adapter.make_example(
                    sample,
                    image_folder=config.image_folder,
                )
                for sample in sample_batch
            ]
            messages_batch = [
                [
                    make_user_message(
                        example.prompt,
                        load_images(example.image_files),
                    )
                ]
                for example in examples
            ]
            responses = generate_responses(
                model=bundle.model,
                processor=bundle.processor,
                messages_batch=messages_batch,
                chat_template=chat_template,
                temperature=config.temperature,
                top_p=config.top_p,
                num_beams=config.num_beams,
                max_new_tokens=config.max_new_tokens,
            )
            for sample, example, response in zip(
                sample_batch, examples, responses, strict=True
            ):
                answer = adapter.process_response(sample, response)
                ans_file.write(
                    json.dumps(
                        {
                            "question_id": example.question_id,
                            "prompt": example.prompt,
                            "text": answer,
                            "answer_id": shortuuid.uuid(),
                            "model_id": model_id,
                            "metadata": example.metadata,
                        }
                    )
                    + "\n"
                )
            ans_file.flush()


def main(args: Sequence[str] | None = None) -> None:
    """Read resolved worker YAML, apply field overrides, and generate answers."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Resolved evaluation YAML.")
    parser.add_argument("overrides", nargs="*", help="OmegaConf dotlist overrides.")
    options = parser.parse_args(args)
    config = load_config(options.config, options.overrides)
    run_generation(EvalArguments(**config))


if __name__ == "__main__":
    main()


__all__ = [
    "get_chunk",
    "iter_batches",
    "main",
    "run_generation",
    "split_list",
    "use_left_padding_for_generation",
]

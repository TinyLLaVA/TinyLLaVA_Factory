"""Batch generation loop for evaluation datasets."""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Any

import shortuuid
import torch
from tqdm import tqdm

from tinyllava.eval.dataset_adapters.auto.loader_auto import AutoDatasetLoader
from tinyllava.eval.dataset_adapters.loader_base import DatasetLoader
from tinyllava.eval.generation import generate_response, load_images, make_user_message
from tinyllava.utils.model_loading import load_tinyllava_checkpoint_bundle


def split_list(items: list[Any], n: int) -> list[list[Any]]:
    chunk_size = math.ceil(len(items) / n)
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


def get_chunk(items: list[Any], n: int, k: int) -> list[Any]:
    return split_list(items, n)[k]


def run_generation(args: Any, adapter: DatasetLoader | str) -> None:
    if isinstance(adapter, str):
        adapter = AutoDatasetLoader.from_name(adapter)

    model_path = os.path.expanduser(args.model_path)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    bundle = load_tinyllava_checkpoint_bundle(model_path, device=device)

    samples = adapter.load_samples(args.question_file)
    samples = get_chunk(samples, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    answer_dir = os.path.dirname(answers_file)
    if answer_dir:
        os.makedirs(answer_dir, exist_ok=True)

    model_id = args.model_base or os.path.basename(model_path)
    with open(answers_file, "w", encoding="utf-8") as ans_file:
        for sample in tqdm(samples):
            example = adapter.make_example(sample, image_folder=args.image_folder, args=args)
            response = generate_response(
                model=bundle.model,
                processor=bundle.processor,
                messages=[make_user_message(example.prompt, load_images(example.image_files))],
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
            )
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


def build_parser(
    default_adapter: str = "vqa",
    *,
    default_max_new_tokens: int = 128,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=str, default=default_adapter)
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", "--top_p", dest="top_p", type=float, default=None)
    parser.add_argument("--num-beams", "--num_beams", dest="num_beams", type=int, default=1)
    parser.add_argument(
        "--max-new-tokens",
        "--max_new_tokens",
        dest="max_new_tokens",
        type=int,
        default=default_max_new_tokens,
    )
    parser.add_argument("--single-pred-prompt", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    return parser


def main(default_adapter: str = "vqa") -> None:
    args = build_parser(default_adapter).parse_args()
    run_generation(args, args.adapter)


if __name__ == "__main__":
    main()


__all__ = ["build_parser", "get_chunk", "main", "run_generation", "split_list"]

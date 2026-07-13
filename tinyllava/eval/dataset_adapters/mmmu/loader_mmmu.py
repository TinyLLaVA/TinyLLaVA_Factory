"""Load MMMU samples for generation and parse multiple-choice responses."""

from __future__ import annotations

import json
import os
import random
from collections.abc import Mapping
from typing import Any

import numpy as np

from tinyllava.data.image_payload import resolve_image_payload
from tinyllava.eval.dataset_adapters.loader_base import GenerationExample
from tinyllava.eval.generation import strip_legacy_image_markers


class MmmuLoader:
    """Loader for the local MMMU evaluation JSON format."""

    def load_samples(self, question_file: str) -> list[Mapping[str, Any]]:
        with open(os.path.expanduser(question_file), encoding="utf-8") as f:
            return json.load(f)

    def make_example(
        self,
        sample: Mapping[str, Any],
        *,
        image_folder: str,
        args: Any,
    ) -> GenerationExample:
        image_files = []
        if "image" in sample:
            image_files.append(resolve_image_payload(sample["image"], image_folder=image_folder))

        return GenerationExample(
            question_id=str(sample["id"]),
            prompt=strip_legacy_image_markers(str(sample["prompt"])),
            image_files=image_files,
            metadata={"has_image": bool(image_files)},
        )

    def process_response(self, sample: Mapping[str, Any], response: str) -> str:
        if sample.get("question_type") != "multiple-choice":
            return response
        return parse_multi_choice_response(
            response,
            sample["all_choices"],
            sample["index2ans"],
        )


def parse_multi_choice_response(
    response: str,
    all_choices: list[str],
    index2ans: Mapping[str, str],
) -> str:
    for char in [",", ".", "!", "?", ";", ":", "'"]:
        response = response.strip(char)
    response = " " + response + " "

    index_ans = True
    ans_with_brack = False
    candidates = []
    for choice in all_choices:
        if f"({choice})" in response:
            candidates.append(choice)
            ans_with_brack = True

    if len(candidates) == 0:
        for choice in all_choices:
            if f" {choice} " in response:
                candidates.append(choice)

    if len(candidates) == 0 and len(response.split()) > 5:
        for index, ans in index2ans.items():
            if ans.lower() in response.lower():
                candidates.append(index)
                index_ans = False

    if len(candidates) == 0:
        return random.choice(all_choices)
    if len(candidates) == 1:
        return candidates[0]

    start_indexes = []
    if index_ans:
        for candidate in candidates:
            pattern = f"({candidate})" if ans_with_brack else f" {candidate} "
            start_indexes.append(response.rfind(pattern))
    else:
        for candidate in candidates:
            start_indexes.append(response.lower().rfind(index2ans[candidate].lower()))
    return candidates[np.argmax(start_indexes)]

import json

import pytest

from tinyllava.eval.tasks.auto.loader_auto import AutoDatasetLoader


@pytest.mark.parametrize("with_image", [False, True])
def test_scienceqa_loader_requests_option_letter(tmp_path, with_image):
    question = "Which color is shown?\n(A) Red\n(B) Blue"
    sample = {
        "id": 7,
        "conversations": [{"from": "human", "value": f"<image>\n{question}"}],
    }
    if with_image:
        sample["image"] = "7/image.png"
    source = tmp_path / "scienceqa.json"
    source.write_text(json.dumps([sample]), encoding="utf-8")
    loader = AutoDatasetLoader.from_name("scienceqa")

    loaded = loader.load_samples(str(source))
    example = loader.make_example(loaded[0], image_folder=str(tmp_path))

    assert example.question_id == "7"
    assert example.prompt == (
        question + "\nAnswer with the option's letter from the given choices directly."
    )
    assert example.image_files == (
        [str(tmp_path / "7/image.png")] if with_image else []
    )
    assert example.metadata == {"has_image": with_image}


@pytest.mark.parametrize(
    "name,sample",
    [
        ("vqa", {"question_id": 7, "text": "<image>\nWhat color?", "image": "7.png"}),
        ("pope", {"question_id": 7, "text": "<image>\nWhat color?", "image": "7.png"}),
        ("mmmu", {"id": 7, "prompt": "<image>\nWhat color?", "image": "7.png"}),
    ],
)
def test_other_loaders_preserve_task_prompt(tmp_path, name, sample):
    loader = AutoDatasetLoader.from_name(name)
    example = loader.make_example(sample, image_folder=str(tmp_path))
    assert example.question_id == "7"
    assert example.prompt == "What color?"
    assert example.image_files == [str(tmp_path / "7.png")]

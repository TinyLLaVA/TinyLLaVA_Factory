import json
from types import SimpleNamespace

from tinyllava.eval import batch_generation
from tinyllava.eval.config import build_eval_config
from tinyllava.eval.tasks.loader_base import GenerationExample


class FakeAdapter:
    def __init__(self):
        self.data_args = None

    def load_samples(self, question_file):
        assert question_file == "questions.jsonl"
        return [{"id": "question-1"}]

    def make_example(self, sample, *, image_folder, args):
        assert image_folder == "images"
        self.data_args = args
        return GenerationExample(question_id=sample["id"], prompt="What is shown?")

    def process_response(self, sample, response):
        assert sample["id"] == "question-1"
        return response.upper()


def test_run_generation_uses_structured_config(monkeypatch, tmp_path):
    answers_file = tmp_path / "answers" / "predictions.jsonl"
    config = build_eval_config(
        {
            "model": {"model_path": "checkpoint", "model_id": "model-name"},
            "data": {
                "question_file": "questions.jsonl",
                "image_folder": "images",
                "single_pred_prompt": True,
            },
            "generation": {
                "temperature": 0.0,
                "top_p": 0.9,
                "num_beams": 2,
                "max_new_tokens": 64,
            },
            "runtime": {"device": "cpu"},
            "output": {"answers_file": str(answers_file)},
        }
    )
    adapter = FakeAdapter()
    generation_kwargs = {}

    monkeypatch.setattr(
        batch_generation,
        "load_tinyllava_checkpoint_bundle",
        lambda model_path, device: SimpleNamespace(model=object(), processor=object()),
    )
    monkeypatch.setattr(batch_generation, "load_images", lambda image_files: [])

    def fake_generate_response(**kwargs):
        generation_kwargs.update(kwargs)
        return "answer"

    monkeypatch.setattr(batch_generation, "generate_response", fake_generate_response)

    batch_generation.run_generation(config, adapter)

    assert adapter.data_args.single_pred_prompt is True
    assert generation_kwargs["temperature"] == 0.0
    assert generation_kwargs["top_p"] == 0.9
    assert generation_kwargs["num_beams"] == 2
    assert generation_kwargs["max_new_tokens"] == 64
    prediction = json.loads(answers_file.read_text(encoding="utf-8"))
    assert prediction["question_id"] == "question-1"
    assert prediction["text"] == "ANSWER"
    assert prediction["model_id"] == "model-name"

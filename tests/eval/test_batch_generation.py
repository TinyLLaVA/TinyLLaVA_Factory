import json
from types import SimpleNamespace

from tinyllava.eval import batch_generation
from tinyllava.utils.config import build_eval_arguments
from tinyllava.eval.tasks.loader_base import GenerationExample


class FakeAdapter:
    def __init__(self):
        self.data_args = None

    def load_samples(self, question_file):
        assert question_file == "questions.jsonl"
        return [
            {"id": "question-1"},
            {"id": "question-2"},
            {"id": "question-3"},
        ]

    def make_example(self, sample, *, image_folder, args):
        assert image_folder == "images"
        self.data_args = args
        return GenerationExample(
            question_id=sample["id"],
            prompt=f"What is shown in {sample['id']}?",
        )

    def process_response(self, sample, response):
        return response.upper()


def test_run_generation_uses_structured_config(monkeypatch, tmp_path):
    answers_file = tmp_path / "answers" / "predictions.jsonl"
    config = build_eval_arguments(
        {
            "model": {"model_name_or_path": "checkpoint", "model_id": "model-name"},
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
            "runtime": {"device": "cpu", "batch_size": 2},
            "output": {"answers_file": str(answers_file)},
        }
    )
    adapter = FakeAdapter()
    generation_batches = []
    tokenizer = SimpleNamespace(padding_side="right")
    processor = SimpleNamespace(tokenizer=tokenizer)

    monkeypatch.setattr(
        batch_generation,
        "load_tinyllava_checkpoint_bundle",
        lambda model_path, device: SimpleNamespace(model=object(), processor=processor),
    )
    monkeypatch.setattr(batch_generation, "load_images", lambda image_files: [])

    def fake_generate_responses(**kwargs):
        generation_batches.append(kwargs)
        return [
            f"answer-{len(generation_batches)}-{idx}"
            for idx, _messages in enumerate(kwargs["messages_batch"])
        ]

    monkeypatch.setattr(batch_generation, "generate_responses", fake_generate_responses)

    batch_generation.run_generation(config, adapter)

    assert adapter.data_args.single_pred_prompt is True
    assert tokenizer.padding_side == "left"
    assert [len(batch["messages_batch"]) for batch in generation_batches] == [2, 1]
    assert generation_batches[0]["temperature"] == 0.0
    assert generation_batches[0]["top_p"] == 0.9
    assert generation_batches[0]["num_beams"] == 2
    assert generation_batches[0]["max_new_tokens"] == 64
    predictions = [
        json.loads(line)
        for line in answers_file.read_text(encoding="utf-8").splitlines()
    ]
    assert [prediction["question_id"] for prediction in predictions] == [
        "question-1",
        "question-2",
        "question-3",
    ]
    assert [prediction["text"] for prediction in predictions] == [
        "ANSWER-1-0",
        "ANSWER-1-1",
        "ANSWER-2-0",
    ]
    assert {prediction["model_id"] for prediction in predictions} == {"model-name"}

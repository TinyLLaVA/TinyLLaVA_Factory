import json

from tinyllava.eval.tasks.pope.evaluation_pope import evaluate_pope


def _write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_pope_evaluation_matches_integer_questions_to_string_predictions(tmp_path):
    question_file = tmp_path / "questions.jsonl"
    prediction_file = tmp_path / "predictions.jsonl"
    annotation_dir = tmp_path / "annotations"
    annotation_dir.mkdir()

    _write_jsonl(
        question_file,
        [{"question_id": 1, "category": "random"}],
    )
    _write_jsonl(
        prediction_file,
        [{"question_id": "1", "text": "Yes"}],
    )
    _write_jsonl(
        annotation_dir / "coco_pope_random.json",
        [{"question_id": 1, "label": "yes"}],
    )

    scores = evaluate_pope(
        annotation_dir=str(annotation_dir),
        question_file=str(question_file),
        prediction_file=str(prediction_file),
    )

    assert len(scores) == 1
    assert scores[0].accuracy == 1.0
    assert scores[0].true_positive == 1

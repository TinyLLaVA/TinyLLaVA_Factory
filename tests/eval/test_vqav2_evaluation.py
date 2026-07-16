import json

from tinyllava.eval.tasks.vqav2.evaluation_vqav2 import convert_vqav2_submission


def _write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_vqav2_conversion_matches_integer_split_to_string_prediction(tmp_path):
    prediction_file = tmp_path / "predictions.jsonl"
    split_file = tmp_path / "split.jsonl"
    output_file = tmp_path / "submission.json"
    _write_jsonl(
        prediction_file,
        [{"question_id": "42", "text": "Two."}],
    )
    _write_jsonl(split_file, [{"question_id": 42}])

    stats = convert_vqav2_submission(
        prediction_file=str(prediction_file),
        split_file=str(split_file),
        output_file=str(output_file),
    )

    assert stats == {
        "predictions": 1,
        "split": 1,
        "missing": 0,
        "error_lines": 0,
    }
    assert json.loads(output_file.read_text()) == [
        {"question_id": 42, "answer": "2"}
    ]

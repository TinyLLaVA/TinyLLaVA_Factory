import argparse
import json
import os


def eval_model(args):
    answers = [json.loads(q) for q in open(os.path.expanduser(args.answers_file), "r")]
    answers_dict = {}
    for answer in answers:
        answers_dict[answer["question_id"]] = answer["text"]
        # print(answer)

    with open(args.answers_output, "w") as f:
        json.dump(answers_dict, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--answers-file",
        type=str,
        required=True
    )
    parser.add_argument(
        "--answers-output",
        type=str,
        required=True
    )
    args = parser.parse_args()

    eval_model(args)

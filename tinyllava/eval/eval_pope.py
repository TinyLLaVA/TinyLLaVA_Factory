import os
import json
import argparse


def eval_pope(answers, label_file):
    label_list = [json.loads(q)["label"] for q in open(label_file)]

    for answer in answers:
        text = answer["text"]

        # Only keep the first sentence
        if text.find(".") != -1:
            text = text.split(".")[0]

        text = text.replace(",", "")
        words = text.split(" ")
        if "No" in words or "not" in words or "no" in words:
            answer["text"] = "no"
        else:
            answer["text"] = "yes"

    for i in range(len(label_list)):
        if label_list[i] == "no":
            label_list[i] = 0
        else:
            label_list[i] = 1

    pred_list = []
    for answer in answers:
        if answer["text"] == "no":
            pred_list.append(0)
        else:
            pred_list.append(1)

    pos = 1
    neg = 0
    yes_ratio = pred_list.count(1) / len(pred_list)

    TP, TN, FP, FN = 0, 0, 0, 0
    for pred, label in zip(pred_list, label_list):
        if pred == pos and label == pos:
            TP += 1
        elif pred == pos and label == neg:
            FP += 1
        elif pred == neg and label == neg:
            TN += 1
        elif pred == neg and label == pos:
            FN += 1

    print("TP\tFP\tTN\tFN\t")
    print(f"{TP}\t{FP}\t{TN}\t{FN}")

    eps = 1e-6
    precision = float(TP) / float(TP + FP + eps)
    recall = float(TP) / float(TP + FN + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    acc = (TP + TN) / (TP + TN + FP + FN)
    print(f"Accuracy: {acc}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 score: {f1}")
    print(f"Yes ratio: {yes_ratio}")
    print("%.3f, %.3f, %.3f, %.3f, %.3f" % (f1, acc, precision, recall, yes_ratio))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation-dir", type=str)
    parser.add_argument("--question-file", type=str)
    parser.add_argument("--result-file", type=str)
    args = parser.parse_args()

    questions = [json.loads(line) for line in open(args.question_file)]
    questions = {question["question_id"]: question for question in questions}
    answers = [json.loads(q) for q in open(args.result_file)]
    for file in os.listdir(args.annotation_dir):
        assert file.startswith("coco_pope_")
        assert file.endswith(".json")
        category = file[10:-5]
        cur_answers = [
            x for x in answers if questions[x["question_id"]]["category"] == category
        ]
        print(f"Category: {category}, # samples: {len(cur_answers)}")
        eval_pope(cur_answers, os.path.join(args.annotation_dir, file))
        print("====================================")

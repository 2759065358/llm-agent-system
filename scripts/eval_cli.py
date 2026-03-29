import argparse
import json
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from eval.auto_evaluator import AutoEvaluator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", required=True)
    parser.add_argument("--answer", required=True)
    parser.add_argument("--reference", default="")
    parser.add_argument("--log_path", default="logs/eval_logs.jsonl")
    args = parser.parse_args()

    evaluator = AutoEvaluator(log_path=args.log_path)
    result = evaluator.evaluate(
        question=args.question,
        answer=args.answer,
        reference_answer=args.reference,
    )

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

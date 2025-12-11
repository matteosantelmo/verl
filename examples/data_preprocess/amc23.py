"""
Preprocess the AMC 2023 dataset to parquet, using the same
final answer format used for the MATH dataset in verl, i.e. prompts that require
boxed answers and a `reward_model` field with the raw (unboxed) answer.

Dataset used by default:
  AMC 2023: zwhe99/amc23
"""

import argparse
import json
import os

import datasets


INSTRUCTION_FOLLOWING = "Let's think step by step and output the final answer within \\boxed{}."


def make_map_fn(
    data_source: str,
    split: str,
    question_key: str,
    answer_key: str,
):
    """
    Build a mapping function that:

    - Reads the problem statement from `question_key`
    - Appends the boxed-answer instruction
    - Reads the ground-truth answer from `answer_key`
    - Emits a verl-style sample:

      {
          "data_source": data_source,
          "prompt": [{"role": "user", "content": question_with_instruction}],
          "ability": "math",
          "reward_model": {
              "style": "rule",
              "ground_truth": <unboxed_answer_str>,
          },
          "extra_info": {
              "split": split,
              "index": idx,
          },
      }
    """

    def process_fn(example, idx):
        # Question text
        question = example[question_key].strip()
        question = question + " " + INSTRUCTION_FOLLOWING

        # Ground-truth answer
        # AMC 23 answers in this dataset seem to be floats (e.g. 27.0).
        # We convert to string. If it ends in .0, we might want to strip it if integer answers are expected,
        # but for safety we just convert to string as in the AIME script.
        # However, for 27.0, usually the answer key is 27. Let's check if we should normalize.
        # The AIME script just does str().strip().
        answer = example[answer_key]
        if isinstance(answer, float) and answer.is_integer():
            answer = int(answer)
        
        answer = str(answer).strip()

        return {
            "data_source": data_source + "_math_boxed",
            "prompt": [{"role": "user", "content": question}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": answer},
            "extra_info": {"split": split, "index": idx},
        }

    return process_fn


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--local_save_dir",
        default="~/data/amc23",
        help="The save directory for the preprocessed AMC 2023 dataset.",
    )
    parser.add_argument(
        "--hdfs_dir",
        default=None,
        help="Optional HDFS directory to copy the preprocessed parquet files to.",
    )

    # Optional: allow overriding the HF dataset paths, if needed.
    parser.add_argument(
        "--amc23_path",
        default="zwhe99/amc23",
        help="HuggingFace datasets path (or local path) for AMC 2023.",
    )

    args = parser.parse_args()

    # Load raw AMC 2023
    print(f"Loading AMC 2023 dataset from {args.amc23_path}...", flush=True)
    amc23_raw = datasets.load_dataset(args.amc23_path)

    # zwhe99/amc23 has a 'test' split.
    amc23_test = amc23_raw["test"]

    # Map to verl / MATH-compatible format (boxed answer instruction).
    amc23_processed = amc23_test.map(
        function=make_map_fn(
            data_source=args.amc23_path,
            split="amc23_test",
            question_key="question",
            answer_key="answer",
        ),
        with_indices=True,
    )

    # Save locally
    local_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(local_dir, exist_ok=True)

    amc23_path = os.path.join(local_dir, "test.parquet")

    print(f"Saving AMC 2023 test parquet to {amc23_path}", flush=True)
    amc23_processed.to_parquet(amc23_path)

    # Save example JSON files
    with open(os.path.join(local_dir, "test_example.json"), "w") as f:
        json.dump(amc23_processed[0], f, indent=2)

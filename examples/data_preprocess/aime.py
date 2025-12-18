"""
Preprocess the AIME 2024, AIME 2025 and a subset of DAPO-Math-17k-Processed datasets
to parquet, using the same final answer format used for the MATH dataset in verl,
i.e. prompts that require boxed answers and a `reward_model` field with the raw (unboxed) answer.

Datasets used by default:
    AIME 2024: HuggingFaceH4/aime_2024
    AIME 2025: opencompass/AIME2025
    DAPO-Math-17k-Processed: open-r1/DAPO-Math-17k-Processed
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

        # Ground-truth answer (AIME answers are short 3-digit strings, sometimes
        # with minor formatting; we just strip whitespace).
        answer = str(example[answer_key]).strip()

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
        default="~/data/aime",
        help="The save directory for the preprocessed AIME datasets.",
    )
    args = parser.parse_args()

    # Dataset paths
    aime_2024_path = "HuggingFaceH4/aime_2024"
    aime_2025_path = "opencompass/AIME2025"
    dapo_path = "open-r1/DAPO-Math-17k-Processed"

    ##################################################
    # Load raw AIME 2024
    print(f"Loading AIME 2024 dataset from {aime_2024_path}...", flush=True)
    aime2024_raw = datasets.load_dataset(aime_2024_path)

    # HuggingFaceH4/aime_2024 only has a 'train' split with 30 problems.
    aime2024_train = aime2024_raw["train"]

    # Map to verl / MATH-compatible format (boxed answer instruction).
    aime2024_processed = aime2024_train.map(
        function=make_map_fn(
            data_source=aime_2024_path,
            split="aime_2024_test",
            question_key="problem",
            answer_key="answer",
        ),
        with_indices=True,
    )

    ##################################################
    # Load raw AIME 2025 (two subsets: I and II)
    print(f"Loading AIME 2025 dataset from {aime_2025_path}...", flush=True)
    # AIME2025-I
    aime2025_I_raw = datasets.load_dataset(aime_2025_path, "AIME2025-I")
    aime2025_I_test = aime2025_I_raw["test"]

    # AIME2025-II
    aime2025_II_raw = datasets.load_dataset(aime_2025_path, "AIME2025-II")
    aime2025_II_test = aime2025_II_raw["test"]

    # Map each subset
    aime2025_I_processed = aime2025_I_test.map(
        function=make_map_fn(
            data_source=aime_2025_path,
            split="aime_2025_test_I",
            question_key="question",
            answer_key="answer",
        ),
        with_indices=True,
    )

    aime2025_II_processed = aime2025_II_test.map(
        function=make_map_fn(
            data_source=aime_2025_path,
            split="aime_2025_test_II",
            question_key="question",
            answer_key="answer",
        ),
        with_indices=True,
    )

    # Concatenate I and II into a single AIME 2025 test set (30 problems total)
    aime2025_processed = datasets.concatenate_datasets(
        [aime2025_I_processed, aime2025_II_processed]
    )

    # Optionally reindex so "index" is unique across the combined test set
    def reindex_fn(example, idx):
        # Preserve existing extra_info but update the index
        extra_info = dict(example["extra_info"])
        extra_info["index"] = idx
        example["extra_info"] = extra_info
        return example

    aime2025_processed = aime2025_processed.map(
        function=reindex_fn, with_indices=True
    )

    ##################################################
    # Load raw DAPO dataset
    print(f"Loading and subsetting DAPO dataset from {dapo_path}...", flush=True)
    dapo_raw = datasets.load_dataset(dapo_path)
    # open-r1/DAPO-Math-17k-Processed only has a 'train' split
    dapo_train = dapo_raw["train"]

    # Filter out all samples that contain chinese characters
    def contains_chinese(example):
        for char in example["prompt"]:
            if "\u4e00" <= char <= "\u9fff":
                return True
        return False
    dapo_train = dapo_train.filter(lambda x: not contains_chinese(x))

    n = 200
    dapo_train = dapo_train.shuffle(seed=85)
    dapo_train = dapo_train.select(range(n))

    # Map to verl / MATH-compatible format (boxed answer instruction).
    dapo_processed = dapo_train.map(
        function=make_map_fn(
            data_source=dapo_path,
            split="dapo_subset",
            question_key="prompt",
            answer_key="solution",
        ),
        with_indices=True,
    )

    # Save locally
    local_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(local_dir, exist_ok=True)

    # Create subdirectories
    aime2024_dir = os.path.join(local_dir, "aime_2024")
    aime2025_dir = os.path.join(local_dir, "aime_2025")
    dapo_dir = os.path.join(local_dir, "dapo_subset")
    os.makedirs(aime2024_dir, exist_ok=True)
    os.makedirs(aime2025_dir, exist_ok=True)
    os.makedirs(dapo_dir, exist_ok=True)

    aime2024_path = os.path.join(aime2024_dir, "test.parquet")
    aime2025_path = os.path.join(aime2025_dir, "test.parquet")
    dapo_path =  os.path.join(dapo_dir, "test.parquet")

    print(f"Saving AIME 2024 test parquet to {aime2024_path}", flush=True)
    aime2024_processed.to_parquet(aime2024_path)

    print(f"Saving AIME 2025 test parquet to {aime2025_path}", flush=True)
    aime2025_processed.to_parquet(aime2025_path)

    print(f"Saving DAPO test parquet to {dapo_path}", flush=True)
    dapo_processed.to_parquet(dapo_path)

    # Save example JSON files
    with open(os.path.join(aime2024_dir, "test_example.json"), "w") as f:
        json.dump(aime2024_processed[0], f, indent=2)
    with open(os.path.join(aime2025_dir, "test_example.json"), "w") as f:
        json.dump(aime2025_processed[0], f, indent=2)
    with open(os.path.join(dapo_dir, "test_example.json"), "w") as f:
        json.dump(dapo_processed[0], f, indent=2)
"""
Day 1: Load MedHop and print statistics.
Run this to understand the dataset structure before writing features.

Usage: python explore.py
"""

import json
from datasets import load_dataset


def print_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


def compute_stats(split, name):
    print_section(f"Split: {name}  ({len(split)} questions)")

    n_candidates = [len(ex["candidates"]) for ex in split]
    n_supports   = [len(ex["supports"])   for ex in split]
    q_lengths    = [len(ex["query"].split()) for ex in split]

    def summarize(values, label):
        print(f"  {label}:")
        print(f"    min={min(values)}  max={max(values)}  "
              f"mean={sum(values)/len(values):.1f}")

    summarize(n_candidates, "candidates per question")
    summarize(n_supports,   "support passages per question")
    summarize(q_lengths,    "question length (words)")


def show_examples(split, n=5):
    print_section(f"First {n} examples (manual inspection)")
    for i, ex in enumerate(split.select(range(n))):
        print(f"\n--- Example {i} ---")
        print(f"  ID:        {ex['id']}")
        print(f"  Query:     {ex['query']}")
        print(f"  Answer:    {ex['answer']}")
        print(f"  Candidates ({len(ex['candidates'])}): {ex['candidates'][:5]}"
              + (" ..." if len(ex['candidates']) > 5 else ""))
        # Show the first support to understand passage format
        if ex["supports"]:
            first = ex["supports"][0]
            preview = first[:120].replace("\n", " ")
            print(f"  Support[0] preview: {preview}...")


def main():
    print("Loading MedHop from HuggingFace datasets (first run downloads ~200MB)...")
    # "qangaroo" is the benchmark name; "medhop" is the subset
    # verification_mode="no_checks" skips the checksum that drifts between dataset versions
    dataset = load_dataset("qangaroo", "medhop", verification_mode="no_checks")

    print(f"\nAvailable splits: {list(dataset.keys())}")
    print(f"Column names: {dataset['train'].column_names}")

    for split_name in dataset.keys():
        compute_stats(dataset[split_name], split_name)

    # Inspect training examples — look for patterns that might predict hop difficulty
    show_examples(dataset["train"], n=5)

    # Save a compact summary to a file so you can refer back without re-running
    summary = {
        split: {
            "n_questions": len(dataset[split]),
            "avg_candidates": round(
                sum(len(ex["candidates"]) for ex in dataset[split]) / len(dataset[split]), 1
            ),
            "avg_supports": round(
                sum(len(ex["supports"]) for ex in dataset[split]) / len(dataset[split]), 1
            ),
        }
        for split in dataset.keys()
    }
    with open("dataset_stats.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nStats saved to dataset_stats.json")


if __name__ == "__main__":
    main()

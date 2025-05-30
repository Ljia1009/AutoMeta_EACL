import json
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt

def load_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]

def compute_label_overlap(data):
    overlap_stats = Counter()
    total = len(data)

    for entry in data:
        pred_labels = [x["label"] for x in entry.get("factcc_pred_labels", [])]
        gold_labels = [x["label"] for x in entry.get("factcc_gold_labels", [])]

        if len(pred_labels) != len(gold_labels):
            continue

        for pred, gold in zip(pred_labels, gold_labels):
            overlap_stats[(pred, gold)] += 1

    return overlap_stats, total

def compute_correct_distribution(data):
    pred_correct_counts = []
    gold_correct_counts = []

    for entry in data:
        pred_labels = [x["label"] for x in entry.get("factcc_pred_labels", [])]
        gold_labels = [x["label"] for x in entry.get("factcc_gold_labels", [])]

        pred_correct = sum(1 for label in pred_labels if label == "CORRECT")
        gold_correct = sum(1 for label in gold_labels if label == "CORRECT")

        pred_correct_counts.append(pred_correct)
        gold_correct_counts.append(gold_correct)

    return pred_correct_counts, gold_correct_counts



def summarize_distribution(counts, label):
    print(f"\n=== {label} Correct Label Distribution ===")
    print(f"Min: {np.min(counts)}, Max: {np.max(counts)}")
    print(f"Mean: {np.mean(counts):.2f}, Median: {np.median(counts)}")
    print(f"Standard Deviation: {np.std(counts):.2f}")
    print(f"Correct label count; {np.sum(counts)}")

def main():
    input_path = "outputs/evaluation/factcc/bart_res_factcc.jsonl"  # adjust as needed
    data = load_jsonl(input_path)

    # Overlap analysis
    overlap_stats, total = compute_label_overlap(data)
    print("=== FactCC Label Agreement ===")
    print(f"Total examples compared: {total}")
    total_pairs = sum(overlap_stats.values())
    correct_match = sum(count for (pred, gold), count in overlap_stats.items() if pred == gold)
    print(f"Matching labels: {correct_match} / {total_pairs} ({100 * correct_match / total_pairs:.2f}%)")

    print("\nDetailed label pair counts:")
    for label_pair, count in overlap_stats.items():
        print(f"{label_pair}: {count}")

    # Distribution analysis
    pred_counts, gold_counts = compute_correct_distribution(data)
    summarize_distribution(pred_counts, "Predicted")
    summarize_distribution(gold_counts, "Gold")


    plt.hist([pred_counts, gold_counts],  label=['pred', 'gold'])
    plt.legend(loc='upper right')
    plt.show()

if __name__ == "__main__":
    main()
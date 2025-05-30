import os
import json
from transformers import pipeline
from typing import Any
from run_evaluation_finetune import *

OUTPUT_DIR = "outputs/generated/finetune"
PREPROCESSED_DATA = "data/preprocessed/standardized_test.json"


def run_factcc_with_labels(
    reviews: list[list[str]], 
    meta_reviews: list[str], 
    gold_summaries: list[str],
    output_filename: str
):
    pipe = pipeline(model="manueldeprada/FactCC")
    enriched_data = []

    for review_list, pred_summary, gold_summary in zip(reviews, meta_reviews, gold_summaries):
        # For predicted summary
        pred_pairs = [[[review, pred_summary]] for review in review_list if review.strip()]
        pred_output = pipe(pred_pairs, truncation='only_first', padding='max_length')
        pred_correct_count = sum(1 for r in pred_output if r["label"] == "CORRECT")
        pred_score_total = sum(r["score"] for r in pred_output)

        # For gold summary
        gold_pairs = [[[review, gold_summary]] for review in review_list if review.strip()]
        gold_output = pipe(gold_pairs, truncation=True, padding='max_length')
        gold_correct_count = sum(1 for r in gold_output if r["label"] == "CORRECT")
        gold_score_total = sum(r["score"] for r in gold_output)

        enriched_data.append({
            "reviews": review_list,
            "meta_review": pred_summary,
            "gold_summary": gold_summary,

            # Prediction comparison
            "factcc_pred_labels": pred_output,
            "pred_correct_percentage": pred_correct_count / len(pred_output) if pred_output else 0.0,
            "pred_avg_score": pred_score_total / len(pred_output) if pred_output else 0.0,

            # Gold comparison
            "factcc_gold_labels": gold_output,
            "gold_correct_percentage": gold_correct_count / len(gold_output) if gold_output else 0.0,
            "gold_avg_score": gold_score_total / len(gold_output) if gold_output else 0.0
        })

    output_dir = os.path.join(os.path.dirname(__file__), "outputs/evaluation/factcc")
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, output_filename)
    with open(output_path, "w") as f:
        for entry in enriched_data:
            json.dump(entry, f)
            f.write("\n")

    return enriched_data


def main():
    review_lists, reference_gold = load_standardized_test_data(PREPROCESSED_DATA)

    for file in os.listdir(OUTPUT_DIR):
        if file.endswith(".txt"):
            model_name = file.replace(".txt", "")
            txt_path = os.path.join(OUTPUT_DIR, file)
            print(f"Evaluating FactCC in {model_name}...")

            predictions, _ = load_predictions_and_references(txt_path)
            predictions = predictions[:len(reference_gold)]
            reviews = review_lists[:len(reference_gold)]
            references = reference_gold[:len(predictions)]

            enriched_data = run_factcc_with_labels(reviews, predictions, references, f"{model_name}_factcc.jsonl")

if __name__ == "__main__":
    main()
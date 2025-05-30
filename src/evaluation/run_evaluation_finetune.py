import os
import json
import pandas as pd
from evaluation import Evaluator 

# === Configuration ===
OUTPUT_DIR = "outputs/generated/finetune"
PREPROCESSED_DATA = "data/preprocessed/standardized_test.json"
METRICS = ["rouge_L", "bertscore", "factCC"]

def load_predictions_and_references(txt_path: str):
    predictions, references = [], []
    with open(txt_path, "r") as f:
        for line in f:
            if line.startswith("Generated:	summarize:"):
                generated = line.strip().replace("Generated:	summarize:", "").strip()
            elif line.startswith("Generated:"):
                generated = line.strip().replace("Generated:", "").strip()
            elif line.startswith("Gold:"):
                gold = line.strip().replace("Gold:", "").strip()
                predictions.append(generated)
                references.append(gold)
    return predictions, references

def load_standardized_test_data(preprocessed_data_path: str):
    with open(preprocessed_data_path, "r") as f:
        data = json.load(f)

    reviews = []
    references = []
    for paper in data:
        if paper.get("Metareview") and paper["ReviewList"]:
            review_texts = [r.get("review", "") for r in paper["ReviewList"] if r.get("review")]
            reviews.append(review_texts)
            references.append(paper["Metareview"])
    return reviews, references

def evaluate_file(model_name, predictions, references, reviews):
    evaluator = Evaluator(predictions=predictions, references=references)
    results = {"Model": model_name, "Count": len(predictions)}

    for metric in METRICS:
        if metric == "bertscore":
            res = evaluator.evaluate(metric, model_type="distilbert-base-uncased")
            results["bertscore_precision"] = round(sum(res["precision"]) / len(res["precision"]), 4)
            results["bertscore_recall"] = round(sum(res["recall"]) / len(res["recall"]), 4)
            results["bertscore_f1"] = round(sum(res["f1"]) / len(res["f1"]), 4)

        elif metric == "rouge_L":
            scores = evaluator.evaluate(metric)
            results["rouge_L"] = round(sum(scores) / len(scores), 4)

        elif metric == "factCC":
            res = evaluator.evaluate(metric, reviews=reviews, meta_reviews=predictions)
            avg_pct = sum(r["CORRECT_Percentage"] for r in res) / len(res)
            avg_score = sum(r["avg_score"] for r in res) / len(res)

            results["factCC_correct_pct"] = round(avg_pct, 2)
            results["factCC_avg_score"] = round(avg_score, 4)

        elif metric == "disco":
            res = evaluator.evaluate(metric, reviews=reviews, meta_reviews=predictions)
            for key in ["EntityGraph", "LexicalChain", "RC", "LC"]:
                values = [r[key] for r in res]
                results[f"disco_{key}"] = round(sum(values) / len(values), 4)

    return results


def main():
    review_lists, reference_gold = load_standardized_test_data(PREPROCESSED_DATA)
    all_results = []

    for file in os.listdir(OUTPUT_DIR):
        if file.endswith(".txt"):
            model_name = file.replace(".txt", "")
            txt_path = os.path.join(OUTPUT_DIR, file)
            print(f"Evaluating {model_name}...")

            predictions, _ = load_predictions_and_references(txt_path)
            predictions = predictions[:len(reference_gold)]
            reviews = review_lists[:len(reference_gold)]
            references = reference_gold[:len(predictions)]
            #print(len(predictions),len(reviews),len(references))

            result = evaluate_file(model_name, predictions, references, reviews)
            all_results.append(result)

    # Save to CSV
    df = pd.DataFrame(all_results)
    output_path = "outputs/analysis/model_evaluation_summary.csv"
    df.to_csv(output_path, index=False)
    print(f"\nSaved summary to {output_path}")
    print(df)

if __name__ == "__main__":
    main()

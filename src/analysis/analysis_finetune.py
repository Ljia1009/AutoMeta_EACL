import os
import nltk
import pandas as pd
from nltk.tokenize import word_tokenize

nltk.download("punkt")

OUTPUT_DIR = "outputs/generated/finetune"
OUTPUT_CSV = "outputs/analysis/length_analysis_summary.csv"

def load_predictions_and_references(txt_path: str):
    predictions, references = [], []
    with open(txt_path, "r") as f:
        for line in f:
            if line.startswith("Generated:"):
                generated = line.strip().replace("Generated:", "").strip()
            elif line.startswith("Gold:"):
                gold = line.strip().replace("Gold:", "").strip()
                predictions.append(generated)
                references.append(gold)
    return predictions, references

def compute_token_stats(predictions, references):
    pred_lens = [len(word_tokenize(p)) for p in predictions]
    ref_lens = [len(word_tokenize(r)) for r in references]

    avg_pred_len = sum(pred_lens) / len(pred_lens) if pred_lens else 0
    avg_ref_len = sum(ref_lens) / len(ref_lens) if ref_lens else 0
    ratio = avg_pred_len / avg_ref_len if avg_ref_len else 0

    return avg_pred_len, avg_ref_len, ratio

def main():
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    results = []

    for file in os.listdir(OUTPUT_DIR):
        if file.endswith(".txt"):
            model_name = file.replace(".txt", "")
            txt_path = os.path.join(OUTPUT_DIR, file)

            predictions, references = load_predictions_and_references(txt_path)

            if not predictions or not references:
                print(f"⚠️ Skipping {model_name}: no valid prediction/reference pairs.")
                continue

            avg_pred_len, avg_ref_len, ratio = compute_token_stats(predictions, references)
            results.append({
                "Model": model_name,
                "Count": len(predictions),
                "AvgPredTokens": round(avg_pred_len, 2),
                "AvgGoldTokens": round(avg_ref_len, 2),
                "LengthRatio": round(ratio, 2),
            })

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ Length analysis saved to: {OUTPUT_CSV}")
    print(df)

if __name__ == "__main__":
    main()

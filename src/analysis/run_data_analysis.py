import os
import json
import nltk
import pandas as pd
from nltk.tokenize import word_tokenize

nltk.download("punkt")

DATA_SPLITS = ['dev', 'test', 'train']
PREPROCESSED_DIR = "data/preprocessed"

def analyze_dataset(split: str) -> dict:
    file_path = os.path.join(PREPROCESSED_DIR, f"standardized_{split}.json")

    with open(file_path, "r") as f:
        data = json.load(f)

    num_papers = len(data)
    num_metareviews = sum(1 for item in data if item.get('Metareview'))
    total_reviews = sum(len(item['ReviewList']) for item in data)
    avg_reviews_per_paper = total_reviews / num_papers if num_papers else 0

    reviews_per_paper = [len(item['ReviewList']) for item in data]
    max_reviews_per_paper = max(reviews_per_paper) if reviews_per_paper else 0
    min_reviews_per_paper = min(reviews_per_paper) if reviews_per_paper else 0

    metareview_lengths = [len(word_tokenize(item['Metareview'])) for item in data if item.get('Metareview')]
    review_lengths = [len(word_tokenize(review['review'])) for item in data for review in item['ReviewList'] if review.get('review')]

    avg_metareview_len = sum(metareview_lengths) / len(metareview_lengths) if metareview_lengths else 0
    avg_review_len = sum(review_lengths) / len(review_lengths) if review_lengths else 0

    return {
        'Split': split,
        '#Papers': num_papers,
        '#Metareviews': num_metareviews,
        '#Reviews': total_reviews,
        'AvgReviewsPerPaper': round(avg_reviews_per_paper, 2),
        'MaxReviewsPerPaper': max_reviews_per_paper,
        'MinReviewsPerPaper': min_reviews_per_paper,
        'AvgMetaReviewTokens': round(avg_metareview_len, 2),
        'AvgReviewTokens': round(avg_review_len, 2),
    }

import os
import json

def analyze_decision_labels(split: str, data_dir="data/preprocessed"):
    file_path = os.path.join(data_dir, f"standardized_{split}.json")

    with open(file_path, "r") as f:
        data = json.load(f)

    total = 0
    accept = 0
    reject = 0
    other = 0
    unknown = 0

    for paper in data:
        decision = paper.get("Decision", "")
        total += 1
        if decision:
            decision_lower = decision.lower()
            if "accept" in decision_lower:
                accept += 1
            elif "reject" in decision_lower:
                reject += 1
            else:
                other += 1
        else:
            unknown += 1

    print(f"\n=== Decision Breakdown for Split: {split} ===")
    print(f"Total papers: {total}")
    print(f"Accept: {accept} ({accept/total:.1%})")
    print(f"Reject: {reject} ({reject/total:.1%})")
    print(f"Other: {other} ({other/total:.1%})")
    print(f"Unknown: {unknown} ({unknown/total:.1%})")

def main():
    results = [analyze_dataset(split) for split in DATA_SPLITS]
    df = pd.DataFrame(results)

    #output_path = "outputs/analysis/dataset_analysis_summary.csv"
    #os.makedirs(os.path.dirname(output_path), exist_ok=True)
    #df.to_csv(output_path, index=False)

    #print(f"Saved dataset summary to {output_path}")
    print(df)

if __name__ == "__main__":
    #main()
    for split in ["train", "dev", "test"]:
        analyze_decision_labels(split)

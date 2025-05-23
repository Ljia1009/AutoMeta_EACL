import os
import json
import nltk
from nltk.tokenize import word_tokenize
from collections import defaultdict
import utils.utils as utils
nltk.download("punkt")
from src.utils import load_data
import pandas as pd

DATA_SPLITS = ['dev', 'test']
KEY_OPTION = 'all'

def analyze_dataset(split: str, key_option: str = 'all'):
    file_path = None
    data = load_data.load_data_from_json(file_path, split, key_option)

    num_papers = len(data)
    num_metareviews = sum(1 for item in data if item['Metareview'])
    total_reviews = sum(len(item['ReviewList']) for item in data)
    avg_reviews_per_paper = total_reviews / num_papers if num_papers else 0

    reviews_per_paper = [len(item['ReviewList']) for item in data]
    max_reviews_per_paper = max(reviews_per_paper) if reviews_per_paper else 0
    min_reviews_per_paper = min(reviews_per_paper) if reviews_per_paper else 0

    metareview_lengths = [len(word_tokenize(item['Metareview'])) for item in data if item['Metareview']]
    review_lengths = [len(word_tokenize(review)) for item in data for review in item['ReviewList']]

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

def main():
    results = [analyze_dataset(split, key_option=KEY_OPTION) for split in DATA_SPLITS]

    df = pd.DataFrame(results)
    df.to_csv("evaluation/analysis/dataset_analysis_summary.csv", index=False)
    print(df)

if __name__ == "__main__":
    main()
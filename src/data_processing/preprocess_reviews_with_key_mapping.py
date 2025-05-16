import json
from collections import defaultdict
from collections import Counter

DATA_PATH_PREFIX = "data/raw/ORSUM_"
JSONL_SUFFIX = ".jsonl"

DEFAULT_REVIEW_FIELD_MAPPING = {
    'title': 'title',
    'review': 'review',
    'rating': 'rating',
    'confidence': 'confidence',
    'recommendation': 'recommendation'
}
VENUE_REVIEW_FIELD_MAPPING_OVERRIDES = {
    'JSYS': {
        'decision': 'decision'
    },
    'AutoML-Conf-2022': {
        'rating': 'review_rating',
        'confidence': 'review_confidence',
    },
    'MIDL-2021':{
        'rating': 'final_rating',
    },
    'MIDL-2022':{
        'rating': 'final_rating_after_the_rebuttal',
    },
    'UAI-2022':{
        'rating': 'Q6 Overall score',
        'confidence': 'Q8 Confidence in your score',
    },
    'LoG-2022':{
        'rating': 'Overall Score',
        'confidence': 'Confidence',
    },
    'CLeaR-2022':{
        'rating': 'Overall score',
    }
}
PAPER_LEVEL_KEYS = ['Title', 'Abstract', 'Decision', 'Metareview']

def extract_field(review, unified_field, mapping):
    raw_key = mapping.get(unified_field, DEFAULT_REVIEW_FIELD_MAPPING.get(unified_field))
    if raw_key and raw_key in review:
        return review[raw_key]
    else:
        return None
def preprocess_dataset_with_paper_and_review_keys(file_full_path: str, file_option: str) -> list:
    if not file_full_path:
        file_full_path = DATA_PATH_PREFIX + file_option + JSONL_SUFFIX

    all_papers = []

    with open(file_full_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            paper_data = json.loads(line)

            venue = paper_data.get("Venue", "UNKNOWN")
            review_field_mapping = VENUE_REVIEW_FIELD_MAPPING_OVERRIDES.get(venue, {})

            # Extract paper-level fields safely
            processed_paper = {
                'Venue': venue,
                'Title': paper_data.get('Title', None),
                'Abstract': paper_data.get('Abstract', None),
                'Decision': paper_data.get('Decision', None),
                'Metareview': paper_data.get('Metareview', None),
                'ReviewList': []
            }

            # Process reviews
            if 'Review' in paper_data:
                for review in paper_data['Review']:
                    if not isinstance(review, dict):
                        continue

                    unified_review = {
                        'title': extract_field(review, 'title', review_field_mapping),
                        'review': extract_field(review, 'review', review_field_mapping),
                        'rating': extract_field(review, 'rating', review_field_mapping),
                        'confidence': extract_field(review, 'confidence', review_field_mapping),
                        'recommendation': extract_field(review, 'recommendation', review_field_mapping),
                    }

                    if unified_review['review'] and unified_review['review'].strip():
                        processed_paper['ReviewList'].append(unified_review)

            # Try to infer Decision from review-level metadata (recommendation without review)
            if not processed_paper['Decision']:
                for review in paper_data.get('Review', []):
                    if isinstance(review, dict):
                        has_recommendation = 'recommendation' in review
                        has_review_text = 'review' in review
                        if has_recommendation and not has_review_text:
                            processed_paper['Decision'] = review['recommendation']
                            break

            if processed_paper['ReviewList']:
                all_papers.append(processed_paper)

    return all_papers


def write_field_completeness_by_venue(processed_data, output_path):
    PAPER_LEVEL_KEYS = ['Title', 'Abstract', 'Metareview', 'Decision']
    REVIEW_LEVEL_KEYS = ['title', 'review', 'rating', 'confidence', 'recommendation']

    with open(output_path, "w") as out:
        for venue, papers in processed_data.items():
            paper_counts = defaultdict(int)
            review_counts = defaultdict(int)
            total_papers = len(papers)
            total_reviews = 0

            for paper in papers:
                for key in PAPER_LEVEL_KEYS:
                    if paper.get(key):
                        paper_counts[key] += 1

                for review in paper['ReviewList']:
                    total_reviews += 1
                    for key in REVIEW_LEVEL_KEYS:
                        if review.get(key):
                            review_counts[key] += 1

            out.write(f"\n=== Venue: {venue} ===\n")
            out.write(f"# Papers: {total_papers}\n")
            out.write("Paper-level field coverage:\n")
            for key in PAPER_LEVEL_KEYS:
                filled = paper_counts[key]
                out.write(f"  {key}: {filled} / {total_papers} ({filled / total_papers:.1%})\n")

            out.write(f"\n# Reviews: {total_reviews}\n")
            out.write("Review-level field coverage:\n")
            for key in REVIEW_LEVEL_KEYS:
                filled = review_counts[key]
                out.write(f"  {key}: {filled} / {total_reviews} ({(filled / total_reviews if total_reviews else 0):.1%})\n")

# Example usage
if __name__ == "__main__":
    processed_data_train = preprocess_dataset_with_paper_and_review_keys(None, "train")
    with open("data/preprocessed/standardized_train.json", "w") as out:
        json.dump(processed_data_train, out, indent=2)
    write_field_completeness_by_venue(processed_data_train, "outputs/analysis/field_completeness_summary.txt")

    '''  
    processed_data_dev = preprocess_dataset_with_paper_and_review_keys(None, "dev")
    processed_data_test = preprocess_dataset_with_paper_and_review_keys(None, "test")

    with open("data/preprocessed/standardized_dev.json", "w") as out:
        json.dump(processed_data_dev, out, indent=2)
    with open("data/preprocessed/standardized_test.json", "w") as out:
        json.dump(processed_data_test, out, indent=2)
  

def inspect_paper_level_keys_per_venue(file_full_path: str, file_option: str):
    """
    Inspect and count all paper-level keys (excluding 'Review') per venue.
    """
    if not file_full_path:
        file_full_path = DATA_PATH_PREFIX + file_option + JSONL_SUFFIX

    # A dictionary of Counters per venue
    venue_keys_counter = defaultdict(Counter)

    with open(file_full_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            paper_data = json.loads(line)

            venue = paper_data.get("Venue", "UNKNOWN")

            # Get all top-level keys except 'Review'
            paper_keys = set(paper_data.keys())
            paper_keys.discard('Review')

            venue_keys_counter[venue].update(paper_keys)

    # Display the result
    for venue, counter in venue_keys_counter.items():
        print(f"\n=== Venue: {venue} ===")
        for key, count in counter.most_common():
            print(f"{key}: {count}")
# Example usage
if __name__ == "__main__":


    inspect_paper_level_keys_per_venue(None, "train")


    processed_data = preprocess_dataset_by_venue(None, "train")

    with open("standardized_train_by_venue.json", "w") as out:
        json.dump(processed_data, out, indent=2)

    # Quick overview
    for venue, papers in processed_data.items():
        print(f"Venue: {venue}, Papers: {len(papers)}")
        print("Sample processed review keys:", list(processed_data[venue][0]['ReviewList'][0].keys()))'''
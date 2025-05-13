import json
import re
from collections import defaultdict
import pandas as pd

DATA_PATH_PREFIX = "data/ORSUM_"
JSONL_SUFFIX = ".jsonl"

def load_raw_data_grouped_by_venue(file_full_path: str, file_option: str) -> dict:
    """
    Load JSONL file and group papers by venue.
    
    Returns a dictionary:
    {
        "ICLR": [ { "Metareview": ..., "ReviewList": [ {...}, {...}, ... ] }, ... ],
        "ACL":  [...],
        ...
    }
    """
    if not file_full_path:
        file_full_path = DATA_PATH_PREFIX + file_option + JSONL_SUFFIX

    venue_data = defaultdict(list)

    with open(file_full_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            paper_data = json.loads(line)

            venue = paper_data.get("Venue", "UNKNOWN")

            if 'Metareview' in paper_data and paper_data['Metareview'] and 'Review' in paper_data:
                extracted_item = {
                    'Metareview': paper_data['Metareview'],
                    'ReviewList': []  # list of raw review dicts
                }

                for review in paper_data['Review']:
                    if isinstance(review, dict):
                        extracted_item['ReviewList'].append(review)

                venue_data[venue].append(extracted_item)

    return venue_data

data_by_venue = load_raw_data_grouped_by_venue(None, "train")

'''
output_path = "evaluation/analysis/venue_review_keys_summary.txt"
with open(output_path, "w") as out:
    for venue, papers in data_by_venue.items():
        out.write(f"Venue: {venue}\n")
        out.write(f"# Papers: {len(papers)}\n")

        all_review_keys = set()
        for paper in papers:
            for review in paper['ReviewList']:
                all_review_keys.update(review.keys())

        out.write(f"All review keys used: {sorted(all_review_keys)}\n")

        key_freq = defaultdict(int)
        for paper in papers:
            for review in paper['ReviewList']:
                for key in review.keys():
                    key_freq[key] += 1

        out.write("Key usage frequency:\n")
        for key, freq in sorted(key_freq.items(), key=lambda x: -x[1]):
            out.write(f"  {key}: {freq}\n")

        out.write("\n" + "="*40 + "\n\n")

print(f"Saved venue review key summary to {output_path}")'''

def detect_structured_review(text: str) -> bool:
    """
    Returns True only if both a Strength section and a Weakness section are present.
    """
    patterns = {
        'strength': r'\bstrengths?\b\s*[:\-]',
        #'weakness': r'\blimitations?\b\s*[:\-]|weakness(es)?\b\s*[:\-]'
        'weakness': r'\bweakness(es)?\b\s*[:\-]'
        
    }

    found_strength = bool(re.search(patterns['strength'], text, re.IGNORECASE))
    found_weakness = bool(re.search(patterns['weakness'], text, re.IGNORECASE))

    return found_strength and found_weakness

def analyze_structured_by_venue(file_option="train"):
    data_by_venue = load_raw_data_grouped_by_venue(None, file_option)

    venue_stats = []
    total_reviews = 0
    total_structured = 0

    for venue, papers in data_by_venue.items():
        venue_total = 0
        venue_structured = 0

        for paper in papers:
            for review in paper['ReviewList']:
                review_text = review.get('review', '')
                if review_text.strip():
                    venue_total += 1
                    if detect_structured_review(review_text):
                        venue_structured += 1

        proportion = venue_structured / venue_total if venue_total > 0 else 0

        total_reviews += venue_total
        total_structured += venue_structured

        venue_stats.append({
            'Venue': venue,
            '#Reviews': venue_total,
            '#StructuredReviews': venue_structured,
            '%Structured': round(proportion * 100, 2)
        })

    # Add total summary
    venue_stats.append({
        'Venue': 'TOTAL',
        '#Reviews': total_reviews,
        '#StructuredReviews': total_structured,
        '%Structured': round((total_structured / total_reviews) * 100, 2) if total_reviews > 0 else 0
    })

    return venue_stats


results = analyze_structured_by_venue("train")
df = pd.DataFrame(results)
df.to_csv("evaluation/analysis/structured_reviews_by_venue.csv", index=False)
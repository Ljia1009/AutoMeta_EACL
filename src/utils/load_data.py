import json

DATA_PATH_PREFIX = "data/raw/ORSUM_"
JSONL_SUFFIX = ".jsonl"

def load_data_from_json(file_full_path:str, file_option:str, key_option:str) -> list:
    """
    Load data from a JSON file and extract relevant information.
    """
    # If full path is not given, construct it with the option e.g. ../data/ORSUM_dev.jsonl
    if not file_full_path:
        file_full_path = DATA_PATH_PREFIX + file_option + JSONL_SUFFIX
    # If key_option is not 'review', we assume it is 'all'
    # and we want to extract all keys from the review
    # including 'review', 'rating', and 'confidence'
    if key_option == 'review':
        keys_to_contain = ['review']
    else:
        keys_to_contain = ['review', 'rating', 'confidence']
    
    data_list = []
    with open(file_full_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            paper_data = json.loads(line)
            extracted_item = {'ReviewList': [], 'Metareview': ''}
            if 'Metareview' in paper_data and paper_data['Metareview'] and 'Review' in paper_data:
                extracted_item['Metareview'] = paper_data['Metareview']
                for review in paper_data['Review']:
                    if 'review' in review:
                        serialized_review = ""
                        for k, v in review.items():
                            if k in keys_to_contain:
                                # Only serialize key names that are not 'review' 
                                if k == 'review':
                                    serialized_review += v + ' '
                                else:
                                    serialized_review += f"{k}: {v} "
                        # Remove the last space
                        serialized_review = serialized_review.strip()
                        # Add the serialized review to the review dictionary
                        extracted_item['ReviewList'].append(serialized_review)
                # Append the extracted item to the data list
                data_list.append(extracted_item)
    return data_list

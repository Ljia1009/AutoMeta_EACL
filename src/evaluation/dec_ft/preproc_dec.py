import json
from .args import get_args

DATA_PATH_PREFIX = "data/raw/ORSUM_"
JSONL_SUFFIX = ".jsonl"
DEC_LABEL = "Decision"
MR_LABEL = "Metareview"

def process_data_for_dec(file_full_path:str, file_option:str) -> list:
    """
    Load data from a JSON file and extract fields for decision prediction.
    """
    # If full path is not given, construct it with the option e.g. ../data/ORSUM_dev.jsonl
    if not file_full_path:
        file_full_path = DATA_PATH_PREFIX + file_option + JSONL_SUFFIX
    
    output_list = []
    with open(file_full_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            paper_data = json.loads(line)
            if DEC_LABEL in paper_data:
                dec_string = paper_data['Metareview']
                dec_string = dec_string.lower()
                if 'accept' in dec_string:
                    output_dec = "1"
                elif 'reject' in dec_string:
                    output_dec = "0"
                else:
                    # filtered out
                    continue
                if MR_LABEL in paper_data:
                    mr_string = paper_data[MR_LABEL]
                    if mr_string:
                        output_line = output_dec + '\t' + mr_string + '\n'
                        output_list.append(output_line)
    return output_list

if __name__ == "__main__":
    args = get_args()
    data_option = args.data_option
    output_path = args.output_path
    output_list = process_data_for_dec(args.train_data_path, args.data_option)
    if not output_path:
        output_path = "data/preprocessed/dec_" + data_option + ".txt"
    with open(output_path, 'w') as f:
        for line in output_list:
            f.write(line)

import argparse
from evaluation.unieval import run_UniEval

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_json_path", type=str)
    parser.add_argument("--orig_data_path", type=str)
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--file_option", type=str)
    parser.add_argument("--key_option", type=str)
    parser.add_argument("--device", type=str)
    args = parser.parse_args()
    
    run_UniEval(args.output_json_path, args.orig_data_path, args.save_path, args.file_option, args.key_option, args.device)
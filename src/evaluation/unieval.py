from UniEval.utils import convert_to_json
from UniEval.metric.evaluator import get_evaluator
from utils import load_data
import json
import os
def get_ref_list(output_json_path):
    with open(output_json_path, "r") as file:
        data_lists = json.load(file)
    ref_list = [lst[1] for lst in data_lists]
    return ref_list

def get_output_list(output_json_path):
    with open(output_json_path, "r") as file:
        data_lists = json.load(file)
    ref_list = [lst[0] for lst in data_lists]
    return ref_list

def get_src_list(orig_data_path, file_option, key_option):
    data_list = load_data.load_data_from_json(orig_data_path, file_option, key_option)
    # data_list = [{"Review List":[], "Meta review":str}]
    src_list = ["\\".join(instance["ReviewList"]) for instance in data_list]
    return src_list


def run_UniEval(output_json_path, orig_data_path, save_path, file_option, key_option, device="mps"):
    os.makedirs(save_path, exist_ok=True)
    task="summarization"
    ref_list = get_ref_list(output_json_path)
    output_list = get_output_list(output_json_path)
    src_list = get_src_list(orig_data_path, file_option, key_option)

    # Prepare data for pre-trained evaluators
    data = convert_to_json(output_list=output_list, 
                        src_list=src_list, ref_list=ref_list)
    # Initialize evaluator for a specific task
    evaluator = get_evaluator(task, device=device)
    # Get multi-dimensional evaluation scores
    eval_scores = evaluator.evaluate(data, print_result=True)

    result = []
    for i in range(len(eval_scores)):
        score_dict = eval_scores[i]
        score_dict["Generated"] = output_list[i]
        score_dict["Gold"] = ref_list[i]
        result.append(score_dict)
    
    with open(save_path, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=4)
    

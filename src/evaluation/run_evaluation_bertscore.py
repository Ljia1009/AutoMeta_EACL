from utils import load_data
from evaluation.evaluation import Evaluator
import pandas as pd
import json
from tqdm import tqdm
import argparse
import os

def load_json_output(file_path):
    with open(file_path, "r") as file:
        return json.load(file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_json_path", type=str)
    parser.add_argument("--orig_data_path", type=str)
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--save_file", type=str)
    parser.add_argument("--file_option", type=str)
    parser.add_argument("--key_option", type=str)
    parser.add_argument("--device", type=str)

    args = parser.parse_args()
   
    data_list = load_data.load_data_from_json(args.orig_data_path, args.file_option, args.key_option)
    # data_list = [{"Review List":[], "Meta review":str}]
    
    
    overall_review_lists = []
    overall_predicted_metareviews = []
    overall_gold_metareviews = []

    os.makedirs(args.save_path, exist_ok=True)

    generated_output = load_json_output(args.output_json_path)
    for idx, doc in enumerate(data_list):
        try:
            # TODO: the format of baseline txt and finetune txt is not the same
            # so the index of generated output is also different...
            overall_predicted_metareviews.append(generated_output[idx][1].split('\n')[0])
            overall_gold_metareviews.append(generated_output[idx][2])
            overall_review_lists.append(doc["ReviewList"])
        except:
            print(idx)
            print(generated_output[idx])
            raise Exception()

    evaluation_result = []
    ev = Evaluator(overall_predicted_metareviews, overall_gold_metareviews)
    rouge_scores = ev.evaluate('rouge_L')
    bert_scores = ev.evaluate('bertscore')
    
    # factcc = ev.evaluate('factCC', reviews=overall_review_lists, meta_reviews=overall_predicted_metareviews)
    # summac = ev.evaluate('summaC', reviews=overall_review_lists, meta_reviews=overall_predicted_metareviews)
    # disco = ev.evaluate('disco', reviews=overall_review_lists, meta_reviews=overall_predicted_metareviews)

    for i in tqdm(range(len(overall_gold_metareviews))):
        evaluation_result.append({"gold": overall_gold_metareviews[i],
                                  "prediction":overall_predicted_metareviews[i],
                                  'rouge_score':rouge_scores[i],
                                  'bertscore_precision': bert_scores['precision'][i],
                                  "bertscore_recall": bert_scores["recall"][i],
                                  "bertscore_f1": bert_scores["f1"][i],
                                  })
        
    df = pd.DataFrame(evaluation_result)
    df.to_csv(args.save_file)
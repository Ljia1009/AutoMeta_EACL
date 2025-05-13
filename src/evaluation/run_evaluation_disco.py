from src.utils import utils
from src.utils import load_data
from evaluation.evaluation import Evaluator
import pandas as pd
import json
from tqdm import tqdm

def load_json_output(file_path):
    with open(file_path, "r") as file:
        return json.load(file)

if __name__ == "__main__":
    args = utils.get_args()
    
    data_list = load_data.load_data_from_json(args.data_path, args.data_option, args.key_option)
    # data_list = [{"Review List":[], "Meta review":str}]
    
    
    overall_review_lists = []
    overall_predicted_metareviews = []
    overall_gold_metareviews = []

    generated_output = load_json_output(args.output_path)
    for idx, doc in enumerate(data_list):
        overall_predicted_metareviews.append(generated_output[idx][1])
        overall_gold_metareviews.append(generated_output[idx][2])
        overall_review_lists.append(doc["ReviewList"])
    
    evaluation_result = []
    ev = Evaluator(overall_predicted_metareviews, overall_gold_metareviews)
    # rouge_scores = ev.evaluate('rougeL')
    # bert_scores = ev.evaluate('bertscore')
    
    # factcc = ev.evaluate('factCC', reviews=overall_review_lists, meta_reviews=overall_predicted_metareviews)
    # summac = ev.evaluate('summaC', reviews=overall_review_lists, meta_reviews=overall_predicted_metareviews)
    disco = ev.evaluate('disco', reviews=overall_review_lists, meta_reviews=overall_predicted_metareviews)

    for i in tqdm(range(len(overall_gold_metareviews))):
        evaluation_result.append({"gold": overall_gold_metareviews[i],
                                  "prediction":overall_predicted_metareviews[i],
                                  "EntityGraph": disco[i]["EntityGraph"],
                                "LexicalChain": disco[i]["LexicalChain"],
                                "RC": disco[i]["RC"],
                                "LC": disco[i]["LC"]})
                                # "DS_Focus_NN": disco[i]["DS_Focus_NN"],# FocusDiff 
                                # "DS_SENT_NN":disco[i]["DS_SENT_NN"]})
        
    df = pd.DataFrame(evaluation_result)
    df.to_csv(args.evaluation_result_path)
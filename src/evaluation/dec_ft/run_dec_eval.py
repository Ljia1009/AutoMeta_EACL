import argparse
from transformers import pipeline, BertTokenizerFast, AutoModelForSequenceClassification
from datasets import Dataset
import torch
import pandas as pd

GE_LABEL = "Generated"
GO_LABEL = "Gold"

def get_data(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    data = []
    for line in lines:
        label, text = line.strip().split('\t')
        if label.startswith(GE_LABEL):
            gen = text
        elif label.startswith(GO_LABEL):
            ref = text
            data.append({'gen': gen, 'ref': ref})
    return data

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path",
        type=str,
        default="",
        help="Full path of the file used for input metareview pairs. ",
    )
    parser.add_argument(
        "--result_path",
        type=str,
        default="",
        help="Full path of the file used for the result. ",
    )
    args = parser.parse_args()
    return args

def preprocess_function(examples):
    gen_mr = tokenizer(examples["gen"], padding=True, truncation=True)
    ref_mr = tokenizer(examples["ref"], padding=True, truncation=True)
    return {
        "gen_input_ids": gen_mr["input_ids"],
        "gen_attention_mask": gen_mr["attention_mask"],
        "ref_input_ids": ref_mr["input_ids"],
        "ref_attention_mask": ref_mr["attention_mask"]
    }

if __name__ == '__main__':
    # load saved model
    dec_model = AutoModelForSequenceClassification.from_pretrained("src/evaluation/dec_ft/model/train")
    tokenizer = BertTokenizerFast.from_pretrained("src/evaluation/dec_ft/model/train")

    # load test mreviews and generated mreviews
    arg = get_args()
    input_path = arg.input_path
    mr_dicts = get_data(input_path)

    # run model to get logits from both mreviews
    mr_data = Dataset.from_list(mr_dicts)
    tokenized_mr_data = mr_data.map(
        preprocess_function,
        batched=True,
        remove_columns=mr_data.column_names
    )

    with torch.no_grad():
        gen_logits = dec_model(
            input_ids=tokenized_mr_data["gen_input_ids"],
            attention_mask=tokenized_mr_data["gen_attention_mask"]
        ).logits
        ref_logits = dec_model(
            input_ids=tokenized_mr_data["ref_input_ids"],
            attention_mask=tokenized_mr_data["ref_attention_mask"]
        ).logits
    
    # compute normalized logit diff
    gen_pos_prob = torch.nn.functional.softmax(gen_logits, dim=-1)[:, 1]  # Assuming class 1 is the positive class
    ref_pos_prob = torch.nn.functional.softmax(ref_logits, dim=-1)[:, 1]  # Assuming class 1 is the positive class
    # compute the absolute difference
    abs_diff = torch.abs(gen_pos_prob - ref_pos_prob)
    # compute the average of the absolute differences
    avg_diff = torch.mean(abs_diff).item()
    print(f"Average absolute difference in positive class probabilities: {avg_diff}")
    # save the logits to a pandas dataframe
    cls_logits = pd.DataFrame({
        "gen_logit_0": gen_logits[:, 0].tolist(),
        "gen_logit_1": gen_logits[:, 1].tolist(),
        "ref_logit_0": ref_logits[:, 0].tolist(),
        "ref_logit_1": ref_logits[:, 1].tolist()
    })
    # save the logits to a csv file
    cls_logits.to_csv(arg.result_path, index=True)

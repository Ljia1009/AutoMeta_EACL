from transformers import pipeline
from ..data.get_data import get_data
import argparse

def run_t5_summarization(data_path):
    test_data = get_data(data_path)
    test_input = []
    gold_metareviews = []
    for d in test_data:
        test_input.append(d['input_text'])
        gold_metareviews.append(d['target_text'])
    summarizer = pipeline("summarization",model='/gscratch/stf/jiamu/LING573_AutoMeta/src/models/finetune/t5_out')
    metareviews = []
    for reviews in test_input:
        metareview = summarizer(reviews,min_length=90,do_sample=False)
        metareviews.append(metareview)
    return metareviews,gold_metareviews

def get_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_path",
        type=str,
        default="",
        help="Full path of the file used for output. ",
    )
    arg = parser.parse_args()
    return arg

if __name__ == '__main__':
    metareviews,gold_metareviews = run_t5_summarization('/gscratch/stf/jiamu/LING573_AutoMeta/src/models/finetune/data/test_data.txt')
    arg = get_arg()
    output_path = arg.output_path
    with open(output_path,'w') as f:
        for generated,gold in zip(metareviews,gold_metareviews):
            f.write('Generated:'+'\t')
            f.write(generated+'\n')
            f.write('Gold:'+'\t')
            f.write(gold+'\n')

from transformers import pipeline
import torch


def run_olmo_summarization(sample_size: int, data_list: list):
    device_id = 0 if torch.cuda.is_available() else -1

    output = []
    gold_metareview = []
    if not sample_size:
        sample_size = len(data_list)
    for review in data_list[:sample_size]:
        gold_metareview.append(review['Metareview'])

    summarizer = pipeline(
        "text-generation", model="allenai/OLMo-2-1124-7B-SFT", device=device_id, trust_remote_code=True)

    # result = 'Below are multiple summaries of a paper\'s reviews. You need to summarize them. \n'
    for paper in data_list[:sample_size]:
        result = 'Below are multiple summaries of a paper\'s reviews. You need to summarize them. \n'
        for review in paper['ReviewList']:
            summary = summarizer('summarize below review: '+review, max_new_tokens=150,
                                 min_length=60, do_sample=False)
            result += summary[0]['generated_text']+'\n'
        final = summarizer(result, max_new_tokens=200,
                           min_length=90, do_sample=False)
        output.append(final[0]['generated_text'])
    return output, gold_metareview

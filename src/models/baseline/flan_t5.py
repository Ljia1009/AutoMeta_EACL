from transformers import pipeline
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer
import torch


def run_flan_t5_summarization(sample_size: int, data_list: list):
    """
    Run the summarization process with BART.
    """
    # Load the data

    # Initialize the output list and gold metareview list

    # device_id = 0 if torch.cuda.is_available() else -1

    output = []
    gold_metareview = []

    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    # max_input_tokens = model.config.max_position_embeddings
    if not sample_size:
        sample_size = len(data_list)

    # Extract the gold metareview from the first specified reviews
    for review in data_list[:sample_size]:
        gold_metareview.append(review['Metareview'])

    # Initialize the summarizer pipeline
    summarizer = pipeline(
        "summarization", model="google/flan-t5-base")

    # Prepare the result string
    # result = 'Below are multiple summaries of a paper\'s reviews. '
    # Iterate through each paper and its reviews to generate summaries
    # if not sample_size:
    #     sample_size = len(data_list)
    for paper in data_list[:sample_size]:
        result = 'Below are multiple summaries of a paper\'s reviews. '
        for review in paper['ReviewList']:
            tokens = tokenizer.encode(review, truncation=True, max_length=512)
            # if len(tokens) > max_input_tokens:
            #     tokens = tokens[:max_input_tokens-1]
            #     text_to_summary = tokenizer.decode(tokens, skip_special_tokens=True)
            # else:
            #     text_to_summary = tokenizer.decode(tokens, skip_special_tokens=True)
            text_to_summary = tokenizer.decode(
                tokens, skip_special_tokens=True)
            summary = summarizer(text_to_summary,
                                 min_length=40, do_sample=False)
            result += summary[0]['summary_text']+'\n'
        # tokens = tokenizer.encode(result, truncation = False)
        # if len(tokens) > max_input_tokens:
        #     tokens = tokens[:max_input_tokens-1]
        #     text_to_summary = tokenizer.decode(tokens, skip_special_tokens=True)
        # else:
        #     text_to_summary = tokenizer.decode(tokens, skip_special_tokens=True)
        tokens = tokenizer.encode(result, truncation=True, max_length=512)
        text_to_summary = tokenizer.decode(tokens, skip_special_tokens=True)
        final = summarizer(text_to_summary,
                           min_length=90, do_sample=False)
        output.append(final[0]['summary_text'])
    return output, gold_metareview

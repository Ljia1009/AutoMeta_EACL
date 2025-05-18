from src.utils.load_data import load_data_from_json
from src.models.finetune.args import get_args
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer

model_name = "facebook/bart-large-cnn"
tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=1024)
model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name, device_map="auto")


def save_data(data_list: list, out_dir: str):
    "data_list: list of dictionaries, each containing 'ReviewList' and 'Metareview' keys"
    data = []
    summarizer = pipeline(
        "summarization", model="facebook/bart-large-cnn")
    for paper in data_list:
        item = {}
        input_text = 'Below are multiple reviews of a paper. '
        for review in paper['ReviewList']:
            tokens = tokenizer.encode(
                review, truncation='longest_first', max_length=1020)
            text_to_summary = tokenizer.decode(
                tokens, skip_special_tokens=True)
            summary = summarizer(text_to_summary, max_length=70,
                                 min_length=20, do_sample=False)
            input_text += summary[0]['summary_text']
        item['input_text'] = input_text.strip().replace(
            '\n', ' ').replace('\t', ' ')
        item['target_text'] = paper['Metareview'].strip().replace(
            '\n', ' ').replace('\t', ' ')
        # print('target:', item['target_text'], '\n')
        data.append(item)
    with open(out_dir, 'w') as f:
        for item in data:
            f.write(f"{item['input_text']}\t{item['target_text']}\n")


if __name__ == "__main__":
    args = get_args()
    output_dir = args.output_dir
    train_data_list = load_data_from_json(
        args.train_data_path, args.data_option, args.key_option)
    test_data_list = load_data_from_json(
        args.test_data_path, args.data_option, args.key_option)
    save_data(train_data_list, output_dir + "/train_data.txt")
    save_data(test_data_list, output_dir + "/test_data.txt")

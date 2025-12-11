from .args import get_args
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
from datasets import Dataset
from .data.get_data import get_data

model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=512)
model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name, device_map="auto")


def preprocess_function(examples):
    # inputs = [ex['input_text'] for ex in examples]
    # targets = [ex['target_text'] for ex in examples]
    model_inputs = tokenizer(
        examples["input_text"], max_length=500, truncation=True)

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["target_text"],
                           max_length=500, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


if __name__ == "__main__":
    # need args: train_data_path, test_data_path, output_dir. train_data_path and test_data_path are full paths
    # to the files used for training and testing under finetune/data. output_dir is the full path to save the output.
    args = get_args()
    output_dir = args.output_dir
    # train_data_list = load_data_from_json(
    #     args.train_data_path, args.data_option, args.key_option)
    # test_data_list = load_data_from_json(
    #     args.test_data_path, args.data_option, args.key_option)
    train_data_original = get_data(
        args.train_data_path)
    test_data_original = get_data(
        args.test_data_path)
    task_prefix = 'summarize: '
    for d in train_data_original:
        d['input_text'] = task_prefix+d['input_text']
        d['target_text'] = task_prefix + d['target_text']
    for d in test_data_original:
        d['input_text'] = task_prefix+d['input_text']
        d['target_text'] = task_prefix + d['target_text']

    train_data = Dataset.from_list(train_data_original)
    test_data = Dataset.from_list(test_data_original)

    max_input_length = 512
    max_output_length = 500

    tokenized_train_data = train_data.map(
        preprocess_function,
        batched=True,
        remove_columns=train_data.column_names,
    )
    tokenized_test_data = test_data.map(
        preprocess_function,
        batched=True,
        remove_columns=test_data.column_names,
    )
    data_collator = DataCollatorForSeq2Seq(
        tokenizer, model=model, padding=True, max_length=max_input_length)
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        save_strategy="epoch",
        eval_strategy="epoch",
        learning_rate=1e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        weight_decay=0.01,
        save_total_limit=2,
        num_train_epochs=20,
        predict_with_generate=True,
        logging_dir='anonymous',
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_train_data,
        eval_dataset=tokenized_test_data,
        tokenizer=tokenizer,
    )
    print('training')
    trainer.train()
    print('saving model')
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print('finished')

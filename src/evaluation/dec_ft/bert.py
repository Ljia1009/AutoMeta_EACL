from .args import get_args
from transformers import BertTokenizerFast, AutoModelForSequenceClassification, TrainingArguments, Trainer
from torch import nn
from datasets import Dataset

#load model and tokenizer
# bert = AutoModel.from_pretrained('bert-base-uncased')
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2 
)
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

'''
#defining new layers
class BERT_architecture(nn.Module):

    def __init__(self, bert):
      super(BERT_architecture, self).__init__()
      self.bert = bert
      # dropout layer
      self.dropout = nn.Dropout(0.2)
      # relu activation function
      self.relu =  nn.ReLU()
      # dense layer 1
      self.fc1 = nn.Linear(768,512)
      # dense layer 2 (Output layer)
      self.fc2 = nn.Linear(512,2)

    #define the forward pass
    def forward(self, input_ids, attention_mask):
      #pass the inputs to the model
      _, cls_hs = self.bert(input_ids, attention_mask=attention_mask, return_dict=False)
      x = self.fc1(cls_hs)
      x = self.relu(x)
      x = self.dropout(x)
      # output layer
      x = self.fc2(x)
      # return the logits
      return x
'''

def process_dec_data(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    pairs = []
    for line in lines:
        _, label, text = line.strip().split('\t')
        pairs.append({'text': text, 'label': label})
    return pairs


def preprocess_function(examples):
    model_inputs = tokenizer(examples["text"], padding='longest', truncation=True)
    model_inputs["labels"] = examples["label"]
    return model_inputs


if __name__ == "__main__":
    # args: train_data_option, valid_data_option, output_path
    args = get_args()
    model_save_path = args.output_path
    if not model_save_path:
        model_save_path = "src/evaluation/dec_ft/model"

    train_data_path = "data/preprocessed/dec_" + args.train_data_option + ".txt"
    valid_data_path = "data/preprocessed/dec_" + args.valid_data_option + ".txt"

    train_dicts = process_dec_data(train_data_path)
    valid_dicts = process_dec_data(valid_data_path)
    train_data = Dataset.from_list(train_dicts)
    valid_data = Dataset.from_list(valid_dicts)

    tokenized_train_data = train_data.map(
        preprocess_function,
        batched=True,
        remove_columns=train_data.column_names,
    )
    tokenized_valid_data = valid_data.map(
        preprocess_function,
        batched=True,
        remove_columns=valid_data.column_names,
    )
    
    '''
    #freeze the pretrained layers
    for param in bert.parameters():
        param.requires_grad = False
    
    model = BERT_architecture(bert)
    '''

    training_args = TrainingArguments(
        output_dir=model_save_path,   # Directory to save the model
        eval_strategy="epoch",    # Evaluate after each epoch
        save_strategy="epoch",    # Save after each epoch
        learning_rate=1e-5,             # Common starting point for BERT
        per_device_train_batch_size=4, # Adjust based on GPU memory
        per_device_eval_batch_size=4, # Adjust based on GPU memory
        num_train_epochs=15,             # You can experiment with more epochs
        weight_decay=0.01,              # L2 regularization
        logging_dir=model_save_path + "/logs",           # TensorBoard logs
        save_total_limit=2,             # Save only the last 2 checkpoints
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_data,
        eval_dataset=tokenized_valid_data,
        processing_class=tokenizer
    )

    print('training')
    trainer.train()
    print('saving model')
    trainer.save_model(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    print('finished')

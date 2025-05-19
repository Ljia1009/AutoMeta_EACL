# LING573_AutoMeta
For full original and training dataset, visit https://drive.google.com/drive/folders/14CXIUZWwPkoUQxVDcN8NLVOaYjwcPc-q?usp=drive_link
For finetuned models, visit: https://drive.google.com/drive/folders/1XDfODOWe_p2xHJ0OQnE3oWeM9l-Bcyqx?usp=sharing

## Summarization
Command for running summarization from the repo root:
```bash
python src/run_summarization.py
```
Arguments:
```
--data_path:    Full path of the file used for testing.
                default=""
--data_option:  Option for the file used for testing;
                ignored when full path is provided;
                valid options are dev, test, or train
                default="dev"
--sample_size:  Number of samples to run summarization for;
                default to dataset length.
--key_option:   Option for the keys extracted from each review;
                Valid options are review, all.
                default="review"
--model:        Model used for summarization;
                valid options are bart, ...TBD
                default="bart"
--output_path:  Path to save the output.
                When unspecified, default to output/<model>_<key_option>_<sample_size>_out.txt
```

## Evaluation
Run evaluation using rougeL, bertscore, and factCC metrics:
```bash
./run_evaluation.sh
```

Run evaluation using disco metrics:
```bash
./run_evaluation_disco.sh
```

Run evaluation using summac metrics: 
```
./run_evaluation_summac.sh
```

The evaluation results are save as csv files under `./evaluation` as `<metric>_<model>_<key_option>_out.txt.csv`

### Environment Issue
The environment required by summac package is different from the rest of others.
To run this, you'll have to have one separate environment.

### Metrics issues
"DS_Focus_NN" and "DS_SENT_NN" require using BERT model that has a limit for input length(512).
It seems that some of our inputs are longer than the limits. So at this moment the two are not includede.

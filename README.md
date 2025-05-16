# LING573_AutoMeta

## Data
The dev and test sets are under `data/raw`.
For full original and training dataset, visit https://drive.google.com/drive/folders/14CXIUZWwPkoUQxVDcN8NLVOaYjwcPc-q?usp=drive_link
Standardized dev and test sets (across venues) are under `data/preprocessed`
## Summarization
The following command runs summarization from the repo root:
```bash
python src/models/run_summarization.py
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
                valid options are bart, pegasus, flan-t5, DistilBart.
                default="bart"
--output_path:  Path to save the output.
                When unspecified, default to output/<model>_<key_option>_<sample_size>_out.txt
```

## Evaluation
The following command runs evaluation using rougeL, bertscore, and factCC metrics from the repo root, for all the output files under `output/`:
```bash
src/evaluation/run_evaluation.sh
```
Before running the disco evaluation, do:
```
pip install "git+https://github.com/AIPHES/DiscoScore.git"
```

The following command runs evaluation using disco metrics from the repo root, for all the output files under `output/`:
```bash
./src/evaluation/run_evaluation_disco.sh
```

Before running the summac evaluation, do:

Run evaluation using summac metrics: 
```
./src/evaluation/run_evaluation_summac.sh
```

The evaluation results are save as csv files under `./outputs/baseline/evaluation` as `<metric>_<model>_<key_option>_out.txt.csv`

### Environment Issue
The environment required by summac package is different from the rest of others.
To run this, you'll have to have one separate environment.

For summac:
```
huggingface-hub<=0.17.0
```

### Metrics issues
"DS_Focus_NN" and "DS_SENT_NN" require using BERT model that has a limit for input length(512).
It seems that some of our inputs are longer than the limits. So at this moment the two are not includede.
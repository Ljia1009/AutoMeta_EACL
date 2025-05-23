# LING573_AutoMeta

## Data
The dev and test sets are under `data/raw`.

Standardized dev and test sets (across venues) are under `data/preprocessed`

For full original and training dataset, visit https://drive.google.com/drive/folders/14CXIUZWwPkoUQxVDcN8NLVOaYjwcPc-q?usp=drive_link
For finetuned models, visit: https://drive.google.com/drive/folders/1U4WhO_MG_uu-d_oxzNWZxJZapuz-YaTD?usp=sharing

## Environment
To use our repo, run:
```
pip install -r requirements.txt
```
If any conflicts or issues arise, you can set up an enviroment that is exactly ours by:
```
conda env create -f environment.yml
```

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
                When unspecified, default to outputs/generated/<model>_<key_option>_<sample_size>_out.txt
```
For our finetuned models, you can run them by using individual files located in src/models/finetune/inference

The results of them locate in src/models/finetune/data

## Evaluation
The following command runs evaluation using rougeL, bertscore, and factCC metrics from the repo root, for all the output files under `outputs/generated/`:
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

The evaluation results are save as csv files under `outputs/evaluation/baseline` as `<metric>_<model>_<key_option>_out.txt.csv`

### Metrics issues
"DS_Focus_NN" and "DS_SENT_NN" require using BERT model that has a limit for input length(512).
It seems that some of our inputs are longer than the limits. So at this moment the two are not includede.

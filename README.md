

## Data
In this anonymous repo the data were deleted. They are from the ORSUM dataset.
## Environment
To use our repo, run:
```
pip install -r requirements.txt
```
If conflicts arise and you want to use the model inference and training, not the evaluation, run:
```
pip install -r requirements_model.txt
```
If any conflicts or issues arise, you can set up an enviroment that is exactly ours by:
```
conda env create -f environment.yml
```
## Model Training
Necessary files for replicate our finetuning process are under `src/models/finetune`

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
For our finetuned models, you can run them by using individual files located in `src/models/finetune/inference`

The results of them locate in `outputs/generated/finetune`

## Evaluation
The generated outputs that are ready to run evaluation on are under:
`outputs/generated/baseline/baseline/comparison/review` (for baseline)
`outputs/generated/finetune/comparison` (for finetuned results)

The following command runs evaluation using rougeL and metrics from the `scripts` directory; Be sure to pass the correct OUTPUT_DIR and EVAL_DIR arguments

```bash
run_bertscore.sh
```
The evaluation results are save as csv files under `outputs/evaluation/<type>/review_bertscore` (type=[baseline, finetune])as `<model>_review_out.txt.csv`

The following command runs evaluation using rougeL and metrics from the `scripts` directory; Be sure to pass the correct OUTPUT_DIR and EVAL_DIR arguments

```bash
run_unieval.sh
```
The evaluation results are save as csv files under `outputs/evaluation/<type>/unieval_comparison` (type=[baseline, finetune]) as `<model>_review_out.txt.json`


For the decision distance evaluation, necessary files are under `src/evaluation/dec_ft`.

To fine-tune BERT for decision prediction on the training set:

```bash
python3 -m src.evaluation.dec_ft.bert --train_data_option train --valid_data_option dev
```

To produce decision distance and save the output logits from the generateed and human meta-review pairs, based on the fine-tuned  BERT decision model:
```bash
python3 -m src.evaluation.dec_ft.run_dec_eval --input_path outputs/generated/<SUM_OUTPUT_FILE_NAME> --result_path src/evaluation/dec_ft/results/<LOGITS_FILE_NAME>
```

To get a classification report based on the output logits from the meta-review pairs:
```bash
python3 src.evaluation.dec_ft.logits_to_acc.py --input_path src/evaluation/dec_ft/results/<LOGITS_FILE_NAME> --result_path src/evaluation/dec_ft/results/<REPORT_FILE_NAME>
```

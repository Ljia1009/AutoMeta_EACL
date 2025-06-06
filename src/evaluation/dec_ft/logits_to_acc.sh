#!/bin/bash

python src/evaluation/dec_ft/logits_to_acc.py --input_path src/evaluation/dec_ft/results/bartb_res_deceval.txt --result_path src/evaluation/dec_ft/results/bartb_acc.txt
python src/evaluation/dec_ft/logits_to_acc.py --input_path src/evaluation/dec_ft/results/bartf_res_deceval.txt --result_path src/evaluation/dec_ft/results/bartf_acc.txt

python src/evaluation/dec_ft/logits_to_acc.py --input_path src/evaluation/dec_ft/results/pegb_res_deceval.txt --result_path src/evaluation/dec_ft/results/pegb_acc.txt
python src/evaluation/dec_ft/logits_to_acc.py --input_path src/evaluation/dec_ft/results/pegf_res_deceval.txt --result_path src/evaluation/dec_ft/results/pegf_acc.txt

python src/evaluation/dec_ft/logits_to_acc.py --input_path src/evaluation/dec_ft/results/t5b_res_deceval.txt --result_path src/evaluation/dec_ft/results/t5b_acc.txt
python src/evaluation/dec_ft/logits_to_acc.py --input_path src/evaluation/dec_ft/results/t5f_res_deceval.txt --result_path src/evaluation/dec_ft/results/t5f_acc.txt

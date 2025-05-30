#!/bin/bash

#SBATCH --job-name=jiamu-flan-t5
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT
#SBATCH --mail-user=jiamuluo@uw.edu

#SBATCH --account=stf
#SBATCH --partition=ckpt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --time=00-10:00:00 # Max runtime in DD-HH:MM:SS format.

#SBATCH --export=all
#SBATCH --output=flan-t5.out # where STDOUT goes
#SBATCH --error=flan-t5.error # where STDERR goes

# Your programs to run.
eval "$(/gscratch/stf/jiamu/gscratch/stf/jiamu/miniconda3/bin/conda shell.bash hook)"
conda activate ling573
python3 /gscratch/stf/jiamu/LING573_AutoMeta/src/run_summarization.py --data_path /gscratch/stf/jiamu/LING573_AutoMeta/data/ORSUM_test.jsonl --key_option review --model flan-t5 --output_path /gscratch/stf/jiamu/LING573_AutoMeta/output/flan-t5_review_out.txt
python3 /gscratch/stf/jiamu/LING573_AutoMeta/src/run_summarization.py --data_path /gscratch/stf/jiamu/LING573_AutoMeta/data/ORSUM_test.jsonl --key_option all --model flan-t5 --output_path /gscratch/stf/jiamu/LING573_AutoMeta/output/flan-t5_all_out.txt
conda deactivate
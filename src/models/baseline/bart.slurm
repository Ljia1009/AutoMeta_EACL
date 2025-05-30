#!/bin/bash

#SBATCH --job-name=jiamu-bart
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
#SBATCH --output=bart.out # where STDOUT goes
#SBATCH --error=bart.error # where STDERR goes

module load cuda

# Your programs to run.
eval "$(/gscratch/stf/jiamu/gscratch/stf/jiamu/miniconda3/bin/conda shell.bash hook)"
conda activate ling573
/gscratch/stf/jiamu/gscratch/stf/jiamu/miniconda3/envs/ling573/bin/python3 /gscratch/stf/jiamu/LING573_AutoMeta/src/run_summarization.py --data_path /gscratch/stf/jiamu/LING573_AutoMeta/data/ORSUM_test.jsonl --key_option review --model bart --output_path /gscratch/stf/jiamu/LING573_AutoMeta/output/bart_review_out.txt
/gscratch/stf/jiamu/gscratch/stf/jiamu/miniconda3/envs/ling573/bin/python3 /gscratch/stf/jiamu/LING573_AutoMeta/src/run_summarization.py --data_path /gscratch/stf/jiamu/LING573_AutoMeta/data/ORSUM_test.jsonl --key_option all --model bart --output_path /gscratch/stf/jiamu/LING573_AutoMeta/output/bart_all_out.txt
conda deactivate
#!/bin/bash

#SBATCH --job-name=jiamu-save
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
#SBATCH --output=save.out # where STDOUT goes
#SBATCH --error=save.error # where STDERR goes

module load cuda

eval "$(/gscratch/stf/jiamu/gscratch/stf/jiamu/miniconda3/bin/conda shell.bash hook)"
conda activate ling573
/gscratch/stf/jiamu/gscratch/stf/jiamu/miniconda3/envs/ling573/bin/python3 -m src.models.finetune.data.save_data --train_data_path /gscratch/stf/jiamu/LING573_AutoMeta/data/raw/ORSUM_train.jsonl --test_data_path /gscratch/stf/jiamu/LING573_AutoMeta/data/raw/ORSUM_test.jsonl --output_dir /gscratch/stf/jiamu/LING573_AutoMeta/src/models/finetune/data --key_option review
conda deactivate
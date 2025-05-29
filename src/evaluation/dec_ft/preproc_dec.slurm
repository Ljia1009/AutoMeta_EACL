#!/bin/bash

#SBATCH --job-name=ngaeghy_preproc_dec
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT
#SBATCH --mail-user=ngaeghy@uw.edu

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

eval "$(/gscratch/stf/ngaeghy/miniconda3/bin/conda shell.bash hook)"
conda activate 573
/gscratch/stf/ngaeghy/miniconda3/envs/573/bin/python3 -m src.evaluation.dec_ft.preproc_dec --train_data_path /gscratch/stf/ngaeghy/LING573_AutoMeta/data/raw/ORSUM_test.jsonl --output_path /gscratch/stf/ngaeghy/LING573_AutoMeta/data/preprocessed/dec_test.txt
conda deactivate

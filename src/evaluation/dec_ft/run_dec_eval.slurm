#!/bin/bash

#SBATCH --job-name=ngaeghy_deceval_bartf
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT
#SBATCH --mail-user=ngaeghy@uw.edu

#SBATCH --account=stf
#SBATCH --partition=ckpt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --time=01-12:00:00 # Max runtime in DD-HH:MM:SS format.

#SBATCH --export=all
#SBATCH --output=deceval_bartf.out # where STDOUT goes
#SBATCH --error=deceval_bartf.error # where STDERR goes

module load cuda

eval "$(/gscratch/stf/ngaeghy/miniconda3/bin/conda shell.bash hook)"
conda activate 573
python3 -m src.evaluation.dec_ft.run_dec_eval --input_path /gscratch/stf/ngaeghy/LING573_AutoMeta/outputs/generated/finetune/bart_res.txt --result_path /gscratch/stf/ngaeghy/LING573_AutoMeta/src/evaluation/dec_ft/results/bart_res_deceval.txt
conda deactivate

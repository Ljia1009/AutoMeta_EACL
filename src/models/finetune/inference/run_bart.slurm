#!/bin/bash

#SBATCH --job-name=run-bart
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT
#SBATCH --mail-user=jiamuluo@uw.edu

#SBATCH --account=stf
#SBATCH --partition=ckpt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --time=01-12:00:00 # Max runtime in DD-HH:MM:SS format.

#SBATCH --export=all
#SBATCH --output=run_bart.out # where STDOUT goes
#SBATCH --error=run_bart.error # where STDERR goes

# Your programs to run.
eval "$(/gscratch/stf/jiamu/gscratch/stf/jiamu/miniconda3/bin/conda shell.bash hook)"
conda activate ling573
python3 -m src.models.finetune.inference.run_bart --output_path /gscratch/stf/jiamu/LING573_AutoMeta/src/models/finetune/inference/bart_res.txt
conda deactivate
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
#SBATCH --output=preproc_dec_tr.out # where STDOUT goes
#SBATCH --error=preproc_dec_tr.error # where STDERR goes

module load cuda

eval "$(/gscratch/stf/ngaeghy/miniconda3/bin/conda shell.bash hook)"
conda activate 573
/gscratch/stf/ngaeghy/miniconda3/envs/573/bin/python3 -m src.evaluation.dec_ft.preproc_dec --data_option train
conda deactivate

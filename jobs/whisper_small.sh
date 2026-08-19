#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH -t 12:00:00
#SBATCH --gpus=v100-32:4

# type 'man sbatch' for more information and options
# this job will ask for 1 full RM node (128 cores) for 5 hours
# this job would potentially charge 640 RM SUs

#echo commands to stdout
set -x

cd /ocean/projects/cis250209p/apham8/asr-thesis-project

# run a pre-compiled program which is already in your project space
conda activate env
python src/train_asr.py model=whisper_small 



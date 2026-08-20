#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH -t 12:00:00
#SBATCH --gpus=v100-32:4

# Echo executed commands for debugging
set -x

# Navigate to project root
cd /ocean/projects/cis250209p/apham8/asr-thesis-project || exit 1

# Load Anaconda and initialize shell hook for non-interactive bash
module load anaconda3
source $(conda info --base)/etc/profile.d/conda.sh
conda activate env

# Launch multi-GPU training
accelerate launch \
    --num_processes=4 \
    --multi_gpu \
    src/train_post_correction.py \
    model=llama \
    training=sft \
    lora=llama \
    use_lora=true \
    dataset=speech2latex_llm
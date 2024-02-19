#!/bin/bash

#SBATCH -p das
#SBATCH -A das
#SBATCH --array=0-7
#SBATCH --qos das-preempt
#SBATCH --gpus 1
#SBATCH --time 24:00:00
#SBATCH --mem 10GB
#SBATCH --nodelist cn104
#SBATCH --output=beam_job_%A_%a.out

# path to checkpoint
CKPT_PATH=$1

# Define the values for BEAM_SIZE
BEAM_SIZE_VALUES=(10 50 100 500 1000 1500 2000 0)

# Get the current BEAM_SIZE value for this array task
BEAM_SIZE=${BEAM_SIZE_VALUES[$SLURM_ARRAY_TASK_ID]}

echo BEAM_SIZE="$BEAM_SIZE"
echo CKPT_PATH="$CKPT_PATH"

if [ "$BEAM_SIZE" -eq 0 ]; then
  python3 run_ft_asr.py use_wandb=false load_from_ckpt="$(realpath "$CKPT_PATH")" fit_model=false use_lm=false 2> /dev/null | grep wer=
else
  python3 run_ft_asr.py use_wandb=false load_from_ckpt="$(realpath "$CKPT_PATH")" fit_model=false use_lm=true beam_size="$BEAM_SIZE" 2> /dev/null | grep wer=
fi

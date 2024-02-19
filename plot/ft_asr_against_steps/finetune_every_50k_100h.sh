#!/bin/bash

#SBATCH -p das
#SBATCH -A das
#SBATCH --qos das-preempt
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=24:00:00
#SBATCH --mem=30GB
#SBATCH --nodelist=cn117
#SBATCH --array=0-7
#SBATCH --mail-type=ALL
#SBATCH --mail-user=nvaessen
#SBATCH --output=offline-100h-%A_%a.out
#SBATCH --error=offline-100h-%A_%a.err

# Check if the CKPT_DIR argument is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <checkpoint_directory> $1 <batch size>"
    exit 1
fi

# Check if the BS argument is provided
if [ -z "$2" ]; then
    echo "Usage: $0 <checkpoint_directory> $1 <batch size>"
    exit 1
fi


# Assign the input arguments
CKPT_DIR="$1"
BS="$2"

# Define the STEPS_ARRAY
STEPS_ARRAY=(050000 100000 150000 200000 250000 300000 350000 400000)

# Check if SLURM_TASK_ARRAY_ID is provided
if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    echo "SLURM_ARRAY_TASK_ID not provided."
    exit 1
fi

# Use SLURM_TASK_ARRAY_ID to select a specific STEP
echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
STEP="${STEPS_ARRAY[$SLURM_ARRAY_TASK_ID]}"
STEP="000000$STEP"
echo "Selected STEP: $STEP"

# Find a file in CKPT_DIR with the STEP value in the filename
CKPT_FILES=$(find "$CKPT_DIR" -type f -name "*${STEP}*.progress.ckpt")

# Count the number of files found
FILE_COUNT=$(echo "$CKPT_FILES" | wc -l)

# Check if more than one file is found
if [ "$FILE_COUNT" -gt 1 ]; then
    echo "Error: Found more than one file in $CKPT_DIR with STEP value $STEP in the filename."
    exit 1
elif [ "$FILE_COUNT" -eq 0 ]; then
    echo "No file found in $CKPT_DIR with STEP value $STEP in the filename."
    exit 1
else
    # If only one file is found, assign it to CKPT_FILE
    CKPT_FILE="$CKPT_FILES"
    echo "Found file: $CKPT_FILE"
fi

# set tag based on SLURM_TASK_ID
STEPS_STR=("50k" "100k" "150k" "200k" "250k" "300k" "350k" "400k")
STEP_STR="${STEPS_STR[$SLURM_ARRAY_TASK_ID]}"
TAG="[$BS,$STEP_STR,ft_time]"

# do a fine-tuning with the checkpoint
WANDB_MODE='offline' python run_ft_asr.py +experiment=ft_100h load_from_ckpt="$CKPT_FILE" tags="$TAG"

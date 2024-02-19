#!/usr/bin/env bash

#SBATCH -p das
#SBATCH -A das
#SBATCH --gres gpu:1
#SBATCH --time 24:00:00
#SBATCH --mem 10GB

# Check if all three arguments are provided
if [ "$#" -ge 3 ]; then
  # 1st arguments points towards a directory of checkpoints
  CKPT_DIR=$1

  # 2nd argument specifies the accumulation
  NUM_ACCUMULATION=$2

  # 3th argument specified where to store data
  CSV_FILE=$3
else
    echo "Please provide CKPT_DIR, NUM_ACCUMULATION and CSV_FILE."
fi

# optional 4th argument to set number of max tokens
if [ -n "$4" ]; then
  MAX_TOKENS=$4
else
  MAX_TOKENS=2_400_000
fi

# optional 5th argument points towards a temp dict to store gradients
if [ -n "$5" ]; then
  TMP_DIR=$5
else
  if [ -n "$SLURM_JOB_ID" ]; then
    TMP_DIR=outputs/gradients/$SLURM_JOB_ID
  else
    TMP_DIR=outputs/gradients/$(date +"%Y-%m-%d_%H-%M-%S")
  fi
fi

# get path of this script
SCRIPT_DIR=plot/ssl_grad_var

# collect all checkpoints
CHECKPOINTS=$(find "$CKPT_DIR"/ -name "*.progress.ckpt" -or -name "*.init.ckpt" | sort)

# loop over each checkpoint file
for file in $CHECKPOINTS;
do

  # compute gradients for 10 different batches
  python run_ssl.py use_wandb=false \
    train.num_steps=1000 \
    train.store_gradients=true \
    train.max_validation_steps=10 \
    train.val_interval=1000 \
    optim.algo.lr=0 \
    data.pipe.train.num_workers=1 \
    data.pipe.train.max_tokens="$MAX_TOKENS" \
    load_from_ckpt="$file" \
    hydra.run.dir="$TMP_DIR" \
    train.accumulation="$NUM_ACCUMULATION"

  # process gradient vectors
  python "$SCRIPT_DIR"/accumulate_variance.py "$TMP_DIR" "$file" "$CSV_FILE" "$NUM_ACCUMULATION"

  # delete tmp dir
  rm -r "$TMP_DIR"

done




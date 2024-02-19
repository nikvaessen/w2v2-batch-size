#!/bin/bash

#SBATCH -p das
#SBATCH -A das
#SBATCH --qos das-preempt
#SBATCH --gpus 1
#SBATCH --time 24:00:00
#SBATCH --mem 10GB
#SBATCH --nodelist cn104

CKPT_PATH=$1

echo "without LM"
for x in $(find "$CKPT_PATH" -name '*.ckpt' | sort)
do
echo "$x"
python3 run_ft_asr.py use_wandb=false load_from_ckpt="$(realpath "$x")" fit_model=false use_lm=false 2> /dev/null | grep wer=
done

echo "with LM"

for x in $(find "$CKPT_PATH" -name '*.ckpt' | sort)
do
echo "$x"
python3 run_ft_asr.py use_wandb=false load_from_ckpt="$(realpath "$x")" fit_model=false use_lm=true 2> /dev/null | grep wer=
done
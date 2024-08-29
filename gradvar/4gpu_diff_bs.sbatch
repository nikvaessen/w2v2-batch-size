#!/usr/bin/env bash

#SBATCH --partition=gpu

#SBATCH --time=0-1
#SBATCH --job-name=gradvar

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1

#SBATCH --cpus-per-task=18
#SBATCH --gpu-bind=verbose,closest

#SBATCH --array=1,2,8,16,32

source "$HOME"/bin/load-modules
source .venv/bin/activate

pip freeze | grep torch

find /home/nvaessen/project/nanow2v2/ssl-every-50k/4gpu/ -type f | sort | xargs -I {} \
python3 grad_var.py +experiment=gradvar load_from_ckpt={} \
lightning.trainer.accumulate_grad_batches="$SLURM_ARRAY_TASK_ID"

#!/usr/bin/env bash

#SBATCH --partition=gpu

#SBATCH --time=0-1
#SBATCH --job-name=gradvar

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1

#SBATCH --cpus-per-task=18
#SBATCH --gpu-bind=verbose,closest

#SBATCH --array=0,1,2,4,8,16,32

if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ];
then
    NUM_ACC=1
    SAMPLES=1_400_000
else
    NUM_ACC=$SLURM_ARRAY_TASK_ID
    SAMPLES=2_800_000
fi

source "$HOME"/bin/load-modules
source .venv/bin/activate

find /home/nvaessen/project/nanow2v2/ssl-every-50k/"$SLURM_ARRAY_TASK_ID"gpu/ -type f | sort | xargs -I {} \
python3 grad_var.py +experiment=gradvar load_from_ckpt={} \
lightning.trainer.accumulate_grad_batches="$NUM_ACC" \
data.train.sampler.max_num_samples=$SAMPLES

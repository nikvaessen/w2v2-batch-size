# An implementation of wav2vec 2.0

This repository contains the code for the paper 
[The effect of batch size on contrastive self-supervised speech representation learning](https://arxiv.org/abs/anon)
by anon. This work can be cited using
```bibtex
very anon
```

## (Pre-training) model checkpoints

Here we provide the best checkpoint (according to validation loss) for each batch size condition:

| batch size | iteration | learning rate | checkpoint                                                                |
|------------|-----------|---------------|---------------------------------------------------------------------------|
| 87.5 sec   | 395k      | 7.29e-6       | [0gpu.ckpt](https://surfdrive.surf.nl/files/index.php/s/eZZy53BXYQHXNLX)  |
| 150 sec    | 400k      | 7.91e-5       | [1gpu.ckpt](https://surfdrive.surf.nl/files/index.php/s/Wji0ZEcYkOnHqq5)  |
| 5 min      | 400k      | 1.12e-4       | [2pgu.ckpt](https://surfdrive.surf.nl/files/index.php/s/NPClooexFIvkgTX)  |
| 10 min     | 400k      | 1.58e-4       | [4gpu.ckpt](https://surfdrive.surf.nl/files/index.php/s/oGS0GWlKqebkqI9)  |
| 20 min     | 400k      | 2.24e-4       | [8gpu.ckpt](https://surfdrive.surf.nl/files/index.php/s/izHq8UebczavmyY)  |
| 40 min     | 400k      | 5e-4          | [16gpu.ckpt](https://surfdrive.surf.nl/files/index.php/s/UTHOiwUzG0U1puu) |
| 80 min     | 305k      | 5e-4          | [32gpu.ckpt](https://surfdrive.surf.nl/files/index.php/s/EMKdgKSCvrv2lzf) |


All intermediary pre-training checkpoints (~230 GB) can be downloaded using the
following torrent: anon

The checkpoint(s) can be converted to fairseq format by using
[nano_to_fairseq.py](convert/nano_to_fairseq.py) and then to huggingface format with
the script [convert_fairseq_to_hf.py](convert_fairseq_to_hf.py).

## Training plots

We used weight and biases to plot various metrics during training. The SSL plots can be found here: https://wandb.ai/anon/nanow2v2-ssl/table?workspace=default

For ASR fine-tuning, the plots are provided here: 
https://wandb.ai/anon/nanow2v2-asr/table?workspace=default. 
Note that we filter by the tag `16gpu` by default.
To get a different batch size, change the filter to the correspond value;
in the table above the filename of each checkpoint is the corresponding tag (e.g., 20 mins = `8gpu`).

## Setup

If you want to run the code to do pre-training and/or fine-tuning, first follow these steps:

1. Create a virtual environment and install all dependencies: `python3 -m venv .venv; source .venv; pip install -r requirements.txt` 
2. Create the environment variables file: `cp .env.example .env`
3. Fill in `.env` accordingly with your favourite text editor and then run `source export_env.sh`
4. Setup the librispeech dataset: `./data/librispeech/all_steps.sh` (takes a few hours)
5. Copy [character_vocabulary.json](character_vocabulary.json) to $LIBRISPEECH_META_DIR: `cp character_vocabulary.json "$LIBRISPEECH_META_DIR"/character_vocabulary.json`

## Running pre-training experiments

All pre-training experiments were run by using the following commands. The `hydra/launcher=x` and `hydra.launcher.timeout_min=x` parameters are specific to the SLURM cluster and need to be changed/removed to your needs. 

### batch size 87.5 seconds

```
python run_ssl.py -m optim.algo.lr=7.29E-06,6.04E-05,5.00E-04 train.devices=1 train.accumulation=1 tags="[0gpu,cyclic]" data.pipe.train.max_tokens=1_400_000 hydra/launcher=das_preempt hydra.launcher.timeout_min=30240
```

### batch size 150 seconds

```
python run_ssl.py -m optim.algo.lr=1.25E-05,7.91E-05,5.00E-04 train.devices=1 train.accumulation=1 tags="[1gpu,cyclic]" network.ssl_cfg.diversity_loss_epsilon=0,1e-7 hydra/launcher=das_preempt hydra.launcher.timeout_min=30240
```

### batch size 5 minutes

```
python run_ssl.py -m optim.algo.lr=7.910E-05,1.25E-04,5.00E-04 train.devices=2 train.accumulation=1 tags="[2gpu,cyclic]" network.ssl_cfg.diversity_loss_epsilon=0,1e-7 hydra/launcher=das_preempt hydra.launcher.timeout_min=30240
```

### batch size 10 minutes

```
python run_ssl.py -m optim.algo.lr=5.00E-05,1.58E-04,5.00E-04 train.devices=4 train.accumulation=1 tags="[4gpu,cyclic]" hydra/launcher=das_preempt hydra.launcher.timeout_min=30240
```

### batch size 20 minutes
```
python run_ssl.py -m optim.algo.lr=1.00E-04,2.24E-04,5.00E-04 train.devices=4 train.accumulation=2 tags="[8gpu,cyclic]" hydra/launcher=icis_preempt hydra.launcher.timeout_min=30240
```

### batch size 40 minutes

```
python run_ssl.py -m optim.algo.lr=2.00E-04,3.16E-04,5.00E-04 train.devices=4 train.accumulation=4 tags="[16gpu,cyclic]" hydra/launcher=icis_preempt hydra.launcher.timeout_min=30240
```

### batch size 80 minutes

```
python run_ssl.py -m optim.algo.lr=5.00E-04 train.devices=4 train.accumulation=8 tags="[32gpu,cyclic]" hydra/launcher=icis_preempt hydra.launcher.timeout_min=30240
```

## Running fine-tuning experiments

To fine-tune a checkpoint `path/to/checkpoint.ckpt` for ASR, the following command can be used:

```
python run_ft_asr.py +experiment=$CONDITION load_from_ckpt="$(realpath path/to/checkpoint.ckpt)"
```

where `$CONDITION` is one of 

1. [`ft_min_10`](config/experiment/ft_min_10.yaml)
2. [`ft_1h`](config/experiment/ft_1h.yaml)
3. [`ft_10h`](config/experiment/ft_10h.yaml)
4. [`ft_100h`](config/experiment/ft_100h.yaml)
5. [`ft_960h`](config/experiment/ft_960h.yaml)

If word decoding is desired, `decoder.use_lm=true` can be added to the command 
(which uses settings of [default.yaml](config/decoder/default.yaml)), or use a decoder 
like [4gram_fair_10min.yaml](config/decoder/4gram_fair_10min.yaml) by setting
`decoder=4gram_fair_10min`. 

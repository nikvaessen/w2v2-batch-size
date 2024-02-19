"""
Run the fine-tuning for ASR phase of wav2vec2.
"""

import json
import pathlib
import time

from collections import defaultdict
from typing import List, Any, Dict

import jiwer
import torch
import tqdm
import wandb

from lightning import Fabric
from hydra.utils import instantiate
from torch.utils.data import DataLoader

from nanow2v2.data.datapipe import DataPipeBuilder, DataBatch
from nanow2v2.data.encode import decode_predictions_greedy, decode_predictions_lm
from nanow2v2.model.wav2vec2_asr import Wav2vec2ForASR
from nanow2v2.trainer import (
    FabricTrainer,
    TrainableModule,
    get_best_ckpt,
    get_last_ckpt,
)


########################################################################################
# Define how to train and validate a model


class TrainableModuleForASRTraining(TrainableModule):
    def __init__(
        self,
        network: Wav2vec2ForASR,
        idx_to_char: Dict,
        save_all_train_predictions: bool = False,
    ):
        super().__init__(network)

        # for decoding
        self.idx_to_char = idx_to_char

        # debug all train predictions
        self.save_all_predictions = save_all_train_predictions

        # temp storage of training results
        self.train_result_dict = defaultdict(lambda: defaultdict(list))
        self.train_transcriptions_dict = {}

        # temp storage of validation results
        self.validation_transcription_dict = defaultdict(list)

    def checkpoint_prefix(self):
        return "w2v2.asr"

    def train_step(self, batch: Any, step: int, accumulation: int) -> torch.Tensor:
        assert isinstance(batch, DataBatch)

        loss, vocab_prob, sample_length = self.network.forward_asr_training(
            batch.audio_tensor,
            batch.audio_num_frames,
            batch.transcriptions_tensor,
            batch.transcriptions_length,
            train_step=step,
        )

        k = str(step)
        self.train_result_dict[k]["loss"].append(loss.detach().cpu())

        # decode all
        with torch.no_grad():
            transcription = decode_predictions_greedy(
                vocab_prob, self.idx_to_char, sample_length
            )

        # safe all transcription as tuples (ref, hyp)
        for idx, key in enumerate(batch.keys):
            self.train_transcriptions_dict[key] = (
                batch.transcriptions[idx],
                transcription[idx],
            )

        return loss

    def debug_train_step(self, batch: Any) -> List[str]:
        all_pairs = []

        for i, k in enumerate(batch.keys):
            ref, hyp = self.train_transcriptions_dict[k]
            all_pairs.append(f"ref[{i}]=\\\\{ref}//")
            all_pairs.append(f"hyp[{i}]=\\\\{hyp}//")

            if not self.save_all_predictions:
                break

        return all_pairs

    def debug_train_batch(self, batch: Any) -> List[str]:
        assert isinstance(batch, DataBatch)

        return [
            f"batch size={len(batch.keys)}",
            f"total audio_frames={sum(batch.audio_num_frames)}",
            f"audio tensor shape={list(batch.audio_tensor.shape)}",
            f"audio lengths={batch.audio_num_frames}",
            f"length difference={batch.audio_frame_difference()}",
            f"ref shape={batch.transcriptions_tensor.shape}",
            f"ref lengths={batch.transcriptions_length}",
            f"keys={batch.keys}",
        ]

    @torch.no_grad()
    def log_training(
        self, fabric: Fabric, current_step: int, num_calls: int, durations: List[float]
    ):
        if fabric.is_global_zero:
            # collect mean loss
            losses = []

            for idx, k in enumerate(self.train_result_dict.keys()):
                loss = torch.mean(torch.stack(self.train_result_dict[k]["loss"]))
                losses.append(loss)

            average_loss = torch.mean(torch.stack(losses)).item()

            # collect mean WER
            references = [tpl[0] for tpl in self.train_transcriptions_dict.values()]
            hypothesis = [tpl[1] for tpl in self.train_transcriptions_dict.values()]
            train_wer = jiwer.wer(references, hypothesis)

            # log loss and WER to wandb/tensorboard
            if fabric.is_global_zero:
                wandb.log(
                    {"train/loss": average_loss, "train/wer": train_wer},
                    step=current_step,
                )

            # log to stdout
            it_per_sec = num_calls / sum(durations)

            total_params = sum(p.numel() for p in self.network.parameters())
            trainable_params = sum(
                p.numel() for p in self.network.parameters() if p.requires_grad
            )

            fabric.print(
                f"{current_step=:>12d} | "
                f"{average_loss=:.2f} over previous {num_calls} iterations | "
                f"{train_wer=:.2f} | "
                f"params: trainable {trainable_params}, total {total_params} | "
                f"it/sec={it_per_sec:.2f} ",
                flush=True,
            )

        self.train_result_dict.clear()
        self.train_transcriptions_dict.clear()

    def validation_step(self, batch: Any, step: int) -> torch.Tensor:
        assert isinstance(batch, DataBatch)

        loss, vocab_prob, sample_lengths = self.network.forward_asr_training(
            batch.audio_tensor,
            batch.audio_num_frames,
            batch.transcriptions_tensor,
            batch.transcriptions_length,
        )

        # decode all
        transcription = decode_predictions_greedy(
            vocab_prob, self.idx_to_char, sample_lengths
        )

        self.validation_transcription_dict["reference"].extend(batch.transcriptions)
        self.validation_transcription_dict["hypothesis"].extend(transcription)

        return loss

    @torch.no_grad()
    def log_validation(
        self,
        fabric: Fabric,
        validation_metric: torch.Tensor,
        current_step: int,
        num_calls: int,
        durations: List[float],
    ):
        if fabric.is_global_zero:
            total_duration = sum(durations)

            # compute val wer
            val_wer = jiwer.wer(
                self.validation_transcription_dict["reference"],
                self.validation_transcription_dict["hypothesis"],
            )

            # log to wandb/tensorboard
            wandb.log(
                {"val/loss": validation_metric, "val/wer": val_wer},
                step=current_step,
            )

            # log to stdout
            fabric.print(
                f"validation at {current_step=} | "
                f"val_loss={validation_metric.item():.2f} | "
                f"val_wer={val_wer:.2f} | "
                f"duration={total_duration:.2f} seconds | "
                f"it/sec={num_calls / total_duration:.2f}",
                flush=True,
            )

        self.validation_transcription_dict.clear()


########################################################################################
# evaluation


@torch.no_grad()
def evaluate_model(
    fabric: Fabric,
    model: Wav2vec2ForASR,
    test_dataloader: DataLoader,
    dataset_name: str,
    idx_to_char: Dict,
    use_lm: bool = False,
    beam_size: int = 50,
    lm_weight: float = 2.0,
    word_score: float = 0.0,
):
    model = model.eval()
    model = model.to(fabric.device)

    references = []
    hypothesis = []
    frame_count = []

    start = time.time()

    for batch in tqdm.tqdm(test_dataloader):
        assert isinstance(batch, DataBatch)

        vocab_prob, sample_length = model.forward(
            batch.audio_tensor, batch.audio_num_frames
        )

        if use_lm:
            transcriptions = decode_predictions_lm(
                vocab_prob, idx_to_char, sample_length, beam_size, lm_weight, word_score
            )
        else:
            transcriptions = decode_predictions_greedy(
                vocab_prob, idx_to_char, sample_length
            )

        references.extend(batch.transcriptions)
        hypothesis.extend(transcriptions)
        frame_count.extend(batch.audio_num_frames)

    end = time.time()
    wer = jiwer.wer(references, hypothesis)

    fabric.print(
        f"{dataset_name} "
        f"{wer=} run_time={end-start:.2f}s "
        f"seen_frames={sum(frame_count)} "
        f"{use_lm=} {beam_size=}"
    )

    if fabric.is_global_zero:
        with (pathlib.Path.cwd() / f"eval_{dataset_name}.txt").open("w") as f:
            for ref, hyp in zip(references, hypothesis):
                f.write(f"{ref} | {hyp}\n")

    return wer


########################################################################################
# entrypoint of training


def train_asr(cfg):
    # create a trainer
    train_cfg = instantiate(cfg.train)
    log_cfg = instantiate(cfg.log)
    trainer = FabricTrainer(cfg, train_cfg, log_cfg)

    # load char to idx map
    char_to_idx_json = cfg.data.set.character_vocabulary_json_path
    with open(char_to_idx_json, "r") as f:
        json_obj = json.load(f)
        char_to_idx = json_obj["char_to_idx"]
        idx_to_char = json_obj["idx_to_char"]
        idx_to_char = {int(k): v for k, v in idx_to_char.items()}

    # load train/val/test dataloaders
    print("Building training datapipe")
    train_builder = DataPipeBuilder(
        instantiate(cfg.data.pipe.train), char_to_idx, shard_workers=True
    )
    train_pipe, train_shard_list = train_builder.get_pipe(
        cfg.data.set.train_shard_path, cfg.data.set.shard_pattern
    )
    train_loader = train_builder.wrap_pipe(train_pipe, len(train_shard_list))

    trainer.fabric.print("Train data pipe consists of:")
    for shard in train_shard_list:
        trainer.fabric.print(f"\t{shard}")

    print("Building validation datapipe")
    val_builder = DataPipeBuilder(
        instantiate(cfg.data.pipe.val), char_to_idx, shard_workers=False
    )
    val_pipe, val_shard_list = val_builder.get_pipe(
        cfg.data.set.val_shard_path, cfg.data.set.shard_pattern
    )
    val_loader = val_builder.wrap_pipe(val_pipe, len(val_shard_list))

    trainer.fabric.print("Validation data pipe consists of:")
    for shard in val_shard_list:
        trainer.fabric.print(f"\t{shard}")

    # init model and optimizer
    network = instantiate(cfg.network, init_frozen=True)

    if cfg.compile:
        network = torch.compile(network, dynamic=True)

    if cfg.load_from_ckpt is not None:
        trainer.fabric.print(f"loading weights from {cfg.load_from_ckpt=}")

        ckpt = trainer.fabric.load(cfg.load_from_ckpt)
        missing_keys, unexpected_keys = network.load_state_dict(
            ckpt["network"], strict=False
        )

        trainer.fabric.print("missing keys:")
        for mk in missing_keys:
            trainer.fabric.print(f"\t{mk}")
        if len(missing_keys) == 0:
            trainer.fabric.print("\tno keys were missing")

        trainer.fabric.print("unexpected keys:")
        for uk in unexpected_keys:
            trainer.fabric.print(f"\t{uk=}")
        if len(unexpected_keys) == 0:
            trainer.fabric.print("\tno keys were unexpected")

    optimizer = instantiate(cfg.optim.algo, params=network.parameters())
    lr_schedule = instantiate(cfg.optim.schedule, optimizer=optimizer)

    # wrap model in TrainableModule
    module = TrainableModuleForASRTraining(
        network, idx_to_char, save_all_train_predictions=cfg.save_all_train_predictions
    )

    # train module
    if cfg.fit_model:
        val_metric = trainer.fit(
            module, optimizer, lr_schedule, train_loader, val_loader
        )
    else:
        val_metric = None

    if cfg.eval_model:
        # potentially reload from a checkpoint (best > last > current)
        best_ckpt = get_best_ckpt()
        last_ckpt = get_last_ckpt()

        if best_ckpt is not None:
            trainer.fabric.print(f"reloading from {best_ckpt}")
            ckpt = trainer.fabric.load(best_ckpt)
        elif last_ckpt is not None:
            trainer.fabric.print(f"reloading from {last_ckpt}")
            ckpt = torch.load(last_ckpt)
        else:
            ckpt = None

        if ckpt is not None:
            module.network.load_state_dict(ckpt["network"])

        # evaluate model
        test_builder = DataPipeBuilder(instantiate(cfg.data.pipe.test), char_to_idx)

        test_result_dict = {}
        for test_path in cfg.data.set.test_shard_path:
            test_name = pathlib.Path(test_path).stem
            print(f"Building test datapipe {test_name}")

            test_pipe, test_shard_list = test_builder.get_pipe(
                test_path, cfg.data.set.shard_pattern
            )

            trainer.fabric.print("Test data pipe consists of:")
            for shard in test_shard_list:
                trainer.fabric.print(f"\t{shard}")

            test_loader = test_builder.wrap_pipe(test_pipe, len(test_shard_list))
            test_loader = trainer.fabric.setup_dataloaders(test_loader)

            wer = evaluate_model(
                trainer.fabric,
                module.network,
                test_loader,
                test_name,
                idx_to_char,
                cfg.decoder.use_lm,
                cfg.decoder.beam_size,
                cfg.decoder.lm_weight,
                cfg.decoder.word_score,
            )
            test_result_dict[f"test/{test_name}_wer"] = wer

        if trainer.fabric.is_global_zero:
            wandb.log(test_result_dict)

    return val_metric

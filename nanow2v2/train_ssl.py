"""
Run the self-supervision phase of wav2vec2.
"""
from typing import List, Any, Dict

import torch
import wandb
import torchmetrics

from lightning import Fabric
from hydra.utils import instantiate
from torchmetrics import Metric

from nanow2v2.model.wav2vec2_ssl import Wav2vec2ForSSL, SSLForwardResult
from nanow2v2.data.datapipe import DataPipeBuilder, DataBatch

from nanow2v2.trainer import FabricTrainer, TrainableModule


########################################################################################
# Define how to train and validate a model


class CodebookPerplexityMetric(Metric):
    def __init__(self, num_entries: int):
        super().__init__()
        self.add_state(
            "count", default=torch.zeros((num_entries)), dist_reduce_fx="sum"
        )
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, count: torch.Tensor):
        self.count += count
        self.total += torch.sum(count)

    def compute(self):
        dist = self.count / self.total
        entropy = -torch.sum(dist * torch.log(dist + 1e-7))
        return torch.exp(entropy)


@torch.no_grad()
def get_entry_similarity(
    quantization_choices: torch.Tensor, num_codebooks: int, num_entries: int
):
    if len(quantization_choices.shape) == 3:
        quantization_choices = quantization_choices.squeeze()

    return_list = []

    for cb_idx in range(num_codebooks):
        start_idx = cb_idx * num_entries
        end_idx = start_idx + num_entries
        cb = quantization_choices[start_idx:end_idx, :]

        pairwise_similarities = []
        for i in range(num_entries):
            vector = cb[i : i + 1, :]
            others = cb[[j for j in range(num_entries) if j != i], :]

            sim = torch.nn.functional.cosine_similarity(vector, others)
            pairwise_similarities.append(sim)

        pairwise_similarities = torch.stack(pairwise_similarities)

        sim_avg = torch.mean(pairwise_similarities).item()
        sim_min = torch.min(pairwise_similarities).item()
        sim_max = torch.max(pairwise_similarities).item()
        sim_std = torch.std(pairwise_similarities).item()

        return_list.append((sim_avg, sim_min, sim_max, sim_std))

    return return_list


class TrainableModuleForSSLTraining(TrainableModule):
    def __init__(self, network: Wav2vec2ForSSL, accumulation: int):
        super().__init__(network)

        self.network = network

        # important for decreasing gumbel temperature
        self.last_accumulation_step = accumulation - 1

        # important for monitoring codebooks
        self.num_codebooks = network.ssl_cfg.num_codebooks
        self.num_entries = network.ssl_cfg.num_entries

        self.is_l2_loss_enabled = network.ssl_cfg.l2_loss_weight > 0
        self.is_div_loss_enabled = network.ssl_cfg.diversity_loss_weight > 0

        # temp storage of training results
        self.train_loss = torchmetrics.MeanMetric()
        self.train_loss_c = torchmetrics.MeanMetric()
        if self.is_div_loss_enabled:
            self.train_loss_d = torchmetrics.MeanMetric()
        if self.is_l2_loss_enabled:
            self.train_loss_l2 = torchmetrics.MeanMetric()
        self.train_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=network.ssl_cfg.num_negative_samples + 1
        )
        self.train_codebook_perplexity = [
            CodebookPerplexityMetric(self.num_entries)
            for _ in range(self.num_codebooks)
        ]

        # temp storage of validation results
        self.val_loss_c = torchmetrics.MeanMetric()
        if self.is_div_loss_enabled:
            self.val_loss_d = torchmetrics.MeanMetric()
        if self.is_l2_loss_enabled:
            self.val_loss_l2 = torchmetrics.MeanMetric()
        self.val_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=network.ssl_cfg.num_negative_samples + 1
        )
        self.val_codebook_perplexity = [
            CodebookPerplexityMetric(self.num_entries)
            for _ in range(self.num_codebooks)
        ]

    def checkpoint_prefix(self):
        return "w2v2.ssl"

    def set_device(self, device):
        self.train_loss.to(device)
        self.train_loss_c.to(device)
        if self.is_div_loss_enabled:
            self.train_loss_d.to(device)
        if self.is_l2_loss_enabled:
            self.train_loss_l2.to(device)
        self.train_acc.to(device)
        [m.to(device) for m in self.train_codebook_perplexity]

        self.val_loss_c.to(device)
        if self.is_div_loss_enabled:
            self.val_loss_d.to(device)
        if self.is_l2_loss_enabled:
            self.val_loss_l2.to(device)
        self.val_acc.to(device)
        [m.to(device) for m in self.val_codebook_perplexity]

    def store_codebook_idx(
        self, codebook_logits, metric_perplexity: List[CodebookPerplexityMetric]
    ):
        batch_size, seq_length, num_codebooks, num_entries = codebook_logits.shape
        assert num_codebooks == self.num_codebooks

        with torch.no_grad():
            probs = torch.nn.functional.softmax(codebook_logits, dim=3)

            for i in range(num_codebooks):
                probs_i = probs[:, :, i, :].view(batch_size * seq_length, -1)
                idx_i = torch.argmax(probs_i, dim=-1).flatten()
                counts_i = torch.bincount(idx_i, None, self.num_entries)

                metric_perplexity[i](counts_i)

    def compute_codebook_perplexity(
        self, prefix: str, metric_perplexity: List[CodebookPerplexityMetric]
    ) -> Dict[str, torch.Tensor]:
        return_dict = {}

        for i in range(self.num_codebooks):
            return_dict[f"{prefix}/perplexity_cb{i}"] = metric_perplexity[i].compute()

        return return_dict

    def compute_codebook_similarity(self):
        return_dict = {}

        vectors, num_codebooks, num_entries = self.network.get_codebooks()
        similarity = get_entry_similarity(vectors, num_codebooks, num_entries)

        for i, (avg_sim, min_sim, max_sim, std_sim) in enumerate(similarity):
            return_dict[f"similarity_cb{i}_avg"] = avg_sim
            return_dict[f"similarity_cb{i}_min"] = min_sim
            return_dict[f"similarity_cb{i}_max"] = max_sim
            return_dict[f"similarity_cb{i}_std"] = std_sim

        return return_dict

    def train_step(self, batch: Any, step: int, accumulation: int) -> torch.Tensor:
        assert isinstance(batch, DataBatch)

        r: SSLForwardResult = self.network.forward(
            batch.audio_tensor, batch.audio_num_frames
        )

        self.train_loss(r.loss)
        self.train_loss_c(r.loss_contrastive)
        if self.is_div_loss_enabled:
            self.train_loss_d(r.loss_diversity)
        if self.is_l2_loss_enabled:
            self.train_loss_l2(r.loss_l2)
        self.train_acc(r.cpc_logits, r.cpc_targets)
        self.store_codebook_idx(r.codebook_logits, self.train_codebook_perplexity)

        if accumulation == self.last_accumulation_step:
            self.network.step_gumbel_temperature()

        return r.loss

    def debug_train_batch(self, batch: Any) -> List[str]:
        assert isinstance(batch, DataBatch)

        return [
            f"batch size={len(batch.keys)}",
            f"total audio_frames={sum(batch.audio_num_frames)}",
            f"audio tensor shape={list(batch.audio_tensor.shape)}",
            f"audio lengths={batch.audio_num_frames}",
            f"keys={batch.keys}",
        ]

    def debug_train_step(self, batch: Any) -> List[str]:
        # get textual representation(s) of properties of the train step
        # which be logged to disk for debug purposes
        return []

    def log_training(
        self, fabric: Fabric, current_step: int, num_calls: int, durations: List[float]
    ):
        # collect metrics
        log_dict = {
            "train/loss": self.train_loss.compute(),
            "train/loss_contrastive": self.train_loss_c.compute(),
            "train/accuracy": self.train_acc.compute(),
            **self.compute_codebook_perplexity("train", self.train_codebook_perplexity),
            **self.compute_codebook_similarity(),
            "train/gumbel_temperature": self.network.get_gumbel_temperature(),
        }

        if self.is_div_loss_enabled:
            log_dict["train/loss_diversity"] = self.train_loss_d.compute()

        if self.is_l2_loss_enabled:
            log_dict["train/loss_l2_speech_features"] = self.train_loss_l2.compute()

        # log to wandb
        if fabric.is_global_zero:
            wandb.log(log_dict, step=current_step)

        # log to stdout
        it_per_sec = num_calls / sum(durations)

        total_params = sum(p.numel() for p in self.network.parameters())
        trainable_params = sum(
            p.numel() for p in self.network.parameters() if p.requires_grad
        )

        fabric.print(
            f"{current_step=:>12d} | "
            f"average_loss={log_dict['train/loss']:.2f}"
            f" over previous {num_calls} iterations | "
            f"params: trainable {trainable_params}, total {total_params} | "
            f"it/sec={it_per_sec:.2f}",
            flush=True,
        )

        # clear metrics
        self.train_acc.reset()
        self.train_loss.reset()
        self.train_loss_c.reset()
        if self.is_div_loss_enabled:
            self.train_loss_d.reset()
        if self.is_l2_loss_enabled:
            self.train_loss_l2.reset()
        self.train_steps_taken = set()
        [m.reset() for m in self.train_codebook_perplexity]

    def validation_step(self, batch: Any, step: int) -> torch.Tensor:
        assert isinstance(batch, DataBatch)

        r: SSLForwardResult = self.network.forward(
            batch.audio_tensor, batch.audio_num_frames
        )

        self.val_loss_c(r.loss_contrastive)
        if self.is_div_loss_enabled:
            self.val_loss_d(r.loss_diversity)
        if self.is_l2_loss_enabled:
            self.val_loss_l2(r.loss_l2)
        self.val_acc(r.cpc_logits, r.cpc_targets)
        self.store_codebook_idx(r.codebook_logits, self.val_codebook_perplexity)

        return r.loss

    def log_validation(
        self,
        fabric: Fabric,
        validation_metric: torch.Tensor,
        current_step: int,
        num_calls: int,
        durations: List[float],
    ):
        total_duration = sum(durations)

        # collect metrics
        log_dict = {
            "val/loss": validation_metric,
            "val/loss_contrastive": self.val_loss_c.compute(),
            "val/accuracy": self.val_acc.compute(),
            **self.compute_codebook_perplexity("val", self.val_codebook_perplexity),
        }

        if self.is_div_loss_enabled:
            log_dict["val/loss_diversity"] = self.val_loss_d.compute()

        if self.is_l2_loss_enabled:
            log_dict["val/loss_l2_speech_features"] = self.val_loss_l2.compute()

        # log to wandb/tensorboard
        if fabric.is_global_zero:
            wandb.log(log_dict, current_step, True)

        # log to stdout
        fabric.print(
            f"validation at {current_step=} | "
            f"val_loss={validation_metric.item():.2f} | "
            f"duration={total_duration:.2f} seconds | "
            f"it/sec={num_calls / total_duration:.2f}",
            flush=True,
        )

        # clear metrics
        self.val_loss_c.reset()
        if self.is_div_loss_enabled:
            self.val_loss_d.reset()
        if self.is_l2_loss_enabled:
            self.val_loss_l2.reset()
        self.val_acc.reset()
        [m.reset() for m in self.val_codebook_perplexity]


########################################################################################
# entrypoint of training


def train_ssl(cfg):
    # create a trainer
    train_cfg = instantiate(cfg.train)
    log_cfg = instantiate(cfg.log)
    trainer = FabricTrainer(cfg, train_cfg, log_cfg)

    # load train/val/test dataloaders

    # train
    train_builder = DataPipeBuilder(instantiate(cfg.data.pipe.train))
    train_pipe, train_shard_list = train_builder.get_pipe(
        cfg.data.set.train_shard_path, cfg.data.set.shard_pattern
    )
    train_loader = train_builder.wrap_pipe(train_pipe, len(train_shard_list))

    trainer.fabric.print("Training data pipe consists of:")
    for shard in train_shard_list:
        trainer.fabric.print(f"\t{shard}")

    # val
    val_builder = DataPipeBuilder(instantiate(cfg.data.pipe.val), shard_workers=False)
    val_pipe, val_shard_list = val_builder.get_pipe(
        cfg.data.set.val_shard_path, cfg.data.set.shard_pattern
    )
    val_loader = val_builder.wrap_pipe(val_pipe, len(val_shard_list))

    trainer.fabric.print("Validation data pipe consists of:")
    for shard in val_shard_list:
        trainer.fabric.print(f"\t{shard}")

    # setup loaders with fabric
    train_loader, val_loader = trainer.fabric.setup_dataloaders(
        train_loader, val_loader
    )

    # init model and optimizer
    network = instantiate(cfg.network)

    if cfg.compile:
        network = torch.compile(network, dynamic=False)

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

    network = trainer.fabric.setup_module(network)

    optimizer = instantiate(cfg.optim.algo, params=network.parameters())
    optimizer = trainer.fabric.setup_optimizers(optimizer)

    lr_schedule = instantiate(cfg.optim.schedule, optimizer=optimizer)

    # wrap model in TrainableModule
    model = TrainableModuleForSSLTraining(network, cfg.train.accumulation)
    model.set_device(trainer.fabric.device)

    # train module
    if cfg.enable_profiling:
        trainer.fit_profile(model, optimizer, lr_schedule, train_loader)
        val_metric = None
    else:
        val_metric = trainer.fit(
            model, optimizer, lr_schedule, train_loader, val_loader, is_setup=True
        )

    return val_metric

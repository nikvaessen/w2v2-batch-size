########################################################################################
#
# Fabric-based trainer for local and slurm-based model training and evaluation.
#
# Author(s): anon
########################################################################################

import abc
import gc
import os
import pathlib
import time

from dataclasses import dataclass
from multiprocessing.pool import Pool
from typing import List, Any, Iterator, Tuple, Optional, Union

import torch
import wandb

from lightning import Fabric
from omegaconf import OmegaConf, ListConfig, DictConfig
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader


########################################################################################
# configs related to training


@dataclass
class FabricTrainerConfig:
    # how to run
    accelerator: str
    devices: int
    num_nodes: int
    strategy: str

    precision: str

    # training length
    num_steps: int
    val_interval: int
    accumulation: int

    # potentially cap the number of validation steps each interval
    # -1 indicates all samples are processed
    max_validation_steps: int

    # sanity validation loop
    num_sanity_validation_steps: int

    # experiment reproducibility
    random_seed: int

    # checkpointing
    save_init_ckpt: bool
    save_last_ckpt: bool
    save_progress_ckpt: bool

    best_ckpt_op: str  # 'min' or 'max'
    save_best_ckpt: bool

    # store gradient vectors
    store_gradients: False

    # gradient clipping strategies (at most one can be not null)
    clip_norm: Optional[float] = None
    clip_value: Optional[float] = None

    # enable unused param check if using layer drop
    ddp_find_unused_parameters: bool = False
    ddp_gradient_as_bucket_view: bool = True

    # early stopping (-1 is disabled)
    early_stopping_patience: int = -1

    def __post_init__(self):
        assert self.best_ckpt_op in ["min", "max"]


@dataclass
class LoggingConfig:
    # tags to easily find experiment
    date_str: str
    tags: List[str]

    # how to log
    project_name: str
    use_wandb: bool

    # how often to log
    log_interval: int


########################################################################################
# implementation


class TrainableModule(abc.ABC):
    def __init__(self, network: torch.nn.Module):
        self.network = network

    @abc.abstractmethod
    def checkpoint_prefix(self):
        pass

    @abc.abstractmethod
    def train_step(self, batch: Any, step: int, accumulation: int) -> torch.Tensor:
        # given some data, return a scalar tensor which to apply backpropagation on
        pass

    @abc.abstractmethod
    def validation_step(self, batch: Any, step: int) -> torch.Tensor:
        # given some data, return a scalar tensor used as a validation metric
        # which is minimized
        pass

    @abc.abstractmethod
    def log_training(
        self, fabric: Fabric, current_step: int, num_steps: int, durations: List[float]
    ):
        # called whenever the trainer deems fit to log results from training steps
        # num_steps is how many times `train_step` was called before this call
        # to `log_training`. Durations is a list in seconds of each training step.
        pass

    @abc.abstractmethod
    def log_validation(
        self,
        fabric: Fabric,
        validation_metric: torch.Tensor,
        current_step: int,
        num_steps: int,
        durations: List[float],
    ):
        # called whenever the trainer deems fit to log results from validation steps
        # num_steps is how many times `validation_step` was called before this call
        # to `log_validation`. Durations is a list in seconds of each validation step.
        pass

    @abc.abstractmethod
    def debug_train_batch(self, batch: Any) -> List[str]:
        # get textual representation(s) of properties of the batch
        # which be logged to disk for debug purposes
        pass

    @abc.abstractmethod
    def debug_train_step(self, batch: Any) -> List[str]:
        # get textual representation(s) of properties of the train step
        # which be logged to disk for debug purposes
        pass

    def train(self):
        self.network.train()

    def eval(self):
        self.network.eval()


class TrainingState:
    def __init__(
        self,
        module: TrainableModule,
        optimizer: torch.optim.Optimizer,
        lr_schedule: torch.optim.lr_scheduler.LRScheduler,
        current_step: Union[torch.Tensor, int] = torch.tensor(-1),
        validation_history: List[float] = None,
    ):
        if isinstance(current_step, int):
            current_step = torch.tensor(current_step, dtype=torch.int)

        self.dictionary = {
            "network": module,
            "optimizer": optimizer,
            "lr_schedule": lr_schedule,
            "current_step": current_step,
            "validation_history": validation_history,
        }

    def state(self, weights_only: bool, for_saving: bool):
        # argument 'for_saving' is temporary fix for
        # https://github.com/Lightning-AI/lightning/issues/18493
        if weights_only:
            return {
                "network": self.module.network,
                "current_step": self.current_step,
                "validation_history": self.validation_history,
            }
        else:
            return {
                "network": self.module.network,
                "optimizer": self.optimizer,
                "lr_schedule": self.lr_schedule.state_dict() if for_saving else {},
                "current_step": self.current_step,
                "validation_history": self.validation_history,
            }

    @property
    def module(self) -> TrainableModule:
        return self.dictionary["network"]

    @property
    def lr_schedule(self) -> torch.optim.lr_scheduler.LRScheduler:
        return self.dictionary["lr_schedule"]

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        return self.dictionary["optimizer"]

    @property
    def validation_history(self) -> List[float]:
        return self.dictionary["validation_history"]

    @property
    def current_step(self) -> int:
        return self.dictionary["current_step"].item()

    @current_step.setter
    def current_step(self, value):
        self.dictionary["current_step"] = torch.tensor(value)

    def load(self, fabric: Fabric, ckpt_path: pathlib.Path):
        state = self.state(weights_only=False, for_saving=False)
        fabric.load(ckpt_path, state)

        self.lr_schedule.load_state_dict(state["lr_schedule"])
        self.current_step = state["current_step"]


class FabricTrainer:
    def __init__(
        self, cfg: DictConfig, train_cfg: FabricTrainerConfig, log_cfg: LoggingConfig
    ):
        # extracted train and log settings from hydra config
        # we pass the complete dictionary config for hparam logging purposes
        self.train_cfg = train_cfg
        self.log_cfg = log_cfg

        # use lightning's Fabric to enable (multi-)gpu training and mixed precision
        self.fabric = Fabric(
            accelerator=train_cfg.accelerator,
            devices=train_cfg.devices,
            strategy=train_cfg.strategy,
            precision=train_cfg.precision,
            num_nodes=train_cfg.num_nodes,
        )

        # find grad scaler when using mixed precision
        if hasattr(self.fabric.strategy._precision, "scaler"):
            self.grad_scaler: torch.cuda.amp.GradScaler = (
                self.fabric.strategy._precision.scaler
            )
        else:
            self.grad_scaler = None

        # required to make layer drop work
        if hasattr(self.fabric.strategy, "_ddp_kwargs"):
            self.fabric.strategy._ddp_kwargs = {
                "find_unused_parameters": self.train_cfg.ddp_find_unused_parameters,
                "gradient_as_bucket_view": self.train_cfg.ddp_gradient_as_bucket_view,
            }
        else:
            print(
                f"WARNING: {self.train_cfg.ddp_find_unused_parameters=} but "
                f"unable to find DDP strategy in fabric"
            )

        # for some GPUs this speeds up training
        torch.set_float32_matmul_precision("medium")

        # start one or more GPU processes
        self.fabric.launch()

        # init logging
        if self.fabric.is_global_zero:
            specified_tags = log_cfg.tags
            wandb_tags = []

            if isinstance(specified_tags, str):
                wandb_tags.append(specified_tags)

            if isinstance(specified_tags, list) or isinstance(
                specified_tags, ListConfig
            ):
                wandb_tags.extend(specified_tags)

            if cfg.use_wandb is False:
                wandb_mode = "disabled"
            else:
                wandb_mode = (
                    os.environ["WANDB_MODE"] if "WANDB_MODE" in os.environ else "online"
                )

            wandb.init(
                project=log_cfg.project_name,
                config=OmegaConf.to_object(cfg),
                tags=wandb_tags,
                id=str(cfg.run_id),
                name=str(cfg.run_id),
                resume="allow",
                mode=wandb_mode,
            )

            self.fabric.print(OmegaConf.to_yaml(cfg, resolve=True), flush=True)

        # set random seed, unique per GPU
        self.fabric.seed_everything(
            train_cfg.random_seed + self.fabric.global_rank, workers=True
        )

    def fit_profile(
        self,
        module: TrainableModule,
        optimizer: torch.optim.Optimizer,
        lr_schedule: torch.optim.lr_scheduler.LRScheduler,
        train_loader: DataLoader,
    ):
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=10, warmup=10, active=10),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                str(pathlib.Path.cwd() / "profile"),
                worker_name=f"worker_{self.fabric.global_rank}",
            ),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as p:
            # we create a different thread responsible for writing output
            writing_pool = Pool(1)
            current_step = 0

            train_iter = iter(train_loader)
            step_durations = []
            for i in range(30 * self.train_cfg.num_steps):
                current_step += 1

                train_iter, duration = train_step(
                    fabric=self.fabric,
                    pool=writing_pool,
                    model=module,
                    train_loader=train_loader,
                    train_iter=train_iter,
                    optimizer=optimizer,
                    lr_schedule=lr_schedule,
                    current_step=current_step,
                    accumulation=self.train_cfg.accumulation,
                    clip_value=self.train_cfg.clip_value,
                    clip_norm=self.train_cfg.clip_norm,
                )
                step_durations.append(duration)
                p.step()

                if i % 30 == 0 and i != 0:
                    module.log_training(
                        self.fabric, current_step, len(step_durations), step_durations
                    )
                    step_durations.clear()

            writing_pool.close()
            writing_pool.join()

    def fit(
        self,
        module: TrainableModule,
        optimizer: torch.optim.Optimizer,
        lr_schedule: torch.optim.lr_scheduler.LRScheduler,
        train_loader: DataLoader,
        val_loader: DataLoader,
        is_setup: bool = False,
    ) -> float:
        # setup fabric for model and dataloaders
        if not is_setup:
            network = self.fabric.setup_module(module.network)
            optimizer = self.fabric.setup_optimizers(optimizer)

            module.network = network

            train_loader, val_loader = self.fabric.setup_dataloaders(
                train_loader, val_loader
            )

        # we create a different thread responsible for writing output
        writing_pool = Pool(1)

        # global settings/objects for train/val loop
        max_steps = self.train_cfg.num_steps
        val_interval = self.train_cfg.val_interval

        accumulation = self.train_cfg.accumulation
        log_interval = self.log_cfg.log_interval

        # training state
        state = TrainingState(
            module, optimizer, lr_schedule, current_step=-1, validation_history=[]
        )

        # do an initial (partial) validation loop and save initial checkpoint
        if self.train_cfg.best_ckpt_op == "min":
            best_val_metric = torch.tensor(float("inf"))
        else:
            best_val_metric = torch.tensor(float("-inf"))

        if self.train_cfg.num_sanity_validation_steps > 0:
            val_metric = best_val_metric = validation_loop(
                self.fabric,
                state,
                val_loader,
                self.train_cfg.num_sanity_validation_steps,
            )
            state.validation_history.append(val_metric.item())

        if self.train_cfg.save_init_ckpt and get_init_ckpt() is None:
            save_checkpoint(self.fabric, state, "init.ckpt")

        # potentially restart
        if (last_ckpt := get_last_ckpt()) is not None:
            state.load(self.fabric, last_ckpt)

            if self.train_cfg.best_ckpt_op == "min":
                best_val_metric = torch.tensor(min(state.validation_history))
            else:
                best_val_metric = torch.tensor(max(state.validation_history))

            self.fabric.print(
                f"restarting from {last_ckpt} at step {state.current_step}", flush=True
            )
        else:
            pwd = str(pathlib.Path.cwd())
            self.fabric.print(
                f"did not locate '*.last.ckpt' in {pwd}/checkpoints, "
                f"starting from step 0",
                flush=True,
            )

        # start the train loop
        train_iter = iter(train_loader)
        step_durations = []

        while state.current_step < max_steps:
            state.current_step += 1

            # train step
            train_iter, duration = train_step(
                fabric=self.fabric,
                state=state,
                pool=writing_pool,
                train_loader=train_loader,
                train_iter=train_iter,
                accumulation=accumulation,
                clip_value=self.train_cfg.clip_value,
                clip_norm=self.train_cfg.clip_norm,
                grad_scaler=self.grad_scaler,
                store_gradients=self.train_cfg.store_gradients,
            )

            step_durations.append(duration)

            # sporadic logging of training results
            if state.current_step % log_interval == 0 and state.current_step > 0:
                # log learning rate
                if self.fabric.is_global_zero:
                    wandb.log(
                        {
                            "learning_rate": lr_schedule.get_last_lr()[0],
                            "step": state.current_step,
                        },
                        state.current_step,
                        commit=False,
                    )

                # let module log training progress
                module.log_training(
                    self.fabric, state.current_step, len(step_durations), step_durations
                )
                step_durations.clear()

                # always save the last checkpoint for restarting jobs
                if self.train_cfg.save_last_ckpt:
                    save_checkpoint(
                        self.fabric,
                        state,
                        "last.ckpt",
                        delete_existing_postfix=True,
                        print_msg=state.current_step % (10 * log_interval) == 0,
                    )

            # sporadic validation loop
            if (
                state.current_step % val_interval == 0
                or state.current_step >= max_steps
            ) and state.current_step > 0:
                val_metric = validation_loop(
                    self.fabric, state, val_loader, self.train_cfg.max_validation_steps
                )
                state.validation_history.append(val_metric.item())

                write_best_ckpt = False
                if self.train_cfg.best_ckpt_op == "min":
                    if val_metric < best_val_metric:
                        best_val_metric = val_metric
                        write_best_ckpt = True
                elif self.train_cfg.best_ckpt_op == "max":
                    if val_metric > best_val_metric:
                        best_val_metric = val_metric
                        write_best_ckpt = True

                if self.train_cfg.save_best_ckpt and write_best_ckpt:
                    save_checkpoint(
                        self.fabric,
                        state,
                        "best.ckpt",
                        delete_existing_postfix=True,
                        weights_only=True,
                    )
                if self.train_cfg.save_progress_ckpt:
                    save_checkpoint(
                        self.fabric,
                        state,
                        "progress.ckpt",
                        delete_existing_postfix=False,
                        weights_only=True,
                    )

                # decide whether to do early stopping
                if self.train_cfg.early_stopping_patience >= 0:
                    current_idx = len(state.validation_history) - 1
                    operator = min if self.train_cfg.best_ckpt_op == "min" else max
                    best_idx, _ = operator(
                        enumerate(state.validation_history), key=lambda tpl: tpl[1]
                    )
                    idx_difference = abs(current_idx - best_idx)

                    if idx_difference >= self.train_cfg.early_stopping_patience:
                        msg = (
                            "Detected early-stopping condition on the following "
                            "validation history:\n"
                        )
                        for idx, x in enumerate(state.validation_history):
                            msg += f"{idx}={x:.5f}\n"
                        msg += f"Best index was {best_idx}, current index {current_idx}"
                        msg += f" with difference {current_idx - best_idx}. "
                        msg += (
                            f"This difference is equal to "
                            f"{self.train_cfg.early_stopping_patience=}."
                        )
                        self.fabric.print(msg)
                        break
                    else:
                        if idx_difference > 0:
                            msg = f"Validation metric has not improved. "
                            msg += f"Early stopping will trigger if no improvement for "
                            msg += f"{self.train_cfg.early_stopping_patience - idx_difference + 1} "
                            msg += f"more validation loops."
                            self.fabric.print(msg)

        # save the last checkpoint
        if self.train_cfg.save_last_ckpt:
            save_checkpoint(
                self.fabric,
                state,
                "last.ckpt",
                delete_existing_postfix=True,
            )

        # log best validation metric
        if wandb.run is not None:
            wandb.run.summary["best_val_metric"] = best_val_metric

        # close all background processes
        writing_pool.close()
        writing_pool.join()

        return best_val_metric.item()


def train_step(
    fabric: Fabric,
    state: TrainingState,
    pool: Pool,
    train_loader: DataLoader,
    train_iter: Iterator,
    accumulation: int = 1,
    clip_value: Optional[float] = None,
    clip_norm: Optional[float] = None,
    grad_scaler: Optional[GradScaler] = None,
    store_gradients: bool = False,
) -> Tuple[Iterator, float]:
    # extract from state
    model = state.module
    optimizer = state.optimizer
    lr_schedule = state.lr_schedule
    current_step = state.current_step

    # keep track how long step took
    duration = time.perf_counter()

    # set to train mode at start of training step loop
    model.train()

    # zero grad
    optimizer.zero_grad(set_to_none=False)

    # step as many times as desired
    for acc_step in range(accumulation):
        # get data
        batch = next(train_iter, None)
        if batch is None:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        batch_debug_text = model.debug_train_batch(batch)
        pool.apply_async(
            write_debug_txt,
            (fabric.local_rank, current_step, acc_step, "batch", batch_debug_text),
        )

        is_last_accumulation = (acc_step + 1) == accumulation
        with fabric.no_backward_sync(model.network, enabled=not is_last_accumulation):
            # forward
            loss = model.train_step(batch, current_step, acc_step) / accumulation

            # backward
            fabric.backward(loss)

        # log step debug info
        step_debug_text = model.debug_train_step(batch)
        pool.apply_async(
            write_debug_txt,
            (fabric.local_rank, current_step, acc_step, "step", step_debug_text),
        )

    # potentially clip gradients
    if clip_norm is not None or clip_value is not None:
        fabric.clip_gradients(
            model.network, optimizer, clip_val=clip_value, max_norm=clip_norm
        )

    # or save gradients (cannot be combined with clipping as both call unscale_)
    elif store_gradients:
        grad_scaler.unscale_(optimizer)
        grads = {
            name: param.grad
            for name, param in model.network.named_parameters()
            if param.grad is not None
        }
        grad_vector = torch.concat([v.flatten() for v in grads.values()])
        has_nans = torch.sum(torch.isnan(grad_vector)).item() > 0
        has_infs = torch.sum(torch.isinf(grad_vector)).item() > 0
        print(grad_scaler.state_dict())

        if not (has_nans or has_infs):
            save_dir = pathlib.Path.cwd() / "gradients"
            save_dir.mkdir(parents=True, exist_ok=True)
            save_file = save_dir / f"gradients_step_{current_step}.pt"
            torch.save(
                grads,
                save_file,
            )
            print(f"wrote {str(save_file)}")
            current_files = [f for f in save_dir.glob("gradients_step_*.pt")]
            if len(current_files) >= 10:
                print("done training:", current_files)
                exit()

    # update weights
    optimizer.step()
    lr_schedule.step()

    duration = time.perf_counter() - duration

    return train_iter, duration


@torch.no_grad()
def validation_loop(
    fabric: Fabric,
    state: TrainingState,
    val_loader: DataLoader,
    max_steps: int = -1,
):
    model = state.module
    current_step = state.current_step

    # set to eval mode at start of validation loop
    model.eval()
    gc.collect()

    # the validation loop
    durations = []
    metric_over_epoch = []

    for i, batch in enumerate(val_loader):
        if -1 < max_steps <= i:
            break

        start = time.time()
        batch = batch.to(fabric.local_rank)
        metric = model.validation_step(batch, i)

        metric_over_epoch.append(metric.detach().cpu())
        durations.append(time.time() - start)

    # compute validation metric
    val_metric = torch.mean(torch.stack(metric_over_epoch))

    # log validation results
    model.log_validation(fabric, val_metric, current_step, len(durations), durations)

    # set model to train mode at end of validation loop
    model.train()

    return val_metric


def write_debug_txt(
    rank: int,
    step: int,
    accumulation: int,
    prefix: str,
    textual_representations: List[str],
):
    if len(textual_representations) == 0:
        return

    file = pathlib.Path.cwd() / f"{rank}_{prefix}.txt"
    with file.open("a") as f:
        batch_str = f"{rank=} | {step=} | {accumulation=}"

        for v in textual_representations:
            batch_str += " | "
            batch_str += v.strip()

        batch_str += "\n"
        f.write(batch_str)


def _find_ckpt_path(pattern: str) -> Optional[pathlib.Path]:
    ckpt_dir = pathlib.Path.cwd() / "checkpoints"

    potential_files = [f for f in ckpt_dir.glob(pattern)]

    if len(potential_files) == 0:
        return None
    elif len(potential_files) == 1:
        return potential_files[0]
    else:
        raise ValueError(f"multiple checkpoints available for pattern {pattern}")


def get_best_ckpt():
    return _find_ckpt_path("*.best.ckpt")


def get_last_ckpt():
    return _find_ckpt_path("*.last.ckpt")


def get_init_ckpt():
    return _find_ckpt_path("*.init.ckpt")


def save_checkpoint(
    fabric: Fabric,
    state: TrainingState,
    postfix: str = "best.ckpt",
    delete_existing_postfix: bool = False,
    print_msg: bool = True,
    weights_only: bool = False,
):
    if not fabric.is_global_zero:
        return

    ckpt_dir = pathlib.Path.cwd() / "checkpoints"

    # get values for new filename
    if postfix == "init.ckpt":
        current_step = 0
    else:
        current_step = state.current_step

    val_metric_value = state.validation_history[-1]

    prefix = state.module.checkpoint_prefix()

    # new filename and path
    fn = f"{prefix}.step_{current_step:012d}.loss_{val_metric_value:.2f}.{postfix}"
    new_ckpt_path = str(ckpt_dir / fn)

    # get checkpoint to save
    ckpt = state.state(for_saving=True, weights_only=weights_only)

    # first rename the files with same postfix
    # we do this so chance if minimal that program stops between
    # removing and writing new checkpoint
    deleted_files = []
    if delete_existing_postfix:
        for p in [f for f in ckpt_dir.glob("*" + postfix)]:
            p = p.rename(p.parent / (p.name + ".backup"))
            deleted_files.append(p)

    # save new file
    fabric.save(new_ckpt_path, ckpt)

    # actually remove the backups
    for f in deleted_files:
        f.unlink()

    if print_msg:
        if len(deleted_files) > 0:
            fabric.print(
                f"overwrote existing {postfix} checkpoint to {new_ckpt_path}",
                flush=True,
            )
        else:
            fabric.print(f"wrote checkpoint {new_ckpt_path}", flush=True)

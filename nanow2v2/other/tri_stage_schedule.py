"""
Implements a tri-stage learning rate schedule by defining a callable class which can
be given to torch.optim.lr_scheduler.LambdaLR.
"""

import math
import torch as t


class TriStageLearningRateLambdaLRFunction:
    @staticmethod
    def is_valid_ratio(ratio: float):
        return 0 <= ratio <= 1

    def __init__(
        self,
        max_steps: int,
        warmup_stage_ratio: float,
        constant_stage_ratio: float,
        decay_stage_ratio: float,
        initial_lr: float,
        base_lr: float,
        final_lr: float,
        warmup_rate: str = "linear",
        decay_rate: str = "linear",
    ):
        if not (
            self.is_valid_ratio(warmup_stage_ratio)
            and self.is_valid_ratio(constant_stage_ratio)
            and self.is_valid_ratio(decay_stage_ratio)
        ):
            raise ValueError()

        if (
            abs((warmup_stage_ratio + constant_stage_ratio + decay_stage_ratio) - 1)
            >= 1e-9
        ):
            raise ValueError("stage ratio's need to add up to 1")

        # stage computation
        self.max_steps = max_steps

        if self.max_steps is None:
            raise ValueError(
                "TriStage learning rate schedule requires setting `max_steps` "
                "in the trainer"
            )

        self.warmup_stage_steps = math.floor(self.max_steps * warmup_stage_ratio)
        self.constant_stage_steps = math.floor(self.max_steps * constant_stage_ratio)
        self.decay_stage_steps = math.floor(self.max_steps * decay_stage_ratio)

        self.initial_lr = initial_lr
        self.base_lr = base_lr
        self.final_lr = final_lr

        # warmup_stage
        if warmup_rate == "linear":
            self.warmup_stage_space = t.linspace(
                self.initial_lr, self.base_lr, steps=self.warmup_stage_steps
            )
        elif warmup_rate == "exp":
            self.warmup_stage_space = t.logspace(
                math.log(self.initial_lr),
                math.log(self.base_lr),
                steps=self.warmup_stage_steps,
                base=math.e,
            )
        else:
            raise ValueError(f"{warmup_rate=} should be linear or exp")

        # decay stage
        if decay_rate == "linear":
            self.decay_stage_space = t.linspace(
                self.base_lr,
                self.final_lr,
                steps=self.decay_stage_steps + 2,
            )
        elif decay_rate == "exp":
            self.decay_stage_space = t.logspace(
                math.log(self.base_lr),
                math.log(self.final_lr),
                steps=self.decay_stage_steps + 2,
                base=math.e,
            )
        else:
            raise ValueError(f"{decay_rate=} should be linear or exp")

        self.warmup_stage_space = self.warmup_stage_space.cpu().numpy().tolist()
        self.decay_stage_space = self.decay_stage_space.cpu().numpy().tolist()

    def __call__(self, step_count: int):
        if step_count < self.warmup_stage_steps:
            desired_lr = self.warmup_stage_space[step_count]
        elif step_count <= self.warmup_stage_steps + self.constant_stage_steps:
            desired_lr = self.base_lr
        elif step_count <= self.max_steps:
            desired_lr = self.decay_stage_space[
                step_count - (self.warmup_stage_steps + self.constant_stage_steps)
            ]
        else:
            desired_lr = self.final_lr

        factor = desired_lr / self.base_lr
        return factor

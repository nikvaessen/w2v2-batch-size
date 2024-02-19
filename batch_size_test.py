import click
import time
import lightning

import torch
import torchdata

from torch.utils.data import IterableDataset, DataLoader

from nanow2v2.model.wav2vec2_ssl import Wav2vec2ForSSL, Wav2vec2Config, ForSSLConfig


class Dataset(IterableDataset):
    def __init__(
        self,
        iterations: int,
        max_tokens: int,
        min_sample_length: int,
        max_sample_length: int,
        max_length_diff_per_sample: int,
    ):
        self.iterations = iterations

        self.max_tokens = max_tokens
        self.min_sample_length = min_sample_length
        self.max_sample_length = max_sample_length
        self.max_length_diff_per_sample = max_length_diff_per_sample

    def __iter__(self):
        for i in range(self.iterations):
            # determine random batch size
            seq_length = torch.randint(
                low=self.min_sample_length + self.max_length_diff_per_sample,
                high=self.max_sample_length,
                size=(),
            ).item()
            bs = self.max_tokens // seq_length

            sample_lengths = torch.ones((bs,), dtype=torch.int) * seq_length

            if len(sample_lengths) > 1:
                sample_lengths_diff = torch.randint(
                    low=0, high=self.max_length_diff_per_sample - 1, size=(bs - 1,)
                )
                sample_lengths[1:] -= sample_lengths_diff

            sample_lengths = sample_lengths.tolist()
            batch = gen_batch(bs, seq_length)

            yield batch, sample_lengths

    def __len__(self):
        return self.iterations


def gen_batch(bs: int, seq_length: int):
    return torch.rand((bs, seq_length), device="cpu")


def warm_cuda_cache(model, opt, max_tokens, max_sample_length, fabric):
    batch = gen_batch(max_tokens // max_sample_length, max_sample_length)
    batch = batch.to(fabric.device)
    sample_length = [max_sample_length for _ in range(batch.shape[0])]
    fabric.print(f"warming up with batch {batch.shape}")

    opt.zero_grad(set_to_none=True)
    loss = model.forward(batch, sample_length).loss
    fabric.backward(loss)


def increase_counter(prev, count_list):
    now = time.perf_counter()
    count_list.append(now - prev)
    return now


@click.command()
@click.option("--tk", "max_tokens", type=int, default=1_500_000)
@click.option("--min-ln", "min_sample_length", type=int, default=48_000)
@click.option("--max-ln", "max_sample_length", type=int, default=480_000)
@click.option("--diff", "max_length_diff_per_sample", type=int, default=16_000)
@click.option("--it", "iterations", type=int, default=1000)
@click.option("--lr", "learning_rate", type=float, default=1e-6)
@click.option("--seed", type=int, default=123)
@click.option("--warm-min", is_flag=True, default=False)
@click.option("--warm-max", is_flag=True, default=False)
@click.option("--min-first", is_flag=True, default=False)
@click.option("--precision", type=str, default="32-true")
@click.option("--matmul-precision", type=str, default="medium")
@click.option("--empty-cache", is_flag=True, default=False)
@click.option("--n", "devices", type=int, default=1)
def main(
    max_tokens: int,
    min_sample_length: int,
    max_sample_length: int,
    max_length_diff_per_sample: int,
    iterations: int,
    learning_rate: float,
    seed: int,
    warm_min: bool,
    warm_max: bool,
    min_first: bool,
    precision: str,
    matmul_precision: str,
    empty_cache: bool,
    devices: int,
):
    torch.set_float32_matmul_precision(matmul_precision)

    fabric = lightning.fabric.Fabric(precision=precision, devices=devices)

    if hasattr(fabric.strategy, "_ddp_kwargs"):
        fabric.strategy._ddp_kwargs = {
            "gradient_as_bucket_view": True,
        }

    fabric.launch()

    fabric.print(
        f"benchmarking with {max_tokens=}, {min_sample_length=}, {max_sample_length=} "
        f"{max_length_diff_per_sample=}"
    )

    model = Wav2vec2ForSSL(Wav2vec2Config(), ForSSLConfig())
    model = fabric.setup_module(model)

    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    opt = fabric.setup_optimizers(opt)

    ds = Dataset(
        iterations,
        max_tokens,
        min_sample_length,
        max_sample_length,
        max_length_diff_per_sample,
    )

    loader = DataLoader(ds, batch_size=None, num_workers=1)
    loader = fabric.setup_dataloaders(loader)

    duration_counts = []
    counter = time.perf_counter()

    if min_first:
        if warm_min:
            warm_cuda_cache(model, opt, max_tokens, min_sample_length, fabric)
        counter = increase_counter(counter, duration_counts)
        if warm_max:
            warm_cuda_cache(model, opt, max_tokens, max_sample_length, fabric)
            counter = increase_counter(counter, duration_counts)
    else:
        if warm_max:
            warm_cuda_cache(model, opt, max_tokens, max_sample_length, fabric)
            counter = increase_counter(counter, duration_counts)

        if warm_min:
            warm_cuda_cache(model, opt, max_tokens, min_sample_length, fabric)
            counter = increase_counter(counter, duration_counts)

    torch.manual_seed(seed)
    batch_sizes = []
    counter = increase_counter(counter, duration_counts)

    for i, (batch, sample_lengths) in enumerate(loader):
        opt.zero_grad(set_to_none=True)

        try:
            loss = model.forward(batch, sample_lengths).loss
            fabric.backward(loss)
            opt.step()
        except torch.cuda.OutOfMemoryError as e:
            print(f"{fabric.global_rank=}", torch.cuda.memory_summary())
            stats = torch.cuda.memory_stats()
            for k, v in stats.items():
                print(f"{fabric.global_rank=}", k, v)
            raise e

        batch_sizes.append(batch.shape[0])
        counter = increase_counter(counter, duration_counts)

        if i % 10 == 0:
            total_duration = sum(duration_counts)

            last_durations = duration_counts[-10:]
            it_sec = len(last_durations) / sum(last_durations)
            loss = loss.item()

            fabric.print(
                f"{i=: >5} {loss=: >7.2f} {total_duration=: >6.2f}s "
                f"it/s={it_sec:.2f} {batch_sizes=}"
            )
            batch_sizes.clear()

        if empty_cache:
            torch.cuda.empty_cache()

    fabric.print(f"it/s={len(duration_counts)/sum(duration_counts):.2f}")


if __name__ == "__main__":
    main()

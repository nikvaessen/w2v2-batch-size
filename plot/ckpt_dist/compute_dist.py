import gc
import pathlib
from functools import lru_cache

import click
import polars
import torch
import tqdm


def get_iteration(ckpt_file: pathlib.Path):
    return int(ckpt_file.name.split(".")[2].split("_")[1])


@lru_cache(maxsize=5)
def get_parameter_vector(ckpt_file: pathlib.Path):
    ckpt = torch.load(ckpt_file, map_location="cpu")

    parameter_tensors = [(k, v.flatten()) for k, v in ckpt["network"].items()]
    parameter_tensors = sorted(parameter_tensors, key=lambda tpl: tpl[0])

    return torch.cat([v for k, v in parameter_tensors])


@torch.no_grad()
def compute_distances(ckpt1: pathlib.Path, ckpt2: pathlib.Path, use_cuda: bool = False):
    theta1 = get_parameter_vector(ckpt1)
    theta2 = get_parameter_vector(ckpt2)

    if use_cuda:
        theta1 = theta1.to("cuda")
        theta2 = theta2.to("cuda")

    # euclidian distance
    euclid_dist = torch.cdist(theta1[None, :], theta2[None, :]).squeeze()

    # cosine distance
    cosine_similarity = torch.cosine_similarity(theta1, theta2, dim=0)

    # garbage collection to prevent large memory
    # usage in parallelization
    del theta1, theta2
    gc.collect()

    return euclid_dist, cosine_similarity


def iteration(first, second, use_gpu: bool = False):
    it0, ckpt0 = first
    it1, ckpt1 = second
    euc_dist, cos_sim = compute_distances(ckpt0, ckpt1, use_cuda=use_gpu)

    return {
        "it0": it0,
        "it1": it1,
        "euclidian_distance": euc_dist,
        "cosine_similarity": cos_sim,
    }


@click.command
@click.argument("ckpt_dir", type=pathlib.Path)
@click.argument("result_csv_file", type=pathlib.Path)
@click.option("-w", "--workers", type=int, default=None)
@click.option("--gpu", is_flag=True, default=False)
def main(
    ckpt_dir: pathlib.Path,
    result_csv_file: pathlib.Path,
    workers: int = None,
    gpu: bool = False,
):
    # find all ckpt files
    ckpt_files = sorted(
        [(get_iteration(f), f) for f in ckpt_dir.glob("*.progress.ckpt")],
        key=lambda t: t[0],
    )

    if len(ckpt_files) == 0:
        raise ValueError(f"{str(ckpt_files)} did not contain any progress checkpoints")

    # also find the initial checkpoint
    ckpt_first = [f for f in ckpt_dir.glob("*.init.ckpt")]
    assert len(ckpt_first) == 1

    # store all checkpoints in a list
    ckpt_files = [(0, ckpt_first[0])] + ckpt_files

    # compare t to t+5k
    comparisons_to_next = list(zip(ckpt_files[:-1], ckpt_files[1:]))

    # compare t to 0
    comparisons_to_zero = list(
        zip(ckpt_files[1:], [ckpt_files[0]] * (len(ckpt_files) - 1))
    )

    # chain comparisons
    all_comparisons = comparisons_to_zero + comparisons_to_next

    # use joblib to compute euclid distance in parallel
    print("making comparison")
    results = list(
        tqdm.tqdm(
            (iteration(first, second, gpu) for first, second in all_comparisons),
            total=len(all_comparisons),
        )
    )

    # save results in csv file
    polars.DataFrame(results).write_csv(result_csv_file)


if __name__ == "__main__":
    main()

import gc
import pathlib
import joblib

import click
import polars
import torch
import tqdm


def get_iteration(ckpt_file: pathlib.Path):
    return int(ckpt_file.name.split(".")[2].split("_")[1])


def get_parameter_vector(ckpt_file: pathlib.Path, use_cuda: bool = False):
    ckpt = torch.load(ckpt_file, map_location="cuda" if use_cuda else "cpu")

    parameter_tensors = [(k, v.flatten()) for k, v in ckpt["network"].items()]
    parameter_tensors = sorted(parameter_tensors, key=lambda tpl: tpl[0])

    return torch.cat([v for k, v in parameter_tensors])


@torch.no_grad()
def compute_distances(ckpt1: pathlib.Path, ckpt2: pathlib.Path, use_cuda: bool = False):
    theta1 = get_parameter_vector(ckpt1, use_cuda)
    theta2 = get_parameter_vector(ckpt2, use_cuda)

    # euclidian distance
    diff = theta2 - theta1
    power = torch.pow(diff, 2)
    summation = torch.sum(power)
    euclid_dist = torch.sqrt(summation).item()

    # cosine distance
    cosine_similarity = torch.cosine_similarity(theta1, theta2, dim=0)

    # garbage collection to prevent large memory
    # usage in parallelization
    del theta1, theta2
    gc.collect()

    return euclid_dist, cosine_similarity


def iteration(first, second):
    it0, ckpt0 = first
    it1, ckpt1 = second
    euc_dist, cos_sim = compute_distances(ckpt0, ckpt1, use_cuda=False)

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
def main(ckpt_dir: pathlib.Path, result_csv_file: pathlib.Path, workers: int = None):
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
    results = list(
        tqdm.tqdm(
            (
                joblib.Parallel(n_jobs=workers, timeout=99999, return_as="generator")(
                    joblib.delayed(iteration)(first, second)
                    for first, second in all_comparisons
                )
            ),
            total=len(all_comparisons),
        )
    )

    # save results in csv file
    polars.DataFrame(results).write_csv(result_csv_file)


if __name__ == "__main__":
    main()

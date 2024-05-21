#! /usr/bin/env python3
########################################################################################
#
# Script which transformers a CSV file describing a dataset to multiple tar files,
# which each contain only a (small) subset of the dataset. Each tar file is called a
# shard.
#
# Author(s): anon
########################################################################################

import io
import json
import math
import multiprocessing
import pathlib
import subprocess
import tarfile
import time

from dataclasses import dataclass
from typing import Dict, List, Union, Optional

import click

import pandas as pd
import numpy as np

########################################################################################
# dataclass of a sample


@dataclass()
class ShardCsvSample:
    # identifier of sample
    key: str

    # path to audio file
    path: str

    # stats of audio file
    num_frames: int
    sample_rate: int

    # labels
    speaker_id: Optional[str] = None
    recording_id: Optional[str] = None
    gender: Optional[str] = None
    language_tag: Optional[str] = None
    transcription: Optional[str] = None

    @classmethod
    def to_dataframe(cls, samples: List["ShardCsvSample"]):
        df = pd.DataFrame([s.__dict__ for s in samples])

        return df

    @classmethod
    def to_csv(
        cls, path: pathlib.Path, samples: Union[List["ShardCsvSample"], pd.DataFrame]
    ):
        if isinstance(samples, list):
            df = cls.to_dataframe(samples)
        elif isinstance(samples, pd.DataFrame):
            df = samples
        else:
            raise ValueError(f"invalid type {type(samples)=}")

        df = df.sort_values("key", ascending=True, ignore_index=True)
        df.to_csv(str(path), index=False, sep="\t", quotechar='"', escapechar="\\")

    @classmethod
    def from_csv(cls, path: pathlib.Path) -> pd.DataFrame:
        df = pd.read_csv(str(path), sep="\t", quotechar='"', escapechar="\\")

        return df


########################################################################################
# implementation of strategies determining distribution of samples over shards


def apply_strategy(
    strategy: str, df: pd.DataFrame, seed: int, samples_per_shard: int
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    if strategy == "random":
        distribution_df = strategy_random(df, samples_per_shard, rng)
    elif strategy == "length_sorted":
        distribution_df = strategy_length_sorted(df, samples_per_shard)
    elif strategy == "single_shard":
        distribution_df = strategy_single_shard(df)
    else:
        raise ValueError(f"unknown {strategy=}")

    # add extra column 'partial_shard' to detect a shard_idx which does not contain
    # the maximum amount of samples in all shard_indexes
    mark_partial(distribution_df)

    return distribution_df


def mark_partial(dist_df: pd.DataFrame):
    shard_idx = dist_df["shard_idx"]
    count_df = pd.value_counts(shard_idx)
    max_samples = np.max(count_df)

    partial_idx = count_df.loc[count_df < max_samples].index.to_list()

    dist_df.loc[dist_df["shard_idx"].isin(partial_idx), "partial_shard"] = True
    dist_df.loc[~dist_df["shard_idx"].isin(partial_idx), "partial_shard"] = False


def strategy_random(df: pd.DataFrame, samples_per_shard: int, rng: np.random.Generator):
    key = np.array(df["key"])
    rng.shuffle(key)

    num_samples = key.shape[0]
    shard_idx = [i // samples_per_shard for i in range(num_samples)]

    shard_dist_df = pd.DataFrame({"key": key, "shard_idx": shard_idx})

    return shard_dist_df


def strategy_length_sorted(df, samples_per_shard):
    df_sorted = df.sort_values(by="num_frames")

    key = np.array(df_sorted["key"])

    num_samples = key.shape[0]
    shard_idx = [i // samples_per_shard for i in range(num_samples)]

    shard_dist_df = pd.DataFrame({"key": key, "shard_idx": shard_idx})

    return shard_dist_df


def strategy_single_shard(df: pd.DataFrame):
    keys = np.array(df["key"])
    num_samples = keys.shape[0]

    shard_idx = [0 for _ in range(num_samples)]

    shard_dist_df = pd.DataFrame({"key": keys, "shard_idx": shard_idx})

    return shard_dist_df


########################################################################################
# method to write shards


def dict_to_json_bytes(d: Dict):
    return json.dumps(d).encode("utf-8")


def filter_tarinfo(ti: tarfile.TarInfo):
    ti.uname = "research"
    ti.gname = "data"
    ti.mode = 0o0444  # everyone can read
    ti.mtime = 1672527600  # 2023-01-01 00:00:00

    return ti


def _wrap_to_none(x):
    if isinstance(x, float) and math.isnan(x):
        return None
    else:
        return x


def write_shard(tar_file_path: pathlib.Path, data: pd.DataFrame, compress: bool):
    num_digits = len(str(len(data)))
    print(f"started writing {tar_file_path}")
    with tarfile.TarFile(str(tar_file_path), mode="w") as archive:
        for idx, row in [t for t in enumerate(data.itertuples())]:
            # key identifying audio and json pair
            key = f"{idx:0>{num_digits}}/{row.key}"

            # add audio file
            audio_path = row.path
            archive.add(str(audio_path), arcname=f"{key}.wav", filter=filter_tarinfo)

            # add json file with labels
            json_dict = {
                "sample_id": row.key,
                "num_frames": row.num_frames,
                "sample_rate": row.sample_rate,
                "gender": _wrap_to_none(row.gender),
                "transcription": _wrap_to_none(row.transcription),
                "speaker_id": _wrap_to_none(row.speaker_id),
                "recording_id": _wrap_to_none(row.recording_id),
                "language_tag": _wrap_to_none(row.language_tag),
            }
            json_obj_str = dict_to_json_bytes(json_dict)

            json_tarinfo = tarfile.TarInfo(f"{key}.json")
            json_tarinfo.size = len(json_obj_str)

            archive.addfile(
                tarinfo=filter_tarinfo(json_tarinfo), fileobj=io.BytesIO(json_obj_str)
            )

    if compress:
        subprocess.call(
            ["pigz", tar_file_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def write_shards(
    data_df: pd.DataFrame,
    shard_dist_df: pd.DataFrame,
    out_folder: pathlib.Path,
    compress: bool,
    workers: int,
    shard_name_pattern: str = "{:}-{:06d}.tar",
):
    join_df = pd.merge(shard_dist_df, data_df, left_on="key", right_on="key")
    grouped_by_shard_idx = join_df.groupby(by="shard_idx", dropna=True)

    num_shards = len(grouped_by_shard_idx)

    if len([d for d in out_folder.rglob("*.tar*")]):
        raise ValueError(f"{out_folder} already contains tar files")

    with multiprocessing.Pool(processes=workers) as p:
        start_time = time.time()
        print(
            f"writing {len(grouped_by_shard_idx)} shard{'s' if num_shards > 1 else ''}",
        )
        for value, df in grouped_by_shard_idx:
            shard_path = str(out_folder / shard_name_pattern.format(value))

            is_partial = df["partial_shard"].unique().item()

            if is_partial:
                shard_path = shard_path.replace(".tar", ".partial.tar")

            def cb_success(_):
                pass

            def cb_error(e):
                print(e)

                if isinstance(e, subprocess.CalledProcessError):
                    print("stdout:\n", e.stdout.decode("utf-8"))

            p.apply_async(
                write_shard,
                args=(shard_path, df, compress),
                callback=cb_success,
                error_callback=cb_error,
            )

        p.close()
        p.join()

        end_time = time.time()
        print(f"writing shards took {end_time-start_time:.2f} seconds")


########################################################################################
# entrypoint of script


@click.command()
@click.option(
    "--csv",
    type=pathlib.Path,
    required=True,
    help="path to csv file which described the dataset to write shards for",
)
@click.option(
    "--out",
    type=pathlib.Path,
    required=True,
    help="path to root folder where the tar-based shards will be written",
)
@click.option(
    "--strategy",
    type=click.Choice(["random", "length_sorted", "single_shard"]),
    help="strategy to determine how samples are distributed across shards",
    required=True,
)
@click.option(
    "--prefix",
    type=str,
    required=True,
    help="prefix in file-name of shards",
)
@click.option(
    "--compress",
    type=bool,
    default=False,
    help="whether to compress the tar files",
)
@click.option(
    "--samples_per_shard",
    type=int,
    default=5000,
    help="maximum number of samples per shard. Ignored when strategy=single_shard",
)
@click.option(
    "--seed",
    type=int,
    default=1337,
    help="the random seed used within the script",
)
@click.option(
    "--workers",
    type=int,
    default=1,
    help="the amount of processes used to write shards to disk",
)
def main(
    csv: pathlib.Path,
    out: pathlib.Path,
    strategy: str,
    prefix: str,
    compress: bool,
    samples_per_shard: int,
    seed: int,
    workers: int,
):
    print(f"writing shards for {csv} in folder {out}")

    # read and validate csv file
    df = ShardCsvSample.from_csv(csv)

    # distribute samples over shards
    shard_distribution = apply_strategy(
        strategy,
        df,
        seed,
        samples_per_shard,
    )

    # make sure output folder exists
    out.mkdir(parents=True, exist_ok=True)

    # write the shard distribution to file
    shard_distribution.to_csv(str(out / "_shard_distribution.csv"), index=False)

    # write shards
    write_shards(
        df, shard_distribution, out, compress, workers, f"{prefix}" + ".{:06d}.tar"
    )


if __name__ == "__main__":
    main()

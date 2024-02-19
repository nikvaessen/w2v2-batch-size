#! /usr/bin/env python3
########################################################################################
#
# Script to print content of a checkpoint to standard out.
#
# Author(s): Nik Vaessen
########################################################################################

import pathlib

from typing import Dict, Any

import click
import torch


def print_ckpt(ckpt: Dict[str, Any], is_root: bool = False):
    if len(ckpt) is None:
        print("\tempty")

    for k, v in ckpt.items():
        if isinstance(v, dict) and all(isinstance(e, torch.Tensor) for e in v.values()):
            print(k)
            print_ckpt(v)
        elif isinstance(v, list):
            print(k)
            for e in v:
                print("\t", e)
        elif isinstance(v, torch.Tensor):
            print("\t", k, v.shape)
        else:
            print(k)
            print("\t", v)


@click.command()
@click.argument("ckpt", type=pathlib.Path)
def main(ckpt: pathlib.Path):
    ckpt = torch.load(ckpt)
    print("keys:")
    for k in ckpt.keys():
        print(k, type(ckpt[k]))

    # print(ckpt["cfg"])
    # for k, v in ckpt["cfg"].items():
    #     print(k, type(v))
    # exit()

    print("\ncontent:")
    print_ckpt(ckpt)


if __name__ == "__main__":
    main()

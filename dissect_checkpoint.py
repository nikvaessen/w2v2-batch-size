import pathlib
from typing import Dict

import click
import torch

torch.set_printoptions(threshold=0)


def print_dictionary(dic, level=0):
    prefix = "".join(["\t" for _ in range(level)])
    for key, value in dic.items():
        if isinstance(value, dict):
            print(f"{prefix}{key} - {type(value)} - {len(value)} elements")
            print_dictionary(value, level=level + 1)

        elif isinstance(value, torch.Tensor):
            if torch.numel(value) < 3:
                print(f"{prefix}{key} {value.shape} {value}")
            else:
                print(f"{prefix}{key} {value.shape}")

        elif isinstance(value, list):
            print(f"{prefix}{key} - {type(value)} - {len(value)} elements")
            print_list(value, level=level + 1)

        else:
            print(prefix + key, value)


def print_list(lst, level=0):
    prefix = "".join(["\t" for _ in range(level)])

    for idx, e in enumerate(lst):
        if isinstance(e, list):
            print(f"{prefix}{idx=} {type(e)} - {len(e)} elements")
            print_list(e, level=level + 1)
        elif isinstance(e, dict):
            print(f"{prefix}{idx=} {type(e)} - {len(e)} elements")
            print_dictionary(e, level=level + 1)
        else:
            print(f"{prefix}{idx=}", e)


def dissect_checkpoint(ckpt: Dict):
    root_keys = [k for k in ckpt.keys()]
    print("root keys:", root_keys)

    for k in root_keys:
        v = ckpt[k]

        if isinstance(v, dict):
            print(f"\n{k} - {type(v)} - {len(v)} elements")
            print_dictionary(v, level=1)
        elif isinstance(v, list):
            print(f"\n{k} - {type(v)} - {len(v)} elements")
            print_list(v, level=1)
        else:
            print(f"\n{k} - {type(v)}")
            print(f"\t{v}")


@click.command()
@click.argument("file", type=pathlib.Path)
def main(file: pathlib.Path):
    print(f"dissecting {str(file)}")
    ckpt = torch.load(file)
    dissect_checkpoint(ckpt)


if __name__ == "__main__":
    main()

import pathlib

from collections import defaultdict

import torch
import click
import tqdm


@click.command()
@click.argument("folder", type=pathlib.Path)
def main(folder: pathlib.Path):
    log_files = [f for f in folder.glob("*_batch.txt")]

    key_count = defaultdict(lambda: 0)
    key_length = {}

    for file in log_files:
        print(f"processing {str(file)}")
        with file.open("r") as f:
            lines = [ln for ln in f.readlines()]

            for ln in tqdm.tqdm(lines):
                raw_keys = ln.split("|")[7]
                keys = eval(raw_keys.split("=")[1].strip())
                assert isinstance(keys, list) and len(keys) >= 1

                raw_lengths = ln.split("|")[6]
                lengths = eval(raw_lengths.split("=")[1].strip())

                for idx, k in enumerate(keys):
                    assert isinstance(k, str)
                    k_split = k.split("/")
                    assert len(k_split) == 4
                    assert k_split[0] == "ls"
                    try:
                        int(k_split[1])
                        int(k_split[2])
                        int(k_split[3])
                        key_count[k] += 1
                        key_length[k] = lengths[idx]
                    except:
                        raise ValueError(f"invalid key {k=}")

    items = [i for i in key_count.items()]
    items = sorted(items, key=lambda tpl: tpl[1])

    counts = torch.tensor([v for k, v in items])
    lengths = torch.tensor([v for v in key_length.values()])

    print(f"min: {torch.min(counts).item()}")
    print(f"max: {torch.max(counts).item()}")
    print(f"mean: {torch.mean(counts.to(torch.float32)).item()}")

    print("top 10 least used keys:")
    for k, v in items[:10]:
        length = key_length[k]
        print(
            f"sample {k} was used {v} times, and had a length of {length/16_000:.2f} secs"
        )

    print("top 10 most used keys")
    for k, v in items[-10:]:
        length = key_length[k]
        print(
            f"sample {k} was used {v} times, and had a length of {length/16_000:.2f} secs"
        )

    print(f"maximum length: {torch.max(lengths)/16_000:.2f}s")
    print(f"maximum length: {torch.min(lengths)/16_000:.2f}s")
    print(f"mean length: {torch.mean(lengths.to(torch.float32))/16_000:.2f}s")


if __name__ == "__main__":
    main()

import pathlib

import click
import polars

MAPPING = {
    # (93.75 sec)
    "1940035_0": "87.5 sec",
    # actually correct
    "2586508_1": "87.5 sec",
    "1940028_1": "20 min",
    "2536305_1": "5 min",
    "2536333_1": "150 sec",
    "4520555_1": "10 min",
    "4395274": "80 min",
    "1873380_2": "40 min",
}


def sort_legend_value(x: str):
    value, unit = x.split(" ")
    value = float(value)

    if unit not in ["min", "sec"]:
        raise ValueError(f"unknown unit {unit}")

    if unit == "min":
        value *= 60

    return value


@click.command()
@click.option("--in", "in_path", type=pathlib.Path)
@click.option("--out", "out_path", type=pathlib.Path)
def main(in_path: pathlib.Path, out_path: pathlib.Path):
    df = polars.read_csv(in_path)

    step_list = []
    gpu_list = []
    value_list = []

    for row in df.iter_rows(named=True):
        step = row["Step"]
        for k, v in row.items():
            if "Step" == k:
                continue
            if "__MAX" in k or "__MIN" in k:
                continue

            name = k.split(" ")[0]
            if name not in MAPPING:
                raise ValueError(f"unknown {name=}")

            step_list.append(step)
            gpu_list.append(MAPPING[name])
            value_list.append(v)

    df = polars.DataFrame(
        {
            "step": step_list,
            "batch size": gpu_list,
            "value": value_list,
        }
    )

    # sort in correct order
    groups = []

    for name, df in df.group_by("batch size"):
        groups.append((name, df))

    groups = sorted(groups, key=lambda tupl: sort_legend_value(tupl[0]))

    sorted_df = polars.concat([tupl[1] for tupl in groups])

    sorted_df.write_csv(out_path)


if __name__ == "__main__":
    main()

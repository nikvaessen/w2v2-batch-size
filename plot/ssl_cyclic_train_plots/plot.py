import pathlib

import click
import polars
import seaborn

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from scipy.ndimage import gaussian_filter1d


# Define a custom tick formatter function
def format_ticks(x, pos):
    if x >= 1000:
        # Format as 'k' for thousands
        return f"{x/1000:.0f}k"
    else:
        return str(int(x))


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
@click.option("-x", type=str, help="label of x-axis")
@click.option("-y", type=str, help="label of y-axis")
@click.option("-t", type=str, help="title of plot")
def main(in_path: pathlib.Path, out_path: pathlib.Path, x: str, y: str, t: str):
    seaborn.set_style("whitegrid")

    # read dataframe
    df = polars.read_csv(in_path)

    # set size of plot
    plt.figure(figsize=(12, 6))  # Adjust the width (12) and height (6) as needed

    # set theme
    seaborn.set(style="darkgrid", palette="muted", color_codes=True)

    # make sure legend is ordered
    legend_values = df["batch size"].unique().to_list()
    legend_values = sorted(legend_values, key=sort_legend_value, reverse=True)

    # smooth curve
    new_df = []
    for name, g in df.group_by("batch size"):
        g = g.with_columns(
            g.select(
                polars.col("value").map_batches(
                    lambda s: polars.Series(values=gaussian_filter1d(s, 10))
                )
            )
        )
        new_df.append(g)

    df = polars.concat(new_df)

    # make plot
    seaborn.lineplot(df, x="step", y="value", hue="batch size", hue_order=legend_values)

    # display x-axis with 'k' if big values
    ax = plt.gca()
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_ticks))

    # position legend outside of plot
    legend = plt.legend(title="batch size")
    legend.set_bbox_to_anchor((1, 1))

    # put labels on axis
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(t)

    # save figure
    plt.tight_layout()
    plt.savefig(out_path)


if __name__ == "__main__":
    main()

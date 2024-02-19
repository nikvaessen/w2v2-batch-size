import pathlib

import polars
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches

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


def main():
    # set theme
    sns.set(style="darkgrid", palette="muted", color_codes=True)

    # make subplots
    fig, axes = plt.subplots(3, 3, figsize=(12, 8))

    # find files
    directory = pathlib.Path("csv")
    csv_files = [
        directory / "train_loss.csv",
        directory / "train_acc.csv",
        directory / "train_perplexity_cb1.csv",
        directory / "train_loss_contrastive.csv",
        directory / "train_loss_diversity.csv",
        directory / "train_loss_l2.csv",
        directory / "cb0_sim_avg.csv",
        directory / "cb0_sim_min.csv",
        directory / "cb0_sim_max.csv",
    ]

    labels = {
        "train_loss.csv": ("Total loss", "step", "train loss"),
        "train_acc.csv": ("Accuracy", "step", "train accuracy"),
        "train_perplexity_cb1.csv": (
            "Perplexity of codebook 1",
            "step",
            "train perplexity",
        ),
        "train_loss_contrastive.csv": (
            "Contrastive loss",
            "step",
            "train loss",
        ),
        "train_loss_diversity.csv": ("Diversity loss", "step", "train loss"),
        "train_loss_l2.csv": ("L2 loss", "step", "train loss"),
        "cb0_sim_avg.csv": (
            "Average similarity between codewords",
            "step",
            "train similarity",
        ),
        "cb0_sim_min.csv": (
            "Minimum similarity between codewords",
            "step",
            "train similarity",
        ),
        "cb0_sim_max.csv": (
            "Maximum similarity between codewords",
            "step",
            "train similarity",
        ),
    }
    # create all plots
    row_idx = 0
    col_idx = -1

    patches = []

    for csv in csv_files:
        # read dataframe
        df = polars.read_csv(csv)
        col_idx += 1

        if col_idx >= 3:
            row_idx += 1
            col_idx = 0

        should_plot_legend = row_idx == 0 and col_idx == 0

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

        # make sure legend is ordered
        legend_values = df["batch size"].unique().to_list()
        legend_values = sorted(legend_values, key=sort_legend_value, reverse=True)

        # make plot
        ax = sns.lineplot(
            df,
            x="step",
            y="value",
            hue="batch size",
            hue_order=legend_values,
            ax=axes[row_idx, col_idx],
            legend=True,
        )

        if len(patches) == 0:
            print(patches)
            patches.extend(ax.legend().get_patches())
            print(patches)

        ax.legend().remove()

        # display x-axis with 'k' if big values
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_ticks))

        # set title, x-label, y-label
        ax.set_title(labels[csv.name][0])
        ax.set_xlabel(labels[csv.name][1])
        ax.set_ylabel(labels[csv.name][2])


    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

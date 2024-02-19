import polars

from plotnine import ggplot, aes, geom_line, geom_smooth
import plotnine


def sort_legend_value(x: str):
    value, unit = x.split(" ")
    value = float(value)

    if unit not in ["min", "sec"]:
        raise ValueError(f"unknown unit {unit}")

    if unit == "min":
        value *= 60

    return value


def main():
    df = polars.read_csv("csv/cb0_sim_avg.csv")
    print(df)

    legend_values = df["batch size"].unique().to_list()
    legend_values = sorted(legend_values, key=sort_legend_value, reverse=True)
    print(legend_values)

    p = (
        ggplot(df, aes(x="step", y="value", color="batch size", order=legend_values))
        + geom_line()
    )

    p.draw(show=True)
    pass


if __name__ == "__main__":
    main()

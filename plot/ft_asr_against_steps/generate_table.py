import polars as pl


def human_format(num):
    num = float("{:.3g}".format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return "{} {}".format(
        "{:f}".format(num).rstrip("0").rstrip("."), ["", "k", "M", "B", "T"][magnitude]
    )


def transform_df(df):
    all_groups = []
    for hours_seen, df_group in sorted(df.group_by(["hours seen"]), key=lambda t: t[0]):
        if len(df_group) <= 1:
            continue

        df_group = df_group.with_columns(
            [pl.col("hours seen").map_elements(lambda x: human_format(x))]
        )
        df_group = df_group.with_columns(
            [pl.col("iteration").map_elements(lambda x: human_format(x))]
        )
        df_group = df_group.with_columns(
            [pl.col("wer clean").map_elements(lambda x: f"{round(x * 100, 2):.02f}")]
        )
        df_group = df_group.with_columns(
            [pl.col("wer other").map_elements(lambda x: f"{round(x * 100, 2):.02f}")]
        )
        df_group = df_group.drop(["batch size in sec", "num_gpus", "ft_dataset"])
        df_group = df_group.select(
            ["hours seen", "batch size label", "iteration", "wer clean", "wer other"]
        )

        all_groups.append(df_group)

    df = pl.concat(all_groups)
    return df


def main():
    df = pl.read_csv("merged.csv")

    df_ft_10min = df.filter(pl.col("ft_dataset") == "10 minutes of labels")
    df_ft_100h = df.filter(pl.col("ft_dataset") == "100 hours of labels")

    df_ft_10min = transform_df(df_ft_10min)
    df_ft_100h = transform_df(df_ft_100h)

    df = df_ft_10min.join(
        df_ft_100h,
        on=["hours seen", "batch size label", "iteration"],
        how="outer_coalesce",
    )
    df.write_csv("table.csv")
    print(df)


if __name__ == "__main__":
    main()

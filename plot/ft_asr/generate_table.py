import polars as pl


def bs_to_idx(bs: str):
    if bs == "scratch":
        return -1
    elif "gpu" in bs:
        return int(bs.removesuffix("gpu"))
    else:
        raise ValueError(f"cannot parse {bs=}")


def ds_to_idx(ds: str):
    if ds == "ls10m":
        return 0
    else:
        return int(ds.removeprefix("ls").removesuffix("h"))


def create_df(df, eval: str, lm: bool, value_name: str):
    df = df.filter(pl.col("eval_dataset") == eval)
    df = df.filter(pl.col("lm") == lm)

    df = df.with_columns(
        [pl.col("value").map_elements(lambda x: f"{round(x * 100, 2):.02f}")]
    )

    df = df.rename({"value": value_name})
    df = df.drop(["eval_dataset", "lm"])

    all_df = []
    for key, df_group in df.group_by(["ft_dataset"]):
        df_group = df_group.sort(pl.col("batch size").map_elements(bs_to_idx))

        all_df.append((key[0], df_group))

    all_df = sorted(all_df, key=lambda t: ds_to_idx(t[0]))

    return pl.concat([t[1] for t in all_df])


def main():
    df = pl.read_csv("ft_asr_lm.csv")

    df_clean_no_lm = create_df(df, "test-clean", False, "clean-no-lm")
    df_other_no_lm = create_df(df, "test-other", False, "other-no-lm")

    df_clean_lm = create_df(df, "test-clean", True, "clean-lm")
    df_other_lm = create_df(df, "test-other", True, "other-lm")

    df_no_lm = df_clean_no_lm.join(df_other_no_lm, ["batch size", "ft_dataset"])
    df_lm = df_clean_lm.join(df_other_lm, ["batch size", "ft_dataset"])

    df = df_no_lm.join(df_lm, ["batch size", "ft_dataset"])
    print(df)
    df.write_csv("table.csv")


if __name__ == "__main__":
    main()

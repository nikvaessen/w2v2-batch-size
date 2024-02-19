import polars

df0 = polars.read_csv("acc0.csv").with_columns(
    polars.col("accumulation").apply(lambda _: 0)
)
df1 = polars.read_csv("acc1.csv")
df2 = polars.read_csv("acc2.csv")
df4 = polars.read_csv("acc4.csv")
df8 = polars.read_csv("acc8.csv")
df16 = polars.read_csv("acc16.csv")
df32 = polars.read_csv("acc32.csv")

df = polars.concat(
    [
        df0,
        df1,
        df2,
        df4,
        df8,
        df16,
        df32,
    ]
)

mapping_dict = {
    0: "87.5 sec",
    1: "150 sec",
    2: "5 min",
    4: "10 min",
    8: "20 min",
    16: "40 min",
    32: "80 min",
}

df = df.with_columns(
    polars.col("accumulation").map_dict(mapping_dict).alias("batch_size")
)

print(df)
df.write_csv("merged.csv")

import polars

df100h = polars.read_csv("ft100h_over_time.fixed.csv")
df10m = polars.read_csv("ft10min_over_time.fixed.csv")

df100h = df100h.with_columns(polars.lit("100 hours of labels").alias("ft_dataset"))
df10m = df10m.with_columns(polars.lit("10 minutes of labels").alias("ft_dataset"))

print(df100h)
print(df10m)

polars.concat([df10m, df100h]).write_csv("merged.Lcsv")
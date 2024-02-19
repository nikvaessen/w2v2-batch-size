import polars

polars.read_csv("dist_0gpu.csv")
polars.read_csv("dist_1gpu.csv")
polars.read_csv("dist_2gpu.csv")
polars.read_csv("dist_4gpu.csv")
polars.read_csv("dist_8gpu.csv")
polars.read_csv("dist_16gpu.csv")
polars.read_csv("dist_32gpu.csv")

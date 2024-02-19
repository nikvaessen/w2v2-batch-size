import polars as pl
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure

plot_train = True

if plot_train:
    df = pl.read_csv("ssl_train.csv")
else:
    df = pl.read_csv("ssl_val.csv")

length = len(df)
prefix = "train" if plot_train else "val"
print(df)
print(df.columns)
print(df.dtypes)
for c in df.columns[1:]:
    df = df.with_columns(df[c].cast(pl.Float64, strict=False))

df_254139 = (
    df.select((pl.col(["Step", f"254139 - {prefix}/loss"])))
    .with_columns(
        [
            pl.Series("batch size (s)", [720 for _ in range(length)]),
            pl.Series("mask %", [0.50 for _ in range(length)]),
        ]
    )
    .rename({f"254139 - {prefix}/loss": "loss"})
)

df_254140 = (
    df.select((pl.col(["Step", f"254140 - {prefix}/loss"])))
    .with_columns(
        [
            pl.Series("batch size (s)", [720 for _ in range(length)]),
            pl.Series("mask %", [0.05 for _ in range(length)]),
        ]
    )
    .rename({f"254140 - {prefix}/loss": "loss"})
)

df_254137 = (
    df.select((pl.col(["Step", f"254137 - {prefix}/loss"])))
    .with_columns(
        [
            pl.Series("batch size (s)", [720 // 8 for _ in range(length)]),
            pl.Series("mask %", [0.50 for _ in range(length)]),
        ]
    )
    .rename({f"254137 - {prefix}/loss": "loss"})
)

df_254138 = (
    df.select((pl.col(["Step", f"254138 - {prefix}/loss"])))
    .with_columns(
        [
            pl.Series("batch size (s)", [720 // 8 for _ in range(length)]),
            pl.Series("mask %", [0.05 for _ in range(length)]),
        ]
    )
    .rename({f"254138 - {prefix}/loss": "loss"})
)

df = pl.concat([df_254137, df_254138, df_254139, df_254140])

sns.set_style("darkgrid")
# figure(figsize=(6, 4), dpi=80)

sns.lineplot(df, x="Step", y="loss", hue="batch size (s)", style="mask %")
plt.title(f"{prefix} loss SSL training on ls 960h")
plt.show()

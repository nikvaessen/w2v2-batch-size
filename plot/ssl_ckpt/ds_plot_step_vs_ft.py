import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt

df = pl.read_csv("nanow2v2_all_ds.csv", separator=",")

df = df.select(
    pl.col(["log.tags", "load_from_ckpt", "test/dev-clean_wer", "test/dev-other_wer"])
)


def map_fn(value: str):
    if "wav2vec2_base" in value:
        value = 410_000
    elif len(value) == 0:
        value = 0
    else:
        value = int(value.split("_")[2].split(".")[0])

    return value


df = df.with_columns((df["load_from_ckpt"].apply(map_fn)))
df = df.rename(
    {
        "log.tags": "dataset",
        "load_from_ckpt": "steps",
        "test/dev-clean_wer": "wer dev-clean",
        "test/dev-other_wer": "wer dev-other",
    }
)

sns.set_style("darkgrid")
lineplot = sns.lineplot(
    df, x="steps", y="wer dev-clean", hue="dataset", markers=True, marker="o"
)

# 40.73%	50.04%
# sns.scatterplot(
#     x=[410_000, 410_000], y=[0.4073, 0.5004], hue=["clean", "other"], marker="x"
# )

# Get the current axes
ax = plt.gca()

# Modify the x-axis tick labels
ax.set_xticklabels([f"{int(tick/1000)}k" for tick in ax.get_xticks()])

# Modify the y-axis tick labels
ax.set_yticklabels(["{:.0%}".format(tick) for tick in ax.get_yticks()])

plt.title("SSL training steps versus WER after fine-tuning on various subsets")
plt.show()

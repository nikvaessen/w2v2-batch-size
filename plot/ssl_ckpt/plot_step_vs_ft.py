import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt

df = pl.read_csv("hf.csv", separator=",")

print(df)

sns.set_style("darkgrid")
lineplot = sns.lineplot(df, x="steps", y="wer", hue="subset", markers=True, marker="o")

# 40.73%	50.04%
sns.scatterplot(x=[410_000, 410_000], y=[0.4073, 0.5004], hue=['clean', 'other'], marker="x")

# Get the current axes
ax = plt.gca()

# Modify the x-axis tick labels
ax.set_xticklabels([f"{int(tick/1000)}k" for tick in ax.get_xticks()])

# Modify the y-axis tick labels
ax.set_yticklabels(["{:.0%}".format(tick) for tick in ax.get_yticks()])

plt.title(
    "SSL training steps versus WER after fine-tuning on 10 minutes of LS"
)
plt.show()

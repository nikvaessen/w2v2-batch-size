import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df0 = pd.read_csv("dist_0gpu.csv")
df1 = pd.read_csv("dist_1gpu_v2.csv")
df2 = pd.read_csv("2gpu.csv")
df4 = pd.read_csv("dist_4gpu.csv")
df8 = pd.read_csv("dist_8gpu.csv")
df16 = pd.read_csv("dist_16gpu.csv")
df32 = pd.read_csv("dist_32gpu.csv")

dfs = [df0, df1, df2, df4, df8, df16, df32]
batch_size = ["87.5 sec", "150 sec", "5 min", "10 min", "20 min", "40 min", "80 min"]
seconds = [87.5, 150, 5 * 60, 10 * 60, 20 * 60, 40 * 60, 80 * 60]

# set new column for eachdf
for i, df in enumerate(dfs):
    df["gpu"] = batch_size[i]
    df["hours_seen"] = df["it0"].apply(lambda x: x * seconds[i] / 60 / 60)


# dist to init
df_init = []
for i, df in enumerate(dfs):
    df = df.loc[df["it1"] == 0]
    df_init.append(df)

df_init = pd.concat(df_init, ignore_index=True)

print(df_init.to_string())

sns.lineplot(df_init, x="hours_seen", y="euclidian_distance", hue="gpu")
plt.xscale("log")
plt.show()

# cumulative distance
df_cum = []
for i, df in enumerate(dfs):
    df = df.loc[df["it1"] != 0]
    df["gpu"] = batch_size[i]
    df["cum_dist"] = df["euclidian_distance"].cumsum()
    df_cum.append(df)

df_cum = pd.concat(df_cum)
print(df_cum.to_string())

sns.lineplot(df_cum, x="hours_seen", y="cum_dist", hue="gpu")
plt.xscale("log")
plt.show()

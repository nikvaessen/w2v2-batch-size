import pandas as pd

df = pd.read_json("superb.json", lines=True)


def to_batch_size_in_sec(x: str):
    _, gpu, _ = x.split("-")

    if gpu == "0gpu":
        bs = 87.5
    elif gpu == "1gpu":
        bs = 150
    elif gpu == "2gpu":
        bs = 150 * 2
    elif gpu == "4gpu":
        bs = 150 * 4
    elif gpu == "8gpu":
        bs = 150 * 8
    elif gpu == "16gpu":
        bs = 150 * 16
    elif gpu == "32gpu":
        bs = 150 * 32
    else:
        raise ValueError(f"{gpu=} is unknown")

    return bs


def to_steps(x: str):
    _, _, steps = x.split("-")
    steps_as_int = int(steps.removesuffix("k")) * 1000

    return steps_as_int


def to_desc(x):
    if x < 150:
        r = f"{x:.1f} sec"
    elif x <= 150:
        r = f"{int(x)} sec"
    else:
        r = f"{int(x//60):d} min"

    return r


def from_desc(x):
    value, unit = x.split(" ")

    if unit == "sec":
        return int(value)
    else:
        return int(value) * 60


df["batch_size"] = df["checkpoint"].apply(to_batch_size_in_sec)
df["batch size"] = df["batch_size"].apply(to_desc)
df["steps"] = df["checkpoint"].apply(to_steps)
df["hours_seen"] = df["batch_size"] * df["steps"] / 3600

print(df.to_string())
df.to_csv("superb.csv")

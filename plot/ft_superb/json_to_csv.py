import pathlib

import pandas as pd

file = "superb.partial.json"
pathlib.Path("csv").mkdir(exist_ok=True)

df = pd.read_json(file, lines=True)

if pathlib.Path("fairseq.partial.json").exists():
    df = pd.concat([df, pd.read_json("fairseq.json", lines=True)])

print(df)

batch_size_mapping = {
    "0gpu": 87.5,
    "1gpu": 150,
    "2gpu": 300,
    "4gpu": 600,
    "8gpu": 1200,
    "16gpu": 2400,
    "32gpu": 4800,
}

batch_size_string_mapping = {
    "0gpu": "87.5 sec",
    "1gpu": "150 sec",
    "2gpu": "5 min",
    "4gpu": "10 min",
    "8gpu": "20 min",
    "16gpu": "40 min",
    "32gpu": "80 min",
    "wav2vec2": "90 min"
}

def hours_seen(ckpt_name: str):
    if ckpt_name == "wav2vec2-base-fairseq":
        return 1.5 * 400_000

    gpu, iterations_str = ckpt_name.split("-")[0], ckpt_name.split("-")[1]
    print(gpu)
    batch_size_sec = batch_size_mapping[gpu]
    iterations = int(iterations_str.removesuffix('k')) * 1000

    return batch_size_sec * iterations / 60 / 60

df['hours-seen'] = df['checkpoint'].apply(hours_seen)
df['batch-size'] = df['checkpoint'].apply(lambda x: batch_size_string_mapping[x.split("-")[0]])

for name, df_task in df.groupby('superb_task'):
    for metric, df_metric in df_task.groupby('metric'):
        print(name)
        print(df_metric.to_string())
        df_metric.to_csv(f"csv/{name}_{metric}.csv",index=False)
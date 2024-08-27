import pandas as pd

df = pd.read_csv("/home/anon/scratch/downloads/cb0_sim_max.csv")
print(df.columns)

col = df['7638259 - codebook/cb0_max_cos_sim']

with open('tmp.txt', 'w') as f:
    skip = False
    for e in col:
        skip = not skip
        if skip:
            continue
        else:
            print(e, file=f)

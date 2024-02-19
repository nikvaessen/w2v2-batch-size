import matplotlib.pyplot as plt
import seaborn as sns
import polars

bs_sec = [150, 300, 600, 1200, 2400, 4800]
it = [t for t in range(50_000,400_001, 50_000)]
hours_seen = []

df_rows = []

for bs in bs_sec:
    for t in it:
        row = {'batch size': str(bs), "hours seen": round((t * bs)/3600, 2), "iteration": t}
        print(row)
        df_rows.append(row)


df = polars.DataFrame(df_rows)
print(df)

group_list = [(name, df_group) for name, df_group in df.group_by("hours seen")]
group_list = sorted(group_list, key=lambda t:t[0])

for hours_seen, df_group in group_list:
    print("hours seen:", hours_seen)
    for bs, _, it in df_group.iter_rows():
        print(f"batch size {bs}", "iteration:", it)
    print()

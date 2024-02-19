import pathlib
import polars

from string import digits


files = [f for f in pathlib.Path.cwd().rglob("dist_*gpu*.csv")]
files = sorted(files, key=lambda pth: int("".join(c for c in pth.name if c in digits)))

for f in files:
    print(str(f))
    df = polars.read_csv(f)

    end_distance = None
    distance_traveled = 0

    for r in df.iter_rows(named=True):
        diff_it = abs(r["it1"] - r["it0"])
        distance = r["distance"]

        if r["it0"] == 5000 and r["it1"] == 0:  # skip duplicate
            continue

        elif r["it0"] == 400_000 and r["it1"] == 0:  # skip duplicate
            end_distance = distance

        elif diff_it == 5000:
            distance_traveled += distance

        else:
            continue

    print(f"euclidean distance between 0k and 400k checkpoint:\n{end_distance}")
    print(f"sum of euclidian distance between consecutive checkpoints:\n{distance_traveled}")
    print(f"ratio: {distance_traveled/end_distance:.2f}")

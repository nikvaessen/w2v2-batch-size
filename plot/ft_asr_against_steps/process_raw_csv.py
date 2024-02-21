import pathlib
import polars

csv_file = pathlib.Path("ft100h_over_time.csv")
new_file = pathlib.Path("ft100h_over_time.fixed.csv")

df = polars.read_csv(csv_file)

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
}


def get_iteration(ckpt_file: pathlib.Path):
    return int(ckpt_file.name.split(".")[2].split("_")[1])


new_rows = []
for r in df.iter_rows(named=True):
    pth = pathlib.Path(r["load_from_ckpt"])
    print(pth)
    num_gpus = pth.parent.name
    batch_size = batch_size_mapping[num_gpus]
    iteration = get_iteration(pth)
    hours_seen = iteration * batch_size / 60 / 60
    wer_clean = r["test/test-clean_wer"]
    wer_other = r["test/test-other_wer"]

    new_rows.append(
        {
            "num_gpus": num_gpus,
            "batch size in sec": batch_size,
            "batch size label": batch_size_string_mapping[num_gpus],
            "hours seen": hours_seen,
            "wer clean": wer_clean,
            "wer other": wer_other,
            "iteration": iteration
        }
    )

new_rows = sorted(new_rows, key=lambda dct: dct['batch size in sec'])

polars.DataFrame(new_rows).write_csv(new_file)

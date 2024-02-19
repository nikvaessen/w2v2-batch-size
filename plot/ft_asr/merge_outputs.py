import pathlib

import polars

slurm_dir = pathlib.Path("slurm")

batch_size = []
has_language_model = []
ft_dataset = []
eval_dataset = []
value = []

for f in slurm_dir.glob("*.out"):
    with f.open("r") as h:
        lines = h.readlines()

    has_lm = False
    bs = None
    ft_ds = None

    for ln in lines:
        ln = ln.lower().strip()

        if "without lm" in ln:
            has_lm = False
            continue
        if "with lm" in ln:
            has_lm = True
            continue

        if "ft_asr_ssl_syclic" in ln:
            path = pathlib.Path(ln)
            bs = path.name.split(".")[0]
            ft_ds = path.parent.name
            continue

        if ": wer=" in ln:
            eval_ds = ln.split(":")[0]
            wer = float(ln.split("=")[1])

            batch_size.append(bs)
            has_language_model.append(has_lm)
            ft_dataset.append(ft_ds)
            eval_dataset.append(eval_ds)
            value.append(wer)

df = polars.DataFrame(
    {
        "batch size": batch_size,
        "lm": has_language_model,
        "ft_dataset": ft_dataset,
        "eval_dataset": eval_dataset,
        "value": value,
    }
)

df.write_csv("ft_asr_lm.csv")

import pathlib
import click
import polars
import torch


def load_gradient_vector(ckpt_file: pathlib.Path):
    ckpt = torch.load(ckpt_file, map_location="cpu")

    grad_tensors = []
    for k, v in sorted(ckpt.items(), key=lambda tpl: tpl[0]):
        grad_tensors.append(v.flatten())

    return torch.cat(grad_tensors, dim=0).to(torch.float32)


@click.command()
@click.argument("run_dir", type=pathlib.Path)
@click.argument("ckpt_file", type=pathlib.Path)
@click.argument("csv_file", type=pathlib.Path)
@click.argument("accumulation", type=int)
def main(
    run_dir: pathlib.Path,
    ckpt_file: pathlib.Path,
    csv_file: pathlib.Path,
    accumulation: int,
):
    # get the train step of the checkpoint
    train_step = int(ckpt_file.name.split(".")[2].split("_")[1])
    print(str(ckpt_file))

    # find all gradient vectors in run_dir
    print("considering the following files:")
    gradient_vectors = []
    for fn in (run_dir / "gradients").iterdir():
        if fn.is_file() and "gradients_step" in fn.name:
            print("found correct filename:", str(fn))
            step = int(fn.stem.split("_")[2])
            gradients = load_gradient_vector(fn)
            gradient_vectors.append((fn, step, gradients))
        else:
            print(f"skipping {str(fn)}")

    gradient_vectors = sorted(gradient_vectors, key=lambda tpl: tpl[1])

    for fn, step, gradients in gradient_vectors:
        print(
            fn,
            step,
            gradients.shape,
            torch.sum(torch.isnan(gradients)),
            torch.sum(torch.isinf(gradients)),
        )

    print(f"{gradient_vectors=}")
    print(f"{len(gradient_vectors)=}")
    assert len(gradient_vectors) >= 10

    # only consider the last 10
    gradients = torch.stack([grad for fn, step, grad in gradient_vectors[-10:]])

    # compute metrics
    average_variance = torch.mean(torch.std(gradients, dim=0))

    min_grad = torch.min(gradients)
    max_grad = torch.max(gradients)
    mean_grad = torch.mean(gradients)
    std_grad = torch.std(gradients)

    # row to add to dataframe
    row = {
        "train_step": train_step,
        "accumulation": accumulation,
        "average_variance": average_variance,
        "min_grad": min_grad,
        "max_grad": max_grad,
        "mean_grad": mean_grad,
        "std_grad": std_grad,
    }
    print(row)

    df_row = polars.DataFrame(row)

    if csv_file.exists():
        df = polars.concat([polars.read_csv(csv_file), df_row])
    else:
        df = df_row

    df.write_csv(str(csv_file))


if __name__ == "__main__":
    main()

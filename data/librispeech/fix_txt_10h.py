import pathlib
from typing import List

import click


def find_matching(file: pathlib.Path, potential: List[pathlib.Path]):
    all_equal = []

    for f in potential:
        if file.name == f.name:
            all_equal.append(f)

    return all_equal


@click.command()
@click.argument("rsynced_folder", type=pathlib.Path)
@click.argument("original_folder", type=pathlib.Path)
def main(rsynced_folder: pathlib.Path, original_folder: pathlib.Path):
    fix_needed_txt_files = [f for f in rsynced_folder.rglob("*.trans.txt")]
    original_txt_files = [f for f in original_folder.rglob("*.trans.txt")]

    for file in fix_needed_txt_files:
        all_matching = find_matching(file, original_txt_files)

        if len(all_matching) > 1:
            all_lines = []

            for m in all_matching:
                with m.open("r") as fh:
                    all_lines.extend(fh.readlines())

            with file.open("w") as fh:
                fh.writelines(all_lines)


if __name__ == "__main__":
    main()

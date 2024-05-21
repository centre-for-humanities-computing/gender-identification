import itertools
from pathlib import Path
from typing import Iterable, Union

import pandas as pd
from radicli import Arg, Radicli
from tqdm import tqdm
from transformers import pipeline

cli = Radicli()


def load_table(in_file: Path) -> pd.DataFrame:
    extension = in_file.suffix.removeprefix(".")
    if extension == "csv":
        return pd.read_csv(in_file)
    elif extension == "tsv":
        return pd.read_csv(in_file, delimiter="\t")
    elif extension == "jsonl":
        return pd.read_json(in_file, lines=True, orient="records")
    else:
        raise ValueError(
            f"File format not recognized should be one of csv, tsv, jsonl, recieved: {extension}"
        )


def write_table(table: pd.DataFrame, out_file: Path):
    extension = out_file.suffix.removeprefix(".")
    if extension == "csv":
        return table.to_csv(out_file)
    elif extension == "tsv":
        return table.to_csv(out_file, sep="\t")
    elif extension == "jsonl":
        return table.to_json(out_file, lines=True, orient="records")
    else:
        raise ValueError(
            f"File format not recognized should be one of csv, tsv, jsonl, recieved: {extension}"
        )


def batched(iterable: Iterable[str], n: int) -> Iterable[list[str]]:
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    while batch := list(itertools.islice(it, n)):
        yield batch


@cli.command(
    "infer_gender",
    in_file=Arg(help="Input file path."),
    name_column=Arg("--name_column", "-n", help="Column, where names are contained."),
    out_file=Arg(
        "--out_file",
        "-o",
        help="Output file path, if not specified, the original file will be overwritten.",
    ),
    remove_last_name=Arg(
        "--remove_last_name",
        "-r",
        help="Indicates whether last names should be removed.",
    ),
    drop_confidence=Arg(
        "--drop_confidence",
        "-d",
        help="Indicates whether to drop the column indicating the model's confidence in its predictions.",
    ),
)
def infer_gender(
    in_file: Union[str, Path],
    name_column: str,
    out_file: Union[str, Path, None] = None,
    remove_last_name: bool = False,
    drop_confidence: bool = False,
):
    in_file = Path(in_file)
    out_file = Path(out_file) if out_file is not None else in_file
    print("Loading data.")
    data = load_table(in_file)
    names = data[name_column].tolist()
    if remove_last_name:
        names = [" ".join(name.split()[:-1]) for name in names]
    print("Loading model.")
    model = pipeline(model="Amanaccessassist/Gender-Classification")
    results: list[dict] = []
    progress = tqdm(names, desc="Inferring genders for all names.")
    for batch in batched(progress, n=32):
        results.extend(model(batch))  # type: ignore
    results_df = pd.json_normalize(results).rename(
        columns={"label": "gender", "score": "gender_confidence"}
    )
    if drop_confidence:
        results_df = results_df.drop(columns=["gender_confidence"])
    results_df = results_df.set_index(data.index)
    data = data.join(results_df)
    print("Saving results.")
    write_table(data, out_file)
    print("Done.")

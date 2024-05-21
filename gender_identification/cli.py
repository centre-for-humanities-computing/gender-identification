from pathlib import Path
from typing import Union

import pandas as pd
from radicli import Arg, Radicli

from gender_identification.inference import add_gender

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
    batch_size=Arg(
        "--batch_size", "-b", help="Size of the batches to do inference in."
    ),
)
def infer_gender(
    in_file: Union[str, Path],
    name_column: str,
    out_file: Union[str, Path, None] = None,
    remove_last_name: bool = False,
    drop_confidence: bool = False,
    batch_size: int = 32,
):
    in_file = Path(in_file)
    out_file = Path(out_file) if out_file is not None else in_file
    print("Loading data.")
    data = load_table(in_file)
    print("Inferring results.")
    data = add_gender(
        data,
        name_column=name_column,
        remove_last_name=remove_last_name,
        drop_confidence=drop_confidence,
        batch_size=batch_size,
        progress_bar=True,
    )
    print("Saving results.")
    write_table(data, out_file)
    print("Done.")

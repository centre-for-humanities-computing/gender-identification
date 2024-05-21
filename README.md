# gender-identification
Code and pipeline for gender identification based on names.
The repo contains a CLI and a package for easily adding a gender column to tabular data.

## Usage

Install the package:
```bash
pip install gender-identification
```

If you have some tabular data in csv, tsv or jsonl you can just add a `gender` and a `gender_confidence` column to these using the CLI.

```bash
python3 -m gender_identification data.csv --name_column "first_name"
```

Alternatively you can save it to a different file:

```bash
python3 -m gender_identification data.csv --name_column "first_name" -o results.csv
```

You can also just use the package in Python:
```python
from gender_identification import add_gender

df = pd.DataFrame({"name": ["Peter Jørgensen", "Malte Larsen"]})

df = add_gender(df, name_column="name", remove_last_name=True)
```

## Parameters

| Parameter         | Flag(s)             | Description                                                                                         | Default Value             |
|-------------------|---------------------|-----------------------------------------------------------------------------------------------------|---------------------------|
| `in_file`         |                     | Input file path.                                                                                    | -                      |
| `name_column`     | `--name_column`, `-n` | Column where names are contained.                                                                   | -                      |
| `out_file`        | `--out_file`, `-o`  | Output file path. If not specified, the original file will be overwritten.                           | None                      |
| `remove_last_name`| `--remove_last_name`, `-r` | Indicates whether last names should be removed.                                                      | `False`                   |
| `drop_confidence` | `--drop_confidence`, `-d` | Indicates whether to drop the column indicating the model's confidence in its predictions.            | `False`                   |
| `batch_size`      | `--batch_size`, `-b` | Size of the batches to do inference in.                                                              | `32`                      |

import itertools
from typing import Iterable

import pandas as pd
from tqdm import tqdm
from transformers import pipeline

model = pipeline(model="Amanaccessassist/Gender-Classification")


def batched(iterable: Iterable[str], n: int) -> Iterable[list[str]]:
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    while batch := list(itertools.islice(it, n)):
        yield batch


def add_gender(
    data: pd.DataFrame,
    name_column: str,
    remove_last_name: bool = False,
    drop_confidence: bool = False,
    progress_bar: bool = True,
    batch_size: int = 32,
) -> pd.DataFrame:
    """Adds gender and confidence columns to dataframe."""
    results: list[dict] = []
    names = data[name_column].tolist()
    if remove_last_name:
        names = [" ".join(name.split()[:-1]) for name in names]
    if progress_bar:
        names = tqdm(names, desc="Inferring genders for all names.")
    for batch in batched(names, n=batch_size):
        results.extend(model(batch))  # type: ignore
    results_df = pd.json_normalize(results).rename(
        columns={"label": "gender", "score": "gender_confidence"}
    )
    if drop_confidence:
        results_df = results_df.drop(columns=["gender_confidence"])
    results_df = results_df.set_index(data.index)
    data = data.join(results_df)
    return data

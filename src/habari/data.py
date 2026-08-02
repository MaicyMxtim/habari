"""Download and prepare the Swahili split of MasakhaNEWS.

Caches each split to data/ as parquet so later stages never need the network.
"""

from pathlib import Path

import pandas as pd
from datasets import load_dataset

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
DATASET = "masakhane/masakhanews"
LANGUAGE = "swa"
SPLITS = ("train", "validation", "test")


def prepare() -> dict[str, pd.DataFrame]:
    DATA_DIR.mkdir(exist_ok=True)
    dataset = load_dataset(DATASET, LANGUAGE)
    label_names = sorted(set(dataset["train"]["category"]))
    label_to_id = {name: i for i, name in enumerate(label_names)}
    frames = {}
    for split in SPLITS:
        df = dataset[split].to_pandas()
        df["label_name"] = df["category"]
        df["label"] = df["category"].map(label_to_id)
        # The headline carries strong signal; combine it with the body text.
        df["full_text"] = (df["headline"].fillna("") + "\n" + df["text"].fillna("")).str.strip()
        df.to_parquet(DATA_DIR / f"{split}.parquet")
        frames[split] = df
    (DATA_DIR / "labels.txt").write_text("\n".join(label_names))
    return frames


def load(split: str) -> pd.DataFrame:
    path = DATA_DIR / f"{split}.parquet"
    if not path.exists():
        prepare()
    return pd.read_parquet(path)


def label_names() -> list[str]:
    path = DATA_DIR / "labels.txt"
    if not path.exists():
        prepare()
    return path.read_text().splitlines()


if __name__ == "__main__":
    frames = prepare()
    for split, df in frames.items():
        print(f"{split}: {len(df)} articles")
    print("labels:", ", ".join(label_names()))

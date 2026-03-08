"""Loading and preprocessing for the student performance datasets."""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"


def load_math() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "student-mat.csv", sep=";")


def load_portuguese() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "student-por.csv", sep=";")


def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Encode categoricals and split into features / target (G3)."""
    df = df.copy()
    target = df.pop("G3")
    df = pd.get_dummies(df, drop_first=True)
    return df, target

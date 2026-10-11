"""Shared data loading and feature preparation for the house price models."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

TARGET = "price"
BINARY_COLS = ["mainroad", "guestroom", "basement", "hotwaterheating", "airconditioning", "prefarea"]

FEATURE_SETS = {
    "Simple (area only)": ["area"],
    "Multiple (area, bedrooms, bathrooms)": ["area", "bedrooms", "bathrooms"],
    "Full (all 12 features)": [
        "area", "bedrooms", "bathrooms", "stories", "parking",
        *BINARY_COLS,
        "furnishingstatus_semi-furnished", "furnishingstatus_unfurnished",
    ],
}
FULL_FEATURES = FEATURE_SETS["Full (all 12 features)"]


def load_data(path: str | Path = "data/housing.csv") -> pd.DataFrame:
    return pd.read_csv(path)


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode yes/no columns as 1/0 and one-hot encode furnishing status
    (furnished is the baseline, so it is dropped)."""
    df = df.copy()
    for col in BINARY_COLS:
        df[col] = df[col].map({"yes": 1, "no": 0})
    df["furnishingstatus"] = pd.Categorical(
        df["furnishingstatus"], categories=["furnished", "semi-furnished", "unfurnished"]
    )
    return pd.get_dummies(df, columns=["furnishingstatus"], drop_first=True, dtype=int)

"""
Compare simple, multiple and full linear regression models for house prices,
then train and save the full model.

Usage:
    python src/train.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split

from features import FEATURE_SETS, FULL_FEATURES, TARGET, load_data, prepare_features


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the house price model.")
    parser.add_argument("--data", default="data/housing.csv")
    parser.add_argument("--model", default="models/house_price_model.joblib")
    args = parser.parse_args()

    df = prepare_features(load_data(args.data))
    train, test = train_test_split(df, test_size=0.2, random_state=42)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    print(f"{'Model':40s} {'Test R²':>8s} {'CV R²':>7s} {'Test MAE':>12s}")
    for name, cols in FEATURE_SETS.items():
        model = LinearRegression().fit(train[cols], train[TARGET])
        pred = model.predict(test[cols])
        cv_r2 = cross_val_score(LinearRegression(), df[cols], df[TARGET], cv=cv).mean()
        print(f"{name:40s} {r2_score(test[TARGET], pred):8.3f} {cv_r2:7.3f} "
              f"{mean_absolute_error(test[TARGET], pred):12,.0f}")

    final = LinearRegression().fit(df[FULL_FEATURES], df[TARGET])
    Path(args.model).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final, args.model)
    print(f"\nFull model trained on all {len(df)} houses and saved to {args.model}")


if __name__ == "__main__":
    main()

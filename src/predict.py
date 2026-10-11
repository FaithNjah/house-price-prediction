"""
Predict a house price with the trained full model.

Usage:
    python src/predict.py --area 6000 --bedrooms 3 --bathrooms 2 --stories 2 \
        --parking 1 --mainroad --airconditioning --furnishing semi-furnished
"""

from __future__ import annotations

import argparse

import joblib
import pandas as pd

from features import BINARY_COLS, FULL_FEATURES


def main() -> None:
    p = argparse.ArgumentParser(description="Predict a house price.")
    p.add_argument("--area", type=float, required=True, help="plot area in square feet")
    p.add_argument("--bedrooms", type=int, required=True)
    p.add_argument("--bathrooms", type=int, required=True)
    p.add_argument("--stories", type=int, default=1)
    p.add_argument("--parking", type=int, default=0, help="number of parking spaces")
    for col in BINARY_COLS:
        p.add_argument(f"--{col}", action="store_true", help=f"house has {col}")
    p.add_argument("--furnishing", choices=["furnished", "semi-furnished", "unfurnished"],
                   default="furnished")
    p.add_argument("--model", default="models/house_price_model.joblib")
    a = p.parse_args()

    row = {
        "area": a.area, "bedrooms": a.bedrooms, "bathrooms": a.bathrooms,
        "stories": a.stories, "parking": a.parking,
        **{col: int(getattr(a, col)) for col in BINARY_COLS},
        "furnishingstatus_semi-furnished": int(a.furnishing == "semi-furnished"),
        "furnishingstatus_unfurnished": int(a.furnishing == "unfurnished"),
    }
    model = joblib.load(a.model)
    price = model.predict(pd.DataFrame([row])[FULL_FEATURES])[0]
    print(f"Predicted price: {price:,.0f}")


if __name__ == "__main__":
    main()

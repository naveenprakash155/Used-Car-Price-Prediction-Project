"""
model.py - Model loading and prediction utilities.
Loaded once at app startup; thread-safe for concurrent requests.
"""

import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

MODEL_PATH = os.path.join(os.path.dirname(__file__), "car_price_model.pkl")

_artifacts: Dict[str, Any] = {}


def load_model() -> None:
    """Load model artifacts from disk into module-level cache."""
    global _artifacts
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run `python model/train.py` first."
        )
    with open(MODEL_PATH, "rb") as f:
        _artifacts = pickle.load(f)
    print(f"✅ Model loaded from {MODEL_PATH}")


def get_metadata() -> Dict[str, Any]:
    """Return structured metadata for populating the UI dropdowns."""
    if not _artifacts:
        load_model()
    return {
        "car_models":    _artifacts["car_models"],
        "cities":        _artifacts["cities"],
        "fuel_types":    _artifacts["fuel_types"],
        "transmissions": _artifacts["transmissions"],
    }


def _encode(value: str, col: str) -> int:
    """Encode a categorical value using the saved LabelEncoder."""
    enc = _artifacts["encoders"][col]
    if value not in enc.classes_:
        raise ValueError(f"Unknown value '{value}' for feature '{col}'.")
    return int(enc.transform([value])[0])


def predict_price(
    brand: str,
    model_name: str,
    year: int,
    fuel_type: str,
    transmission: str,
    km_driven: int,
    owners: int,
    city: str,
) -> Dict[str, Any]:
    """
    Run the ML model and return a prediction dict.

    Returns
    -------
    {
        "price_lakh"     : float,      # point estimate in ₹ lakhs
        "price_inr"      : int,        # ₹ value
        "range_low_inr"  : int,        # lower bound (confidence range)
        "range_high_inr" : int,        # upper bound
        "deal_rating"    : str,        # Great Deal / Fair Deal / Overpriced
        "deal_score"     : float,      # 0–100 for gauge
    }
    """
    if not _artifacts:
        load_model()

    age         = 2024 - year
    km_per_year = km_driven / max(age, 1)
    age_sq      = age ** 2
    log_km      = np.log1p(km_driven)

    features = {
        "brand":        _encode(brand, "brand"),
        "model":        _encode(model_name, "model"),
        "year":         year,
        "age":          age,
        "age_sq":       age_sq,
        "fuel_type":    _encode(fuel_type, "fuel_type"),
        "transmission": _encode(transmission, "transmission"),
        "km_driven":    km_driven,
        "log_km":       log_km,
        "km_per_year":  km_per_year,
        "owners":       owners,
        "city":         _encode(city, "city"),
    }

    feature_vector = pd.DataFrame(
        [[features[c] for c in _artifacts["feature_cols"]]],
        columns=_artifacts["feature_cols"],
    )
    price_lakh = float(_artifacts["model"].predict(feature_vector)[0])
    price_lakh = max(0.3, price_lakh)   # hard floor

    # Confidence range: ±7% (realistic market spread)
    spread = 0.07
    low_lakh  = price_lakh * (1 - spread)
    high_lakh = price_lakh * (1 + spread)

    # ── Deal rating ────────────────────────────────────────────────────
    # Compare actual km_per_year vs a "well-maintained" baseline of 12,000
    baseline_km = 12_000
    owner_penalty = (owners - 1) * 8      # 0–24 pts
    km_penalty    = min(30, max(0, (km_per_year - baseline_km) / baseline_km * 25))
    age_penalty   = min(20, age * 1.5)

    deal_score = max(0, min(100, 100 - owner_penalty - km_penalty - age_penalty))

    if deal_score >= 70:
        deal_rating = "Great Deal"
    elif deal_score >= 45:
        deal_rating = "Fair Deal"
    else:
        deal_rating = "Overpriced"

    return {
        "price_lakh":     round(price_lakh, 2),
        "price_inr":      int(price_lakh * 1_00_000),
        "range_low_inr":  int(low_lakh * 1_00_000),
        "range_high_inr": int(high_lakh * 1_00_000),
        "deal_rating":    deal_rating,
        "deal_score":     round(deal_score, 1),
    }

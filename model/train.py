"""
train.py - Trains a Random Forest regression model on synthetic Indian used car market data.
Saves the trained model and encoders for use in the Flask app.
"""

import numpy as np
import pandas as pd
import pickle
import os
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# ─────────────────────────────────────────────
# 1. SYNTHETIC DATASET GENERATION
# ─────────────────────────────────────────────

np.random.seed(42)

# Indian car brands & popular models with base price bands (in INR lakhs)
CAR_MODELS = {
    "Maruti Suzuki": {
        "Alto":       (2.5,  4.5),
        "Swift":      (5.0,  9.0),
        "Dzire":      (5.5,  9.5),
        "Baleno":     (6.0, 10.5),
        "Vitara Brezza": (8.0, 14.0),
        "Ertiga":     (8.5, 13.5),
    },
    "Hyundai": {
        "i10":        (3.5,  6.0),
        "i20":        (6.5, 11.5),
        "Venue":      (8.0, 14.0),
        "Creta":      (10.0, 18.0),
        "Verna":      (9.0, 15.0),
        "Tucson":     (18.0, 28.0),
    },
    "Honda": {
        "Brio":       (4.0,  7.0),
        "Amaze":      (6.0, 10.0),
        "City":       (9.0, 16.0),
        "Jazz":       (7.5, 12.0),
        "WR-V":       (8.5, 13.5),
        "CR-V":       (22.0, 32.0),
    },
    "Tata": {
        "Tiago":      (4.5,  7.5),
        "Tigor":      (5.5,  9.0),
        "Nexon":      (7.5, 14.0),
        "Harrier":    (14.0, 22.0),
        "Safari":     (15.0, 24.0),
        "Altroz":     (6.0, 10.5),
    },
    "Mahindra": {
        "KUV100":     (5.0,  8.5),
        "XUV300":     (8.0, 14.5),
        "XUV500":     (12.0, 20.0),
        "Thar":       (14.0, 22.0),
        "Scorpio":    (10.0, 18.0),
        "Bolero":     (8.0, 13.0),
    },
    "Toyota": {
        "Glanza":     (7.0, 11.0),
        "Urban Cruiser": (9.0, 14.5),
        "Innova":     (14.0, 25.0),
        "Fortuner":   (28.0, 42.0),
        "Camry":      (35.0, 48.0),
        "Yaris":      (8.5, 13.0),
    },
    "Kia": {
        "Sonet":      (7.5, 14.5),
        "Seltos":     (10.0, 19.5),
        "Carnival":   (25.0, 35.0),
        "Carens":     (9.0, 16.5),
    },
    "Renault": {
        "Kwid":       (3.5,  6.0),
        "Triber":     (5.5,  9.5),
        "Duster":     (8.0, 14.0),
        "Kiger":      (5.5, 10.0),
    },
    "Volkswagen": {
        "Polo":       (7.0, 11.5),
        "Vento":      (9.0, 14.5),
        "Taigun":     (11.0, 18.0),
        "Tiguan":     (28.0, 40.0),
    },
    "Skoda": {
        "Rapid":      (9.0, 14.5),
        "Slavia":     (11.0, 18.0),
        "Kushaq":     (11.0, 18.5),
        "Octavia":    (26.0, 38.0),
    },
    "Ford": {
        "Figo":       (5.0,  8.5),
        "Aspire":     (6.5, 10.5),
        "EcoSport":   (8.0, 13.5),
        "Endeavour":  (25.0, 38.0),
    },
    "MG": {
        "Hector":     (14.0, 22.0),
        "ZS EV":      (18.0, 28.0),
        "Gloster":    (28.0, 40.0),
        "Astor":      (12.0, 20.0),
    },
}

FUEL_TYPES    = ["Petrol", "Diesel", "CNG", "Electric"]
TRANSMISSIONS = ["Manual", "Automatic"]
CITIES        = [
    "Mumbai", "Delhi", "Bangalore", "Hyderabad", "Chennai",
    "Pune", "Kolkata", "Ahmedabad", "Jaipur", "Lucknow",
]

# City cost-of-living multiplier (premium cities → slightly higher resale)
CITY_MULTIPLIER = {
    "Mumbai":    1.08, "Delhi":      1.05, "Bangalore":  1.06,
    "Hyderabad": 1.03, "Chennai":    1.04, "Pune":       1.05,
    "Kolkata":   1.00, "Ahmedabad":  1.02, "Jaipur":     0.98,
    "Lucknow":   0.97,
}

# Fuel price premium / discount (relative to Petrol = 1.0)
FUEL_MULTIPLIER = {
    "Petrol":   1.00,
    "Diesel":   1.06,   # Diesel depreciation is slightly less for high-km cars
    "CNG":      0.90,   # CNG cheaper but lower resale
    "Electric": 1.15,   # EV premium in Indian market
}

# Automatic transmission commands a premium
TRANS_MULTIPLIER = {"Manual": 1.00, "Automatic": 1.10}


def generate_dataset(n_samples: int = 8000) -> pd.DataFrame:
    """Generate a realistic synthetic dataset of Indian used cars."""
    records = []
    brands = list(CAR_MODELS.keys())

    for _ in range(n_samples):
        brand = np.random.choice(brands)
        model_name = np.random.choice(list(CAR_MODELS[brand].keys()))
        base_low, base_high = CAR_MODELS[brand][model_name]

        year = int(np.random.randint(2010, 2024))
        age  = 2024 - year

        fuel_type    = np.random.choice(FUEL_TYPES, p=[0.52, 0.32, 0.10, 0.06])
        transmission = np.random.choice(TRANSMISSIONS, p=[0.70, 0.30])
        city         = np.random.choice(CITIES)

        # Km driven: roughly 12,000–15,000 km per year, with noise
        avg_km = np.random.randint(10_000, 18_000)
        km_driven = max(500, int(np.random.normal(avg_km * age, avg_km * 0.4 * age)))

        # Previous owners: skewed toward 1
        owners = int(np.random.choice([1, 2, 3, 4], p=[0.55, 0.30, 0.12, 0.03]))

        # ── PRICE CALCULATION ──────────────────────────────────────────
        # Start with a random base (in lakhs)
        base_price = np.random.uniform(base_low, base_high)

        # Depreciation: ~15% first year, ~10% subsequent years
        depr = 0.85 if age > 0 else 1.0
        for y in range(1, age):
            depr *= 0.90
        price = base_price * depr

        # Mileage penalty: every 10k km over expected reduces value ~1.5%
        expected_km = avg_km * age
        km_excess   = max(0, km_driven - expected_km)
        price *= max(0.60, 1 - (km_excess / 10_000) * 0.015)

        # Owner penalty
        price *= (0.92 ** (owners - 1))

        # Multipliers
        price *= FUEL_MULTIPLIER[fuel_type]
        price *= TRANS_MULTIPLIER[transmission]
        price *= CITY_MULTIPLIER[city]

        # Clip to realistic range (in lakhs)
        price = np.clip(price, 0.8, 55.0)

        # Add realistic market noise (~±3%)
        price *= np.random.normal(1.0, 0.03)
        price = max(0.5, price)

        records.append({
            "brand":        brand,
            "model":        model_name,
            "year":         year,
            "fuel_type":    fuel_type,
            "transmission": transmission,
            "km_driven":    km_driven,
            "owners":       owners,
            "city":         city,
            "price_lakh":   round(price, 4),   # in ₹ lakhs
        })

    return pd.DataFrame(records)


# ─────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features to the dataframe."""
    df = df.copy()
    df["age"]            = 2024 - df["year"]
    df["km_per_year"]    = (df["km_driven"] / df["age"].replace(0, 1)).astype(int)
    df["age_sq"]         = df["age"] ** 2
    df["log_km"]         = np.log1p(df["km_driven"])
    return df


# ─────────────────────────────────────────────
# 3. TRAINING
# ─────────────────────────────────────────────

def train_and_save_model(output_dir: str = "model") -> None:
    """Train GradientBoostingRegressor and persist model + encoders."""
    print("📦 Generating synthetic dataset …")
    df = generate_dataset(n_samples=10_000)
    df = engineer_features(df)

    print(f"   Dataset size: {len(df):,} rows")
    print(f"   Price range : ₹{df['price_lakh'].min():.2f}L – ₹{df['price_lakh'].max():.2f}L")

    # Encode categoricals
    cat_cols    = ["brand", "model", "fuel_type", "transmission", "city"]
    encoders    = {}
    df_encoded  = df.copy()

    for col in cat_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df[col])
        encoders[col]   = le

    feature_cols = [
        "brand", "model", "year", "age", "age_sq",
        "fuel_type", "transmission", "km_driven", "log_km",
        "km_per_year", "owners", "city",
    ]
    X = df_encoded[feature_cols]
    y = df_encoded["price_lakh"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

    print("\n🚀 Training Gradient Boosting Regressor …")
    model = GradientBoostingRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=5,
        subsample=0.8,
        random_state=42,
        verbose=0,
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    mae    = mean_absolute_error(y_test, y_pred)
    r2     = r2_score(y_test, y_pred)
    print(f"\n✅ Model Metrics:")
    print(f"   MAE : ₹{mae:.4f} lakhs  (~₹{mae*100_000:,.0f})")
    print(f"   R²  : {r2:.4f}")

    # Save artifacts
    os.makedirs(output_dir, exist_ok=True)
    artifacts = {
        "model":        model,
        "encoders":     encoders,
        "feature_cols": feature_cols,
        "car_models":   CAR_MODELS,
        "cities":       CITIES,
        "fuel_types":   FUEL_TYPES,
        "transmissions": TRANSMISSIONS,
    }
    with open(os.path.join(output_dir, "car_price_model.pkl"), "wb") as f:
        pickle.dump(artifacts, f)

    print(f"\n💾 Model saved to {output_dir}/car_price_model.pkl")


if __name__ == "__main__":
    # When run from project root, save into model/
    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_and_save_model(output_dir=script_dir)

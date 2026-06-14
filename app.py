"""
app.py - Flask web server for the Used Car Price Predictor.

Usage:
    python app.py

The model trains automatically on first run if the pickle is missing.
"""

import os
import sys
import subprocess
from flask import Flask, render_template, request, jsonify

# ── Bootstrap: train model if missing ────────────────────────────────
MODEL_PKL = os.path.join("model", "car_price_model.pkl")
if not os.path.exists(MODEL_PKL):
    print("🔧 Model not found — training now (takes ~30 s) …")
    result = subprocess.run(
        [sys.executable, os.path.join("model", "train.py")],
        capture_output=False,
    )
    if result.returncode != 0:
        print("❌ Training failed. Check model/train.py and dependencies.")
        sys.exit(1)

# ── Import model utilities after pickle exists ────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "model"))
from model import load_model, get_metadata, predict_price  # noqa: E402

app = Flask(__name__, template_folder="templates", static_folder="static")

# Load model once at startup
load_model()


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the main SPA page."""
    metadata = get_metadata()
    return render_template("index.html", metadata=metadata)


@app.route("/api/metadata")
def api_metadata():
    """Return UI metadata (brands, models, cities …) as JSON."""
    return jsonify(get_metadata())


@app.route("/predict", methods=["POST"])
def predict():
    """
    POST /predict
    Body (JSON):
        brand, model, year, fuel_type, transmission,
        km_driven, owners, city
    Returns JSON with price prediction.
    """
    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({"error": "Invalid JSON body."}), 400

    required = ["brand", "model", "year", "fuel_type", "transmission",
                "km_driven", "owners", "city"]
    missing  = [f for f in required if f not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {', '.join(missing)}"}), 400

    # ── Type coercion & validation ────────────────────────────────────
    try:
        year      = int(data["year"])
        km_driven = int(data["km_driven"])
        owners    = int(data["owners"])
    except (ValueError, TypeError) as exc:
        return jsonify({"error": f"Type error: {exc}"}), 400

    if not (1990 <= year <= 2024):
        return jsonify({"error": "Year must be between 1990 and 2024."}), 422
    if not (0 <= km_driven <= 5_00_000):
        return jsonify({"error": "km_driven must be between 0 and 5,00,000."}), 422
    if not (1 <= owners <= 5):
        return jsonify({"error": "Number of owners must be between 1 and 5."}), 422

    try:
        result = predict_price(
            brand        = str(data["brand"]),
            model_name   = str(data["model"]),
            year         = year,
            fuel_type    = str(data["fuel_type"]),
            transmission = str(data["transmission"]),
            km_driven    = km_driven,
            owners       = owners,
            city         = str(data["city"]),
        )
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 422
    except Exception as exc:
        app.logger.exception("Prediction error")
        return jsonify({"error": f"Prediction failed: {exc}"}), 500

    return jsonify(result)


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n🚗  Car Price Predictor running → http://localhost:{port}\n")
    app.run(host="0.0.0.0", port=port, debug=False)

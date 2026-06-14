# CarMūlya — Used Car Price Predictor 🚗

AI-powered resale price estimator for the Indian used car market.  
Built with **Flask + scikit-learn (Gradient Boosting)** and a clean dark-UI frontend.

---

## Features

| | |
|---|---|
| 🤖 **ML Model** | Gradient Boosting Regressor trained on 10,000 synthetic Indian market records |
| 🏙️ **12 Brands, 60+ Models** | Maruti, Hyundai, Honda, Tata, Mahindra, Toyota, Kia, Renault, VW, Skoda, Ford, MG |
| ⛽ **4 Fuel Types** | Petrol, Diesel, CNG, Electric |
| 📍 **10 Cities** | Mumbai, Delhi, Bangalore, Hyderabad, Chennai, Pune, Kolkata, Ahmedabad, Jaipur, Lucknow |
| 📊 **Deal Rating** | Great Deal / Fair Deal / Overpriced (with visual gauge) |
| 💰 **Indian Pricing** | Outputs in ₹ with confidence range |

---

## Quick Start

### 1. Clone / extract the project

```bash
cd car-price-predictor
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
python app.py
```

The model trains **automatically** on first run (~30 seconds).  
Open **http://localhost:5000** in your browser.

---

## File Structure

```
car-price-predictor/
├── app.py                    # Flask web server & /predict endpoint
├── requirements.txt
├── model/
│   ├── train.py              # Synthetic data generation + GBR training
│   ├── model.py              # Inference utilities (load, predict)
│   └── car_price_model.pkl   # Saved model (auto-generated)
├── templates/
│   └── index.html            # Single-page UI
└── static/
    ├── css/style.css
    └── js/app.js
```

---

## API

### `POST /predict`

**Request body (JSON):**
```json
{
  "brand":        "Hyundai",
  "model":        "Creta",
  "year":         2020,
  "fuel_type":    "Petrol",
  "transmission": "Manual",
  "km_driven":    45000,
  "owners":       1,
  "city":         "Bangalore"
}
```

**Response:**
```json
{
  "price_lakh":     9.85,
  "price_inr":      985000,
  "range_low_inr":  915450,
  "range_high_inr": 1054550,
  "deal_rating":    "Great Deal",
  "deal_score":     82.5
}
```

### `GET /api/metadata`

Returns all valid brands, models, cities, fuel types, and transmissions for populating UI dropdowns.

---

## Model Details

| Metric | Value |
|--------|-------|
| Algorithm | Gradient Boosting Regressor (scikit-learn) |
| Training samples | 10,000 |
| Features | 12 (brand, model, year, age, age², fuel type, transmission, km driven, log(km), km/year, owners, city) |
| MAE | ~₹92,000 |
| R² | ~0.945 |

---

## Production Deployment

```bash
gunicorn -w 4 -b 0.0.0.0:8000 app:app
```

---

*Price estimates are AI-generated and for reference only. Actual resale prices vary by vehicle condition, service history, and market demand.*

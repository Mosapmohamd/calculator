from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, List, Dict
import pandas as pd
import numpy as np
import re
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import t
from fuzzywuzzy import fuzz, process
import json
import pickle
from datetime import datetime
import hashlib
import os
import requests

# =====================
# Paths & URLs
# =====================

CSV_PATH = "test2.csv"
MODEL_PATH = "car_price_model.pkl"
FEEDBACK_PATH = "feedback_data.json"

MODEL_URL = "https://drive.google.com/uc?export=download&id=1kLCgHJ0Jm-FawLrfBqV3bcei8IATaCd7"

# =====================
# Constants
# =====================

FUZZY_THRESHOLD = 80
MIN_SAMPLES = 2
YEAR_DELTA = 2
CONFIDENCE = 0.9
MIN_MODEL_YEAR = 2010
HYBRID_BONUS = 3000

MILEAGE_RATE_MORE_KM = 1000
MILEAGE_RATE_LESS_KM = 700

app = FastAPI(title="Vehicle Price Prediction API")

# =====================
# Utils
# =====================

def download_model():
    if not os.path.exists(MODEL_PATH):
        r = requests.get(MODEL_URL, stream=True)
        with open(MODEL_PATH, "wb") as f:
            for chunk in r.iter_content(8192):
                if chunk:
                    f.write(chunk)

app = FastAPI(title="Vehicle Price Prediction API")

def clean(s: Optional[str]) -> Optional[str]:
    if s is None or pd.isna(s):
        return None
    s = str(s).lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^a-z0-9 ]", "", s)
    return s if s else None

def match_make_smart(value: Optional[str], makes: List[str]):
    if not value or not makes:
        return None
    val = clean(value)
    for m in makes:
        if val == m:
            return m
    for m in makes:
        if val in m or m in val:
            return m
    hit = process.extractOne(val, makes, scorer=fuzz.partial_ratio)
    if hit and hit[1] >= FUZZY_THRESHOLD:
        return hit[0]
    return None

def match_model_smart(value: Optional[str], models: List[str]):
    if not value or not models:
        return None
    val = clean(value)
    for m in models:
        if val == m:
            return m
    for m in models:
        if val in m or m in val:
            return m
    hit = process.extractOne(val, models, scorer=fuzz.partial_ratio)
    if hit and hit[1] >= FUZZY_THRESHOLD:
        return hit[0]
    return None

def is_hybrid(p: BaseModel) -> bool:
    text = clean(f"{p.make} {p.model} {p.trim or ''}")
    if not text:
        return False
    return "hybrid" in text

def calculate_mileage_adjustment(target_km: int, comparable_km: int, comparable_price: float) -> float:
    km_difference = target_km - comparable_km
    
    if km_difference > 0:
        adjustment = - (km_difference / 10000) * MILEAGE_RATE_MORE_KM
    else:
        adjustment = - (km_difference / 10000) * MILEAGE_RATE_LESS_KM
    
    adjusted_price = comparable_price + adjustment
    return max(adjusted_price, 0)

def trim_similarity(trim1: Optional[str], trim2: Optional[str]) -> float:
    if not trim1 or not trim2:
        return 0.0
    
    clean1 = clean(trim1)
    clean2 = clean(trim2)
    
    if not clean1 or not clean2:
        return 0.0
    
    if clean1 == clean2:
        return 100.0
    
    if clean1 in clean2 or clean2 in clean1:
        return 90.0
    
    return fuzz.partial_ratio(clean1, clean2)

class PriceModel:
    def __init__(self):
        self.model = None
        self.features = ['year', 'odometer', 'make_encoded', 'model_encoded']
        self.is_trained = False
        
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['make_encoded'] = pd.factorize(df['make'])[0]
        df['model_encoded'] = pd.factorize(df['model'])[0]
        df['age'] = datetime.now().year - df['year']
        df['odometer_per_year'] = df['odometer'] / (df['age'] + 1)
        return df
    
    def train(self, X: pd.DataFrame, y: pd.Series):
        X_processed = self.create_features(X)
        X_features = X_processed[['year', 'odometer', 'make_encoded', 'model_encoded', 'age', 'odometer_per_year']]
        
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X_features, y)
        self.is_trained = True
        
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump(self.model, f)
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        X_processed = self.create_features(X)
        X_features = X_processed[['year', 'odometer', 'make_encoded', 'model_encoded', 'age', 'odometer_per_year']]
        
        return self.model.predict(X_features)
    
    def load(self):
        try:
            with open(MODEL_PATH, 'rb') as f:
                self.model = pickle.load(f)
            self.is_trained = True
        except FileNotFoundError:
            self.is_trained = False

def save_feedback(request: Dict, predicted_price: float, actual_price: Optional[float] = None):
    try:
        with open(FEEDBACK_PATH, 'r') as f:
            feedback_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        feedback_data = []
    
    feedback_entry = {
        'id': hashlib.md5(json.dumps(request, sort_keys=True).encode()).hexdigest(),
        'request': request,
        'predicted_price': predicted_price,
        'actual_price': actual_price,
        'timestamp': datetime.now().isoformat(),
        'error': actual_price - predicted_price if actual_price else None
    }
    
    feedback_data.append(feedback_entry)
    
    with open(FEEDBACK_PATH, 'w') as f:
        json.dump(feedback_data, f, indent=2)
    
    if len(feedback_data) >= 50 and actual_price is not None:
        retrain_model_with_feedback()

def retrain_model_with_feedback():
    try:
        with open(FEEDBACK_PATH, 'r') as f:
            feedback_data = json.load(f)
    except FileNotFoundError:
        return
    
    train_data = []
    for entry in feedback_data:
        if entry.get('actual_price') is not None:
            req = entry['request']
            train_data.append({
                'year': req.get('year'),
                'make': req.get('make'),
                'model': req.get('model'),
                'odometer': req.get('odometer'),
                'price': entry['actual_price']
            })
    
    if len(train_data) >= 20:
        df_train = pd.DataFrame(train_data)
        
        global df
        df_combined = pd.concat([df, df_train], ignore_index=True).drop_duplicates()
        
        X = df_combined[['year', 'make', 'model', 'odometer']]
        y = df_combined['price']
        
        model = PriceModel()
        model.train(X, y)

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={
        "Year": "year",
        "Make": "make",
        "Model": "model",
        "Trim": "trim",
        "Odometer": "odometer",
        "Price": "price",
        "Province": "province",
    })
    
    for c in ["year", "odometer", "price"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    
    for c in ["make", "model", "trim", "province"]:
        df[c] = df[c].apply(clean)
    
    df = df[(df['price'] > 1000) & (df['price'] < 200000)]
    df = df[(df['odometer'] >= 0) & (df['odometer'] <= 500000)]
    
    df = df.dropna(subset=["year", "make", "model", "odometer", "price"])
    return df.reset_index(drop=True)

df = load_data(CSV_PATH)
price_model = PriceModel()
price_model.load()

class EstimateRequest(BaseModel):
    year: int
    make: str
    model: str
    odometer: int
    trim: Optional[str] = None
    province: Optional[str] = None
    actual_price: Optional[float] = None

class FeedbackRequest(BaseModel):
    request: EstimateRequest
    predicted_price: float
    actual_price: float

def estimate_value(p: EstimateRequest):
    
    if p.year < MIN_MODEL_YEAR:
        return {
            "status": "not_supported",
            "title": f"{p.year} {p.make} {p.model}",
            "price": None,
            "note": "Model year below supported range"
        }

    make = match_make_smart(p.make, df.make.unique().tolist())
    if not make:
        return {"status": "error", "note": "Make not found"}

    model = match_model_smart(
        p.model,
        df[df.make == make].model.unique().tolist()
    )
    if not model:
        return {"status": "error", "note": "Model not found"}

    title = f"{p.year} {make.title()} {model.title()}"
    
    comps = df[
        (df.make == make) &
        (df.model == model) &
        (df.year >= p.year - YEAR_DELTA) &
        (df.year <= p.year + YEAR_DELTA)
    ].copy()
    
    if p.trim and 'trim' in comps.columns:
        comps['trim_similarity'] = comps['trim'].apply(
            lambda x: trim_similarity(p.trim, x)
        )
        
        similar_comps = comps[comps['trim_similarity'] >= 70]
        if len(similar_comps) >= MIN_SAMPLES:
            comps = similar_comps
    
    n = len(comps)
    
    def apply_hybrid(price: float) -> float:
        return price + HYBRID_BONUS if is_hybrid(p) else price
    
    if n == 0:
        if price_model.is_trained:
            try:
                pred_df = pd.DataFrame([{
                    'year': p.year,
                    'make': make,
                    'model': model,
                    'odometer': p.odometer
                }])
                ml_price = price_model.predict(pred_df)[0]
                ml_price = apply_hybrid(max(ml_price, 0))
                
                return {
                    "status": "success",
                    "mode": "ml_model",
                    "title": title,
                    "price": round(ml_price, 0),
                    "comparables": 0,
                    "note": "Using ML model (no direct comparables)"
                }
            except:
                pass
        
        return {
            "status": "no_comparables",
            "title": title,
            "price": None,
            "comparables": 0
        }
    
    if n == 1:
        base = comps.iloc[0]
        price = calculate_mileage_adjustment(p.odometer, base.odometer, base.price)
        price = apply_hybrid(price)
        
        if p.actual_price:
            save_feedback(p.dict(), price, p.actual_price)
        
        return {
            "status": "success",
            "mode": "single_comparable",
            "title": title,
            "price": round(price, 0),
            "comparables": 1,
            "used_years": [int(base.year)],
            "base_price": float(base.price),
            "mileage_adjustment": round(price - base.price, 0)
        }
    
    adjusted_prices = []
    for _, comp in comps.iterrows():
        adjusted_price = calculate_mileage_adjustment(
            p.odometer, comp.odometer, comp.price
        )
        adjusted_prices.append(adjusted_price)
    
    mean_price = np.mean(adjusted_prices)
    median_price = np.median(adjusted_prices)
    
    weights = []
    for _, comp in comps.iterrows():
        year_diff = abs(comp.year - p.year)
        weight = 1 / (1 + year_diff)
        weights.append(weight)
    
    weighted_price = np.average(adjusted_prices, weights=weights)
    
    if n >= 5:
        trimmed_mean = np.mean(sorted(adjusted_prices)[1:-1]) if n > 2 else mean_price
        final_price = trimmed_mean
    else:
        final_price = weighted_price
    
    final_price = apply_hybrid(max(final_price, 0))
    
    if n >= 3:
        std_price = np.std(adjusted_prices, ddof=1)
        t_val = t.ppf((1 + CONFIDENCE) / 2, df=n - 1)
        margin = t_val * std_price / np.sqrt(n)
        
        low = max(final_price - margin, 0)
        high = final_price + margin
    else:
        low = high = final_price
    
    if p.actual_price:
        save_feedback(p.dict(), final_price, p.actual_price)
    
    return {
        "status": "success",
        "mode": f"multiple_comparables_{'weighted' if n < 5 else 'trimmed'}",
        "title": title,
        "price": round(final_price, 0),
        "low": round(low, 0),
        "high": round(high, 0),
        "comparables": n,
        "used_years": sorted(comps.year.unique().astype(int).tolist()),
        "price_range": f"{min(adjusted_prices):.0f} - {max(adjusted_prices):.0f}",
        "std_dev": round(np.std(adjusted_prices), 0) if n >= 2 else None
    }

@app.get("/")
def health():
    return {
        "status": "ok", 
        "records": len(df),
        "model_trained": price_model.is_trained,
        "feedback_count": get_feedback_count()
    }

@app.post("/estimate")
def estimate(p: EstimateRequest):
    result = estimate_value(p)
    return result

@app.post("/estimate/batch")
def estimate_batch(vehicles: List[EstimateRequest]):
    return [estimate_value(v) for v in vehicles]

@app.post("/feedback")
def submit_feedback(feedback: FeedbackRequest):
    save_feedback(
        feedback.request.dict(),
        feedback.predicted_price,
        feedback.actual_price
    )
    return {"status": "feedback_received"}

@app.get("/model/retrain")
def retrain_model():
    retrain_model_with_feedback()
    return {"status": "retraining_triggered"}

@app.get("/model/status")
def model_status():
    feedback_count = get_feedback_count()
    
    return {
        "status": "trained" if price_model.is_trained else "untrained",
        "feedback_samples": feedback_count,
        "training_samples": len(df),
        "next_retraining_at": max(50 - feedback_count, 0) if feedback_count < 50 else "ready"
    }

def get_feedback_count() -> int:
    try:
        with open(FEEDBACK_PATH, 'r') as f:
            feedback_data = json.load(f)
        return len(feedback_data)
    except (FileNotFoundError, json.JSONDecodeError):
        return 0

def initial_model_training():
    if len(df) >= 20:
        try:
            X = df[['year', 'make', 'model', 'odometer']]
            y = df['price']
            price_model.train(X, y)
            print(f"Initial model trained with {len(df)} samples")
        except Exception as e:
            print(f"Initial training failed: {e}")

initial_model_training()

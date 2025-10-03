import pandas as pd
from prophet import Prophet
from fastapi import HTTPException
import joblib
import os

# Load pre-trained model if it exists
PRETRAINED_MODEL_PATH = "models/prophet_model.pkl"
pretrained_model = None
if os.path.exists(PRETRAINED_MODEL_PATH):
    try:
        pretrained_model = joblib.load(PRETRAINED_MODEL_PATH)
    except Exception as e:
        print(f"Warning: Failed to load pre-trained model: {e}")
        pretrained_model = None

def run_forecast(req, use_pretrained=True):
    """
    Generate a monthly demand forecast for a product.
    
    Args:
        req: ForecastRequest object from FastAPI
        use_pretrained: bool, whether to try using pre-trained model first
    
    Returns:
        dict with product_id and monthly forecast
    """
    # Convert sales records to DataFrame
    df = pd.DataFrame([{"ds": pd.to_datetime(r.date), "y": r.quantity} for r in req.sales])

    if df.empty:
        raise HTTPException(status_code=400, detail="No sales data provided.")

    # Decide which model to use
    if use_pretrained and pretrained_model is not None:
        model = pretrained_model
        # Note: Prophet cannot incrementally update with new data, so we warn if request has new sales
        if df["ds"].max() > model.history['ds'].max():
            # Optional: retrain on new data if recent sales exist
            model = Prophet()
            model.fit(df)
    else:
        model = Prophet()
        model.fit(df)

    # Forecast horizon in days
    days = req.months * 30
    future = model.make_future_dataframe(periods=days, freq="D")
    forecast = model.predict(future)

    # Prevent negative demand
    forecast.loc[forecast["yhat"] < 0, "yhat"] = 0.0

    # Only take future months after last observed date
    last_date = df["ds"].max()
    fc_future = forecast[forecast["ds"] > last_date].copy()

    # Aggregate daily predictions into monthly sums
    monthly = fc_future.set_index("ds")["yhat"].resample("M").sum().head(req.months)
    forecast_list = [
        {"month": d.strftime("%Y-%m"), "demand": int(v)}
        for d, v in monthly.items()
    ]

    return {
        "product_id": req.product_id,
        "forecast": forecast_list
    }

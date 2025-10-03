import pandas as pd
from fastapi import HTTPException

def run_forecast(req, model):
    # Convert sales records to DataFrame
    df = pd.DataFrame([{"ds": pd.to_datetime(r.date), "y": r.quantity} for r in req.sales])

    if df.empty:
        raise HTTPException(status_code=400, detail="No sales data provided.")

    # Forecast horizon
    days = int(req.months * 30)
    future = model.make_future_dataframe(periods=days, freq="D")
    forecast = model.predict(future)

    # Prevent negative demand
    forecast.loc[forecast["yhat"] < 0, "yhat"] = 0.0

    # Only take future months after last observed date
    last_date = df["ds"].max()
    fc_future = forecast[forecast["ds"] > last_date].copy()

    # Aggregate by month
    monthly = fc_future.set_index("ds")["yhat"].resample("M").sum().head(req.months)
    forecast_list = [
        {"month": d.strftime("%Y-%m"), "demand": int(v)}
        for d, v in monthly.items()
    ]

    return {
        "product_id": req.product_id,
        "forecast": forecast_list
    }

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from datetime import date
from prophetAI import run_forecast  # our business logic

app = FastAPI(title="AI Backend API", version="1.0")

class SaleRecord(BaseModel):
    date: date
    quantity: int

class ForecastRequest(BaseModel):
    product_id: str
    sales: List[SaleRecord]
    months: int = 3  # default forecast horizon

@app.get("/")
def root():
    return {"status": "ok", "message": "Go to /docs to try the API"}

@app.post("/demand")
def forecast_post(req: ForecastRequest):
    return run_forecast(req)

# Fayrouz and Mayar add your endpoint details here
@app.get('/similar-search')
def similar_search_get():
    return
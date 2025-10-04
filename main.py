# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from datetime import date
from prophetAI import run_forecast
from labse_ai import generate_embedding, find_similar_items
import joblib
import os

app = FastAPI(title="AI Backend API", version="1.0")

# ---------- Demand DTOs ----------
class SaleRecord(BaseModel):
    date: date
    quantity: int

class ForecastRequest(BaseModel):
    product_id: str
    sales: List[SaleRecord]
    months: int = 3


# ---------- Smart Search DTOs ----------
class Candidate(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    category_name: Optional[str] = None
    embedding: Optional[List[float]] = None

class SimilarSearchRequest(BaseModel):
    text: str                       # required: the text used in the request -> written by the user to search for similar items
    item_id: Optional[str] = None   # optional: the item being searched id (if any)
    embedding: Optional[List[float]] = None  # optional: the item being searched embedding (if any)
    candidates: List[Candidate]

class EmbedRequest(BaseModel):
    name: str
    description: Optional[str] = None
    category_name: Optional[str] = None


# ---------- Prophet endoint ----------
PROP_MODEL_PATH = "models/prophet_model.pkl"
prophet_model = None
if os.path.exists(PROP_MODEL_PATH):
    try:
        prophet_model = joblib.load(PROP_MODEL_PATH)
    except Exception as e:
        print(f"Warning: failed to load Prophet model: {e}")
        prophet_model = None


@app.get("/")
def root():
    return {"status": "ok", "message": "Go to /docs to try the API"}


@app.post("/demand")
def forecast_post(req: ForecastRequest):
    if prophet_model is None:
        raise HTTPException(status_code=502, detail="Prophet model not available")
    return run_forecast(req, use_pretrained=True)


# ---------- LaBSE endpoints ----------
@app.post("/similar-search")
def similar_search_post(req: SimilarSearchRequest):
    try:
        # req.dict() is not required — pass fields explicitly
        results = find_similar_items(
            text=req.text,
            item_id=req.item_id,
            embedding=req.embedding,
            candidates=[c.dict() for c in req.candidates],
            top_k=10,
        )
        return results
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to process similarity search: {e}")


@app.post("/embed")
def embed_item_post(req: EmbedRequest):
    try:
        emb = generate_embedding(req.name, req.description, req.category_name)
        return {"embedding": emb}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to generate embedding: {e}")

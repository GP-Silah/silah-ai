# labse_ai.py
import os
import joblib
from typing import List, Dict, Optional
from fastapi import HTTPException
import torch
from sentence_transformers import SentenceTransformer, util

# Paths / constants
TRAINED_MODEL_PATH = "models/labse_finetuned.pkl"
PRETRAINED_MODEL_NAME = "sentence-transformers/LaBSE"

# Try to load fine-tuned model (joblib)
finetuned_model = None
if os.path.exists(TRAINED_MODEL_PATH):
    try:
        finetuned_model = joblib.load(TRAINED_MODEL_PATH)
        print("Loaded fine-tuned LaBSE model from disk.")
    except Exception as e:
        print(f"Warning: Failed to load fine-tuned LaBSE model: {e}")
        finetuned_model = None

# Load pretrained LaBSE (fallback)
pretrained_model = None
try:
    pretrained_model = SentenceTransformer(PRETRAINED_MODEL_NAME)
    print("Loaded pretrained LaBSE model.")
except Exception as e:
    print(f"Warning: Failed to load pretrained LaBSE model: {e}")
    pretrained_model = None


def get_model():
    """
    Returns the model instance to use: finetuned if available else pretrained.
    Raises HTTPException(502) if neither is available.
    """
    if finetuned_model is not None:
        return finetuned_model
    if pretrained_model is not None:
        return pretrained_model
    raise HTTPException(status_code=502, detail="No embedding model available")


def _to_tensor(x) -> torch.Tensor:
    """
    Convert numpy array / list / torch.Tensor to torch.Tensor on CPU.
    """
    if isinstance(x, torch.Tensor):
        return x
    return torch.tensor(x)


def generate_embedding(name: str, description: Optional[str] = None, category_name: Optional[str] = None) -> List[float]:
    """
    Create a single embedding for concatenated text.
    Returns a Python list of floats.
    """
    text_parts = [name]
    if description:
        text_parts.append(description)
    if category_name:
        text_parts.append(category_name)
    text = " ".join([p for p in text_parts if p and p.strip()]).strip()

    if not text:
        raise HTTPException(status_code=400, detail="Name (or name+description+category) required for embedding")

    model = get_model()

    # Try to get a tensor result; if the loaded model doesn't support convert_to_tensor, fallback to numpy
    try:
        emb = model.encode(text, convert_to_tensor=True)
        emb = emb.detach().cpu().numpy()
    except TypeError:
        emb = model.encode(text, convert_to_numpy=True)
    except Exception:
        # as a final fallback try encode without arguments
        emb = model.encode(text)
        if isinstance(emb, torch.Tensor):
            emb = emb.cpu().numpy()

    return emb.tolist()


def find_similar_items(
    text: str,
    item_id: Optional[str],
    embedding: Optional[List[float]],
    candidates: List[Dict],
    top_k: int = 10,
) -> List[Dict]:
    """
    Main similarity function.
    - text: the search text (string) — always present.
    - item_id: optional
    - embedding: optional embedding for the query (list of floats)
    - candidates: list of dicts with keys:
        - id (str), name (str), description (opt), category_name (opt), embedding (opt list)
    Returns top_k results sorted by similarity as:
        [{"id": ..., "rank": 1, "score": 0.87}, ...]
    """

    if not isinstance(candidates, list) or len(candidates) == 0:
        raise HTTPException(status_code=400, detail="Candidates list required")

    model = get_model()

    # 1) Build query embedding
    try:
        if embedding:
            # Convert provided embedding to tensor
            query_emb = _to_tensor(embedding).float()
            if query_emb.dim() == 1:
                query_emb = query_emb.unsqueeze(0)  # shape (1, dim)
        else:
            # compute from text
            try:
                query_emb = model.encode(text, convert_to_tensor=True)
            except TypeError:
                # model doesn't accept convert_to_tensor param
                q_np = model.encode(text, convert_to_numpy=True)
                query_emb = torch.tensor(q_np)
            if query_emb.dim() == 1:
                query_emb = query_emb.unsqueeze(0)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to compute query embedding: {e}")

    # 2) Prepare candidate embeddings: use provided ones where available; compute missing ones in batch
    candidate_emb_tensors = [None] * len(candidates)
    missing_texts = []
    missing_indices = []

    for idx, c in enumerate(candidates):
        if c.get("embedding"):
            try:
                tensor = _to_tensor(c["embedding"]).float()
                if tensor.dim() == 1:
                    tensor = tensor.unsqueeze(0)
                candidate_emb_tensors[idx] = tensor
            except Exception:
                # if conversion fails, treat as missing
                missing_indices.append(idx)
                missing_texts.append(" ".join(filter(None, [c.get("name",""), c.get("description") or "", c.get("category_name") or ""])).strip())
        else:
            missing_indices.append(idx)
            missing_texts.append(" ".join(filter(None, [c.get("name",""), c.get("description") or "", c.get("category_name") or ""])).strip())

    # compute missing embeddings in batch (if any)
    if missing_indices:
        try:
            try:
                missing_embs = model.encode(missing_texts, convert_to_tensor=True)
            except TypeError:
                missing_embs_np = model.encode(missing_texts, convert_to_numpy=True)
                missing_embs = torch.tensor(missing_embs_np)
            # missing_embs should be shape (m, dim)
            for i, idx in enumerate(missing_indices):
                vec = missing_embs[i]
                if vec.dim() == 1:
                    vec = vec.unsqueeze(0)
                candidate_emb_tensors[idx] = vec
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Failed to compute candidate embeddings: {e}")

    # 3) Stack candidate tensors into a single tensor (n, dim)
    try:
        # ensure all candidate_emb_tensors are set
        candidate_embs = [c.squeeze(0) if c.dim()==2 and c.size(0)==1 else c for c in candidate_emb_tensors]
        candidate_matrix = torch.stack(candidate_embs)  # shape (n, dim)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create candidate embedding matrix: {e}")

    # 4) Compute cosine similarity
    try:
        # util.cos_sim returns (1, n) for query_emb (1, dim) vs candidate_matrix (n, dim)
        sims = util.cos_sim(query_emb, candidate_matrix)[0]  # tensor length n
        sims_np = sims.detach().cpu().numpy().tolist()
    except Exception as e:
        # fallback to manual numpy cosine if util fails
        try:
            import numpy as np
            q = query_emb.detach().cpu().numpy().reshape(-1)
            cm = candidate_matrix.detach().cpu().numpy()
            dot = (cm @ q)
            qnorm = (q ** 2).sum() ** 0.5
            cmnorm = (cm ** 2).sum(axis=1) ** 0.5
            sims_np = (dot / (cmnorm * qnorm + 1e-10)).tolist()
        except Exception as e2:
            raise HTTPException(status_code=502, detail=f"Failed to compute similarity: {e} / {e2}")

    # 5) Build result list and sort
    results = []
    for idx, c in enumerate(candidates):
        score = float(sims_np[idx]) if idx < len(sims_np) else 0.0
        results.append({"id": c.get("id"), "score": score})

    results = [r for r in results if r["id"] is not None]
    results.sort(key=lambda x: x["score"], reverse=True)

    # 6) Take top_k and assign ranks (1 = most similar)
    top = results[:top_k]
    out = []
    for i, r in enumerate(top):
        out.append({"id": r["id"], "rank": i + 1, "score": r["score"]})

    return out

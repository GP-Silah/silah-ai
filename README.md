# Silah AI Backend

_Last Updated: October 2025_

This repository contains the backend for the **Silah** project AI features.  
It currently provides:

- **Demand forecasting** using a pre-trained Facebook Prophet model
- **semantic similarity search** using a fine-tuned LaBSE model and cosine similarity algorithem
- FastAPI endpoints to expose these features to NestJS backend service

This is a production-ready setup that **uses a pre-trained model**, so there is no training happening on the server.

---

## Libraries Used

- [`fastapi`](https://fastapi.tiangolo.com/) – Web framework for building APIs

- [`uvicorn`](https://www.uvicorn.org/) – ASGI server to run FastAPI

- [`pandas`](https://pandas.pydata.org/) – Data loading and manipulation

- [`numpy`](https://numpy.org/) – Numerical computations

- [`prophet`](https://facebook.github.io/prophet/) – Time series forecasting

- [`joblib`](https://joblib.readthedocs.io/) – Model serialization / deserialization

- [`matplotlib`](https://matplotlib.org/) – Plotting (optional, used by Prophet)

- [`torch`](https://pytorch.org/) – Required for LaBSE embeddings

- [`sentence-transformers`](https://www.sbert.net/) – Pre-trained and fine-tuned LaBSE models

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/GP-Silah/silah-ai.git
cd silah-ai
```

### 2. Create a Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows
```
This ensures all dependencies are installed in an isolated environment.

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

This installs all required Python packages for the AI backend.

### 4. Place the Pre-Trained Model

Create a `models` folder in the project root:

```arduino
silah-ai/
├─ main.py
├─ prophetAI.py
├─ models/
│  └─ prophet_model.pkl       # Optional trained Prophet model
│  └─ labse_finetuned.pkl     # optional fine-tuned LaBSE model
```

The backend will load this model at startup. If a fine-tuned model is not available for any feature (Prophet or LaBSE), the backend will automatically fall back to the corresponding pre-trained model to ensure functionality.

### 5. Run the FastAPI Server

```bash
uvicorn main:app --reload
```

Visit `http://127.0.0.1:8000/docs` for the interactive Swagger UI documentation.

---

## API Endpoints

| Endpoint               | Method | Description                                                                           |
|------------------------|--------|---------------------------------------------------------------------------------------|
| `/`                    | GET    | Health check / root                                                                   |
| `/demand`              | POST   | Returns a monthly demand forecast for a given product based on sales records          |
| `/similar-search`      | POST   | Returns top-N most similar items for a given query and optional candidates            |
| `/embed`               | POST   | Returns an embedding vector for a given item (`name`, `description`, `category_name`) |

### `/demand` Endpoint:

#### Example Request Body:

```json
{
  "product_id": "239284-4324-efsd",
  "months": 3,
  "sales": [
    {"date": "2025-09-01", "quantity": 5},
    {"date": "2025-09-02", "quantity": 2}
  ]
}
```
#### Example Response Body:

```json
{
  "product_id": "239284-4324-efsd",
  "forecast": [
    {"month": "2025-10", "demand": 30},
    {"month": "2025-11", "demand": 25},
    {"month": "2025-12", "demand": 28}
  ]
}
```

#### `/similar-search` Endpoint:

#### Example Request Body:

```json
{
  "text": "Dog food for puppies",
  "item_id": null,
  "embedding": null,
  "candidates": [
    {
      "id": "prod-123",
      "name": "Premium Puppy Food",
      "description": "High protein puppy food",
      "category_name": "Pet Supplies",
      "embedding": null
    }
  ]
}
```

#### Example Response Body:

```json
[
  {"id": "prod-123", "rank": 1, "score": 0.87}
]
```

### `/embed` Endpoint:

#### Example Request Body:

```json
{
  "name": "Premium Puppy Food",
  "description": "High protein puppy food",
  "category_name": "Pet Supplies"
}
```

#### Example Response Body:

```json
{
  "embedding": [0.123, 0.432, 0.987, ...]
}
```

---

## Files Overview

| File                         | Description                                                                         |
|------------------------------|-------------------------------------------------------------------------------------|
| `main.py`                    | FastAPI entry point; defines endpoints and loads the pre-trained models             |
| `prophetAI.py`               | Business logic for demand forecasting using the pre-trained Prophet model           |
| `labse_ai.py`                | Business logic for semantic similarity search and embedding generation using LaBSE. |
| `models/prophet_model.pkl`   | Trained Prophet model (loaded at startup)                                           |
| `models/labse_finetuned.pkl` | fine-tuned LaBSE model (loaded at startup)                                          |
| `requirements.txt`           | Python dependencies                                                                 |
| `README.md`                  | Project documentation                                                               |

---

> Built with care as part of a graduation project requirement, by an amazing team.

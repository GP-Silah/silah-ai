# Silah AI Backend

This repository contains the backend for the **Silah** project AI features.  
It currently provides:

- **Demand forecasting** using a pre-trained Facebook Prophet model
- **semantic similarity search** using a fine-tuned LaBSE model and cosine similarity algorithem
- FastAPI endpoints to expose these features to NestJS backend service

This is a production-ready setup that **uses a pre-trained model**, so there is no training happening on the server.

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

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Place the Pre-Trained Model

Create a `models` folder in the project root:

```arduino
silah-ai/
├─ main.py
├─ prophetAI.py
├─ models/
│  └─ prophet_model.pkl   # pre-trained Prophet model
```

The backend will load this model at startup.

### 5. Run the FastAPI Server

```bash
uvicorn main:app --reload
```

Visit `http://127.0.0.1:8000/docs` for the interactive Swagger UI.

---
## API Endpoints

| Endpoint           | Method | Description                                                      |
|-------------------|--------|------------------------------------------------------------------|
| `/`               | GET    | Health check / root                                              |
| `/demand`         | POST   | Returns a monthly demand forecast for a given product based on sales records |

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
#### Response Body:

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

## Files Overview

| File                       | Description                                                                 |
|----------------------------|-----------------------------------------------------------------------------|
| `main.py`                  | FastAPI entry point, defines endpoints and loads the pre-trained model     |
| `prophetAI.py`             | Business logic for demand forecasting using the pre-trained Prophet model  |
| `models/prophet_model.pkl` | Pre-trained Prophet model (loaded at startup)                               |
| `requirements.txt`         | Python dependencies                                                         |
| `README.md`                | This file                                                                   |


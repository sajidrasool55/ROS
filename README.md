# Orders Processing Web App

FastAPI + SQLite + Pandas web app to:

- Upload and persist `Calculator.xlsx` (shipping references).
- Upload `orders.csv` and cutoff order.
- Clean and process orders with shipping/handling logic.
- Detect and fix missing AU postcodes in browser.
- Download `cleaned_orders.xlsx`.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --reload
```

Open <http://127.0.0.1:8000>.

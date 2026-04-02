# Feature Runner App

This project includes:

- A FastAPI backend with `GET /features` and `POST /run`
- A React + Tailwind frontend with S3 URL verification, model run flow, and save-to-local plus save-to-S3 actions

## Project Structure

```text
project/
├── backend/
│   ├── main.py
│   ├── features.py
│   ├── runner.py
│   └── requirements.txt
├── frontend/
│   ├── index.html
│   ├── package.json
│   ├── postcss.config.js
│   ├── tailwind.config.js
│   ├── vite.config.js
│   └── src/
│       ├── App.jsx
│       ├── MainUI.jsx
│       ├── index.css
│       └── main.jsx
└── README.md
```

## Backend Setup

1. Open a terminal in `backend/`
2. Create and activate a virtual environment if you want one
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Start the API:

```bash
uvicorn main:app --reload
```

The backend runs at `http://127.0.0.1:8000`.

## Frontend Setup

1. Open a second terminal in `frontend/`
2. Install dependencies:

```bash
npm install
```

3. Start the frontend:

```bash
npm run dev
```

The frontend runs at `http://127.0.0.1:5173`.

## Edit Your Python Logic

### `backend/features.py`

Replace the placeholder `get_features()` function with your own logic that returns a list of feature names:

```python
def get_features() -> list[str]:
    return ["feature_1", "feature_2", "feature_3"]
```

### `backend/runner.py`

Replace the placeholder `run_model()` function with your own logic that accepts the selected features and returns JSON-serializable model output:

```python
def run_model(selected_features: list[str]) -> dict:
    return {
        "accuracy": 0.92,
        "message": "model ran successfully",
        "selected_features": selected_features,
    }
```

## Save Output Format

When you click `Save`, the frontend creates this JSON payload automatically:

```json
{
  "user": "Alice",
  "timestamp": "2024-01-15T14:32:00.000Z",
  "notes": "Selected high correlation features",
  "features_used": ["age", "income", "score"],
  "output": {
    "accuracy": 0.92,
    "message": "model ran successfully"
  }
}
```

## Notes

- CORS is enabled for the FastAPI app.
- The S3 upload uses the presigned URL directly from the browser.
- The timestamp is generated automatically at save payload creation time.
- The local save is handled as a direct browser download.
- The verification screen sends a direct request to the provided presigned URL, so the URL should be valid for `PUT` uploads.

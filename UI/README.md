# MindMetrics — Feature Runner UI

Interactive web tool for running stress prediction models with selected feature subsets and saving results.

## Structure

```
ui/
├── backend/
│   ├── main.py          # FastAPI app — /features, /run, /save endpoints
│   ├── features.py      # Reads features.json and returns categories
│   ├── features.json    # Feature list organized by category with tooltips
│   └── runner.py        # Calls model_comparison.run_all() and reads result CSVs
└── frontend/
    └── src/
        ├── LandingPage.jsx   # Screen 1 — project intro
        ├── App.jsx           # Screen 2 — S3 URL verification
        └── MainUI.jsx        # Screen 3 — feature selection, run, results
```

## Running

**Backend** — from `ui/backend/`:
```bash
uvicorn main:app --reload
```
Runs at `http://127.0.0.1:8000`

**Frontend** — from `ui/frontend/`:
```bash
npm install   # first time only
npm run dev
```
Runs at `http://127.0.0.1:5173`

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/features` | Returns feature categories with tooltips from `features.json` |
| POST | `/run` | Trains all 4 models with selected features, returns metrics |
| POST | `/save` | Saves run JSON to `results/runs/<user>_<timestamp>.json` |
| GET | `/results/<file>` | Serves static plot images from `results/metrics/` |

## Run Output

```json
{
  "selected_features": ["RR", "PNSindex", "Cadence"],
  "model_metrics": [
    { "Model": "XGBoost", "Accuracy": 0.86, "Precision": 0.84, "Recall": 0.83, "F1-Score": 0.83, "TN": 5458, "FP": 799, "FN": 747, "TP": 3951 }
  ],
  "fold_metrics": [
    { "model": "XGBoost", "fold": 1, "accuracy": 0.87, "precision": 0.90, "recall": 0.74, "f1_score": 0.81, "roc_auc": 0.96 }
  ]
}
```

## Saved Run Format

Each run auto-saves to `results/runs/<user>_<timestamp>.json`:

```json
{
  "user": "Harish",
  "timestamp": "2026-04-05T12:00:00.000Z",
  "notes": "Testing HRV features only",
  "features_used": ["RR", "PNSindex", "SNSindex"],
  "features_excluded": ["Cadence", "Speed", "BMI"],
  "output": { "model_metrics": [...], "fold_metrics": [...] }
}
```

## Notes

- Runs take several minutes — a live timer shows elapsed time on the Run button
- Results auto-save locally and attempt S3 upload simultaneously after each run
- S3 upload shows a Retry button if it fails
- Plot images in the UI are always from the latest run (`results/metrics/`)
- Full run history is preserved in `results/runs/`

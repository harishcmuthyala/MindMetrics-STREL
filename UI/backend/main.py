import sys
import os
import json
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src/evaluation'))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from features import get_features
from runner import run_model

METRICS_DIR = os.path.join(os.path.dirname(__file__), '../../results/metrics')
RUNS_DIR    = Path(os.path.join(os.path.dirname(__file__), '../../results/runs'))
RUNS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Feature Runner API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/results", StaticFiles(directory=METRICS_DIR), name="results")


class RunRequest(BaseModel):
    selected_features: list[str]
    user: str
    notes: str


class SaveRequest(BaseModel):
    user: str
    notes: str
    selected_features: list[str]
    excluded_features: list[str]
    output: dict
    timestamp: str


@app.get("/features")
def features() -> dict:
    return get_features()


@app.post("/run")
def run(request: RunRequest):
    return run_model(request.selected_features)


@app.post("/save")
def save(request: SaveRequest):
    safe_user = request.user.strip().replace(" ", "_") or "user"
    ts = request.timestamp.replace(":", "-").replace(".", "-")
    run_name = f"{safe_user}_{ts}"

    payload = {
        "user": request.user,
        "timestamp": request.timestamp,
        "notes": request.notes,
        "features_used": request.selected_features,
        "features_excluded": request.excluded_features,
        "output": request.output,
    }

    # JSON sits outside, plots folder has same name
    json_file = RUNS_DIR / f"{run_name}.json"
    json_file.write_text(json.dumps(payload, indent=2))
    return {"saved_as": str(json_file)}

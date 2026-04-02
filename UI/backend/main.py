from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from features import get_features
from runner import run_model


app = FastAPI(title="Feature Runner API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RunRequest(BaseModel):
    selected_features: list[str]
    user: str
    notes: str


@app.get("/features")
def features() -> dict[str, list[str]]:
    return {"features": get_features()}


@app.post("/run")
def run(request: RunRequest):
    return run_model(request.selected_features)

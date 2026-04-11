import json
from pathlib import Path

FEATURES_PATH = Path(__file__).resolve().parent / "features.json"


def get_features() -> dict:
    return json.loads(FEATURES_PATH.read_text())

from model_service import load_model_once
from fastapi import FastAPI
from pathlib import Path

MODEL_PATH = (Path(__file__).parent.parent / "week07" / "model.json").resolve()

load_model_once(MODEL_PATH)

app = FastAPI()

@app.get('/health')
def health():
    info = load_model_once(MODEL_PATH)
    return {'status':'ok','model_version':info['model_version']}
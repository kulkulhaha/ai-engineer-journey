from model_service import load_model_once, predict_proba_from_model
from fastapi import FastAPI, HTTPException
from pathlib import Path
from pydantic import BaseModel, Field
import pandas as pd

MODEL_PATH = (Path(__file__).parent.parent / "week07" / "model.json").resolve()

load_model_once(MODEL_PATH)

app = FastAPI()

@app.get('/health')
def health():
    info = load_model_once(MODEL_PATH)
    return {'status':'ok','model_version':info['model_version']}

# week08/main.py

class PredictRequest(BaseModel):
    """
    예측 요청 본문.

    rows: 각 행이 {열 이름: 숫자}인 딕셔너리의 리스트. 열 순서는 보장되지 않는다.
          최소 1행, 최대 100행.
    예: {"rows": [{"mean radius": 17.99, "mean texture": 10.38}]}
    """
    rows: list[dict[str,float]] = Field(min_length=1, max_length=100)


class PredictResponse(BaseModel):
    """
    예측 응답 본문.

    probabilities: 각 행의 양성 확률, 입력 행과 같은 순서.
    labels: threshold를 적용한 0/1 라벨, probabilities와 같은 길이.
    model_version: 이 예측을 낸 모델의 버전 문자열.
    """
    probabilities: list[float]
    labels: list[int]
    model_version: str

@app.post('/predict')
def predict(request: PredictRequest):
    df = pd.DataFrame(request.rows)
    model = load_model_once(MODEL_PATH)
    try:
        result = predict_proba_from_model(model, df)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    response = PredictResponse(probabilities=result, labels=(result >= model['threshold']), model_version=model['model_version']) 
    return response

    
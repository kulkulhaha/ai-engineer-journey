import json
import numpy as np

def sigmoid(z: np.ndarray) -> np.ndarray:
    pos_mask = z>=0
    neg_mask = ~pos_mask
    result = z.astype(float)
    result[pos_mask] =  1/(1+np.exp(-result[pos_mask]))
    result[neg_mask] = np.exp(result[neg_mask])/(1+np.exp(result[neg_mask]))
    return result

def load_model(path: str) -> dict:
    REQUIRED_KEYS = {"model_version", "train_at", "feature_names", "coefficients",
                 "intercept", "scaler", "threshold", "metrics"}
    
    with open(path, 'r', encoding='utf-8') as f:
        result = json.load(f)
    for i in REQUIRED_KEYS:
        if i not in result:
            raise ValueError(f"{i} is not in file")
    result["coefficients"] = np.array(result["coefficients"],dtype=float)
    result["scaler"]["mean"] = np.array(result["scaler"]["mean"],dtype=float)
    result["scaler"]["scale"] = np.array(result["scaler"]["scale"],dtype=float)
    n = len(result["feature_names"])
    for key, arr in [("coefficients", result["coefficients"]),
                    ("scaler.mean", result["scaler"]["mean"]),
                    ("scaler.scale", result["scaler"]["scale"])]:
        if len(arr) != n:
            raise ValueError(f"{key} 길이 {len(arr)} != feature_names {n}")
    return result


def predict_proba_from_model(model: dict, df) -> np.ndarray:
    names = model["feature_names"]
    missing = [c for c in names if c not in df.columns]
    if missing:
        raise ValueError(f"df에 없는 열: {missing}")
    sorted_df = df[names]
    scaled = (sorted_df-model['scaler']['mean'])/model['scaler']['scale']
    z = scaled.to_numpy(dtype=float) @ model["coefficients"] + model["intercept"]
    return sigmoid(z)

from functools import lru_cache

@lru_cache(maxsize=None)
def load_model_once(path: str) -> dict:
    """
    모델 JSON을 한 번만 읽어 캐시하고, 이후 호출에는 같은 객체를 돌려준다.

    path: W7이 저장한 모델 JSON 경로.
    반환: load_model(path)의 반환값과 같은 구조의 딕셔너리.
          같은 path로 두 번 호출하면 두 반환값이 같은 객체여야 한다.
    functools.lru_cache이용 : dict으로 직접구현보다 데코레이터를 이용 비용을 줄임. 대신 캐시 내부 탐색 불가
    """
    return load_model(path)

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="인자 테스트 스크립트")
    parser.add_argument('--path', type=str, required=True, help='파일을 입력하세요')
    args = parser.parse_args()
    result1 = load_model_once(args.path)
    result2 = load_model_once(args.path)
    print(result1 is result2)
    print(isinstance(result1, dict))
    print(result1["model_version"])
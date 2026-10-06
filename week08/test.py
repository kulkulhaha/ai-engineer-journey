import json, pandas as pd
from sklearn.datasets import load_breast_cancer

X = load_breast_cancer(as_frame=True).frame.drop(columns="target")
print(json.dumps({"rows": X.iloc[[0]].to_dict(orient="records")}, indent=2))
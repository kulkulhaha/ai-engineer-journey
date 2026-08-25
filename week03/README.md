# W3: 로지스틱 회귀 & 분류 평가

## 진행 상황 (2026-08-25 기준 — 진행 중)

**완료:**
- Day1: 클래스 불균형 베이스라인 & 특성 스케일이 로지스틱 회귀에 미치는 영향 확인, `LogisticRegression`(sklearn) 학습 + 혼동행렬 + ROC/AUC (`LogisticRegression_+_ConfusionMatrix_+_ROC.ipynb`) — `predict_proba` shape/의미, data leakage(스케일러를 test에 다시 fit하면 안 되는 이유), AUC와 threshold에 따라 달라지는 recall의 구분까지 확인
- Day2: 베이즈 정리 직접 계산(사전확률 vs 사후확률 그래프) + 동전 던지기 로그우도 MLE (`Bayes_theorem+MLE.ipynb`), 베르누이 우도 → 로그우도 → NLL(=cross-entropy) 동치성 유도
- Day3: `sigmoid`(오버플로 방지 포함) + `train_logistic_regression`(배치 경사하강) 직접 구현, `sklearn.LogisticRegression`과 테스트 정확도·log_loss 비교까지 완료 (`GradientDescent_LogisticRegression.ipynb`) — 여러 차례 디버깅(리스트/타입 문제, shape mismatch, 부호 반전 2회, 이중 정규화, `sigmoid` 부호 누락)을 거쳐 완성. 정규화 없는 sklearn과 log_loss가 소수점 여러 자리까지 일치함을 확인해 구현을 검증함

**더 공부 필요 / 다음에 다시 볼 것:**
- 로지스틱 회귀에서 MSE 대신 cross-entropy를 쓰는 이유 (키워드: sigmoid+MSE 결합 시 손실함수의 non-convexity, gradient vanishing 구간)
- log(0) 방지 방식 비교: `np.clip(p, eps, 1-eps)`(gradient에도 영향, 극단적으로 확신하는 샘플에 미세한 힘이 계속 남음) vs `log(p+eps)`(gradient는 그대로, 확신한 샘플은 완전히 0으로 수렴) — 트레이드오프는 확인했으나 실제 채택 기준은 미정

**다음 액션:**
- 로지스틱 회귀에서 MSE 대신 cross-entropy를 쓰는 이유 이어서 보기

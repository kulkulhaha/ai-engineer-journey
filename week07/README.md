# W7: Gradient Descent from Scratch | 경사하강법 직접 구현

> ⚠️ **초안** — 2026-09-30에 폴더의 노트북 코드·저장된 출력을 읽고 소급 작성. 실제 진행 날짜와 완료 판정은 직접 확인·수정할 것.

## 진행 상황 (2026-09-30 기준 — 진행 중)

**완료 (저장된 출력으로 확인됨):**
- Day1: `numerical_gradient` (중심차분) — 검사 3개 PASS (`f=v0²+v1² → [2,4]`, 성분별 섭동, `x` 원본 유지) (`gradient_descent.ipynb`)
- Day2: `batch_gradient_descent` (MSE) — η = 0.001 / 0.01 / 0.5 학습곡선 비교 그래프
- Day3: `sigmoid`(큰 음수 오버플로 분기) + `minibatch_sgd_logistic` — `losses[0]=0.7326`(w=0이면 ln2≈0.693 근처), 학습 결과 `w=[0.322, 1.452, -1.872]` vs `w_true=[0.5, 1.5, -2.0]`, seed 재현 True, `batch_size>n` 처리 OK, **미니배치 최종손실 0.3934 < 풀배치 0.4877**. batch_size 1/32/64/128 곡선 비교
- Day4: `minibatch_sgd_momentum` — β = 0 / 0.9 / 0.99 곡선 비교. `minibatch_sgd_adam` 작성
- Day5: `save_model` / `load_model` / `predict_proba_from_model` (`model_export.ipynb`) — `model.json` 생성 (v0.1, feature_names 30개, scaler mean·scale, threshold 0.5, metrics acc 0.9737 / f1 0.9793 / n_test 114). 검증 3개 통과: **왕복 일치 True (최대차 3.33e-16), 열 순서 무관 True, 열 누락 시 에러 PASS**

**아직 안 한 것 / 확인 필요:**
- **Day1 h 민감도 표가 이상하다** — h = 1e-1 ~ 1e-12에서 「절대 오차」가 전부 `2.400e+01`로 동일하다. 수치 그래디언트 값은 h에 따라 바뀌는데 오차만 고정인 건 **기준값(해석적 그래디언트)이 잘못 들어갔을 가능성**. 재확인 필요. 절단오차↓·소거오차↑가 만드는 V자 그래프도 아직 없음
- **유도 기록 없음** — MSE·이진 교차엔트로피 그래디언트 손유도를 남긴 파일(`derivations.md` 등)이 폴더에 없음
- **`minibatch_sgd_adam`의 docstring이 momentum 함수 것 그대로** — β1·β2·편향보정 설명으로 교체 필요. 실제 Adam 구현 여부도 같이 확인
- **W8 인계 문제** — `save_model`/`load_model`이 노트북 안에만 있다. W8이 `import`하려면 `.py` 모듈로 빼야 함

**다음 액션:** h 민감도 표 기준값 재확인 → 그래디언트 유도 기록 → save/load를 `.py`로 분리 (W8 선행조건)

## 배운 것

- 중심차분은 h를 줄인다고 정확해지지 않는다 — 절단오차는 줄지만 소거오차가 커진다
- 같은 epoch 수에서 미니배치가 풀배치보다 손실이 낮았다 (0.3934 vs 0.4877). 한 epoch 안의 갱신 횟수가 많기 때문
- 모델을 저장할 때 계수만으로는 부족하다. `feature_names`·scaler 통계량·threshold·version이 같이 있어야 재로딩 예측이 재현되고, 열 순서가 바뀌어도 같은 결과가 나온다

## W7 완료 항목

- [x] 중심차분 그래디언트를 해석적 그래디언트와 `np.allclose`로 대조
- [ ] MSE·BCE 그래디언트 손유도 기록
- [x] 배치 GD · 미니배치 SGD · 모멘텀 SGD 학습곡선을 같은 조건에서 비교
- [ ] SGD와 Adam의 차이를 모멘텀 관점에서 설명 (구현은 있으나 설명 기록 없음)
- [x] 학습한 모델을 열 이름·버전과 함께 저장·재로딩해 같은 예측 확인

## 최소 보장 체크

- [ ] 왜 그래디언트의 반대 방향이 가장 빠르게 감소하는 방향인지 설명 가능
- [ ] SGD와 Adam의 차이를 설명 가능

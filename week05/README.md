# W5: 트리·앙상블·정보이득·중심극한정리

> ⚠️ **초안** — 2026-09-30에 폴더의 노트북 코드·저장된 출력을 읽고 소급 작성. 실제 진행 날짜와 완료 판정은 직접 확인·수정할 것.

## 진행 상황 (2026-09-30 기준 — 완료 여부 확인 필요)

**완료 (저장된 출력으로 확인됨):**
- Day1: `compare_rf_xgb` — RandomForest와 XGBoost를 같은 train/test 분할로 학습, accuracy·ROC-AUC·특성중요도 top5를 dict로 반환 (`RandomForest vs XGBoost.ipynb`)
- Day2: `entropy` — `np.unique`로 클래스 비율을 구해 `-Σ p·log2(p)` 계산, 순수 노드 0 처리 (`Entropy and Information gain.ipynb`)
- Day3: `best_split` — 특성별 정렬 유니크값의 중간점을 임계값 후보로 만들고 정보이득 최대인 `(특성 idx, 임계값, ig)` 반환. sklearn `DecisionTreeClassifier(max_depth=1, criterion="entropy")`와 대조 → **둘 다 feature 22 / threshold 105.95로 일치** (직접 구현 ig=0.5620)
- Day4: `load_wine`(3클래스)에 `compare_rf_xgb` 적용 — 이진분류 전제였던 `roc_auc_score` 부분을 다중클래스용으로 수정 (`RandomForest vs XGBoost.ipynb`)
- CLT: `simulate_clt(dist_sampler, sample_size, n_trials)` — 지수분포 샘플러로 표본평균 분포를 sample_size별 5개 패널로 시각화 (`CLT simulation.ipynb`)

**아직 안 한 것 / 확인 필요:**
- **`review.ipynb`의 `entropy` 버그** — 루프 안이 `result = ...`(누적 아님)이라 **마지막 클래스 항만 반환**한다. breast_cancer target에서 0.4219가 나왔는데 실제 엔트로피는 약 0.953. `+=`로 고쳐야 하고, 이 값을 쓰는 `information_gain`·`best_split`에 그대로 전파됨
- **`review.ipynb`의 `best_split`이 임계값을 버린다** — `(feature_idx, gain)`만 반환하고 `idxs[idx]`(해당 임계값)를 반환하지 않음. Day3 원본은 3개를 반환하므로 복습본만의 문제
- Day4 wine 결과가 accuracy·ROC-AUC 모두 1.0000 — 데이터가 쉬워서인지, 다중클래스 평균 방식 때문인지 확인 필요
- 「배운 것」·W5 회고 미작성

**다음 액션:** review.ipynb `entropy` 수정 → best_split 반환값 맞추기 → 아래 「배운 것」 채우기

## 배운 것

- (직접 작성)

## W5 완료 항목

- [x] 단일 결정트리·RandomForest·XGBoost를 같은 분할에서 비교
- [x] 엔트로피·정보이득으로 **한 번의 최적 분할**을 직접 찾고 sklearn과 대조
- [ ] 트리 복잡도를 바꿔 학습/검증 점수 차이 관찰 — 해당 실험 기록 못 찾음
- [x] 중심극한정리를 시뮬레이션으로 확인

## 최소 보장 체크

- [ ] 엔트로피가 왜 "불확실성"인지, 정보이득이 왜 분할 기준이 되는지 설명 가능
- [ ] RandomForest(배깅)와 XGBoost(부스팅)의 차이를 학습 방식 관점에서 설명 가능

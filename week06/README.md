# W6: 시계열 EDA · 최소제곱 · MLE · 회귀 평가 · 자동 리포트

> ⚠️ **초안** — 2026-09-30에 폴더의 노트북 코드·저장된 출력을 읽고 소급 작성. 실제 진행 날짜와 완료 판정은 직접 확인·수정할 것. 노트북을 실행해보지는 않았고, **저장된 출력이 있는 셀만 완료로 표시**했다.

## 진행 상황 (2026-09-30 기준 — 진행 중으로 추정)

**완료 (저장된 출력으로 확인됨):**
- `time_series_eda.ipynb` — `load_yearly_series`(연도 결측·중복 오류 처리, 원본 보존), `summarize_series`(시작/끝·행 수·결측·누락 연도·중복), sunspots 1700–1899 구간 5년 이동평균 시각화, `lag_pearson`(시차 Pearson 상관, 상수 구간 NaN), `make_lag1_table`(`previous_value`/`target`)
- `linear_regression_mle.ipynb` — `fit_least_squares`(SVD 기반, 상대 특이값 컷오프 `eps·max(n, d+1)·s_max`로 rank 결손 처리), `gaussian_log_likelihood`(차원·길이 검증 포함), `regression_metrics`(MSE·MAE·R²)
- validation 비교 — 199/50/나머지로 분할해 **지속성(persistence) 기준선 vs 선형회귀**의 지표를 같은 구간에서 대조
- `automated_eda_report.ipynb` — `generate_ts_report`가 `week06/eda/report.md`를 실제로 생성 (1991~2004, 14행, 누락 0, 결측 0)

**아직 안 한 것 / 확인 필요:**
- **경계 검증 실행 기록 없음** — `fit_least_squares`, `regression_metrics`, `make_lag1_table` 셀에 저장된 출력이 없다(`execution_count`가 비어 있음). 중복 특성·0 특성·상수 정답 같은 경계 사례를 실제로 돌린 근거를 못 찾음
- **test 구간 최종 평가 출력 미확인** — validation 비교까지만 확인됨
- **정규방정식 ↔ MLE 유도 기록 없음** — 손유도를 남긴 파일이 폴더에 없음
- **`eda/report.md`가 3줄짜리** — 계획서가 요구한 기간·빈도·행 수·누락 시점 수 + **그림 2개 상대경로 링크**가 들어가 있지 않다. `series.png`는 `week06/` 루트에 있고 리포트에서 참조되지 않음
- **리포트 구간이 데이터와 안 맞는다** — `output.csv`의 lag1 표는 1701~2008(308행)인데 `eda/report.md`는 "1991 ~ 2004 / 14행"만 담고 있다. `generate_ts_report`에 전체 Series를 넘겼는데 14행만 나온 이유, 그리고 `label`을 손으로 적은 "1991 ~ 2004"가 실제 구간과 일치하는지 확인 필요

**다음 액션:** 경계 검증 셀 실행·저장 → test 구간 평가 → report.md에 그림 링크 추가

## 배운 것

- (직접 작성 — 후보: 누락된 연도 vs NaN 관측값의 차이, 시간 순서를 섞으면 미래로 과거를 예측하게 되는 문제, 시차 Pearson 상관이 인과를 뜻하지 않는다는 점)

## W6 완료 항목

- [x] 연간 시계열의 정렬·중복·누락 시점·결측값을 구분해 로드
- [x] 공분산·시차 상관의 의미와 한계 확인
- [x] 최소제곱을 SVD로 구현하고 MLE와의 연결을 코드로 확인
- [ ] MSE·MAE·R²의 **경계 사례** 실행 기록
- [ ] test 구간 최종 평가 + 기준선 대비 결론
- [ ] 미니프로젝트 #2: 자동 EDA 리포트 (생성은 되나 내용 미완)

## 최소 보장 체크

- [ ] 정규방정식과 MLE가 왜 같은 목적함수로 이어지는지 설명 가능
- [ ] R²가 음수가 될 수 있는 경우를 설명 가능

# W6 구체적 실행 계획 (학습 가이드형)

> **주제**: Pandas 시계열 EDA + 공분산·자기상관 재활성화 + 선형회귀를 정규방정식·MLE 두 관점에서 재구현 + 미니프로젝트 #2(시계열 EDA 자동 리포트)
>
> **사용 데이터셋 주의**: 이번 주는 공개 시계열 데이터(예: 월별 항공 승객 수 데이터셋 "AirPassengers", 또는 접근이 어려우면 `yfinance`로 받은 임의 종목의 일별 종가)를 씁니다. 어떤 데이터를 쓰든 "날짜 인덱스가 있고 시간순으로 정렬된 데이터"라는 조건만 맞으면 됩니다. 데이터 출처가 매주 바뀔 수 있으니, 아래 코드의 구체적 수치는 참고용이 아니라 본인이 직접 돌려 나온 값을 근거로 삼으세요.
>
> **총 목표 시간**: 10–12시간 (v7 커리큘럼 블록 A 기준)
> **기준**: 평일 1.5시간 + 토요일 2.5시간 + 일요일 2시간
>
> **코드 제시 방식**: 완성된 실행 코드를 주지 않습니다. 학습 목표·핵심 개념·구현 요구사항(스펙)·확인 질문만 제시합니다. 코드는 스스로 작성하세요. Git/설치 같은 보일러플레이트만 그대로 제시합니다.
>
> **전제**: W1의 `numpy` 행렬 연산(전치·역행렬)과 W2의 SVD/고유값분해 감각을 이번 주 정규방정식 유도에 재사용합니다. W7(다음 주)에서는 이번 주 만든 정규방정식 결과를 SGD로 다시 학습시킨 결과와 비교하게 되니, 이번 주 코드를 잘 보관해두세요.

---

## W6 목표 (이것만 달성하면 성공)

1. **실습**: Pandas로 공개 시계열 데이터를 로드해 추세·계절성·결측치를 파악하는 EDA를 완료한다
2. **이론**: 공분산·자기상관(autocorrelation)이 무엇이고 시계열에서 왜 중요한지 설명할 수 있다
3. **재구현**: 선형회귀를 정규방정식(closed-form)과 MLE(로그우도) 두 관점에서 각각 numpy로 구현하고 같은 결과가 나오는지 확인한다
4. **미니프로젝트 #2**: 시계열 EDA 자동 리포트 도구를 완성해 GitHub에 커밋한다
5. **최소 보장**: MLE를 로그우도 최대화로 손유도 가능

---

## Day 1 (월요일) — Pandas 시계열 EDA 첫걸음 [1.5시간]

### 00:00–00:45 | 개념: 시계열 데이터의 특성

```
자료: "time series trend seasonality stationarity" 검색 (10~15분 영상)

메모할 것 (개념만, 코드 없음):
- 추세(trend): 장기적으로 증가/감소하는 패턴
- 계절성(seasonality): 일정한 주기로 반복되는 패턴 (월별, 요일별 등)
- 정상성(stationarity): 시간이 지나도 평균·분산이 일정한 성질
  (많은 통계 모델이 이 가정을 전제로 함 — 지금은 이름만 익혀두면 충분)
```

**막히는 지점 예상**: "정상성 검정(예: ADF test)"은 이번 주 범위 밖입니다. 개념만 알아두고 넘어가세요.

### 00:45–01:30 | 데이터 로드 + 기본 EDA 함수 작성

- **학습 목표**: 날짜를 인덱스로 갖는 Pandas DataFrame을 다루고, 결측치·기간·빈도를 스스로 점검하는 함수를 만들 수 있다.
- **구현 과제 (스스로 작성)**:
  ```python
  def load_timeseries(path_or_url: str, date_col: str, value_col: str) -> "pd.Series":
      """
      요구사항:
      - CSV를 읽어 date_col을 datetime으로 파싱하고 인덱스로 설정한다.
      - 시간순으로 정렬한다 (원본이 정렬되어 있지 않을 수 있음을 가정).
      - value_col만 담은 pandas Series를 반환한다.
      """
      ...

  def basic_ts_summary(series: "pd.Series") -> dict:
      """
      요구사항:
      - 시작일·종료일·전체 기간(일수)을 계산한다.
      - 결측치 개수와 비율을 계산한다.
      - 데이터 빈도(일별/월별 등)를 index의 차이(diff)로 추정한다.
      - 위 정보를 dict로 반환한다.
      """
      ...
  ```
- **시각화 과제**: 원본 시계열을 선 그래프로 그리고 `week06/raw_timeseries.png`로 저장하세요. 눈으로 추세나 계절성이 보이는지 메모하세요.
- **확인 질문**:
  - 결측치가 있다면, 그냥 버릴지(dropna) 보간(interpolate)할지 어떻게 판단할 것인가? 이번 데이터에서는 어떤 선택을 했고 왜인가?
  - 데이터 빈도가 일정하지 않은 구간이 있다면(예: 주말 데이터 없음), `basic_ts_summary`가 그것을 어떻게 알려주는가?

**막히면**: 공개 시계열 데이터를 못 찾겠으면 `statsmodels.datasets`에 내장된 예제(예: `sunspots`, `co2`)를 써도 됩니다. `pip install statsmodels`.

```bash
mkdir -p week06
git add week06/
git commit -m "W6 Day1: time series EDA basics"
git push
```

---

## Day 2 (화요일) — 공분산·자기상관 이론 역추적 [1.5시간]

### 00:00–00:45 | 개념: 공분산이 왜 중요한가

```
자료: Harvard Stat 110 Lec 14–20 (공분산·상관계수 관련 부분)
URL: https://projects.iq.harvard.edu/stat110/youtube (강의 목록에서 해당 주차 검색)

메모할 것 (개념만, 코드 없음):
- 공분산: 두 변수가 "함께" 얼마나 변하는지 (W1에서 PCA에 쓴 공분산 "행렬"의
  각 원소가 바로 이것 — 대각 원소는 분산, 비대각 원소는 공분산)
- 상관계수: 공분산을 각 변수의 표준편차로 나눠 정규화한 것 (-1~1 범위)
- 자기상관(autocorrelation): 시계열 자기 자신을 k 시점만큼 밀어서(lag)
  비교했을 때의 상관계수. "어제 값이 오늘 값과 얼마나 비슷한가"를 측정
```

**막히는 지점 예상**: "왜 공분산을 표준편차로 나누면 -1~1 범위가 되는가?"는 코시-슈바르츠 부등식과 관련되는데, 지금은 증명 없이 "정규화된 값이라 스케일에 안 흔들린다"는 결론만 받아들이고 넘어가세요.

### 00:45–01:15 | 공분산 행렬 + 자기상관을 numpy로 직접 계산

- **학습 목표**: W1에서 이미 구현한 공분산 행렬 계산을 다시 떠올리고, 그 개념을 시계열의 "시간 축 자기상관"으로 확장할 수 있다.
- **구현 과제 (스스로 작성)**:
  ```python
  def covariance_matrix(X: np.ndarray) -> np.ndarray:
      """
      요구사항:
      - W1에서 만든 것과 같은 형태. (X.T @ X_centered) / (n-1) 방식으로 계산.
      - 이번 주에는 "복습 겸 재작성"이 목적이므로 W1 코드를 보지 않고 다시 짜볼 것.
      """
      ...

  def autocorrelation(series: np.ndarray, lag: int) -> float:
      """
      요구사항:
      - series[lag:]와 series[:-lag] 사이의 상관계수를 계산해 반환한다.
      - np.corrcoef를 써도 되고, 공분산·표준편차를 직접 계산해서 만들어도 된다.
      - lag=0일 때는 항상 1.0이 나와야 함 (자기 자신과의 상관).
      """
      ...

  def plot_acf_manual(series: np.ndarray, max_lag: int = 20, save_path: str = "week06/acf_plot.png"):
      """
      요구사항:
      - lag=0부터 max_lag까지 autocorrelation을 계산해 막대그래프로 그린다.
      - (선택) statsmodels.graphics.tsaplots.plot_acf 결과와 비교해볼 것.
      """
      ...
  ```
- **확인 질문**:
  - 자기상관이 lag=1에서 매우 높다면(예: 0.9 이상) 이 시계열에 대해 무엇을 알 수 있는가? (힌트: "오늘 값은 어제 값과 거의 비슷하게 움직인다"는 것이 예측에 어떤 의미인지)
  - 계절성이 있는 데이터라면(예: 월별 데이터에 연 단위 패턴), 자기상관 그래프에서 어떤 lag 근처에서 다시 값이 커질 것으로 예상하는가? 직접 실행해서 확인해보세요.

### 01:15–01:30 | LeetCode: Move Zeroes

- **구현 과제 (스스로 작성, 정답 코드 없음)**:
  ```python
  def move_zeroes(nums: list[int]) -> None:
      """
      요구사항:
      - 0이 아닌 원소들의 상대적 순서를 유지한 채, 모든 0을 배열 끝으로 이동시킨다.
      - 리스트를 제자리(in-place)에서 수정한다 (반환값 없음).
      - 투 포인터로 O(n) 시간, 추가 배열 없이 구현해볼 것.
      - 테스트: [0,1,0,3,12] → [1,3,12,0,0]
      """
      ...
  ```
- **확인 질문**: 두 포인터 중 하나는 "다음에 값을 써넣을 위치"를, 다른 하나는 "탐색 위치"를 가리킵니다. 이 패턴이 이전 주(예: W2의 슬라이딩 윈도우)와 어떻게 비슷하거나 다른가?

```bash
git add week06/
git commit -m "W6 Day2: covariance and autocorrelation + Move Zeroes"
git push
```

---

## Day 3 (수요일) — 정규방정식 vs MLE로 선형회귀 재구현 [1.5시간]

### 00:00–00:45 | 개념: 정규방정식과 MLE, 두 가지 다른 길

- **학습 목표**: 선형회귀의 최적 가중치를 구하는 두 가지 방법(닫힌 형태 해 vs 확률 모델 관점)이 왜 같은 답에 도달하는지 이해한다.
- **핵심 개념**:
  - 정규방정식: MSE 손실을 `w`로 미분해 0으로 놓고 풀면 `w = (X^T X)^{-1} X^T y`라는 닫힌 형태 해가 나옵니다. W7에서 반복적으로(iteratively) 찾은 답을, 이번 주는 "한 번의 계산"으로 바로 구합니다.
  - MLE 관점: "오차가 평균 0인 정규분포를 따른다"고 가정하면, 데이터의 우도(likelihood)를 최대화하는 `w`를 찾는 문제가 됩니다. 로그우도를 전개하면 MSE를 최소화하는 것과 수학적으로 동치임이 드러납니다.
- **과제 (종이 먼저)**:
  1. MSE 손실 `L(w) = ||Xw - y||^2`을 `w`로 미분해 0으로 놓고 풀어, 정규방정식 `w = (X^T X)^{-1} X^T y`를 직접 유도하세요. (W1의 전치·역행렬 감각을 재사용)
  2. 오차 `y - Xw`가 `N(0, sigma^2)`를 따른다고 가정했을 때의 로그우도 `log L(w)`를 전개하고, 이를 최대화하는 것이 MSE를 최소화하는 것과 같아지는 이유를 한 문단으로 정리하세요. (이번 주 최소 보장의 핵심)

**막히는 지점 예상**: 행렬 미분 규칙(`d(w^T A w)/dw = 2Aw` 등)이 낯설면, 먼저 스칼라(1차원) 경우로 유도한 뒤 벡터/행렬로 일반화하세요. 규칙을 통째로 외우기보다 "이번 문제에 필요한 한두 개 규칙"만 확인하고 넘어가도 충분합니다.

### 00:45–01:30 | 정규방정식과 로그우도를 numpy로 구현

- **구현 과제 (스스로 작성)**:
  ```python
  def normal_equation(X: np.ndarray, y: np.ndarray) -> np.ndarray:
      """
      요구사항:
      - w = (X^T X)^{-1} X^T y를 계산해 반환한다.
      - X^T X가 특이(singular)하거나 조건수가 나쁠 수 있으므로,
        np.linalg.inv 대신 np.linalg.pinv 또는 np.linalg.lstsq 사용을 고려할 것
        (둘 중 무엇을 쓸지, 그리고 왜인지 스스로 판단해 결정할 것).
      """
      ...

  def log_likelihood_linear_regression(
      w: np.ndarray, X: np.ndarray, y: np.ndarray, sigma: float = 1.0
  ) -> float:
      """
      요구사항:
      - 오차 residual = y - X @ w를 계산한다.
      - 정규분포 가정 하의 로그우도 공식을 그대로 코드로 옮긴다
        (Day3에서 종이로 유도한 식을 사용할 것).
      - sigma를 다르게 주면 로그우도 값이 어떻게 달라지는지 실험해볼 것.
      """
      ...
  ```
- **검증 과제**: `normal_equation`으로 구한 `w`가, 같은 데이터에 대해 `w`를 조금씩 바꿔가며 `log_likelihood_linear_regression`을 계산했을 때 실제로 로그우도를 최대화하는 지점 근처에 있는지 확인하세요. (예: `w`의 한 성분만 -1~1 범위에서 조금씩 바꿔가며 로그우도를 그래프로 그려보고, 정규방정식 해가 그 그래프의 꼭대기 근처에 있는지 확인)
- **확인 질문**:
  - `X^T X`의 역행렬이 존재하지 않는 경우(다중공선성, 특성 수 > 샘플 수)는 언제 발생하는가? `pinv`는 그럴 때 어떤 답을 주는가?
  - 정규방정식은 "한 번에" 풀리는데, W7에서는 왜 굳이 반복적인 SGD를 쓰는가? (힌트: 특성 수가 매우 많을 때 `(X^T X)^{-1}` 계산 비용을 생각해보세요)

**막히면**: `LinAlgError: Singular matrix`가 나오면 `np.linalg.inv` 대신 `np.linalg.pinv`로 바꿔보기. "normal equation vs gradient descent when to use" 검색.

```bash
git add week06/
git commit -m "W6 Day3: normal equation vs MLE for linear regression"
git push
```

---

## Day 4 (목요일) — 미니프로젝트 #2 착수: 시계열 EDA 자동 리포트 [1.5시간]

### 00:00–01:00 | 자동 EDA 리포트 도구 만들기

- **학습 목표**: 지금까지 따로따로 만든 EDA 함수들(요약 통계, 시각화, 자기상관)을 하나의 재사용 가능한 리포트 생성 도구로 묶을 수 있다.
- **구현 과제 (스스로 작성)**:
  ```python
  def generate_ts_report(
      series: "pd.Series",
      output_dir: str = "week06/report"
  ) -> str:
      """
      요구사항 (Day1~2에서 만든 함수들을 조합):
      - basic_ts_summary로 요약 통계를 얻는다.
      - 원본 시계열 그래프, 이동평균(rolling mean, window는 데이터 빈도에 맞게 선택) 그래프,
        자기상관 그래프를 output_dir에 각각 이미지로 저장한다.
      - (선택, 여유 있으면) statsmodels.tsa.seasonal.seasonal_decompose로
        추세·계절성·잔차를 분해해 그래프 하나 더 저장.
      - 위 결과를 요약한 markdown 문자열(제목, 요약 통계표, 이미지 링크 포함)을
        만들어 output_dir/report.md로 저장하고, 그 경로를 반환한다.
      """
      ...
  ```
- **확인 질문**:
  - 이동평균의 `window` 크기를 어떻게 정했는가? (데이터가 월별이면 12, 일별이면 7 또는 30 등 — 이유를 설명할 수 있어야 함)
  - `seasonal_decompose`를 썼다면, 분해된 "추세" 성분과 원본 시계열의 이동평균이 비슷하게 보이는 이유는 무엇인가?

### 01:00–01:30 | LeetCode: Container With Most Water

- **구현 과제 (스스로 작성, 정답 코드 없음)**:
  ```python
  def max_area(height: list[int]) -> int:
      """
      요구사항:
      - 두 선을 골라 만들 수 있는 컨테이너의 최대 물 저장량(면적)을 구한다.
        면적 = min(height[i], height[j]) * (j - i).
      - 양끝에서 시작하는 투 포인터로 O(n) 시간에 구현해볼 것 (모든 쌍을
        비교하는 O(n^2)보다 나은 방법).
      - 테스트: [1,8,6,2,5,4,8,3,7] → 49
      """
      ...
  ```
- **확인 질문**: 두 포인터 중 더 짧은 쪽을 안쪽으로 옮기는 것이 왜 최적의 선택인가? (반대로 긴 쪽을 옮기면 면적이 절대 커질 수 없는 이유를 스스로 설명해보세요.)

```bash
git add week06/
git commit -m "W6 Day4: mini-project #2 auto EDA report + Container With Most Water"
git push
```

---

## Day 5 (금요일) — 복습 + LSTM 논문 [1.5시간]

### 00:00–00:30 | W6 최소 보장 자가 점검

자료를 보지 않고 아래 질문에 답해보세요. 막히면 해당 Day로 돌아가세요.

```
□ MLE를 로그우도 최대화로 손유도할 수 있는가?
  → Day3에서 종이로 정리한 유도 과정을 다시 한번, 아무것도 안 보고 적어보기

□ 정규방정식이 어떻게 유도되는지 설명할 수 있는가?
  → "MSE를 w로 미분해서 0으로 놓는다"는 절차를 말로 설명

□ 공분산과 자기상관의 관계를 설명할 수 있는가?
  → "같은 변수를 시간차를 두고 비교한 것"이라는 표현을 써서

□ 이동평균과 계절성 분해가 시계열 EDA에서 하는 역할은?
  → generate_ts_report에서 그것들을 왜 넣었는지 스스로 설명
```

### 00:30–01:00 | 논문: LSTM — Vanishing Gradient와 게이트 구조

```
읽을 것: Hochreiter & Schmidhuber (1997) "Long Short-Term Memory"
- 전체 읽을 필요 없음 (원 논문은 표기법이 오래되어 읽기 어려울 수 있음 —
  후속 설명 자료를 병행해도 무방, 예: colah.github.io의 "Understanding LSTM Networks")
- 읽을 부분: 왜 기존 RNN이 긴 시퀀스에서 학습이 안 되는지(vanishing gradient),
  LSTM의 게이트(forget/input/output gate)가 이를 어떻게 해결하는지
- 시간: 20–30분

읽으면서 메모할 것:
1. Vanishing gradient 문제란 무엇인가? (긴 시퀀스에서 그래디언트가
   역전파되며 왜 점점 작아지는가 — W7의 연쇄법칙과 연결해서 생각해보기)
2. LSTM의 forget gate가 하는 역할을 한 문장으로: _______________
3. 이번 주 다룬 시계열 데이터와 LSTM이 어떤 관계가 있는가?
   (힌트: 둘 다 "시간 순서가 있는 데이터"를 다룬다는 공통점)

영어 한 문장 준비 (자기 말로 — 아래는 참고용 뼈대):
"LSTMs use gating mechanisms to control how much information flows
through the cell state over time, which mitigates the vanishing
gradient problem that plain RNNs suffer from on long sequences."
```

### 01:00–01:30 | 마무리 커밋 + README

```bash
git add .
git commit -m "W6 완료: 시계열 EDA, 공분산·자기상관, 정규방정식/MLE, 미니프로젝트#2, LSTM"
git push

cat >> week06/README.md << 'EOF'
# W6: Time series EDA, covariance, normal equation vs MLE

## W6 완료 항목 (스스로 구현)
- [ ] load_timeseries, basic_ts_summary
- [ ] covariance_matrix(재구현), autocorrelation, plot_acf_manual
- [ ] normal_equation, log_likelihood_linear_regression
- [ ] 미니프로젝트 #2: generate_ts_report (자동 EDA 리포트)
- [ ] LeetCode: Move Zeroes, Container With Most Water

## 최소 보장 체크
- [ ] MLE를 로그우도 최대화로 손유도 가능
EOF

git add week06/README.md
git commit -m "W6: update README"
git push
```

---

## 주말 — 심화 [토요일 2.5시간 / 일요일 2시간]

### 토요일 [2.5시간]

**[00:00–01:15] 미니프로젝트 #2 마무리 — GitHub 공개용으로 정리**

- **학습 목표**: 개인 실습 코드를 다른 사람이 읽어도 이해할 수 있는 작은 프로젝트로 정리할 수 있다.
- **과제**: `week06/README.md`에 "무엇을 위한 도구인지 → 사용법(함수 호출 예시) → 생성된 리포트 예시 이미지"를 정리하세요. `generate_ts_report`가 다른 시계열 데이터(예: 다른 종목, 다른 기간)에도 수정 없이 동작하는지 한 번 더 테스트하세요.
- **확인 질문**: 이 도구를 "다른 데이터에도 통하는 일반적인 도구"라고 부르려면 어떤 조건(입력 형식)을 갖춰야 하는가? 지금 코드가 그 조건을 만족하지 못하는 부분이 있다면 어디인가?

**[01:15–02:30] 정규방정식·SVD 연결 심화 (필요 시)**

```
Day3에서 pinv 또는 lstsq를 언급만 했다면, 오늘 왜 그것이 다중공선성
상황에서 inv보다 안정적인지 W2의 SVD 지식과 연결해 정리해보세요.
(힌트: pinv는 내부적으로 SVD를 사용합니다 — X = U S V^T일 때
pseudo-inverse가 어떻게 정의되는지 찾아보세요.)

이미 충분히 이해했다고 느끼면 이 항목은 건너뛰고 Harvard Stat 110의
조건부확률·베이즈 정리 관련 강의를 복습하는 데 시간을 써도 됩니다.
30분 이상 막히면 → 넘어가고 W13 복습 구간에서 다시 만나기.
```

### 일요일 [2시간]

**[00:00–01:00] W6 총복습 — 아무것도 보지 않고 재구현**

이번 주 코드를 하나도 열지 않고, 아래 흐름을 처음부터 다시 짜보세요.

```
1. 시계열 데이터 로드 + 결측치·기간 요약 (5분)
2. 원본 시계열 + 이동평균 시각화 (5분)
3. 자기상관을 lag 몇 개에 대해 직접 계산 (10분)
4. 정규방정식으로 선형회귀 w 계산 (10분)
5. 같은 w 근처에서 로그우도를 몇 개 지점 계산해, w가 최댓값 근처인지 확인 (15분)
6. generate_ts_report를 새 데이터에 실행 (10분)

30분 이상 막히면 Day1~4 코드 참고 가능.
목표는 코드를 외우는 게 아니라 "정규방정식으로 구한 답 = 로그우도를
최대화하는 답"이라는 연결을 손에 익히는 것.
```

**영어로 설명 연습** (혼자 소리 내어, `___` 부분은 직접 측정한 값으로 채우기 — 외운 숫자 금지):

```
"The normal equation gives a closed-form solution for linear regression
by setting the derivative of the MSE loss to zero. This is mathematically
equivalent to maximizing the log-likelihood under the assumption that
residuals are normally distributed, because maximizing that log-likelihood
reduces to minimizing the same squared error term. Separately, in my
time series, the autocorrelation at lag ___ was the first lag where
the correlation dropped below 0.2 (fill in your measured lag), which
suggests how far back the series' memory extends."
```

**[01:00–02:00] 주간 회고 + W7 준비**

```markdown
## W6 회고 (일요일에 작성)

### 달성한 것
- [ ] 시계열 EDA (요약 통계, 시각화, 자기상관)
- [ ] 정규방정식·MLE로 선형회귀 재구현
- [ ] 미니프로젝트 #2: 시계열 EDA 자동 리포트 완성
- [ ] LeetCode: Move Zeroes, Container With Most Water

### 최소 보장 체크
- [ ] MLE를 로그우도 최대화로 손유도 가능

### 스스로 짠 코드에서 막힌 지점
(개념을 몰라서 막힌 것 / 파이썬·pandas 문법을 몰라서 막힌 것을 구분해서 적기 —
 이 구분이 다음 주에 어디에 시간을 쓸지 알려줍니다)

### 로그우도 유도가 안 됐다면 어느 단계였는가
(정규분포 확률밀도함수 대입 / 로그 전개 / MSE와의 연결 중 어디)

### W7에 가져갈 것
(이번 주에 이해 못 하고 넘어간 것)

### 다음 주 첫 번째 할 일
(W7 Day1 무엇부터 시작할지)
```

---

## 막힐 때 대응 가이드

**개념이 막히면**:
```
1단계 (5분): 구글에 영어로 검색
  예: "why is maximizing likelihood equivalent to minimizing MSE"
2단계 (10분): StatQuest 관련 영상 검색 ("MLE", "autocorrelation" 등)
3단계 (20분): Harvard Stat 110 해당 강의 노트 확인
30분 넘어도 안 되면 → "아직 모름: [개념]" 메모하고 넘어가기
→ W13 복습 구간에서 다시 만납니다. 막히는 것은 정상 과정.
```

**직접 짠 코드가 막히면 (완성 코드를 찾아 베끼지 말고)**:
```
1. 에러 메시지 전체를 그대로 구글에 검색
2. 자주 나오는 문제:
   - LinAlgError: Singular matrix → np.linalg.inv 대신 pinv 사용
   - 날짜 파싱 실패 → pd.to_datetime(..., errors="coerce")로 원인 확인
   - 로그우도가 nan → sigma가 0이거나 음수인지 확인
3. Stack Overflow에서 "접근 방식"만 참고하고, 자기 코드로 다시 작성 (복붙 금지)
```

---

## W6 완료 기준

일요일 저녁에 아래를 할 수 있으면 W6 성공:

```
□ week06/ 폴더에 시계열 EDA 코드와 raw_timeseries.png가 있다
□ autocorrelation·acf_plot.png가 있고, lag=0에서 값이 1.0으로 나온다
□ normal_equation으로 구한 w와 log_likelihood 최댓값 근처의 w가 (거의) 일치한다
□ week06/report/report.md (미니프로젝트 #2 자동 리포트)가 생성된다
□ "MLE를 로그우도 최대화로" 손으로 유도한 과정을 3문장으로 설명할 수 있다 (한국어 가능)
□ LeetCode 2문제(Move Zeroes, Container With Most Water)를 스스로 다시 풀 수 있다

절반(4개 이상) 달성하면 W7로 진행.
전부 못 해도 W7로 진행 — 이해 못 한 부분은 이후 주차에서 다시 나옴.
```

---

## W7 첫 할 일 미리 보기

W7 Day1에 열어야 할 것:
1. `week07/` 폴더 생성
2. 편미분·그래디언트 개념 복습 (numerical_gradient로 그래디언트 검증부터 시작)
3. 막히면 → "그래디언트가 왜 이 방향인가?" → 다변수 미적분(편미분·연쇄법칙)으로 역추적
4. 이번 주 만든 `normal_equation` 결과를 W7의 SGD 결과와 비교할 준비를 해둘 것

---

*이 계획대로 완벽하게 안 돼도 됩니다.
W6의 진짜 목표는 "같은 문제(선형회귀)를 서로 다른 두 관점(닫힌 형태 해, 확률 모델)에서 풀어도 같은 답에 도달한다"는 경험입니다.
그리고 그 경험은 정답 공식을 외워서가 아니라, 로그우도를 직접 전개하다가 MSE가 튀어나오는 순간을 스스로 발견하는 과정에서 만들어집니다.*

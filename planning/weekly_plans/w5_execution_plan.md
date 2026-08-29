# W5 구체적 실행 계획 (학습 가이드형)

> **주제**: RandomForest·XGBoost 앙상블 비교 + 엔트로피·정보이득 재활성화 + 결정트리 분할 기준 재구현 + CLT(중심극한정리)
>
> **사용 데이터셋 주의**: 이번 주는 `sklearn.datasets.load_breast_cancer`(569개, 이진분류)를 기본으로 씁니다. 실제 의료 진단 데이터이지만 학습용 공개 벤치마크이며, 실제 임상 판단에는 쓰이지 않습니다. 주말에는 W1~W2에서 쓴 `load_digits`(이진분류로 축소) 또는 `load_wine`에도 같은 파이프라인을 적용해봅니다.
>
> **총 목표 시간**: 10–12시간 (v7 커리큘럼 블록 A 기준)
> **기준**: 평일 1.5시간 + 토요일 2.5시간 + 일요일 2시간
>
> **코드 제시 방식**: 완성된 실행 코드를 주지 않습니다. 학습 목표·핵심 개념·구현 요구사항(스펙)·확인 질문만 제시합니다. 코드는 스스로 작성하세요. Git/설치 같은 보일러플레이트만 그대로 제시합니다.
>
> **전제**: W3에서 만든 로지스틱 회귀·ROC 평가 코드가 있다면 이번 주 RandomForest·XGBoost 평가에도 그대로 재사용하세요(같은 혼동행렬·ROC 함수를 다른 모델에 적용하는 연습이 됩니다). 없어도 이번 주 코드는 독립적으로 완성됩니다.

---

## W5 목표 (이것만 달성하면 성공)

1. **실습**: RandomForest와 XGBoost를 같은 데이터로 학습·평가해 성능·특성중요도를 비교한다
2. **이론**: 엔트로피·정보이득이 무엇이고, 결정트리가 왜 그 값을 기준으로 분할점을 고르는지 설명할 수 있다
3. **재구현**: 결정트리의 최적 분할점을 찾는 로직을 numpy로 직접 구현한다
4. **시뮬레이션**: 중심극한정리(CLT)를 코드로 시뮬레이션해 눈으로 확인한다
5. **최소 보장**: 정규분포가 왜 자주 등장하는지(CLT 관점) 설명 가능

---

## Day 1 (월요일) — RandomForest vs XGBoost 첫 비교 [1.5시간]

### 00:00–00:45 | 개념: 배깅 vs 부스팅

```
읽을 것/시청: "bagging vs boosting" 검색 (StatQuest 관련 영상, 각 10분 내외)
- Random Forest: 여러 트리를 병렬로, 각기 다른 부트스트랩 샘플과
  특성 부분집합으로 학습시킨 뒤 투표/평균으로 합침 (배깅)
- XGBoost: 트리를 순차적으로 추가하며, 이전 트리들이 틀린 부분을
  다음 트리가 보완하도록 학습 (부스팅)

메모할 것 (개념만, 코드 없음):
- 왜 단일 결정트리는 과적합되기 쉬운가?
- 배깅이 분산(variance)을 줄이는 원리 / 부스팅이 편향(bias)을 줄이는 원리
  (정확한 이유는 아직 몰라도 됨 — B② 이론 보강 트랙에서 다시 다룸)
```

**막히는 지점 예상**: "왜 부트스트랩 샘플링이 분산을 줄이는가?"는 아직 통계 이론이 부족해 직관만으로는 완전히 이해되지 않을 수 있습니다. 지금은 "여러 모델의 평균은 하나보다 변동이 적다"는 결론만 받아들이고 넘어가세요.

### 00:45–01:30 | RandomForest vs XGBoost 실습 비교

- **학습 목표**: 같은 데이터·같은 평가지표로 두 앙상블 모델을 비교하고, 성능과 특성중요도의 차이를 관찰할 수 있다.
- **핵심 개념**: 두 모델 모두 `feature_importances_` 속성을 제공하지만 계산 방식이 다릅니다(RF는 불순도 감소 평균, XGBoost는 기본적으로 gain/weight/cover 중 선택). 같은 순위가 나오지 않을 수 있습니다.
- **구현 과제 (스스로 작성)**:
  ```python
  def compare_rf_xgb(X: np.ndarray, y: np.ndarray, random_state: int = 42) -> dict:
      """
      요구사항:
      - train_test_split으로 데이터를 나눈다.
      - RandomForestClassifier와 XGBClassifier(또는 xgboost 미설치 시 GradientBoostingClassifier)를
        각각 같은 train 세트로 학습한다.
      - 테스트 세트에서 accuracy와 ROC-AUC를 각각 계산한다.
      - 두 모델의 feature_importances_ 상위 5개를 각각 반환한다.
      - 결과를 dict로 반환 (모델명 → {accuracy, roc_auc, top5_features}).
      """
      ...
  ```
- **확인 질문**:
  - 두 모델의 정확도 차이가 크지 않다면, 어떤 상황에서 그 작은 차이가 실무적으로 중요해지는가?
  - 두 모델의 top5 특성이 다르게 나왔다면, 그것이 "특성이 안 중요해서"인지 "계산 방식이 달라서"인지 어떻게 구분할 것인가?

**막히면**: `xgboost` 설치가 안 되면 `pip install xgboost` (Colab은 기본 설치되어 있는 경우가 많음). 설치가 계속 막히면 `sklearn.ensemble.GradientBoostingClassifier`로 대체해도 이번 주 목표(배깅 vs 부스팅 비교)는 충분히 달성됩니다.

```bash
mkdir -p week05
git add week05/
git commit -m "W5 Day1: RandomForest vs XGBoost comparison"
git push
```

---

## Day 2 (화요일) — 엔트로피·정보이득 이론 역추적 [1.5시간]

### 00:00–00:45 | 개념: 엔트로피와 정보이득

```
자료: "entropy information gain decision tree" 검색 (StatQuest 관련 영상, 15분)
병행 (선택): Harvard Stat 110 Lec 7–13 (확률의 기본 성질 복습 — 엔트로피 자체보다는
"불확실성"을 다루는 감각을 재활성화하는 용도)

메모할 것 (개념만, 코드 없음):
- 엔트로피: H(y) = -sum(p_i * log2(p_i)). 클래스가 균등하게 섞여있을수록 값이 커짐
  (불확실성이 최대)
- 정보이득: 분할 전 엔트로피 - 분할 후 (가중평균) 엔트로피. 클수록 "좋은 분할"
- 결정트리는 매 노드에서 정보이득이 최대가 되는 (특성, 임계값) 쌍을 고름
```

**막히는 지점 예상**: 로그의 밑(base)이 2인지 자연로그인지는 라이브러리마다 다를 수 있습니다. 지금은 "어느 밑을 쓰든 상대적인 대소 비교 결과는 같다"는 점만 이해하고 넘어가세요.

### 00:45–01:30 | 엔트로피·정보이득을 numpy로 직접 계산

- **학습 목표**: 라이브러리 없이 엔트로피와 정보이득을 계산할 수 있고, 그 값으로 "어떤 분할이 더 나은지" 판단할 수 있다.
- **구현 과제 (스스로 작성)**:
  ```python
  def entropy(y: np.ndarray) -> float:
      """
      요구사항:
      - y는 클래스 레이블 배열(예: 0과 1).
      - 각 클래스의 비율 p_i를 구해 -sum(p_i * log2(p_i))를 계산.
      - 클래스가 한 종류만 있으면(순수 노드) 엔트로피가 0이 되어야 함.
      - p_i가 0인 경우 log2(0)이 되지 않도록 처리할 것.
      """
      ...

  def information_gain(y_parent: np.ndarray, y_left: np.ndarray, y_right: np.ndarray) -> float:
      """
      요구사항:
      - 부모 노드의 엔트로피에서, 왼쪽/오른쪽 자식 노드 엔트로피의
        (샘플 수 비례) 가중평균을 뺀 값을 반환.
      """
      ...
  ```
- **검증 과제**: `load_breast_cancer`에서 특정 특성 하나를 골라 임의의 임계값으로 둘로 나눈 뒤, `information_gain`을 계산해보세요. 임계값을 여러 개 시도했을 때 정보이득이 어떻게 달라지는지 관찰하세요.
- **확인 질문**:
  - 부모 노드가 이미 순수(엔트로피 0)하다면 어떤 분할을 해도 정보이득이 어떻게 되는가?
  - 지니 불순도(`1 - sum(p_i^2)`)와 엔트로피는 계산식이 다른데, 결정트리 분할 기준으로서 왜 비슷한 역할을 하는가? (직접 두 값을 여러 `p`에 대해 계산해 비교해보면 감이 옵니다)

**막히면**: 결과가 `nan`이 되면 `p_i == 0`인 클래스에서 `log2(0)`을 계산했는지 확인. "gini vs entropy decision tree" 검색.

```bash
git add week05/
git commit -m "W5 Day2: entropy and information gain from scratch"
git push
```

---

## Day 3 (수요일) — 결정트리 최적 분할점 재구현 [1.5시간]

### 00:00–00:45 | 개념: 최적 분할점을 찾는 알고리즘

```
메모할 것 (개념만, 코드 없음):
- 연속형 특성에서 "가능한 모든 임계값"은 무한하지만, 실제로 의미 있는
  후보는 정렬된 유니크값들 사이의 중간점들뿐입니다 (그 사이에서는
  분할 결과가 바뀌지 않으므로)
- 결정트리는 모든 특성 × 모든 후보 임계값 조합에 대해 정보이득을 계산하고,
  가장 큰 값을 주는 조합을 선택합니다 (탐욕적 알고리즘)
- 이 과정을 각 노드마다 재귀적으로 반복하는 것이 결정트리 학습의 본질
```

### 00:45–01:15 | best_split 함수 구현

- **학습 목표**: sklearn 내부에서 일어나는 "분할점 탐색"을 직접 구현하고, 그 결과가 sklearn의 깊이 1짜리 트리(decision stump)와 같은 분할을 찾는지 검증할 수 있다.
- **구현 과제 (스스로 작성)**:
  ```python
  def best_split(X: np.ndarray, y: np.ndarray) -> tuple[int, float, float]:
      """
      요구사항:
      - 모든 특성(열)에 대해, 정렬된 유니크값들의 중간점을 임계값 후보로 생성.
      - 각 (특성 인덱스, 임계값) 조합에 대해 Day2의 information_gain을 계산.
      - 정보이득이 최대인 (특성 인덱스, 임계값, 그때의 정보이득)을 반환.
      - 힌트: 이중 반복문으로 시작해도 되고, 익숙하면 벡터화해도 됨.
      """
      ...
  ```
- **검증 과제**: `load_breast_cancer` 데이터로 `best_split`을 실행한 결과와, `DecisionTreeClassifier(max_depth=1, criterion="entropy")`가 실제로 선택한 특성·임계값이 일치하는지 비교하세요. (sklearn 모델 객체의 `tree_.feature[0]`, `tree_.threshold[0]`으로 확인 가능합니다.)
- **확인 질문**:
  - 데이터가 30개 특성 × 569개 샘플일 때, 이 방식의 시간복잡도는 대략 어느 정도인가? (특성 수 × 샘플 수 × 정렬 비용) 특성이나 샘플이 100배로 늘어나면 왜 이 방식이 느려질까?
  - sklearn과 결과가 정확히 일치하지 않는다면, 임계값 후보를 만드는 방식(중간점 vs 실제 값 자체)의 차이 때문은 아닌지 확인해보세요.

### 01:15–01:30 | LeetCode: Binary Search

- **구현 과제 (스스로 작성, 정답 코드 없음)**:
  ```python
  def binary_search(nums: list[int], target: int) -> int:
      """
      요구사항:
      - 정렬된 배열에서 target의 인덱스를 O(log n)에 찾는다. 없으면 -1.
      - 재귀 대신 반복문(while)으로 구현해볼 것.
      - 테스트: nums=[-1,0,3,5,9,12], target=9 → 4
      """
      ...
  ```
- **확인 질문**: `best_split`에서 정렬된 유니크값 위에서 후보를 만드는 것과, Binary Search가 정렬된 배열을 전제로 하는 것 사이에 공통점이 있는가? ("정렬"이 왜 여러 알고리즘에서 전처리로 쓰이는지 생각해보세요.)

```bash
git add week05/
git commit -m "W5 Day3: best_split from scratch + Binary Search"
git push
```

---

## Day 4 (목요일) — 중심극한정리(CLT) 시뮬레이션 [1.5시간]

### 00:00–00:45 | 개념: 왜 정규분포가 자주 등장하는가

```
자료: 3Blue1Brown 또는 StatQuest "Central Limit Theorem" 검색 (10~15분)

메모할 것 (개념만, 코드 없음, 이번 주 최소 보장의 핵심):
- 중심극한정리: 원래 분포가 무엇이든(균등분포든 지수분포든), 그 분포에서
  뽑은 표본의 "평균"을 여러 번 반복해서 모으면, 표본 크기가 커질수록
  그 표본평균들의 분포는 정규분포에 가까워진다
- 이것이 왜 중요한가: 통계 검정·신뢰구간·많은 ML 가정(오차가 정규분포를
  따른다는 가정 등)이 이 정리에 근거함
```

### 00:45–01:15 | CLT를 코드로 시뮬레이션

- **학습 목표**: 원래 분포가 정규분포가 아니어도, 표본평균의 분포가 정규분포에 가까워지는 것을 직접 시뮬레이션으로 확인할 수 있다.
- **구현 과제 (스스로 작성)**:
  ```python
  def simulate_clt(
      dist_sampler,
      sample_size: int,
      n_trials: int = 5000
  ) -> np.ndarray:
      """
      요구사항:
      - dist_sampler: size를 인자로 받아 그 크기만큼 표본을 뽑는 함수
        (예: lambda size: np.random.exponential(scale=2.0, size=size)).
      - n_trials번 반복해서, 매번 sample_size개를 뽑아 평균을 계산하고 기록.
      - n_trials개의 표본평균 배열을 반환한다.
      """
      ...

  def plot_clt_comparison(dist_sampler, sample_sizes: list[int], save_path: str = "week05/clt_simulation.png"):
      """
      요구사항:
      - sample_sizes(예: [1, 5, 30, 100]) 각각에 대해 simulate_clt를 실행.
      - 각 결과의 히스토그램을 나란히(subplot) 그려, 표본 크기가 커질수록
        분포가 종 모양(정규분포)에 가까워지는 것을 시각적으로 비교.
      """
      ...
  ```
- **실행 과제**: 원본 분포로 균등분포(`np.random.uniform`)와 지수분포(`np.random.exponential`)를 각각 하나씩 시도해보세요. 둘 다 정규분포와 전혀 다른 모양인데도, 표본평균의 분포가 결국 종 모양으로 수렴하는지 확인하세요.
- **확인 질문**:
  - `sample_size=1`일 때 히스토그램은 원본 분포와 어떻게 다른가(또는 같은가)?
  - `sample_size`가 커질수록 히스토그램의 "폭(분산)"은 어떻게 변하는가? 왜 그럴까? (힌트: 표본평균의 분산 = 원본 분산 / n)

### 01:15–01:30 | LeetCode: Search in Rotated Sorted Array

- **구현 과제 (스스로 작성, 정답 코드 없음)**:
  ```python
  def search_rotated(nums: list[int], target: int) -> int:
      """
      요구사항:
      - 오름차순 정렬된 배열을 어떤 지점에서 회전시킨 배열에서 target의 인덱스를
        O(log n)에 찾는다. 없으면 -1.
      - 이진탐색을 변형: 매 스텝에서 "어느 절반이 정렬되어 있는지" 먼저 판단할 것.
      - 테스트: nums=[4,5,6,7,0,1,2], target=0 → 4
      """
      ...
  ```
- **확인 질문**: 이 문제와 Day3의 `binary_search`의 공통점과 차이점은? 회전된 배열에서 "정렬된 절반"을 어떻게 판별하는지 자기 말로 설명해보세요.

```bash
git add week05/
git commit -m "W5 Day4: CLT simulation + Search in Rotated Sorted Array"
git push
```

---

## Day 5 (금요일) — 복습 + ResNet 논문 [1.5시간]

### 00:00–00:30 | W5 최소 보장 자가 점검

자료를 보지 않고 아래 질문에 답해보세요. 막히면 해당 Day로 돌아가세요.

```
□ 정규분포가 왜 자주 등장하는지(CLT 관점) 설명할 수 있는가?
  → Day4의 시뮬레이션 결과를 예로 들어 3문장으로 설명해보기

□ 엔트로피와 정보이득이 무엇인지 설명할 수 있는가?
  → "불확실성"이라는 단어를 써서 한 문단으로

□ 결정트리가 분할점을 어떻게 고르는지 설명할 수 있는가?
  → best_split의 로직을 코드 없이 말로 설명

□ 배깅과 부스팅의 차이를 설명할 수 있는가?
  → RandomForest와 XGBoost를 예로 들어
```

### 00:30–01:00 | 논문: ResNet — 왜 깊을수록 나빠지는가

```
읽을 것: He et al. (2016) "Deep Residual Learning for Image Recognition" (ResNet)
- 전체 읽을 필요 없음
- 읽을 부분: Abstract, Section 1(Introduction)의 "degradation problem" 설명 부분
- 시간: 20–30분

읽으면서 메모할 것:
1. "깊은 네트워크가 얕은 네트워크보다 학습 오차조차 더 크다"는
   degradation 문제란 무엇인가? (과적합 문제와 어떻게 다른가?)
2. Skip connection(잔차 연결)이 이 문제를 어떻게 완화하는가?
   (힌트: "항등 함수를 학습하기 쉽게 만든다"는 표현의 의미)
3. 이번 주 배운 앙상블(RF/XGBoost)과 ResNet의 skip connection 사이에
   "여러 개의 약한 신호를 합친다"는 공통된 직관이 있는가?

영어 한 문장 준비 (자기 말로 — 아래는 참고용 뼈대):
"ResNet introduces skip connections that let each layer learn a residual
function relative to its input, making it easier to preserve information
even as the network gets very deep."
```

### 01:00–01:30 | 마무리 커밋 + README

```bash
git add .
git commit -m "W5 완료: RF/XGBoost 비교, 엔트로피·정보이득, CLT 시뮬레이션, ResNet"
git push

cat >> week05/README.md << 'EOF'
# W5: Ensembles, entropy, information gain, CLT

## W5 완료 항목 (스스로 구현)
- [ ] RandomForest vs XGBoost 비교 (compare_rf_xgb)
- [ ] entropy·information_gain 직접 구현
- [ ] best_split으로 결정트리 분할 로직 재구현 + sklearn과 비교
- [ ] CLT 시뮬레이션 (simulate_clt, plot_clt_comparison)
- [ ] LeetCode: Binary Search, Search in Rotated Sorted Array

## 최소 보장 체크
- [ ] 정규분포가 왜 자주 등장하는지(CLT) 설명 가능
EOF

git add week05/README.md
git commit -m "W5: update README"
git push
```

---

## 주말 — 심화 [토요일 2.5시간 / 일요일 2시간]

### 토요일 [2.5시간]

**[00:00–01:15] 다른 데이터셋에 파이프라인 적용 (일반화 검증)**

- **학습 목표**: Day1~3에서 만든 `compare_rf_xgb`, `best_split`이 breast_cancer 말고 다른 데이터에도 수정 없이 통하는지 확인한다.
- **구현 과제 (스스로 작성)**: `load_wine`(다중 클래스, 3개)에 `compare_rf_xgb`를 적용해보세요. 이진분류를 가정하고 짠 부분(예: ROC-AUC 계산)이 있다면 다중 클래스에서 에러가 나거나 의미가 달라질 수 있습니다 — 그 지점을 스스로 찾아 수정하세요(`roc_auc_score`의 `multi_class` 옵션을 찾아보는 것도 방법입니다).
- **확인 질문**: 이진분류용으로 짠 코드가 다중분류에서 어디서 깨졌는가? 그 부분이 "이진분류라는 암묵적 가정"을 코드에 심어놓은 신호였다는 것을 어떻게 알아챘는가?

**[01:15–02:30] XGBoost 하이퍼파라미터 튜닝 실험 (필요 시)**

```
Day1에서는 기본 하이퍼파라미터로만 비교했습니다. 오늘은 XGBoost의
n_estimators, max_depth, learning_rate를 몇 가지 조합으로 바꿔가며
성능이 어떻게 달라지는지 표로 정리해보세요 (GridSearchCV까지 안 써도
됩니다 — 수동으로 3~4개 조합만 비교해도 충분).

이미 앙상블 비교를 충분히 했다고 느끼면 이 항목은 건너뛰고
Harvard Stat 110의 관련 강의(조건부확률·베이즈 정리 복습)를
보충하는 데 시간을 써도 됩니다.
30분 이상 막히면 → 넘어가고 W13 복습 구간에서 다시 만나기.
```

### 일요일 [2시간]

**[00:00–01:00] W5 총복습 — 아무것도 보지 않고 재구현**

이번 주 코드를 하나도 열지 않고, 아래 흐름을 처음부터 다시 짜보세요.

```
1. breast_cancer 로드 + RandomForest/XGBoost 학습·평가 (10분)
2. entropy 함수 구현 + 순수 노드에서 0이 나오는지 확인 (5분)
3. information_gain 함수 구현 (5분)
4. best_split으로 특정 데이터의 최적 분할점 찾기 + sklearn과 비교 (15분)
5. CLT 시뮬레이션: 지수분포에서 표본평균 분포가 정규분포에 가까워지는지 확인 (15분)

30분 이상 막히면 Day1~4 코드 참고 가능.
목표는 코드를 외우는 게 아니라 "불확실성(엔트로피) → 분할 기준 →
앙상블"이라는 흐름과, CLT가 왜 통계 전반의 기반이 되는지를 손에 익히는 것.
```

**영어로 설명 연습** (혼자 소리 내어, `___` 부분은 직접 측정한 값으로 채우기 — 외운 숫자 금지):

```
"A decision tree chooses the split that maximizes information gain,
which is the reduction in entropy after splitting the data by a feature
and threshold. Random Forest reduces variance by averaging many trees
trained on bootstrapped samples, while XGBoost reduces bias by
sequentially adding trees that correct the errors of previous ones.
Separately, the Central Limit Theorem explains why sample means tend
toward a normal distribution regardless of the original distribution.
In my simulation, the sample mean distribution looked visibly closer
to normal once the sample size reached ___ (fill in your measured
value)."
```

**[01:00–02:00] 주간 회고 + W6 준비**

```markdown
## W5 회고 (일요일에 작성)

### 달성한 것
- [ ] RandomForest vs XGBoost 비교
- [ ] entropy·information_gain·best_split 직접 구현
- [ ] CLT 시뮬레이션
- [ ] LeetCode: Binary Search, Search in Rotated Sorted Array

### 최소 보장 체크
- [ ] 정규분포가 왜 자주 등장하는지(CLT) 설명 가능

### 스스로 짠 코드에서 막힌 지점
(개념을 몰라서 막힌 것 / 파이썬·numpy 문법을 몰라서 막힌 것을 구분해서 적기 —
 이 구분이 다음 주에 어디에 시간을 쓸지 알려줍니다)

### best_split이 sklearn과 달랐다면 어느 지점이었는가
(임계값 후보 생성 방식 / 정보이득 계산 / 둘 다 아닌 다른 이유)

### W6에 가져갈 것
(이번 주에 이해 못 하고 넘어간 것)

### 다음 주 첫 번째 할 일
(W6 Day1 무엇부터 시작할지)
```

---

## 막힐 때 대응 가이드

**개념이 막히면**:
```
1단계 (5분): 구글에 영어로 검색
  예: "why does central limit theorem matter in statistics"
2단계 (10분): StatQuest 또는 3Blue1Brown 관련 영상 검색
3단계 (20분): Harvard Stat 110 해당 강의 노트 확인
30분 넘어도 안 되면 → "아직 모름: [개념]" 메모하고 넘어가기
→ W13 복습 구간에서 다시 만납니다. 막히는 것은 정상 과정.
```

**직접 짠 코드가 막히면 (완성 코드를 찾아 베끼지 말고)**:
```
1. 에러 메시지 전체를 그대로 구글에 검색
2. 자주 나오는 문제:
   - entropy 계산에서 nan → p_i == 0인 클래스에서 log2(0) 발생 확인
   - best_split이 너무 느림 → 특성/샘플 수를 줄여서 먼저 동작 확인 후 최적화
   - xgboost ImportError → pip install xgboost, 계속 안 되면 GradientBoostingClassifier로 대체
3. Stack Overflow에서 "접근 방식"만 참고하고, 자기 코드로 다시 작성 (복붙 금지)
```

---

## W5 완료 기준

일요일 저녁에 아래를 할 수 있으면 W5 성공:

```
□ week05/ 폴더에 compare_rf_xgb 결과와 비교표가 있다
□ entropy·information_gain 함수가 breast_cancer 데이터에서 정상 동작한다
□ best_split의 결과가 sklearn DecisionTreeClassifier(max_depth=1)와 (거의) 일치한다
□ clt_simulation.png가 week05/ 폴더에 있고, 표본 크기가 커질수록 종 모양에 가까워진다
□ "정규분포가 왜 자주 등장하는가"를 CLT로 3문장 설명할 수 있다 (한국어 가능)
□ LeetCode 2문제(Binary Search, Search in Rotated Sorted Array)를 스스로 다시 풀 수 있다

절반(4개 이상) 달성하면 W6로 진행.
전부 못 해도 W6로 진행 — 이해 못 한 부분은 이후 주차에서 다시 나옴.
```

---

## W6 첫 할 일 미리 보기

W6 Day1에 열어야 할 것:
1. `week06/` 폴더 생성
2. Pandas로 공개 시계열 데이터(예: 항공 승객 수, 주가, 날씨 데이터 등) 로드 + 기본 EDA
3. 막히면 → "공분산이 왜 중요한가?" → Harvard Stat 110 Lec 14–20으로 역추적
4. 이번 주 목표 중 하나는 미니프로젝트 #2(시계열 EDA 자동 리포트) 착수

---

*이 계획대로 완벽하게 안 돼도 됩니다.
W5의 진짜 목표는 "왜 이 분할이 저 분할보다 나은가"라는 질문에 숫자(엔트로피)로 답할 수 있게 되는 것입니다.
그리고 그 감각은 RandomForest를 한 줄로 호출해서가 아니라, 분할 기준을 직접 계산하다가 sklearn과 결과를 맞춰보는 과정에서 만들어집니다.*

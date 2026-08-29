# W2 구체적 실행 계획 (v2 — 학습 가이드형)

> **주제**: LU분해·영공간 재활성화 + numpy PCA 파이프라인 완성 + Python 함수형·예외처리
>
> **사용 데이터셋 주의**: W1과 동일하게 sklearn `load_digits()`(8×8, 1797개, "미니 MNIST")를 기본으로 쓰되,
> 주말에 비교용으로 `load_wine()`(178개, 13차원)을 하나 더 씁니다. 둘 다 진짜 MNIST가 아닙니다.
>
> **총 목표 시간**: 10–12시간 (v7 커리큘럼 블록 A 기준)
> **기준**: 평일 1.5시간 + 토요일 2.5시간 + 일요일 2시간
>
> **v1과의 차이**: 완성된 실행 코드를 주지 않습니다. 학습 목표·핵심 개념·구현 요구사항(스펙)·확인 질문만 제시합니다. 코드는 스스로 작성하세요. Git/설치 같은 보일러플레이트만 그대로 제시합니다.
>
> **전제**: W1에서 만든 `SimplePCA` 클래스와 `manual_pca_via_eigh` 함수를 이번 주에 확장합니다. W1 코드가 없으면 Day 1에 먼저 복구하세요.

---

## W2 목표 (이것만 달성하면 성공)

1. **이론**: LU분해·영공간(null space)의 의미를 설명할 수 있다
2. **재구현**: 공분산 → 고유값분해(또는 SVD) → 투영 → 재구성까지 PCA 전 과정을 스스로 함수로 완성
3. **판단력**: 설명 분산 누적 그래프로 "몇 개의 주성분을 쓸지" 스스로 결정할 수 있다
4. **Python**: comprehension·함수형·예외처리를 자기 PCA 코드에 적용
5. **최소 보장**: PCA를 고유값분해로 **손계산** / SVD와 PCA 차이를 행렬 분해 관점에서 설명 가능

---

## Day 1 (월요일) — LU분해·영공간 [1.5시간]

W1에서 익힌 행렬 곱·역행렬 위에 "해가 없거나 무한히 많은 경우"를 다룹니다.

### 00:00–00:45 | 개념: LU분해 · 영공간

```
시청: MIT 18.06 Lecture 4 (Gilbert Strang) — LU 분해
URL: https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/

병행 (선택): 3Blue1Brown "Essence of Linear Algebra" Ch.7 (Inverse, column space, null space)
URL: https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab

메모할 것 (개념만, 코드 없음):
- LU분해: A = L(하삼각) @ U(상삼각). 가우스 소거법을 행렬로 기록한 것
- 영공간(null space): A @ x = 0 을 만족하는 x들의 집합
- rank(계수)와 영공간의 차원 사이 관계는? (rank-nullity theorem)
```

**막히는 지점 예상**: "왜 가우스 소거법을 행렬 곱으로 표현하나?" → 지금은 "그렇게 정리된다"만 받아들이고 넘어가세요. 증명은 ROI가 낮습니다.

### 00:45–01:30 | numpy/scipy로 직접 확인

- **학습 목표**: 특이(singular) 행렬에 영공간이 실제로 존재함을 코드로 확인하고, 그것이 "정보가 사라지는 방향"이라는 의미와 연결할 수 있다.
- **핵심 개념**: 영공간에 속한 벡터 `v`는 `A @ v = 0`입니다. 즉 `A`라는 변환을 거치면 `v`의 정보가 완전히 사라집니다. 되돌릴 수 없으니 역행렬도 없습니다. PCA에서는 공분산 행렬이 특이하면 일부 고유값이 0이 되고, 그 방향에는 데이터가 전혀 퍼져 있지 않다는 뜻이 됩니다.
- **구현 과제 (스스로 작성)**:
  ```python
  def explore_lu(A: np.ndarray):
      """
      요구사항:
      - scipy.linalg.lu로 A를 P, L, U로 분해한다.
      - P @ L @ U 가 원래 A와 같은지 np.allclose로 검증하고 결과를 출력한다.
      - L이 하삼각, U가 상삼각 형태인지 눈으로 확인할 수 있게 출력한다.
      """
      ...

  def explore_null_space(B: np.ndarray):
      """
      요구사항:
      - 일부러 선형 종속인 행(예: 어떤 행이 다른 행의 배수)을 가진 3x3 행렬 B를 만든다.
      - np.linalg.matrix_rank(B)를 출력해 rank가 3보다 작음을 확인한다.
      - scipy.linalg.null_space(B)로 영공간 basis를 구한다.
      - basis 벡터 v 하나에 대해 B @ v 가 0 벡터에 가까운지 np.allclose(..., atol=1e-8)로 검증한다.
      """
      ...
  ```
- **확인 질문**:
  - `rank(B)`가 2이고 B가 3×3일 때, 영공간의 차원은 몇인가? 실제 `null_space(B)`의 shape와 일치하는가?
  - 만약 샘플 수가 차원 수보다 적다면(예: 데이터 50개, 특성 100개) 공분산 행렬의 rank는 최대 얼마인가? PCA에서 어떤 문제가 생기는가?

**막히면**: `scipy` 미설치면 `pip install scipy` (Colab은 기본 설치). "rank nullity theorem" 키워드로 검색.

```bash
mkdir -p week02
git add week02/
git commit -m "W2 Day1: LU decomposition + null space exploration"
git push
```

---

## Day 2 (화요일) — PCA를 SVD로도 풀어보기 [1.5시간]

W1에서는 고유값분해로 PCA를 했습니다. 오늘은 SVD로 같은 결과를 얻고 차이를 비교합니다.

### 00:00–00:45 | SVD vs 고유값분해

- **학습 목표**: "공분산 행렬을 고유값분해하는 것"과 "데이터 행렬을 SVD하는 것"이 수학적으로 동치임을 코드로 직접 검증한다.
- **핵심 개념**: `X = U @ S @ Vᵀ`일 때, `V`의 열이 주성분 방향이고 특이값 `S`와 고유값 `λ` 사이에는 일정한 관계가 있습니다. 그 관계식이 무엇인지는 스스로 찾아내는 것이 오늘의 과제입니다. SVD는 공분산 행렬을 명시적으로 만들지 않으므로 수치적으로 더 안정적입니다.
- **구현 과제 (스스로 작성)**:
  ```python
  def compare_eig_vs_svd(X_scaled: np.ndarray, k: int = 2):
      """
      요구사항:
      - 방법 A: 공분산 행렬 → np.linalg.eigh → 고유값 내림차순 정렬 → 상위 k개 고유벡터로 투영.
      - 방법 B: np.linalg.svd(X_scaled, full_matrices=False) → U, S, Vt 획득 →
                상위 k개 성분으로 투영.
      - 두 방법의 "고유값"과 "특이값" 사이 변환 공식을 스스로 찾아내서,
        두 값이 일치하는지 np.allclose로 검증한다.
        (힌트: 특이값 S와 샘플 수 n을 사용해 고유값을 만들어낼 수 있다. 공식은 직접 유도할 것.)
      - 투영 결과도 절댓값 기준으로 일치하는지 확인한다 (부호는 다를 수 있음).
      """
      ...
  ```
- **확인 질문**:
  - 고유값과 특이값 사이 관계식은 무엇이었는가? 왜 그런 관계가 성립하는가 (`XᵀX`를 SVD로 전개해보면 보입니다)?
  - sklearn의 `PCA`는 내부적으로 둘 중 어느 방법을 쓸까? 문서나 소스를 확인해보세요.

**막히면**: "SVD PCA equivalence covariance matrix" 검색 → StatQuest "SVD" 영상 (12분). 관계식을 찾지 못하겠으면 `XᵀX = V S² Vᵀ`를 종이에 전개해보세요.

### 00:45–01:30 | 손계산: 2×2 행렬로 PCA 전 과정 (이번 주 최소 보장의 핵심)

- **학습 목표**: 코드 없이 종이와 펜만으로 작은 데이터의 PCA를 끝까지 계산할 수 있다.
- **핵심 개념**: 고유값은 특성방정식 `det(Cov - λI) = 0`의 해입니다. 2×2면 이차방정식이 되므로 손으로 풀 수 있습니다. 각 λ에 대해 `(Cov - λI)v = 0`을 풀면 고유벡터가 나옵니다.
- **과제 (종이 먼저, 그다음 코드로 검증)**:
  1. 아래 10개 2차원 데이터를 쓰세요.
     ```
     (2.5, 2.4) (0.5, 0.7) (2.2, 2.9) (1.9, 2.2) (3.1, 3.0)
     (2.3, 2.7) (2.0, 1.6) (1.0, 1.1) (1.5, 1.6) (1.1, 0.9)
     ```
  2. **종이에서**: 평균을 빼서 중심화 → 2×2 공분산 행렬 계산 → `det(Cov - λI) = 0`을 이차방정식으로 풀어 고유값 2개 → 각 고유값에 대해 고유벡터 구하기 (정규화까지).
  3. **그다음 코드로**: 위 데이터를 numpy 배열로 만들고 공분산과 `np.linalg.eigh` 결과를 계산해, 종이 풀이와 소수점 2자리까지 맞는지 비교하는 스크립트를 작성하세요. (검증용이므로 코드는 짧아도 됩니다.)
- **확인 질문**:
  - 두 고유값 중 큰 쪽이 PC1인 이유는? 만약 두 고유값이 거의 같다면 PCA 결과를 어떻게 해석해야 하는가?
  - 고유벡터를 정규화(길이 1)하는 이유는 무엇인가?

**막히면 (30분 넘게 특성방정식이 안 풀리면)**: StatQuest "PCA main ideas" (20분)의 특성방정식 부분만 다시 보기. 그래도 안 되면 numpy 결과를 정답으로 받아들이고, "공분산 → det(Cov-λI)=0 → 고유벡터" 순서만 기억한 뒤 넘어가세요. W13 복습 구간에서 다시 만납니다.

```bash
git add week02/
git commit -m "W2 Day2: SVD vs eigendecomposition + hand-calculated PCA"
git push
```

---

## Day 3 (수요일) — 주성분 개수 결정 + 재구성 [1.5시간]

"2차원으로 줄인다"는 W1에서의 임의의 선택이었습니다. 오늘은 몇 개가 적절한지 데이터로 결정합니다.

### 00:00–00:45 | 누적 설명 분산 + Scree plot

- **학습 목표**: 임의로 정하던 `n_components`를 데이터 기반으로 결정하는 방법을 익힌다.
- **핵심 개념**: 누적 설명 분산(cumulative explained variance)이 목표 임계값(예: 90%)을 넘는 최소 주성분 개수를 고르는 것이 표준적인 방법입니다. Scree plot은 고유값이 급격히 꺾이는 지점(elbow)을 눈으로 찾는 보조 도구입니다.
- **구현 과제 (스스로 작성)**:
  ```python
  def find_k_for_variance(X_scaled: np.ndarray, threshold: float = 0.90) -> int:
      """
      요구사항:
      - n_components를 지정하지 않은 PCA()를 fit해 전체 성분을 계산한다.
      - explained_variance_ratio_의 누적합을 구한다.
      - 누적합이 threshold 이상이 되는 최소 k를 반환한다.
      """
      ...

  def plot_variance_analysis(X_scaled, k, save_path='week02/scree_plot.png'):
      """
      요구사항:
      - 왼쪽 subplot: 누적 설명 분산 곡선 + threshold 수평선 + k 수직선.
      - 오른쪽 subplot: 상위 10개 주성분의 설명 분산 비율 막대그래프 (scree plot).
      - save_path로 저장.
      """
      ...
  ```
- **확인 질문**:
  - W1에서 쓴 2차원은 digits 분산의 몇 %만 설명하는가? 그렇다면 W1의 시각화는 "정보 보존"이 아니라 무엇을 위한 것이었는가?
  - Scree plot에서 elbow가 뚜렷하지 않다면 어떻게 판단해야 하는가?

### 00:45–01:15 | 재구성(reconstruction) 오차 확인

- **학습 목표**: 차원 축소가 "무손실이 아니라는 것"을 직접 눈으로 확인한다.
- **핵심 개념**: `inverse_transform`은 k개 주성분으로 압축한 결과를 원래 차원으로 되돌립니다. 하지만 버린 (64-k)개 방향의 정보는 복구되지 않으므로 오차가 남습니다.
- **구현 과제 (스스로 작성)**:
  ```python
  def reconstruct_and_compare(X_scaled, digits, k):
      """
      요구사항:
      - PCA(n_components=k)로 fit_transform → inverse_transform으로 복원.
      - 원본과 복원본 사이 MSE를 계산해 출력.
      - 상위 5개 샘플에 대해 (원본 8x8 이미지, 복원된 8x8 이미지)를 2행 5열로 나란히 그려
        'week02/reconstruction_comparison.png'로 저장.
      - 주의: 복원된 값은 스케일된 공간에 있으므로 원본 픽셀값과 완전히 같지는 않다.
        시각 비교가 목적이라면 스케일을 되돌리는 것도 고려해볼 것.
      """
      ...
  ```
- **확인 질문**:
  - k를 5, 20, 40으로 바꿔가며 MSE와 복원 이미지가 어떻게 변하는가? 실행 전에 먼저 예상하고 비교하세요.
  - "SVD와 PCA의 차이"를 면접에서 설명할 때, 이 재구성 실습을 어떻게 예로 들 수 있는가?

### 01:15–01:30 | LeetCode: Valid Palindrome

- **구현 과제 (스스로 작성, 정답 코드 없음)**:
  ```python
  def is_palindrome(s: str) -> bool:
      """
      요구사항:
      - 영숫자만 남기고, 대소문자를 무시했을 때 앞뒤가 같은지 판정.
      - 리스트 comprehension을 한 번 이상 사용해볼 것 (W2 Python 목표와 연결).
      - 테스트: "A man, a plan, a canal: Panama" → True, "race a car" → False
      """
      ...
  ```
- **확인 질문**: 문자열을 뒤집어 비교하는 방법과 양끝에서 좁혀오는 투 포인터 방법 중, 공간 복잡도가 더 나은 쪽은? 왜인가?

```bash
git add week02/
git commit -m "W2 Day3: explained variance analysis + reconstruction"
git push
```

---

## Day 4 (목요일) — Python 함수형·예외처리로 파이프라인 정리 [1.5시간]

### 00:00–00:45 | comprehension·함수형 스타일

- **학습 목표**: 흩어져 있던 PCA 절차를 재사용 가능한 작은 함수들로 나누고, 두 개 이상의 데이터셋에 같은 코드를 적용할 수 있다.
- **핵심 개념**: 함수를 인자로 받는 함수(고차 함수)를 쓰면 "어떤 데이터셋인가"를 바꿔 끼울 수 있습니다. 방어적 코딩(예: 표준편차 0인 열)도 여기서 연습합니다.
- **구현 과제 (스스로 작성)**:
  ```python
  def load_and_scale(loader_fn):
      """
      요구사항:
      - sklearn dataset loader 함수(예: load_digits)를 인자로 받는다.
      - 평균 0, 표준편차 1로 직접 스케일링한다 (StandardScaler를 쓰지 말고 numpy로).
      - 표준편차가 0인 열이 있으면 0으로 나누기가 발생한다 — 어떻게 방어할지 스스로 결정할 것.
      - (X_scaled, y) 튜플 반환.
      """
      ...

  def compute_cumulative_variance(eigenvalues: np.ndarray) -> np.ndarray:
      """
      요구사항:
      - 고유값 배열(내림차순 가정)을 받아 누적 설명 분산 비율 배열을 반환한다.
      - np.cumsum을 쓰지 말고, comprehension 또는 functools.reduce로 한 번 구현해볼 것 (연습용).
      - 구현 후 np.cumsum 결과와 일치하는지 스스로 검증할 것.
      """
      ...

  def k_for_variance_threshold(cum_var: np.ndarray, threshold: float = 0.9) -> int:
      """
      요구사항:
      - 누적 분산 배열에서 threshold를 처음 넘는 인덱스 + 1을 반환한다.
      - 만족하는 값이 하나도 없으면 ValueError를 명확한 메시지와 함께 발생시킨다.
      """
      ...
  ```
- **실행 과제**: 위 함수들을 조합해 `load_digits`와 `load_wine` 두 데이터셋에 대해 "원래 차원 / 90% 설명에 필요한 차원"을 각각 출력하세요.
- **확인 질문**:
  - digits(64차원)와 wine(13차원)에서 90%에 필요한 차원 비율이 어떻게 다른가? 왜 그런 차이가 날까?
  - `reduce`로 짠 누적합과 `np.cumsum` 중 무엇을 실무에서 쓸 것인가? 왜인가?

### 00:45–01:15 | 예외처리: SimplePCA 견고하게 만들기

- **학습 목표**: 잘못된 사용을 조용히 통과시키지 않고, 명확한 에러로 알려주는 클래스를 설계할 수 있다.
- **핵심 개념**: 커스텀 예외 클래스를 만들면 호출하는 쪽에서 "어떤 종류의 실패인지" 구분해 처리할 수 있습니다. `raise ... from e`는 원래 원인을 보존합니다.
- **구현 과제 (스스로 작성)**: W1의 `SimplePCA`를 확장하세요.
  ```python
  class PCANotFittedError(Exception):
      """fit() 호출 전에 transform()을 호출했을 때"""

  class InvalidComponentsError(ValueError):
      """n_components가 유효 범위를 벗어났을 때"""

  # SimplePCA에 추가할 요구사항:
  # - fit(): n_components가 1 이상 n_features 이하가 아니면 InvalidComponentsError를
  #   "무엇이 잘못됐고 어떤 범위여야 하는지" 담은 메시지와 함께 raise.
  # - fit(): np.linalg.LinAlgError가 발생하면 원인을 보존한 채(raise ... from e)
  #   더 읽기 쉬운 에러로 감싸서 raise.
  # - transform(): fit되지 않았으면 PCANotFittedError raise.
  ```
- **검증 과제**: `n_components=999`로 fit을 시도했을 때, 그리고 fit 없이 transform을 호출했을 때 각각 의도한 예외가 실제로 발생하는지 `try/except`로 확인하는 테스트 코드를 스스로 작성하세요.
- **확인 질문**:
  - `InvalidComponentsError`를 `ValueError`를 상속해서 만든 이유는? 그냥 `Exception`을 상속하면 무엇이 달라지는가?
  - `raise ... from e`를 쓴 것과 안 쓴 것의 트레이스백 차이를 직접 실행해서 비교해보세요.

### 01:15–01:30 | LeetCode: Reverse Linked List

- **구현 과제 (스스로 작성, 정답 코드 없음)**:
  ```python
  class ListNode:
      def __init__(self, val=0, next=None):
          self.val = val
          self.next = next

  def reverse_list(head: ListNode) -> ListNode:
      """
      요구사항:
      - 연결 리스트를 제자리에서 뒤집어 새 head를 반환한다. O(n) 시간, O(1) 추가 공간 목표.
      - 검증용으로 리스트→파이썬 list 변환 헬퍼를 만들어 [1,2,3] → [3,2,1] 확인.
      """
      ...
  ```
- **확인 질문**: 반복(iterative) 방식과 재귀 방식 중 공간 복잡도가 다른 이유는? 노드가 100만 개면 어느 쪽이 위험한가?

```bash
git add week02/
git commit -m "W2 Day4: functional-style pipeline + custom exceptions"
git push
```

---

## Day 5 (금요일) — 복습 + 논문 [1.5시간]

### 00:00–00:30 | W2 최소 보장 자가 점검

아래 질문에 **자료를 보지 않고** 답해보세요. 답이 막히면 해당 Day로 돌아가세요.

```
□ PCA를 고유값분해로 손계산할 수 있는가?
  → Day2의 2x2 예제를 아무것도 안 보고 다시 풀어보기 (10분)

□ SVD와 PCA의 차이를 행렬 분해 관점에서 설명할 수 있는가?
  → 종이에 3문장으로 적어보기

□ 영공간(null space)이 PCA와 어떻게 연결되는가?
  → 공분산 행렬이 특이할 때 무슨 일이 일어나는지 설명

□ 몇 개의 주성분을 쓸지 어떻게 정하는가?
  → 자기가 구현한 find_k_for_variance의 논리를 말로 설명
```

### 00:30–01:00 | 논문: 역전파의 탄생

```
읽을 것: Rumelhart, Hinton, Williams (1986)
"Learning representations by back-propagating errors"
읽을 부분: Abstract + 핵심 아이디어 (전체 읽을 필요 없음)
시간: 20–30분

메모할 것:
1. 이 논문 이전에는 신경망을 어떻게 학습시켰는가? (또는 왜 못 시켰는가)
2. 역전파의 핵심 아이디어를 한 문장으로: _______________
3. "연쇄법칙(chain rule)"이 왜 등장하는가?
   → W7에서 SGD를 직접 구현할 때 다시 나옵니다. 지금은 이름만 익혀두면 충분.

영어 한 문장 준비 (자기 말로 쓸 것 — 아래는 참고용 뼈대):
"Backpropagation made it possible to train multi-layer neural networks by
efficiently computing gradients using the chain rule."
```

### 01:00–01:30 | 마무리 커밋 + README

```bash
git add .
git commit -m "W2 완료: LU분해·영공간, SVD로 PCA 검증, 함수형·예외처리"
git push

cat >> week02/README.md << 'EOF'
# W2: LU decomposition, null space, PCA via SVD

## W2 완료 항목 (스스로 구현)
- [ ] LU분해·영공간 탐구 (explore_lu, explore_null_space)
- [ ] SVD와 고유값분해 동치 검증 (compare_eig_vs_svd)
- [ ] 2x2 PCA 손계산 + numpy 검증
- [ ] 누적 설명 분산으로 k 결정 (find_k_for_variance) + 재구성 오차
- [ ] 함수형 파이프라인 + 커스텀 예외 (SimplePCA 확장)
- [ ] LeetCode: Valid Palindrome, Reverse Linked List

## 최소 보장 체크
- [ ] PCA를 고유값분해로 손계산 가능
- [ ] SVD와 PCA 차이를 행렬 분해 관점에서 설명 가능
EOF

git add week02/README.md
git commit -m "W2: update README"
git push
```

---

## 주말 — 심화 [토요일 2.5시간 / 일요일 2시간]

### 토요일 [2.5시간]

**[00:00–01:15] 다른 데이터셋에 파이프라인 적용 (일반화 검증)**

- **학습 목표**: Day4에서 만든 함수형 파이프라인이 digits 말고 다른 데이터에도 수정 없이 통하는지 확인한다. 통하지 않는다면 왜인지 찾아 고친다.
- **구현 과제 (스스로 작성)**: Day4의 `load_and_scale`, `compute_cumulative_variance`, `k_for_variance_threshold`를 **그대로 재사용해서** `load_wine`에 적용하고, 2차원 투영 결과를 클래스(0~2)별 색으로 산점도로 그려 `week02/wine_pca.png`로 저장하세요. 그래프 제목에 상위 2개 성분의 설명 분산 비율을 표시할 것.
- **확인 질문**:
  - Wine은 digits보다 클래스가 더 뚜렷하게 분리되는가? 차원 수와 클래스 수를 고려해 이유를 추론해보세요.
  - 파이프라인 함수를 하나라도 수정해야 했다면, 왜였는가? 그 수정이 "digits 전용 가정"을 코드에 심어놨다는 신호는 아니었는가?

**[01:15–02:30] MIT 18.06 Lecture 5–6 보충 (필요 시)**

```
Day1에서 LU분해가 충분히 이해됐으면 스킵 가능.
막혔던 부분(치환행렬, 가우스 소거 순서)이 있으면 지금 다시 보기.
Lecture 6: 열공간(column space)·영공간(null space) 통합 정리.
30분 넘게 막히면 → 넘어가고 "아직 모름: [개념]" 메모만 남기기.
```

### 일요일 [2시간]

**[00:00–01:00] W2 총 복습 — 아무것도 보지 않고 재구현**

이번 주 코드를 하나도 열지 않고, 아래 흐름을 처음부터 다시 짜보세요.

```
1. digits 또는 wine 로드 + 직접 표준화 (5분)
2. 공분산 행렬 계산 (5분)
3. 고유값분해로 고유값·고유벡터 (5분)
4. 누적 설명 분산으로 90% 임계값의 k 찾기 (10분)
5. k개로 투영 후 재구성, MSE 계산 (10분)
6. SVD로도 같은 결과가 나오는지 검증 (10분)

30분 이상 막히면 Day2~3 코드 참고 가능.
목표는 코드를 외우는 게 아니라 "순서와 이유"를 손에 익히는 것.
```

**영어로 설명 연습** (혼자 소리 내어, `___` 부분은 직접 측정한 값으로 채우기 — 외운 숫자 금지):

```
"PCA and SVD are closely related but not the same thing.
PCA is a technique for dimensionality reduction — it finds the directions
of maximum variance in the data. SVD is a more general matrix factorization
that decomposes any matrix into U, S, and V^T.
Eigendecomposing the covariance matrix gives the same principal directions
as applying SVD directly to the centered data matrix.
In my digits experiment, ___ principal components were enough to explain
90% of the variance, out of 64 original dimensions."
```

**[01:00–02:00] 주간 회고 + W3 준비**

```markdown
## W2 회고 (일요일에 작성)

### 달성한 것
- [ ] LU분해·영공간 이해
- [ ] SVD와 고유값분해 동치 검증
- [ ] 2x2 PCA 손계산 완료
- [ ] 주성분 개수 결정 (누적 분산 90%)
- [ ] 함수형 + 예외처리 파이프라인
- [ ] LeetCode: Valid Palindrome, Reverse Linked List

### 최소 보장 체크
- [ ] PCA를 고유값분해로 손계산 가능
- [ ] SVD와 PCA 차이 설명 가능

### 스스로 짠 코드에서 막힌 지점
(개념을 몰라서 막힌 것 / 파이썬 문법을 몰라서 막힌 것을 구분해서 적기 —
 이 구분이 다음 주에 어디에 시간을 쓸지 알려줍니다)

### 손계산이 안 됐다면 어느 단계였는가
(공분산 계산 / 특성방정식 / 고유벡터 구하기 중 어디)

### W3에 가져갈 것
(이번 주에 이해 못 하고 넘어간 것)

### 다음 주 첫 번째 할 일
(W3 Day1 무엇부터 시작할지)
```

---

## 막힐 때 대응 가이드

**개념이 막히면**:
```
1단계 (5분): 구글에 영어로 검색
  예: "why SVD equivalent to eigendecomposition of covariance matrix"
2단계 (10분): 3Blue1Brown 또는 StatQuest 관련 영상 검색
3단계 (20분): MIT 18.06 해당 강의 노트 확인
30분 넘어도 안 되면 → "아직 모름: [개념]" 메모하고 넘어가기
→ W13 복습 구간에서 다시 만납니다. 막히는 것은 정상 과정.
```

**직접 짠 코드가 막히면 (완성 코드를 찾아 베끼지 말고)**:
```
1. 에러 메시지 전체를 그대로 구글에 검색
2. 자주 나오는 에러:
   - LinAlgError: Singular matrix → np.linalg.matrix_rank로 rank 확인
   - RuntimeWarning: invalid value / divide by zero → 표준편차 0인 열 확인
   - Shape mismatch → print(array.shape)로 각 단계 차원 추적
3. Stack Overflow에서 "접근 방식"만 참고하고, 자기 코드로 다시 작성 (복붙 금지)
```

---

## W2 완료 기준

일요일 저녁에 아래를 할 수 있으면 W2 성공:

```
□ week02/ 폴더에 LU분해·영공간 탐구 코드가 있다 (스스로 작성)
□ SVD와 고유값분해가 같은 PCA 결과를 준다는 것을 직접 검증했다
□ 2x2 PCA를 종이로 풀고 numpy로 검증했다
□ scree_plot.png, reconstruction_comparison.png가 week02/ 폴더에 있다
□ 파이프라인 함수가 digits와 wine 양쪽에 수정 없이 적용된다
□ SimplePCA가 잘못된 입력에 대해 명확한 커스텀 예외를 던진다
□ LeetCode 2문제를 스스로 다시 풀 수 있다

절반(4개 이상) 달성하면 W3로 진행.
전부 못 해도 W3로 진행 — 이해 못 한 부분은 이후 주차에서 다시 나옴.
```

---

## W3 첫 할 일 미리 보기

W3 Day1에 열어야 할 것:
1. `week03/` 폴더 생성
2. sklearn `LogisticRegression` + 혼동행렬·ROC 실습 (공개 이진분류 데이터)
3. 막히면 → "로그우도가 왜 손실함수인가?" → 베이즈 정리·MLE 역추적 시작

---

*이 계획대로 완벽하게 안 돼도 됩니다.
W2의 진짜 목표는 "PCA를 여러 각도(고유값분해·SVD·손계산)에서 보고 같은 결론에 도달하는 경험"입니다.
그리고 그 경험은 남이 짜준 코드를 실행해서가 아니라, 스펙만 보고 직접 짜다가 막히고 고치는 과정에서 만들어집니다.*

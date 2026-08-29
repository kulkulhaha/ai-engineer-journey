# W1 구체적 실행 계획 (v2 — 학습 가이드형)

> **주제**: 선형대수 재활성화 + 환경 세팅 + sklearn PCA 실습
>
> **사용 데이터셋 주의**: 이 계획은 sklearn `load_digits()`(8×8 픽셀, 1797개)를 씁니다.
> 이건 "미니 MNIST"이지, 진짜 MNIST(28×28, 7만 개)가 아닙니다. 아래에서 편의상 "digits"로 부릅니다.
>
> **총 목표 시간**: 10–12시간 (v7 커리큘럼 블록 A 기준)
> **기준**: 평일 1.5시간 + 토요일 2.5시간 + 일요일 2시간
>
> **v1과의 차이**: 이 버전은 완성된 실행 코드를 주지 않습니다. 대신 학습 목표·핵심 개념·구현 요구사항(스펙)·확인 질문을 제시합니다. 코드는 스스로 작성하는 것이 이번 주의 진짜 과제입니다. 환경 설정처럼 학습 내용과 무관한 명령어만 그대로 제시합니다.

---

## W1 목표 (이것만 달성하면 성공)

1. **환경**: Colab, GitHub repo, Git 워크플로우 작동
2. **실습**: sklearn PCA로 digits(미니 MNIST) 차원축소 → 시각화를 스스로 구현
3. **이론**: "PCA가 왜 고유벡터인가" 설명 가능
4. **재구현**: numpy로 행렬 곱·전치·역행렬의 기하학적 의미를 직접 확인
5. **최소 보장**: 행렬 곱의 기하학적 의미 (회전·스케일링) 설명 가능
6. **⭐영어**: IELTS 진단 모의고사 1회 완료

---

## Day 1 (월요일) — 환경 세팅 + 데이터 이해 [1.5시간]

환경 세팅은 생각보다 오래 걸립니다. 처음 30분이 막히더라도 정상입니다.

### 00:00–00:30 | GitHub + Git 세팅 (그대로 실행)

```bash
# 1. github.com 접속 → 새 레포 생성
#    이름: ai-engineer-journey, Public, README 체크, .gitignore → Python

# 2. 로컬 터미널 (또는 Colab 터미널)
git config --global user.name "Young Eun Kim"
git config --global user.email "your@email.com"
git clone https://github.com/YOUR_USERNAME/ai-engineer-journey.git
cd ai-engineer-journey

# 3. 첫 커밋
echo "# W1: Linear Algebra & PCA" > week01/README.md
git add .
git commit -m "W1: initialize week01"
git push origin main
```

**막히면**: GitHub Docs "create a repo" 검색 → 5분 이내 해결 안 되면 넘어가고 Colab만 씀.

### 00:30–01:00 | PCA 실습 준비 — 데이터 이해하기

- **학습 목표**: digits 데이터셋의 구조(shape, label)를 이해하고, "차원(dimension)"이 무엇을 의미하는지 설명할 수 있다.
- **핵심 개념**: 차원 = feature 개수. digits는 8×8 = 64차원. 사람은 3차원까지만 직관적으로 볼 수 있는데, 64차원 데이터를 어떻게 "볼" 것인가 — 이게 이번 주 전체의 질문입니다.
- **구현 과제 (스스로 작성)**: 아래 시그니처를 채우는 스크립트를 작성하세요.
  ```python
  def load_and_inspect_digits():
      """
      요구사항:
      - sklearn.datasets.load_digits()로 데이터를 불러온다.
      - X.shape, y.shape을 출력한다 (X: (1797, 64) 예상, y: (1797,) 예상).
      - X의 전체 최소/최대 픽셀 값을 출력한다.
      - StandardScaler로 스케일링 전/후 값의 범위를 비교해서 출력한다.
      """
      ...
  ```
- **확인 질문**:
  - 64라는 숫자는 정확히 어디서 나왔는가?
  - 스케일링 전/후 값의 범위가 왜 달라지는가? 스케일링을 안 하면 PCA 결과가 왜 왜곡될 수 있는가?

### 01:00–01:30 | ⭐영어: IELTS 진단

```bash
# 코드 아님 — 체크리스트
# 1. 무료 IELTS 모의 진단 테스트 검색 (예: British Council, IELTS.com 진단)
# 2. 리스닝/리딩 최소 1개 섹션 풀기
# 3. 현재 대략적인 레벨과 B2까지 격차를 한 줄로 메모
```

---

## Day 2 (화요일) — 선형대수 이론 + PCA 파이프라인 [1.5시간]

### 00:00–00:45 | MIT 18.06 핵심 강의

```
시청: MIT 18.06 Lecture 1 (Gilbert Strang)
URL: https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/

오늘 집중할 것 (전체 다 볼 필요 없음):
- 행렬 곱의 4가지 해석 방법 (column picture 특히 중요)
- 연립방정식과 행렬의 관계

메모 원칙: 이미 아는 것(체크만) / 잊었던 것(한 줄 정리) / 처음 보는 것(별표+설명)
```

**막히는 지점 예상**: "열 공간이 뭔가?", "선형 독립이 왜 중요한가?" → 넘어가세요. W2에서 다시 나옵니다.

### 00:45–01:30 | sklearn PCA 파이프라인 구현

- **학습 목표**: 표준화 → PCA 적용 → 결과 해석까지 전체 파이프라인을 스스로 조립할 수 있다.
- **핵심 개념**: `explained_variance_ratio_`는 각 주성분이 전체 분산 중 몇 %를 설명하는지를 뜻합니다. 왜 첫 두 주성분만으로 64차원을 "요약"할 수 있다고 기대하는지 생각해보세요.
- **구현 과제 (스스로 작성)**:
  ```python
  def run_pca_pipeline(X, n_components=2):
      """
      요구사항:
      - StandardScaler로 X를 스케일링한다.
      - sklearn.decomposition.PCA(n_components=n_components)로 fit_transform한다.
      - 반환값: (X_reduced, explained_variance_ratio) 튜플.
      - 별도로 시각화 함수를 만들어, X_reduced를 라벨(y)별로 색을 다르게 해서
        scatter plot으로 그리고 'week01/pca_digits.png'로 저장한다.
      """
      ...
  ```
- **확인 질문**:
  - `n_components=2`일 때와 `n_components=10`일 때 설명 분산이 얼마나 달라지는가? 실행 전에 먼저 예상하고, 실행 후 비교하세요.
  - 시각화에서 같은 숫자끼리 뭉쳐 있는가? 뭉쳐 있지 않다면 왜일까?

**막히면**: 에러 메시지 전체를 그대로 구글에 검색 → Stack Overflow 상위 답변 확인.

**커밋**:
```bash
git add week01/pca_digits.png week01/day2_pca_pipeline.py
git commit -m "W1 Day2: PCA pipeline on digits dataset"
git push
```

---

## Day 3 (수요일) — 고유값·고유벡터 + numpy 재구현 [1.5시간]

### 00:00–00:45 | 3Blue1Brown 고유벡터

```
시청: 3Blue1Brown YouTube "Eigenvectors and eigenvalues" (13분)
URL: https://www.youtube.com/watch?v=PFDu9oVAE-g

시청하며 메모할 것:
- 고유벡터란 행렬이 적용됐을 때 방향이 바뀌지 않는 벡터
- 고유값이란 그 방향으로 얼마나 늘어나는지
- PCA는 공분산 행렬의 고유벡터 방향으로 데이터를 투영
```

### 00:45–01:30 | numpy 행렬 연산 직접 확인

- **학습 목표**: 행렬 곱·전치·역행렬을 코드로 직접 실행해보고, 그 결과가 "기하학적으로" 무엇을 의미하는지 눈으로 확인한다.
- **핵심 개념**: `A @ v`는 벡터 `v`를 행렬 `A`가 정의하는 변환(회전·스케일링 등)으로 옮기는 것입니다. 역행렬이 없다는 것은 그 변환이 정보를 "압축"해서 되돌릴 수 없다는 뜻입니다.
- **구현 과제 (스스로 작성)**: 아래 다섯 가지를 확인하는 스크립트를 작성하세요. 함수로 나누든 순서대로 스크립트로 짜든 자유입니다.
  1. 임의의 2×2 행렬 `A`, `B`를 정의하고 `A @ B`를 계산해 출력한다.
  2. 단위벡터 `[1,0]`, `[0,1]`에 `A`를 적용한 결과가 `A`의 첫 번째/두 번째 열과 같은지 비교한다.
  3. `A.T`(전치)를 출력한다.
  4. `A`의 역행렬을 계산하고, `A @ A_inv`가 (부동소수점 오차 범위 내에서) 단위행렬인지 `np.allclose` 또는 `np.round`로 확인한다.
  5. 두 번째 행이 첫 번째 행의 배수인 "특이(singular) 행렬"을 하나 만들어 `np.linalg.det()`으로 행렬식을 확인하고, 역행렬 계산을 시도했을 때 어떤 에러가 나는지 관찰한다.
- **확인 질문**:
  - `A @ B`는 기하학적으로 "B를 먼저 적용하고 A를 적용하는 것"이라는 말이 왜 맞는지, 위 2번 결과로 설명할 수 있는가?
  - 행렬식이 0이면 왜 역행렬이 존재하지 않는가?

**막히면**: "왜 역행렬이 존재하지 않는가 determinant zero" 같은 키워드로 검색 → 3Blue1Brown "The determinant" 영상 참고.

**커밋**:
```bash
git add week01/day3_matrix_ops.py
git commit -m "W1 Day3: matrix multiply/transpose/inverse exploration"
git push
```

---

## Day 4 (목요일) — PCA 수학 연결 + Python OOP [1.5시간]

### 00:00–00:45 | 공분산 행렬 → 고유값 분해 → 투영

- **학습 목표**: sklearn의 PCA가 내부적으로 무엇을 계산하는지 numpy만으로 재현하고, 결과가 sklearn과 (부호를 제외하고) 일치함을 스스로 확인한다.
- **핵심 개념**: PCA = ① 공분산 행렬 계산 → ② 고유값 분해 → ③ 고유값이 큰 순서로 상위 k개 고유벡터를 골라 투영. 고유벡터의 방향은 ±로 자유롭기 때문에 sklearn 결과와 부호가 다를 수 있습니다.
- **구현 과제 (스스로 작성)**:
  ```python
  def manual_pca_via_eigh(X_scaled, k=2):
      """
      요구사항:
      - 공분산 행렬을 (X_scaled.T @ X_scaled) / (n_samples - 1) 로 계산한다.
      - np.linalg.eigh로 고유값·고유벡터를 구하고, 고유값 내림차순으로 정렬한다.
      - 상위 k개 고유벡터로 X_scaled를 투영한 결과를 반환한다.
      - sklearn의 PCA(n_components=k) 결과와 첫 샘플을 비교해,
        np.allclose(np.abs(manual), np.abs(sklearn_result))가 True인지 확인한다.
      """
      ...
  ```
- **확인 질문**:
  - 두 결과의 부호가 다르게 나올 수 있는데, 왜 "같은 결과"로 봐도 되는가?
  - `eigh`를 쓰는 이유는 무엇인가 (`eig`와 무슨 차이가 있는가)?

### 00:45–01:30 | Python OOP: SimplePCA 클래스

- **학습 목표**: scikit-learn 스타일의 `fit`/`transform` 인터페이스를 왜 그렇게 나누는지 이해하고 직접 설계할 수 있다.
- **핵심 개념**: `fit`은 학습(파라미터 계산), `transform`은 그 파라미터로 새 데이터를 변환. 이 분리 덕분에 학습 데이터로 `fit`한 뒤 다른 데이터에도 같은 변환을 재사용할 수 있습니다.
- **구현 과제 (스스로 작성)**: 아래 시그니처만 주어집니다. 본문은 스스로 채우세요.
  ```python
  class SimplePCA:
      def __init__(self, n_components: int):
          ...

      def fit(self, X: np.ndarray) -> "SimplePCA":
          """Day4 오전에 만든 manual_pca_via_eigh 로직을 재사용."""
          ...

      def transform(self, X: np.ndarray) -> np.ndarray:
          """fit되지 않은 상태로 호출되면 명확한 에러를 발생시킬 것."""
          ...

      def fit_transform(self, X: np.ndarray) -> np.ndarray:
          ...
  ```
- **확인 질문**:
  - `fit`을 호출하지 않고 `transform`을 호출하면 어떤 에러를 어떻게 발생시켰는가? 왜 조용히 넘기면 안 되는가?
  - 이 클래스에 `random_state`처럼 나중에 필요할 수 있는 설정값을 추가하고 싶다면 어떻게 구조를 바꿀 것인가?

**커밋**:
```bash
git add week01/day4_manual_pca.py week01/simple_pca.py
git commit -m "W1 Day4: manual PCA via eigh + SimplePCA class"
git push
```

---

## Day 5 (금요일) — 알고리즘 + 논문 + 마무리 [1.5시간]

### 00:00–00:30 | LeetCode (시간제한: 문제당 20분)

- **구현 과제 (스스로 작성, 정답 코드 없음)**:
  ```python
  def two_sum(nums: list[int], target: int) -> list[int]:
      """두 수의 합이 target이 되는 두 인덱스를 반환. O(n) 목표."""
      ...

  def max_profit(prices: list[int]) -> int:
      """한 번 사고 한 번 팔아서 얻을 수 있는 최대 이익. O(n) 목표."""
      ...
  ```
- **확인 질문**:
  - Two Sum을 O(n²)이 아니라 O(n)으로 풀려면 어떤 자료구조가 필요한가? 왜 그 자료구조가 조회를 빠르게 해주는가?
  - 20분 안에 못 풀었다면, 어느 지점에서 막혔는지 한 줄로 적어두세요 (다음에 같은 유형을 만났을 때 참고).

**막히면**: 문제를 다시 정확히 읽기 → 예제를 손으로 따라가기 → 그래도 안 되면 "two sum approach" 검색(정답 코드를 바로 보지 말고 접근 방식만 확인).

### 00:30–01:00 | Turing (1950) 논문 맥락 파악

```
읽을 것: Turing (1950) "Computing Machinery and Intelligence"
읽을 부분: Section 1 (The Imitation Game), Section 6 (Objections)
시간: 20분

메모:
1. Turing Test가 "지능"을 어떻게 정의하는가?
2. 왜 이 논문이 AI의 시작으로 불리는가?
3. 2026년 LLM 관점에서 Turing Test를 통과했다고 볼 수 있는가?
```

### 01:00–01:30 | 마무리 커밋 + README 업데이트

```bash
git add .
git commit -m "W1 완료: PCA 파이프라인, 행렬 연산 탐구, SimplePCA, Two Sum/Max Profit"
git push

cat >> week01/README.md << 'EOF'

## W1 완료 항목 (스스로 구현)
- [ ] sklearn PCA 파이프라인 (run_pca_pipeline)
- [ ] 행렬 곱·전치·역행렬 탐구 (day3_matrix_ops.py)
- [ ] numpy 고유값분해로 PCA 재구현 (manual_pca_via_eigh)
- [ ] SimplePCA 클래스
- [ ] Two Sum / Best Time to Buy and Sell Stock
- [ ] Git 워크플로우 정착
EOF
```

---

## 주말 — 심화 [토요일 2.5시간 / 일요일 2시간]

### 토요일 [2.5시간]

**[00:00–01:30] 고유값·고유벡터를 눈으로 검증**

- **학습 목표**: `A @ v = λ @ v`가 왜 성립하는지, 그리고 왜 고유벡터만 방향이 유지되는지 시각적으로 확인한다.
- **구현 과제 (스스로 작성)**:
  ```python
  def verify_eigen(A: np.ndarray):
      """
      요구사항:
      - np.linalg.eig(A)로 고유값·고유벡터를 구한다.
      - 각 고유벡터 v, 고유값 λ에 대해 A @ v ≈ λ * v 인지 np.allclose로 확인한다.
      - 몇 개의 일반 벡터(예: [1,1], [-1,1])와 고유벡터 각각에 A를 적용한 뒤
        (변환 전 화살표, 변환 후 화살표)를 그려 비교한다 — 고유벡터만 방향이
        유지되는 것을 그림으로 확인.
      """
      ...
  ```
- **확인 질문**: 일반 벡터는 변환 후 방향이 바뀌는데, 고유벡터는 왜 바뀌지 않는가? 이걸 3Blue1Brown 영상 설명과 연결해서 자기 말로 설명해보세요.

**[01:30–02:30] MIT 18.06 Lecture 6 보충 (필요 시)**

Day 2에서 잘 이해됐으면 스킵해도 됩니다. 막혔던 개념(열 공간, 선형 독립)이 있으면 지금 해결하세요. 30분 이상 막히면 넘어가고 W2에서 다시 만날 때 해결.

### 일요일 [2시간]

**[00:00–01:00] W1 총 복습 — 아무것도 보지 않고 재구현**

- **구현 과제**: Day1~Day4에서 만든 것을 하나도 열어보지 않고, 아래 흐름을 처음부터 다시 짜보세요.
  1. digits 로드 + 스케일링 (5분)
  2. 공분산 행렬 계산 (5분)
  3. 고유값 분해 (5분)
  4. 상위 2개 고유벡터로 투영 (5분)
  5. sklearn PCA와 결과 비교 (5분)
  - 30분 이상 막히면 이번 주에 만든 코드 참고 가능. 중요한 건 정답을 외우는 게 아니라 흐름을 기억하는 것입니다.

**영어로 설명 연습** (혼자 소리 내어 말하기). `___%` 부분은 반드시 직접 코드를 돌려 나온 값으로 채우세요 (외운 숫자 금지):

```
"PCA is a dimensionality reduction technique.
It finds the directions of maximum variance in the data,
which are the eigenvectors of the covariance matrix.
By projecting data onto the top k eigenvectors,
we can reduce dimensions while preserving the most information.
For example, with the sklearn digits dataset (64 dimensions),
the top 2 principal components explain about ___% of the variance
(fill in the number you measured yourself),
which is still enough to visually separate different digit clusters."
```

**[01:00–02:00] 주간 회고 + W2 준비**

```markdown
## W1 회고 (일요일에 작성)

### 달성한 것
- [V] sklearn PCA 파이프라인 스스로 구현
- [V] 행렬 연산의 기하학적 의미 확인
- [V] numpy 고유값분해로 PCA 재구현
- [V] SimplePCA 클래스 작성
- [V] Two Sum / Max Profit 구현
- [ ] IELTS 진단 완료

### 최소 보장 체크
- [ ] 행렬 곱의 기하학적 의미 설명 가능
- [ ] "PCA가 왜 고유벡터인가" 설명 가능 (한국어 가능, 영어 시도)

### 예상보다 오래 걸린 것
(솔직하게 적기 — 코드를 못 짜서 막힌 부분과 개념을 몰라서 막힌 부분을 구분해서 적으면 좋음)

### 스스로 짠 코드에서 정답 코드(참고 자료)와 다르게 접근한 부분
(있다면 — 완전히 다른 접근이었는지, 왜 그렇게 했는지)

### W2에 가져갈 것
(이번 주에 이해 못 하고 넘어간 것)

### 다음 주 첫 번째 할 일
(W2 Day1 무엇부터 시작할지)
```

---

## 막힐 때 대응 가이드

**개념이 막히면**:
```
1단계 (5분): 구글에 영어로 검색 (예: "why PCA uses eigenvectors covariance matrix")
2단계 (10분): 3Blue1Brown 관련 영상 검색
3단계 (20분): MIT 18.06 해당 강의 노트 확인
30분 넘어도 해결 안 되면 → "아직 모름: [개념]" 메모하고 다음으로 넘어가기
→ W2 또는 W3에서 다시 만날 때 해결. 막히는 것은 실력 부족이 아니라 정상 과정.
```

**직접 짠 코드가 막히면 (완성 코드를 찾아 베끼지 말고)**:
```
1. 에러 메시지 전체를 그대로 구글에 검색
2. 자주 나오는 에러:
   - ModuleNotFoundError: pip install [라이브러리명]
   - Shape mismatch: print(array.shape)로 차원 확인
   - CUDA/GPU 에러: Colab 런타임 → GPU로 변경
3. Stack Overflow에서 "접근 방식"만 참고하고, 자기 코드로 다시 옮겨서 작성 (그대로 복붙 금지)
```

---

## W1 완료 기준

일요일 저녁에 아래를 할 수 있으면 W1 성공:

```
□ github.com/YOUR_USERNAME/ai-engineer-journey 에 코드가 올라가 있다
□ digits(미니 MNIST) PCA 시각화 이미지를 스스로 만든 코드로 생성했다
□ 행렬 곱·전치·역행렬 탐구 스크립트를 스스로 작성했다
□ SimplePCA 클래스가 fit/transform 없이 에러 없이 동작한다
□ "PCA가 왜 고유벡터인가"를 3문장으로 설명할 수 있다
□ Two Sum / Max Profit을 스스로 다시 풀 수 있다
□ IELTS 진단 모의고사를 완료했다

절반(4개 이상) 달성하면 W2로 진행.
전부 못 해도 W2로 진행 — 이해 못 한 부분은 W2에서 다시 나옴.
```

---

## W2 첫 할 일 미리 보기

W2 Day1에 열어야 할 것:
1. `week02/` 폴더 생성
2. numpy로 PCA를 처음부터 구현 시도 (W1보다 더 적은 가이드로)
3. 막히면 → "LU분해·영공간이 뭔가?" → MIT 18.06 Lec 4–9 역추적

---

*이 계획대로 완벽하게 안 돼도 됩니다. W1의 진짜 목표는 "직접 만들어보는 습관"이에요.
코드를 남이 준 대로 실행하는 것이 아니라, 스펙만 보고 스스로 짜고, 틀리고, 고치는 과정 자체가 W1의 핵심입니다.*

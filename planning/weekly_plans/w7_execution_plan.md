# W7 구체적 실행 계획 (학습 가이드형)

> **주제**: 다변수 미적분(편미분·그래디언트·연쇄법칙) 재활성화 + numpy SGD 직접 구현 + 선형/로지스틱 회귀 재학습
>
> **사용 데이터셋 주의**: 이번 주는 그래디언트가 "맞게" 계산됐는지 스스로 검증하는 것이 핵심이라, `sklearn.datasets.make_regression`·`make_classification`으로 만든 합성 데이터를 기본으로 씁니다. 정답(진짜 계수)을 알고 시작하기 때문에 SGD가 그 값에 수렴하는지 직접 확인할 수 있습니다. 주말에는 W1~W3에서 쓴 `load_digits`/`load_wine`에도 같은 코드를 적용해봅니다.
>
> **총 목표 시간**: 10–12시간 (v7 커리큘럼 블록 A 기준)
> **기준**: 평일 1.5시간 + 토요일 2.5시간 + 일요일 2시간
>
> **코드 제시 방식**: 완성된 실행 코드를 주지 않습니다. 학습 목표·핵심 개념·구현 요구사항(스펙)·확인 질문만 제시합니다. 코드는 스스로 작성하세요. Git/설치 같은 보일러플레이트만 그대로 제시합니다.
>
> **전제**: W3에서 만든 로지스틱 회귀 코드, W1의 `SimplePCA`처럼 W3의 모델 래퍼가 있다면 이번 주에 "직접 만든 옵티마이저로 다시 학습시키는" 방식으로 재사용합니다. 없어도 문제 없습니다 — 이번 주 코드는 독립적으로 완성됩니다.

---

## W7 목표 (이것만 달성하면 성공)

1. **이론**: 편미분·그래디언트가 무엇이고, 왜 "그래디언트의 반대 방향"이 함수를 가장 빠르게 줄이는 방향인지 설명할 수 있다
2. **재구현**: numpy만으로 SGD 옵티마이저를 처음부터 구현하고, 선형회귀·로지스틱 회귀 학습에 적용한다
3. **시각화**: 학습 곡선(loss vs iteration)을 그려 SGD가 실제로 수렴하는지 눈으로 확인한다
4. **비교**: 배치 GD·미니배치 SGD·모멘텀의 차이를 직접 실험으로 비교한다
5. **최소 보장**: 연쇄법칙을 역전파에 적용해 설명 가능 / SGD·Adam 차이를 모멘텀 관점에서 설명 가능

---

## Day 1 (월요일) — 편미분·그래디언트 재활성화 [1.5시간]

W1~W6은 "이미 계산된 손실"을 다뤘습니다. 이번 주는 "그 손실을 어떻게 줄이는가"로 넘어갑니다.

### 00:00–00:45 | 개념: 편미분 · 그래디언트 벡터 · 방향 도함수

```
시청: 3Blue1Brown "Essence of calculus" 중 "Gradient descent" 챕터 (약 20분)
URL: https://www.youtube.com/watch?v=IHZwWFHWa-w
병행 (선택): 같은 시리즈의 "Partial derivatives" 챕터

메모할 것 (개념만, 코드 없음):
- 편미분: 다른 변수를 고정하고 한 변수에 대해서만 변화율을 본 것
- 그래디언트 벡터: 모든 편미분을 모아놓은 벡터. 함수가 가장 가파르게 "증가"하는 방향을 가리킴
- 그래서 손실을 줄이려면 그래디언트의 "반대" 방향으로 이동
- 연쇄법칙: 합성함수의 미분. f(g(x))의 미분은 f'(g(x)) * g'(x)
```

**막히는 지점 예상**: "왜 그래디언트가 가장 가파른 증가 방향인가?"는 방향 도함수(directional derivative)와 코사인 유사도로 증명되는데, 지금은 증명보다 "그렇게 정리된다"는 결론과 직관(등고선에 수직인 방향)만 받아들이고 넘어가세요.

### 00:45–01:30 | 수치미분으로 그래디언트 직접 확인

- **학습 목표**: 해석적으로 미분식을 유도하지 않고도, 수치적으로 그래디언트를 근사할 수 있음을 이해하고, 그 근사가 해석적 미분과 얼마나 일치하는지 검증할 수 있다.
- **핵심 개념**: `f(x+h) - f(x-h)) / (2h)` 형태의 중심차분(central difference)이 전진차분보다 오차가 작습니다. 이 방법은 "내 손으로 유도한 그래디언트 공식이 맞는지" 검증하는 용도로 이번 주 내내 씁니다(gradient checking).
- **구현 과제 (스스로 작성)**:
  ```python
  def numerical_gradient(f, x: np.ndarray, h: float = 1e-5) -> np.ndarray:
      """
      요구사항:
      - f: np.ndarray를 받아 스칼라를 반환하는 함수 (예: f(x) = x[0]**2 + x[1]**2).
      - x의 각 성분에 대해 중심차분으로 편미분을 근사해 그래디언트 벡터를 반환한다.
      - 반복문을 써도 되고, 벡터화해도 된다.
      """
      ...
  ```
- **검증 과제**: `f(x, y) = x**2 + y**2` 처럼 손으로 미분식을 아는 함수를 하나 고르고, 특정 점에서 `numerical_gradient`의 결과와 직접 손으로 유도한 해석적 그래디언트를 `np.allclose`로 비교하세요.
- **확인 질문**:
  - `h`를 `1e-2`처럼 크게 하면, 또는 `1e-12`처럼 너무 작게 하면 각각 어떤 오차가 생기는가? (힌트: 하나는 근사 오차, 하나는 부동소수점 반올림 오차)
  - 3차원 이상의 입력에도 이 함수가 그대로 동작하는가?

**막히면**: "gradient checking central difference" 검색. 결과가 안 맞으면 먼저 `h` 값을 바꿔보고, 그래도 안 맞으면 해석적 미분식 자체를 종이에 다시 유도해보세요.

```bash
mkdir -p week07
git add week07/
git commit -m "W7 Day1: numerical gradient checking"
git push
```

---

## Day 2 (화요일) — 선형회귀를 배치 경사하강법으로 [1.5시간]

### 00:00–00:45 | MSE 손실의 그래디언트를 손으로 유도

- **학습 목표**: 선형회귀의 MSE 손실을 가중치에 대해 직접 미분해, "왜 그래디언트 공식이 그렇게 생겼는지" 설명할 수 있다.
- **핵심 개념**: `L(w) = (1/n) * sum((Xw - y)^2)`을 `w`로 미분하면 `dL/dw = (2/n) * X^T @ (Xw - y)`가 나옵니다. 이 유도 과정에 연쇄법칙이 쓰입니다(제곱 함수의 미분 × 내부 함수의 미분).
- **과제 (종이 먼저)**:
  1. 위 손실함수를 `w`에 대해 직접 미분해 그래디언트 공식을 유도하세요. 벡터/행렬 미분이 낯설면 먼저 1차원(특성 1개, 절편 없음)으로 유도한 뒤 행렬 형태로 일반화하세요.
  2. Day1에서 만든 `numerical_gradient`로 작은 합성 데이터(`make_regression`, 특성 2~3개)에서 이 손실함수의 그래디언트를 수치적으로 계산하고, 손으로 유도한 해석적 공식의 결과와 비교하세요.
- **확인 질문**:
  - 왜 손실함수 앞에 `1/n`을 붙이는가? 안 붙이면 무엇이 달라지는가? (힌트: 데이터 개수에 따라 그래디언트 크기가 어떻게 변하는지)
  - `X^T @ (Xw - y)`에서 `X^T`가 왜 필요한가? (힌트: 행렬 곱이 성립하려면 shape이 맞아야 합니다 — W1에서 다룬 내용과 연결)

### 00:45–01:30 | 배치 경사하강법으로 선형회귀 학습

- **핵심 개념**: 배치 경사하강법(Batch GD)은 매 스텝마다 전체 데이터로 그래디언트를 계산해 한 번 업데이트합니다. 학습률(learning rate)이 너무 크면 발산하고, 너무 작으면 수렴이 느립니다.
- **구현 과제 (스스로 작성)**:
  ```python
  def batch_gradient_descent(
      X: np.ndarray, y: np.ndarray,
      lr: float = 0.01, n_iters: int = 500
  ) -> tuple[np.ndarray, list[float]]:
      """
      요구사항:
      - w를 0 또는 작은 난수로 초기화한다.
      - 매 iteration마다 Day2에서 유도한 그래디언트 공식으로 w를 업데이트한다.
      - 매 iteration의 MSE 손실을 리스트에 기록한다.
      - (최종 w, 손실 기록 리스트)를 반환한다.
      """
      ...

  def plot_learning_curve(losses: list[float], save_path: str = "week07/gd_learning_curve.png"):
      """
      요구사항:
      - x축: iteration, y축: loss. 로그 스케일(y축)로도 하나 더 그려서 비교.
      - save_path로 저장.
      """
      ...
  ```
- **검증 과제**: `make_regression`으로 만든 데이터는 사이킷런이 내부적으로 쓴 "진짜 계수"를 알고 있습니다(`coef` 반환값). 학습된 `w`가 그 값에 가까워지는지 확인하세요.
- **확인 질문**:
  - 학습률을 `0.001`, `0.01`, `0.5`로 바꿔가며 학습 곡선이 어떻게 달라지는가? 발산하는 학습률을 하나 찾아보세요.
  - `n_iters`를 늘리면 손실이 계속 줄어드는가, 아니면 어느 지점부터 거의 안 줄어드는가? 왜 그런가?

**막히면**: 학습이 발산(loss가 점점 커짐)하면 → 학습률을 10배 낮춰보기. 손실이 전혀 안 줄어들면 → 그래디언트 부호가 반대로 들어갔는지(`+=` vs `-=`) 확인.

```bash
git add week07/
git commit -m "W7 Day2: batch gradient descent for linear regression"
git push
```

---

## Day 3 (수요일) — SGD·미니배치로 로지스틱 회귀 학습 [1.5시간]

### 00:00–00:45 | 왜 "확률적"인가 — SGD와 미니배치

- **학습 목표**: 배치 GD와 SGD·미니배치 SGD의 차이를 계산 비용과 수렴 패턴(노이즈) 관점에서 설명할 수 있다.
- **핵심 개념**: SGD는 매 스텝마다 데이터 1개(또는 미니배치)만으로 그래디언트를 "근사"해 업데이트합니다. 그래서 학습 곡선이 배치 GD보다 들쭉날쭉하지만, 한 iteration당 계산량이 훨씬 적어 대규모 데이터에 유리합니다.
- **읽을 것**: W3에서 만든 로지스틱 회귀 코드(또는 sklearn `LogisticRegression`)를 다시 열어, 손실함수가 로그우도(음의 로그우도, cross-entropy)였음을 복습하세요. 오늘은 이 손실의 그래디언트를 SGD로 최적화합니다.
- **확인 질문 (자료 없이)**:
  - 데이터가 100만 개일 때 배치 GD 한 스텝과 SGD 한 스텝 중 어느 쪽이 더 빠른가? 왜 그런데도 배치 GD를 아예 안 쓰지는 않는가?
  - 미니배치 크기를 1(순수 SGD)에서 전체 데이터 크기(배치 GD)로 늘려가면 학습 곡선의 "노이즈"는 어떻게 변할 것 같은가? (실행하기 전에 먼저 예상해보세요)

### 00:45–01:30 | 로지스틱 회귀를 미니배치 SGD로 재구현

- **핵심 개념**: 로지스틱 회귀의 그래디언트는 선형회귀와 형태가 비슷합니다(`X^T @ (sigmoid(Xw) - y)`). 왜 그렇게 되는지는 연쇄법칙으로 유도됩니다: 시그모이드 함수의 미분과 로그우도 미분이 곱해지면서 상당 부분이 상쇄되기 때문입니다.
- **구현 과제 (스스로 작성)**:
  ```python
  def sigmoid(z: np.ndarray) -> np.ndarray:
      """요구사항: 오버플로우 방지(예: z가 매우 큰 음수일 때)를 고려해 구현."""
      ...

  def minibatch_sgd_logistic(
      X: np.ndarray, y: np.ndarray,
      lr: float = 0.1, batch_size: int = 32, n_epochs: int = 50
  ) -> tuple[np.ndarray, list[float]]:
      """
      요구사항:
      - 매 epoch마다 데이터를 셔플한 뒤 batch_size 단위로 나눠 그래디언트 업데이트.
      - 각 미니배치의 손실(binary cross-entropy)을 기록해 epoch별 평균 손실 리스트를 반환.
      - 그래디언트 공식은 오늘 배운 "X^T @ (sigmoid(Xw) - y) / batch_size" 형태를 스스로 코드로 옮길 것.
      """
      ...
  ```
- **검증 과제**: `make_classification`으로 이진분류 데이터를 만들고, 위 함수로 학습한 정확도를 `sklearn.linear_model.LogisticRegression`의 정확도와 비교하세요. 완전히 같을 필요는 없지만 비슷한 범위여야 합니다.
- **확인 질문**:
  - `batch_size=1`(순수 SGD)과 `batch_size=len(X)`(사실상 배치 GD)로 각각 돌려보면 학습 곡선의 노이즈가 예상과 맞았는가?
  - 셔플을 안 하면(매 epoch마다 같은 순서로 미니배치를 나누면) 어떤 문제가 생길 수 있는가?

**막히면**: 손실이 `nan`이 되면 → `sigmoid`에서 오버플로우 확인 (`np.exp`에 너무 큰 음수/양수가 들어가는지). 정확도가 50%에서 안 움직이면 → 그래디언트 부호나 학습률을 의심.

```bash
git add week07/
git commit -m "W7 Day3: logistic regression via minibatch SGD"
git push
```

---

## Day 4 (목요일) — 모멘텀·Adam 직관 + 학습률 실험 [1.5시간]

### 00:00–00:45 | 모멘텀과 적응적 학습률의 직관

- **학습 목표**: SGD에 모멘텀을 추가하면 왜 더 안정적으로 수렴하는지, Adam이 SGD·모멘텀과 어떻게 다른지 개념 수준에서 설명할 수 있다. (이번 주 최소 보장의 핵심)
- **핵심 개념**:
  - 모멘텀: 이전 스텝의 이동 방향을 일부 기억해 현재 그래디언트에 더합니다. 공이 언덕을 굴러 내려가며 관성을 갖는 것과 비슷한 비유가 자주 쓰입니다. 진동(zigzag)을 줄여줍니다.
  - Adam: 모멘텀(1차 모멘트)과 함께, 그래디언트 크기의 이동평균(2차 모멘트)으로 파라미터마다 "개별" 학습률을 조정합니다. 자주 업데이트되는 파라미터는 학습률을 줄이고, 드물게 업데이트되는 파라미터는 상대적으로 키웁니다.
- **자료**: "SGD momentum Adam explained" 검색 → 3Blue1Brown 또는 다른 시각화 영상 (10~15분). 수식을 전부 외울 필요는 없고, "모멘텀 = 관성", "Adam = 파라미터별 적응적 학습률"이라는 두 문장을 자기 말로 재구성할 수 있으면 충분합니다.
- **확인 질문 (자료 없이, 최소 보장 체크)**:
  - SGD와 Adam의 차이를 "모멘텀 관점"에서 한 문단으로 설명해보세요.
  - 학습률이 너무 크면 손실 곡선이 어떻게 되는가? 너무 작으면?

### 00:45–01:15 | 순수 SGD vs 모멘텀 SGD 비교 실험

- **구현 과제 (스스로 작성)**: Day3의 `minibatch_sgd_logistic`을 확장하세요.
  ```python
  def minibatch_sgd_with_momentum(
      X: np.ndarray, y: np.ndarray,
      lr: float = 0.1, momentum: float = 0.9,
      batch_size: int = 32, n_epochs: int = 50
  ) -> tuple[np.ndarray, list[float]]:
      """
      요구사항:
      - velocity(속도) 벡터를 0으로 초기화하고, 매 스텝마다
        velocity = momentum * velocity - lr * gradient 로 갱신 후 w += velocity.
      - Day3과 같은 방식으로 epoch별 평균 손실을 기록해 반환.
      - momentum=0일 때는 순수 SGD와 동일해야 함 (직접 확인할 것).
      """
      ...
  ```
- **비교 과제**: 같은 데이터·같은 학습률로 (a) `momentum=0`, (b) `momentum=0.9` 두 경우의 학습 곡선을 한 그래프에 겹쳐 그리세요.
- **확인 질문**:
  - 두 곡선 중 어느 쪽이 더 빨리 수렴했는가? 예상과 같았는가?
  - `momentum=0.99`처럼 너무 크게 하면 어떤 문제가 생기는가? 직접 실행해서 확인하세요.

### 01:15–01:30 | LeetCode: Longest Substring Without Repeating Characters

- **구현 과제 (스스로 작성, 정답 코드 없음)**:
  ```python
  def length_of_longest_substring(s: str) -> int:
      """
      요구사항:
      - 반복되는 문자가 없는 가장 긴 부분 문자열의 길이를 반환한다.
      - 슬라이딩 윈도우(투 포인터) + set 또는 dict로 O(n) 시간에 구현해볼 것.
      - 테스트: "abcabcbb" → 3, "bbbbb" → 1, "pwwkew" → 3
      """
      ...
  ```
- **확인 질문**: 윈도우의 왼쪽 포인터를 언제, 얼마나 이동시켜야 하는가? dict에 문자의 "마지막 등장 인덱스"를 저장하면 왼쪽 포인터를 한 번에 점프시킬 수 있는데, 이 방식과 한 칸씩 이동하는 방식의 시간복잡도 차이는?

```bash
git add week07/
git commit -m "W7 Day4: momentum SGD comparison + LeetCode"
git push
```

---

## Day 5 (금요일) — 복습 + Word2Vec 논문 [1.5시간]

### 00:00–00:30 | W7 최소 보장 자가 점검

자료를 보지 않고 아래 질문에 답해보세요. 막히면 해당 Day로 돌아가세요.

```
□ 연쇄법칙을 역전파에 적용해 설명할 수 있는가?
  → "손실 → 예측값 → 가중치" 순서로 미분이 어떻게 곱해지며 전달되는지 한 문단으로 말해보기

□ SGD와 Adam의 차이를 모멘텀 관점에서 설명할 수 있는가?
  → Day4에서 자기 말로 정리한 두 문장을 다시 말해보기 (토씨까지 외울 필요 없음)

□ 그래디언트의 반대 방향으로 이동하는 이유는?
  → "가장 가파르게 증가하는 방향의 반대 = 가장 가파르게 감소하는 방향"을 설명

□ 학습률이 너무 크거나 작으면 어떻게 되는가?
  → Day2·Day4에서 직접 관찰한 결과를 근거로 말하기
```

### 00:30–01:00 | 논문: Word2Vec — 의미의 벡터 연산

```
읽을 것: Mikolov et al. (2013) "Efficient Estimation of Word Representations
in Vector Space" (Word2Vec)
- 전체 읽을 필요 없음
- 읽을 부분: Abstract, Introduction 앞부분, 그리고 유명한
  "king - man + woman ≈ queen" 예시가 나오는 부분(관련 후속 논문이나
  블로그 설명을 참고해도 무방)
- 시간: 20–30분

읽으면서 메모할 것:
1. 이 논문 이전에는 단어를 어떻게 표현했는가? (원-핫 인코딩의 한계와 비교)
2. "king - man + woman ≈ queen"이 의미하는 것은? 벡터 공간에서 "의미"가
   어떻게 산술 연산으로 표현되는가?
3. 이 임베딩과 W10(RAG)에서 쓸 문장 임베딩의 관계는? (힌트: 둘 다
   "의미가 비슷하면 벡터도 가깝다"는 원리를 공유)

영어 한 문장 준비 (자기 말로 — 아래는 참고용 뼈대):
"Word2Vec learns dense vector representations of words such that
semantic relationships are captured as vector arithmetic, unlike
sparse one-hot encodings which treat every word as equally distant
from every other word."
```

### 01:00–01:30 | 마무리 커밋 + README

```bash
git add .
git commit -m "W7 완료: 그래디언트 검증, SGD/모멘텀 선형·로지스틱 회귀, Word2Vec"
git push

cat >> week07/README.md << 'EOF'
# W7: Gradient descent, SGD, momentum

## W7 완료 항목 (스스로 구현)
- [ ] numerical_gradient로 그래디언트 검증
- [ ] 배치 경사하강법으로 선형회귀 학습 + 학습 곡선
- [ ] 미니배치 SGD로 로지스틱 회귀 학습
- [ ] 모멘텀 SGD 구현 + 순수 SGD와 비교
- [ ] LeetCode: Longest Substring Without Repeating Characters

## 최소 보장 체크
- [ ] 연쇄법칙을 역전파에 적용해 설명 가능
- [ ] SGD·Adam 차이를 모멘텀 관점에서 설명 가능
EOF

git add week07/README.md
git commit -m "W7: update README"
git push
```

---

## 주말 — 심화 [토요일 2.5시간 / 일요일 2시간]

### 토요일 [2.5시간]

**[00:00–01:15] LeetCode: Minimum Size Subarray Sum + 다른 데이터셋에 파이프라인 적용**

- **LeetCode 구현 과제 (스스로 작성, 정답 코드 없음)**:
  ```python
  def min_subarray_len(target: int, nums: list[int]) -> int:
      """
      요구사항:
      - 합이 target 이상이 되는 가장 짧은 연속 부분 배열의 길이를 반환. 없으면 0.
      - 슬라이딩 윈도우로 O(n) 시간에 구현해볼 것 (Day4 슬라이딩 윈도우 감각 재사용).
      - 테스트: target=7, nums=[2,3,1,2,4,3] → 2 ([4,3])
      """
      ...
  ```
- **일반화 과제**: 이번 주 만든 `minibatch_sgd_with_momentum`을 W1~W3에서 쓴 `load_digits`(이진분류로 축소: 예를 들어 "3인가 아닌가") 또는 `load_wine`에 적용해, sklearn `LogisticRegression`과 정확도를 비교하세요.
- **확인 질문**:
  - 슬라이딩 윈도우 문제를 이번 주에 두 번(Day4, 오늘) 풀면서, 윈도우를 언제 넓히고 언제 좁힐지 판단하는 공통 패턴을 발견했는가?
  - 직접 만든 로지스틱 회귀가 sklearn과 정확도가 비슷하지 않다면, 그 이유가 "학습이 덜 됐어서"인지 "그래디언트 계산이 틀려서"인지 어떻게 구분할 것인가?

**[01:15–02:30] 학습률 스케줄링 실험 (필요 시)**

```
Day2·Day4에서 고정 학습률만 써봤습니다. 오늘은 학습이 진행될수록
학습률을 점차 줄이는 방식(learning rate decay)을 실험해보세요.

과제:
- 매 epoch마다 lr을 조금씩 줄이는 규칙을 하나 정해서(예: 1/epoch에
  비례, 또는 일정 epoch마다 절반으로) minibatch_sgd_with_momentum에
  적용해보고, 고정 학습률 대비 학습 곡선이 어떻게 달라지는지 비교.

이미 학습률 실험을 충분히 했다고 느끼면 이 항목은 건너뛰고
Day1의 다변수 미적분 영상을 복습하는 데 시간을 써도 됩니다.
30분 이상 막히면 → 넘어가고 W13 복습 구간에서 다시 만나기.
```

### 일요일 [2시간]

**[00:00–01:00] W7 총복습 — 아무것도 보지 않고 재구현**

이번 주 코드를 하나도 열지 않고, 아래 흐름을 처음부터 다시 짜보세요.

```
1. 임의의 2변수 함수에 대해 numerical_gradient로 그래디언트 확인 (5분)
2. 선형회귀 MSE 손실의 그래디언트 공식을 종이에 다시 유도 (5분)
3. 배치 경사하강법으로 선형회귀 학습 (10분)
4. 시그모이드 + 로지스틱 회귀 손실의 그래디언트로 미니배치 SGD 구현 (15분)
5. 모멘텀 추가 + 순수 SGD와 학습 곡선 비교 (15분)

30분 이상 막히면 Day2~4 코드 참고 가능.
목표는 코드를 외우는 게 아니라 "그래디언트 → 업데이트 → 반복"이라는
흐름과 그 안에 연쇄법칙이 어디서 쓰이는지를 손에 익히는 것.
```

**영어로 설명 연습** (혼자 소리 내어, `___` 부분은 직접 측정한 값으로 채우기 — 외운 숫자 금지):

```
"Gradient descent updates parameters by moving in the direction opposite
to the gradient, because the gradient points toward the steepest increase
of the loss. SGD approximates this gradient using a small batch instead
of the full dataset, which is noisier but much cheaper per step.
Momentum smooths out this noise by accumulating a moving average of past
updates, while Adam additionally adapts the learning rate per parameter
based on the recent magnitude of its gradients.
In my experiment, plain SGD took ___ epochs to converge while momentum
SGD reached a similar loss in about ___ epochs (fill in your measured
values)."
```

**[01:00–02:00] 주간 회고 + W8 준비**

```markdown
## W7 회고 (일요일에 작성)

### 달성한 것
- [ ] numerical_gradient로 그래디언트 검증
- [ ] 배치 GD로 선형회귀, 미니배치 SGD로 로지스틱 회귀 학습
- [ ] 모멘텀 SGD 구현 + 비교 실험
- [ ] LeetCode: Longest Substring Without Repeating Characters, Minimum Size Subarray Sum

### 최소 보장 체크
- [ ] 연쇄법칙을 역전파에 적용해 설명 가능
- [ ] SGD·Adam 차이를 모멘텀 관점에서 설명 가능

### 스스로 짠 코드에서 막힌 지점
(개념을 몰라서 막힌 것 / 파이썬·numpy 문법을 몰라서 막힌 것을 구분해서 적기 —
 이 구분이 다음 주에 어디에 시간을 쓸지 알려줍니다)

### 그래디언트 유도가 안 됐다면 어느 단계였는가
(손실함수 미분 / 연쇄법칙 적용 / 행렬 형태로 정리 중 어디)

### W8에 가져갈 것
(이번 주에 이해 못 하고 넘어간 것)

### 다음 주 첫 번째 할 일
(W8 Day1 무엇부터 시작할지)
```

---

## 막힐 때 대응 가이드

**개념이 막히면**:
```
1단계 (5분): 구글에 영어로 검색
  예: "why does gradient point in direction of steepest ascent"
2단계 (10분): 3Blue1Brown 관련 영상 검색 (Essence of calculus 시리즈)
3단계 (20분): MIT 18.02(다변수 미적분) 관련 강의 노트 확인
30분 넘어도 안 되면 → "아직 모름: [개념]" 메모하고 넘어가기
→ W13 복습 구간에서 다시 만납니다. 막히는 것은 정상 과정.
```

**직접 짠 코드가 막히면 (완성 코드를 찾아 베끼지 말고)**:
```
1. 에러 메시지 전체를 그대로 구글에 검색
2. 자주 나오는 문제:
   - 손실이 nan/inf로 발산 → 학습률을 10배 낮추거나 sigmoid 오버플로우 확인
   - 손실이 전혀 안 줄어듦 → 그래디언트 부호(+=/-=) 확인, numerical_gradient로 검산
   - Shape mismatch → print(array.shape)로 각 단계 차원 추적
3. Stack Overflow에서 "접근 방식"만 참고하고, 자기 코드로 다시 작성 (복붙 금지)
```

---

## W7 완료 기준

일요일 저녁에 아래를 할 수 있으면 W7 성공:

```
□ week07/ 폴더에 numerical_gradient 검증 코드가 있다 (스스로 작성)
□ 배치 경사하강법으로 선형회귀를 학습시켜 진짜 계수에 근접한 결과를 얻었다
□ 미니배치 SGD로 로지스틱 회귀를 학습시켜 sklearn과 비슷한 정확도를 얻었다
□ gd_learning_curve.png 등 학습 곡선 그래프가 week07/ 폴더에 있다
□ 모멘텀 SGD와 순수 SGD의 학습 곡선을 비교한 그래프가 있다
□ "SGD와 Adam의 차이"를 모멘텀 관점에서 3문장으로 설명할 수 있다 (한국어 가능)
□ LeetCode 2문제(Longest Substring, Minimum Size Subarray Sum)를 스스로 다시 풀 수 있다

절반(4개 이상) 달성하면 W8로 진행.
전부 못 해도 W8로 진행 — 이해 못 한 부분은 이후 주차에서 다시 나옴.
```

---

## W8 첫 할 일 미리 보기

W8 Day1에 열어야 할 것:
1. `week08/` 폴더 생성
2. W3에서 만든 로지스틱 회귀 모델을 FastAPI로 서빙하는 엔드포인트 만들기 (`/predict`)
3. 막히면 → 알고리즘의 시간복잡도(Big-O) 감이 필요해지면 CLRS 핵심 챕터로 역추적
4. ⭐영어: IELTS 스피킹·라이팅 주 1회 루틴을 이번 주부터 시작 (W1 진단 결과와 목표 갭 기준)

---

*이 계획대로 완벽하게 안 돼도 됩니다.
W7의 진짜 목표는 "손실을 어떻게 줄이는가"를 sklearn 뒤에 숨기지 않고 직접 손으로 만들어보는 경험입니다.
그리고 그 경험은 완성된 옵티마이저 코드를 실행해서가 아니라, 그래디언트 공식을 스스로 유도하고 코드로 옮기다가 막히고 고치는 과정에서 만들어집니다.*

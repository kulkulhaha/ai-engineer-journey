# W3 구체적 실행 계획 (v2 — 학습 가이드형)

> **주제**: 로지스틱 회귀 재활성화 + 베이즈 정리·MLE 이론 역추적 + Python OOP
>
> **사용 데이터셋 주의**: sklearn `load_breast_cancer()`(569개 샘플, 30개 특성, 악성/양성 이진 분류)를 씁니다.
> W1–2의 digits/wine과 달리 **이진 분류**라서 혼동행렬·ROC·AUC를 그대로 쓸 수 있습니다.
>
> **총 목표 시간**: 10–12시간 (v7 커리큘럼 블록 A 기준)
> **기준**: 평일 1.5시간 + 토요일 2.5시간 + 일요일 2시간
>
> **v1과의 차이**: 완성된 실행 코드를 주지 않습니다. 학습 목표·핵심 개념·구현 요구사항(스펙)·확인 질문만 제시합니다. 수식 유도와 코드 작성이 이번 주의 진짜 과제입니다. Git/설치 같은 보일러플레이트만 그대로 제시합니다.
>
> **전제**: W1–2에서 만든 `SimplePCA`를 토요일 심화에서 재사용합니다.

---

## W3 목표 (이것만 달성하면 성공)

1. **실습**: sklearn LogisticRegression으로 breast_cancer 분류 + 혼동행렬·ROC 곡선을 스스로 구현
2. **이론**: "로그우도가 왜 손실함수인가"를 베이즈 정리·MLE 관점에서 설명 가능
3. **재구현**: numpy로 로지스틱 회귀를 MLE(로그우도 최대화) 관점에서 직접 구현
4. **Python**: 추상클래스·상속·decorator로 모델 래퍼를 스스로 설계
5. **최소 보장**: 베이즈 정리를 예시로 **손으로 유도** / MLE를 로그우도로 **손으로 유도**

---

## Day 1 (월요일) — 로지스틱 회귀 첫 실습 [1.5시간]

W1–2에서 익힌 "실습 먼저, 막히면 이론" 패턴을 그대로 이어갑니다.

### 00:00–00:20 | 데이터 로드 + 빠른 EDA

- **학습 목표**: 이진 분류 데이터의 클래스 불균형과 특성 스케일 차이를 먼저 파악한다.
- **핵심 개념**: 클래스 분포가 한쪽으로 치우쳐 있으면 "정확도(accuracy)"만으로는 모델을 평가할 수 없습니다. 오늘 ROC/AUC를 배우는 이유가 여기 있습니다.
- **구현 과제 (스스로 작성)**:
  ```python
  def inspect_breast_cancer():
      """
      요구사항:
      - load_breast_cancer()로 X, y, target_names를 가져온다.
      - X.shape, 클래스 이름, 클래스별 샘플 수를 출력한다.
      - pandas DataFrame으로 만들어 앞 5개 특성의 describe()를 출력한다.
      - 특성들 사이 스케일(평균, 표준편차) 차이가 얼마나 큰지 확인한다.
      """
      ...
  ```
- **확인 질문**:
  - 클래스 비율이 대략 몇 대 몇인가? 모든 샘플을 다수 클래스로만 찍으면 정확도가 몇 %가 되는가?
  - 특성 스케일 차이가 크면 로지스틱 회귀 학습에 어떤 문제가 생기는가?

### 00:20–01:10 | LogisticRegression + 혼동행렬 + ROC

- **학습 목표**: 분류 모델을 학습하고, 정확도가 아닌 혼동행렬·ROC/AUC로 평가할 수 있다.
- **핵심 개념**: `predict`는 0/1 라벨을, `predict_proba`는 확률을 반환합니다. ROC 곡선은 판정 임계값(threshold)을 0에서 1까지 움직이며 (FPR, TPR)을 찍은 것이고, AUC는 그 아래 면적입니다. AUC 0.5는 무작위 추측, 1.0은 완벽한 순위 매기기를 뜻합니다.
- **구현 과제 (스스로 작성)**:
  ```python
  def train_and_evaluate():
      """
      요구사항:
      - train_test_split으로 8:2 분할 (random_state=42, stratify=y).
      - StandardScaler: train에만 fit, train/test 모두 같은 scaler로 transform.
        (⚠️ test에 다시 fit하면 안 되는 이유를 스스로 설명할 수 있어야 함)
      - LogisticRegression(max_iter=1000)으로 학습.
      - classification_report 출력.
      - 혼동행렬을 그려 'week03/confusion_matrix.png'로 저장.
      - predict_proba의 양성 클래스 확률로 ROC 곡선을 그리고 AUC를 제목/범례에 표시,
        대각선(무작위 추측) 기준선도 함께 그려 'week03/roc_curve.png'로 저장.
      """
      ...
  ```
- **확인 질문**:
  - `predict_proba`가 반환하는 배열의 shape은 왜 (n, 2)인가? 두 열의 합은 얼마인가?
  - test 데이터에 `scaler.fit_transform`을 쓰면 정확히 무엇이 잘못되는가? (키워드: data leakage)
  - AUC가 0.99인데 재현율(recall)이 낮을 수도 있는가? 가능하다면 어떤 경우인가?

**막히면**: 에러 메시지 전체를 그대로 구글 검색. 그래프가 안 뜨면 `plt.show()` 위치와 저장 순서 확인(`savefig`는 `show` 앞에).

### 01:10–01:30 | 이론 질문 도출 (내일 이어짐)

```
코드를 돌리면 반드시 생기는 질문:
"sklearn은 내부에서 뭘 최소화하길래 이 확률값이 나오는가?"
"왜 제곱오차가 아니라 로그우도인가?"

이 질문을 메모장에 적어두세요. 내일(Day 2)의 출발점입니다.

(시간 남으면) StatQuest "Logistic Regression Details Pt1: Coefficients" (9분)
URL: https://www.youtube.com/watch?v=vN5cNN2-HWE
```

```bash
mkdir -p week03
git add week03/
git commit -m "W3 Day1: logistic regression baseline + confusion matrix + ROC"
git push
```

---

## Day 2 (화요일) — 베이즈 정리 + MLE 손유도 [1.5시간]

**오늘이 이번 주 최소 보장의 핵심입니다. 코드보다 종이가 먼저입니다.**

### 00:00–00:45 | 베이즈 정리를 예시로 손유도

- **학습 목표**: 베이즈 정리를 외운 공식이 아니라, 구체적 숫자로 직접 계산해 직관을 얻는다.
- **핵심 개념**: `P(A|B) = P(B|A)·P(A) / P(B)`. 분모 `P(B)`는 전체확률의 법칙으로 분해합니다. 사전확률(prior)이 낮으면, 검사가 아무리 정확해도 사후확률은 생각보다 낮습니다.
- **과제 (종이 먼저)**: 아래 조건으로 `P(질병|양성)`을 **손으로** 계산하세요. 공식을 찾아보지 말고, "양성 판정을 받은 사람 중 실제 환자의 비율"이 무엇인지부터 생각해보세요.
  ```
  P(질병) = 1%              (사전확률)
  P(양성|질병) = 99%         (민감도)
  P(양성|질병 아님) = 5%      (위양성률)
  → P(질병|양성) = ?
  ```
  - 힌트가 필요하면: 인구 10,000명을 가정하고 표를 그려보세요 (실제 환자 몇 명, 그중 양성 몇 명, 비환자 중 양성 몇 명).
- **구현 과제 (손계산 후 코드로 검증)**:
  ```python
  def bayes_posterior(prior: float, sensitivity: float, false_positive_rate: float) -> float:
      """
      요구사항:
      - 전체확률의 법칙으로 P(양성)을 계산한 뒤 사후확률을 반환한다.
      - 손으로 계산한 값과 소수점 4자리까지 일치하는지 확인할 것.
      - 추가: prior를 0.001 ~ 0.5로 바꿔가며 사후확률이 어떻게 변하는지
        그래프로 그려볼 것.
      """
      ...
  ```
- **확인 질문**:
  - 검사가 99% 정확한데 사후확률이 20%도 안 되는 이유를 한 문장으로 설명할 수 있는가? (키워드: base rate fallacy)
  - 사전확률이 몇 %가 되어야 사후확률이 50%를 넘는가? 위 그래프에서 찾아보세요.

**막히면**: Harvard Stat 110 Lecture 5–6 "Conditional Probability / Bayes' Rule" (https://projects.iq.harvard.edu/stat110/youtube). "law of total probability" 키워드 검색.

### 00:45–01:30 | MLE를 로그우도로 손유도

- **학습 목표**: "우도를 최대화한다"는 말의 의미를 동전 던지기로 이해하고, 왜 로그를 씌우는지 스스로 설명할 수 있다.
- **핵심 개념**: 우도(likelihood)는 "파라미터가 이 값일 때 관측된 데이터가 나올 확률"입니다. MLE는 그 값을 최대로 만드는 파라미터를 고르는 것입니다. 로그는 단조증가 함수라 argmax를 바꾸지 않으면서 곱을 합으로 바꿔줍니다.
- **과제 (종이 먼저)**: 동전을 10번 던져 앞면이 7번 나왔습니다. 앞면 확률 `p`의 MLE를 **미분으로 직접 유도**하세요.
  1. 우도 `L(p) = C(10,7)·p⁷·(1-p)³`를 씁니다.
  2. 로그를 씌워 `log L(p)`를 정리합니다.
  3. `p`로 미분해서 0이 되는 지점을 구합니다.
  4. 결과가 `k/n = 0.7`이 나오는지 확인합니다.
- **구현 과제 (손유도 후 코드로 검증)**:
  ```python
  def log_likelihood(p: float, n: int, k: int) -> float:
      """이항분포의 로그우도. p가 0 또는 1이면 -inf 반환."""
      ...

  def find_mle_numerically(n: int, k: int):
      """
      요구사항:
      - p를 0.01~0.99로 촘촘히 훑으며 로그우도를 계산한다.
      - argmax인 p를 찾아 이론값 k/n과 비교 출력한다.
      - 로그우도 곡선과 argmax 위치를 그려 'week03/mle_coin_flip.png'로 저장.
      """
      ...
  ```
- **로지스틱 회귀로 연결 (종이에 직접 쓰기)**:
  ```
  1. 샘플 하나의 베르누이 우도를 y_i와 p_i로 쓰면?   (힌트: 지수 형태로 한 줄에 표현 가능)
  2. 전체 로그우도는 그 합으로 어떻게 되는가?
  3. 이것을 "최대화"하는 것이 무엇을 "최소화"하는 것과 같은가?  (이 손실함수의 이름은?)
  → 이 세 줄을 스스로 쓸 수 있으면 이번 주 최소 보장의 절반은 끝났습니다.
  ```
- **확인 질문**:
  - 로그를 씌우는 이유를 세 가지 대보세요 (계산 안정성 / argmax 보존 / 미분 편의).
  - 로그우도를 최대화하는 것과 크로스엔트로피를 최소화하는 것이 같은 이유는?

```bash
git add week03/
git commit -m "W3 Day2: Bayes theorem + MLE derivation with numerical verification"
git push
```

---

## Day 3 (수요일) — numpy로 로지스틱 회귀 MLE 재구현 [1.5시간]

### 00:00–00:30 | sigmoid + gradient 유도 (종이)

- **학습 목표**: 손실함수의 gradient를 연쇄법칙으로 직접 유도하고, 왜 그렇게 단순한 형태가 되는지 이해한다.
- **핵심 개념**: sigmoid의 도함수는 `σ'(z) = σ(z)(1-σ(z))`라는 특이한 성질이 있습니다. 이것이 로그우도의 미분과 상쇄되면서 최종 gradient가 놀랍도록 단순해집니다.
- **과제 (종이에서 직접 유도)**:
  ```
  1. σ(z) = 1/(1+e^{-z}) 를 z로 미분해서 σ(z)(1-σ(z))가 나오는지 확인.
  2. 손실 L(w) = -Σ[y·log(σ(Xw)) + (1-y)·log(1-σ(Xw))] 를 w로 미분.
  3. 최종 형태가 어떤 모양으로 정리되는지 직접 구하세요.
     (스포일러 없음 — 아주 단순한 형태가 나옵니다. "예측 - 실제"라는 표현이
      왜 나오는지 유도 과정에서 확인하는 게 목적입니다.)
  ```
- **확인 질문**: 최종 gradient에 sigmoid의 도함수가 남아 있지 않은 이유는 무엇인가?

**막히면**: "logistic regression gradient derivation" 검색 — 단, 최종 결과만 보지 말고 중간 유도 과정을 따라가며 자기 종이에 다시 쓰세요.

### 00:30–01:15 | Gradient Descent 구현 + sklearn과 비교

- **학습 목표**: 유도한 gradient로 학습 루프를 직접 짜서 sklearn과 비슷한 성능을 낸다.
- **구현 과제 (스스로 작성)**:
  ```python
  def sigmoid(z: np.ndarray) -> np.ndarray:
      """
      요구사항: 수치적 오버플로를 방지할 것 (z가 매우 크거나 작을 때 exp가 터짐).
      어떻게 방지할지는 스스로 결정 (힌트: np.clip 또는 조건 분기).
      """
      ...

  def train_logistic_regression(X, y, lr=0.1, n_iters=2000):
      """
      요구사항:
      - w를 0으로, b를 0.0으로 초기화.
      - 매 반복마다: z 계산 → p = sigmoid(z) → 음의 로그우도 손실 계산 → 손실 기록
        → Day3 오전에 유도한 gradient로 w, b 업데이트.
      - log(0) 방지책을 넣을 것 (어떻게 할지 스스로 결정).
      - (w, b, losses) 반환.
      """
      ...
  ```
- **검증 과제**:
  1. 손실이 실제로 감소했는지 첫 값과 마지막 값을 비교 출력.
  2. 손실 곡선을 그려 `week03/loss_curve.png`로 저장 — 곡선 모양이 매끄럽게 수렴하는가?
  3. 내 구현의 테스트 정확도와 sklearn `LogisticRegression`의 정확도를 나란히 출력.
- **확인 질문**:
  - 두 정확도 차이가 크다면 원인은 무엇일까? (반복 횟수 / 학습률 / 정규화 항 유무)
  - sklearn은 기본적으로 L2 정규화를 적용합니다. 내 구현에는 없는데도 성능이 비슷하다면 왜일까?

**막히면**: 손실이 NaN이 되면 → 학습률을 1/10로 낮추기, sigmoid 입력 clip 확인, log에 eps 더했는지 확인. 손실이 안 내려가면 → gradient 부호와 shape을 `print`로 추적.

### 01:15–01:30 | LeetCode: Valid Parentheses

- **구현 과제 (스스로 작성, 정답 코드 없음)**:
  ```python
  def is_valid(s: str) -> bool:
      """
      요구사항: (), [], {} 가 올바르게 짝지어지고 중첩됐는지 판정.
      테스트: "()[]{}" → True, "(]" → False, "([)]" → False, "" → True
      """
      ...
  ```
- **확인 질문**: 왜 스택 자료구조가 이 문제에 자연스러운가? 큐로는 왜 안 되는가?

```bash
git add week03/
git commit -m "W3 Day3: numpy logistic regression via MLE gradient descent"
git push
```

---

## Day 4 (목요일) — Python OOP: 추상클래스·상속·decorator [1.5시간]

### 00:00–00:50 | 모델 래퍼 설계

- **학습 목표**: 여러 모델을 같은 인터페이스로 다룰 수 있는 구조를 직접 설계하고, decorator로 공통 관심사(로깅·타이밍)를 분리할 수 있다.
- **핵심 개념**:
  - **추상 베이스 클래스(ABC)**: 하위 클래스가 반드시 구현해야 할 메서드를 강제합니다.
  - **decorator**: 함수를 감싸서 원래 로직을 건드리지 않고 부가 기능(시간 측정 등)을 추가합니다. `functools.wraps`를 쓰는 이유도 확인해보세요.
- **구현 과제 (스스로 작성)**:
  ```python
  def log_timing(func):
      """
      요구사항:
      - 감싼 함수의 실행 시간을 측정해 "[함수이름] 0.0123초" 형태로 출력.
      - functools.wraps를 사용해 원래 함수의 __name__과 docstring을 보존할 것.
      """
      ...

  class BaseClassifierWrapper(ABC):
      """
      요구사항:
      - __init__(name): 이름과 fitted 상태를 저장.
      - _build_model(): @abstractmethod. 하위 클래스가 실제 sklearn 모델을 반환.
      - fit(X, y): @log_timing 적용. _build_model()로 모델을 만들고 학습, self 반환.
      - predict_proba(X): @log_timing 적용. fit되지 않았으면 명확한 에러.
                          양성 클래스 확률 1차원 배열 반환.
      - __repr__: 클래스명, name, fitted 상태가 보이게.
      """
      ...

  # 위를 상속해서 LogisticWrapper(C=1.0)와 RandomForestWrapper(n_estimators=100)를
  # 각각 구현할 것. 두 클래스 모두 _build_model()만 다르게 채우면 되도록 설계.
  ```
- **실행 과제**: 두 래퍼를 리스트에 담고 반복문으로 fit → AUC 계산 → 출력하세요. **주의**: train/test 스케일링 시 반드시 **같은 scaler 인스턴스**를 사용할 것 (train에 fit, 둘 다 transform).
- **확인 질문**:
  - `_build_model()`을 `@abstractmethod`로 만들지 않으면 무엇이 위험해지는가? 실제로 지우고 하위 클래스에서 구현을 빼먹어보세요.
  - `functools.wraps`를 빼면 `LogisticWrapper.fit.__name__`이 무엇으로 나오는가? 직접 확인해보세요.
  - 이 구조에 XGBoost 래퍼를 추가하려면 몇 줄을 써야 하는가? 그것이 이 설계의 이점인가?

### 00:50–01:30 | LeetCode: Min Stack

- **구현 과제 (스스로 작성, 정답 코드 없음)**:
  ```python
  class MinStack:
      """
      요구사항:
      - push(val), pop(), top(), getMin() 네 연산 모두 O(1).
      - getMin()이 O(1)이려면 무엇을 함께 저장해야 하는지 스스로 설계할 것.
      - 테스트: push(-2), push(0), push(-3) → getMin()==-3 → pop() → top()==0, getMin()==-2
      """
      ...
  ```
- **확인 질문**:
  - 매번 `min(stack)`을 계산하면 시간복잡도가 얼마인가? 왜 그것이 문제인가?
  - 최솟값을 함께 저장하는 방식의 공간 오버헤드는 얼마인가? 더 아낄 방법이 있는가?

25분 안에 안 풀리면: 접근 방식만 검색해서 확인하고(정답 코드 복붙 금지), 금요일에 다시 시도하세요.

```bash
git add week03/
git commit -m "W3 Day4: classifier wrapper (ABC + decorator) + LeetCode"
git push
```

---

## Day 5 (금요일) — 복습 + LeNet 논문 + 마무리 [1.5시간]

### 00:00–00:30 | W3 최소 보장 자가 점검

아래 질문에 **자료를 보지 않고** 종이에 답해보세요. 막히면 해당 Day로 돌아가세요.

```
□ 베이즈 정리를 의료검사 예시로 처음부터 유도할 수 있는가? (숫자까지)
□ 검사가 99% 정확한데 사후확률이 낮았던 이유를 한 문장으로?
□ 로지스틱 회귀의 손실함수는 무엇이고, 왜 제곱오차가 아닌가?
□ 로그우도의 gradient가 단순해지는 이유는? (sigmoid 도함수와의 관계)
□ ROC 곡선의 각 점은 무엇을 바꿔가며 찍은 것인가?
```

### 00:30–00:50 | Min Stack 재도전 (안 풀렸으면)

아무것도 보지 않고 다시 시도. 여전히 안 되면 접근 방식만 확인하고 **자기 손으로** 다시 작성하세요 (복붙 금지).

### 00:50–01:10 | LeNet 논문 맥락 파악

```
읽을 것: LeCun et al. (1998) "Gradient-Based Learning Applied to Document Recognition"
읽을 부분: Abstract, Section 1 (Introduction), Figure 2 (LeNet-5 구조)
시간: 20분

메모할 것:
1. Fully-connected 신경망으로 이미지를 처리하면 왜 비효율적인가?
   (스스로 답해보고, 논문에서 확인하세요. 힌트가 필요하면: 파라미터 수를 세어보세요 —
    28x28 이미지에 1000개 은닉 뉴런이면 가중치가 몇 개인가?)
2. Convolution이 무엇을 "공유"해서 파라미터를 줄이는가?
3. Pooling의 역할은?

Convolution 자체는 블록 C(W51)에서 깊게 다룹니다. 지금은 개념만.
```

### 01:10–01:30 | 마무리 커밋 + README

```bash
git add .
git commit -m "W3 완료: 로지스틱 회귀, 베이즈 정리, MLE, OOP 모델 래퍼"
git push

cat >> week03/README.md << 'EOF'
# W3: Logistic Regression, Bayes, MLE

## W3 완료 항목 (스스로 구현)
- [ ] LogisticRegression + 혼동행렬 + ROC/AUC (train_and_evaluate)
- [ ] 베이즈 정리 손유도 + 코드 검증 (bayes_posterior)
- [ ] MLE 손유도 (동전 던지기) + 수치 검증
- [ ] gradient 손유도 + numpy 로지스틱 회귀 (train_logistic_regression)
- [ ] BaseClassifierWrapper (ABC + decorator)
- [ ] LeetCode: Valid Parentheses, Min Stack

## 최소 보장 체크
- [ ] 베이즈 정리를 예시로 유도 가능
- [ ] MLE를 로그우도로 유도 가능
EOF

git add week03/README.md
git commit -m "W3: update README"
git push
```

---

## 주말 — 심화 [토요일 2.5시간 / 일요일 2시간]

### 토요일 [2.5시간]

**[00:00–01:15] 혼동행렬 지표를 손으로 계산 + 임계값의 의미**

- **학습 목표**: precision·recall·F1·specificity를 라이브러리 없이 혼동행렬 숫자만으로 계산하고, 도메인에 따라 무엇을 우선해야 하는지 판단할 수 있다.
- **구현 과제 (스스로 작성)**:
  ```python
  def metrics_from_confusion(tn, fp, fn, tp) -> dict:
      """
      요구사항:
      - sklearn.metrics의 precision_score 등을 쓰지 말고 정의식으로 직접 계산.
      - precision, recall(=TPR=sensitivity), specificity(=TNR), f1을 dict로 반환.
      - 분모가 0이 되는 경우를 어떻게 처리할지 스스로 결정할 것.
      - sklearn 함수 결과와 비교해 일치하는지 검증할 것.
      """
      ...

  def threshold_sweep(y_true, y_proba, thresholds=(0.3, 0.5, 0.7)):
      """
      요구사항:
      - 각 임계값에서 예측 라벨을 만들고 위 지표들을 계산해 표로 출력한다.
      - 임계값이 올라갈수록 precision과 recall이 각각 어떻게 움직이는지 관찰.
      """
      ...
  ```
- **확인 질문**:
  - 암 진단에서 FN(암을 놓침)과 FP(오진) 중 어느 쪽이 더 위험한가? 그렇다면 임계값을 0.5보다 높여야 하는가 낮춰야 하는가?
  - F1이 precision과 recall의 **산술평균이 아니라 조화평균**인 이유는? 한쪽이 0에 가까울 때 어떤 일이 벌어지는지 숫자를 넣어 확인해보세요.

**[01:15–02:30] Decision boundary 시각화 (W1 PCA 재활용)**

- **학습 목표**: 로지스틱 회귀가 "선형" 분류기라는 말의 의미를 그림으로 확인한다. 동시에 W1의 PCA가 여기서 도구로 쓰이는 것을 경험한다.
- **구현 과제 (스스로 작성)**:
  ```python
  def plot_decision_boundary():
      """
      요구사항:
      - breast_cancer를 표준화한 뒤 PCA로 2차원 축소 (W1/W2에서 만든 SimplePCA를
        써도 되고 sklearn PCA를 써도 됨 — 직접 만든 걸 쓰면 더 좋음).
      - 2차원 데이터로 LogisticRegression 학습.
      - meshgrid로 평면 전체의 P(양성)을 계산해 contourf로 확률 등고선을 그린다.
      - 실제 샘플을 클래스별 색으로 산점도로 겹쳐 그린다.
      - 'week03/decision_boundary.png'로 저장.
      """
      ...
  ```
- **확인 질문**:
  - 확률이 0.5가 되는 경계선이 직선으로 보이는가? 왜 직선일 수밖에 없는지 `w·x + b = 0`이라는 식으로 설명할 수 있는가?
  - 만약 데이터가 원형으로 둘러싸인 형태라면 로지스틱 회귀로 분리할 수 있는가? 어떻게 하면 가능해지는가? (키워드: 특성 공학, 커널)

30분 이상 한 개념에 막히면 넘어가고 메모만 남기세요. W5(엔트로피·트리)에서 비선형 분류기를 만나며 다시 연결됩니다.

### 일요일 [2시간]

**[00:00–01:00] W3 총 복습 — 아무것도 보지 않고 재구현**

```
1. breast_cancer 로드 + train/test 분리 + 스케일링 (같은 scaler 재사용!) (5분)
2. sigmoid 정의 (오버플로 방지 포함) (5분)
3. 음의 로그우도 손실 정의 (5분)
4. gradient descent 학습 루프 (15분)
5. sklearn과 정확도·AUC 비교 (10분)

30분 이상 막히면 Day3 코드 참고 가능.
목표는 코드 암기가 아니라 "로그우도 최대화 = 손실 최소화" 흐름이 손에 남는 것.
```

**영어로 설명 연습** (혼자 소리 내어, `___`는 직접 측정한 값으로 — 외운 숫자 금지):

```
"Logistic regression models the probability of a binary outcome using the
sigmoid function. Its parameters are found via Maximum Likelihood Estimation:
we maximize the log-likelihood of the observed data, which is mathematically
equivalent to minimizing the cross-entropy loss.
On the breast cancer dataset, my model achieved an AUC of ___,
meaning it ranks a randomly chosen malignant case above a randomly chosen
benign case about that fraction of the time."
```

**[01:00–02:00] 주간 회고 + W4 준비**

```markdown
## W3 회고 (일요일에 작성)

### 달성한 것
- [ ] LogisticRegression + 혼동행렬 + ROC/AUC
- [ ] 베이즈 정리 손유도
- [ ] MLE 손유도 (동전 던지기)
- [ ] gradient 손유도 + numpy 로지스틱 회귀
- [ ] OOP 모델 래퍼 (ABC + decorator)
- [ ] LeetCode: Valid Parentheses, Min Stack

### 최소 보장 체크
- [ ] 베이즈 정리를 예시로 유도 가능
- [ ] MLE를 로그우도로 유도 가능

### 손유도가 막힌 지점
(베이즈 / MLE 미분 / gradient 유도 중 어디였는지 — 이게 다음 복습 대상입니다)

### 코드가 막힌 지점
(개념을 몰라서 / 파이썬 문법을 몰라서 / 라이브러리 사용법을 몰라서 — 구분해서 적기)

### W4에 가져갈 것
(이번 주에 이해 못 하고 넘어간 것)

### 다음 주 첫 번째 할 일
(W4 Day1 무엇부터 시작할지)
```

---

## 막힐 때 대응 가이드

**개념이 막히면**:
```
1단계 (5분): 구글에 영어로 검색
  예: "why logistic regression uses log likelihood not squared error"
2단계 (10분): StatQuest 관련 영상 검색
3단계 (20분): Harvard Stat 110 강의 노트 (https://projects.iq.harvard.edu/stat110/youtube)
30분 넘어도 안 되면 → "아직 모름: [개념]" 메모하고 넘어가기
→ W13 복습 구간에서 다시 만납니다.
```

**수식 유도가 막히면 (정답을 바로 보지 말고)**:
```
1. 더 작은 케이스로 축소 (샘플 1개일 때 먼저 유도)
2. 각 기호가 무엇을 뜻하는지 옆에 한글로 적기
3. 미분 대상과 미분 변수를 명확히 표시
4. 그래도 안 되면 유도 "과정"이 나온 자료를 찾되, 결과만 베끼지 말고
   자기 종이에 처음부터 다시 쓰기
```

**직접 짠 코드가 막히면 (완성 코드를 찾아 베끼지 말고)**:
```
에러 메시지 전체를 그대로 구글 검색.
자주 나오는 에러:
- log(0) → -inf/NaN: eps를 더했는지 확인
- gradient descent 손실 발산: learning rate를 1/10로
- RuntimeWarning: overflow in exp → sigmoid 입력 clip
- Shape mismatch → 각 단계에서 print(arr.shape)로 추적
Stack Overflow에서는 "접근 방식"만 참고하고 자기 코드로 다시 작성.
```

---

## W3 완료 기준

일요일 저녁에 아래를 할 수 있으면 W3 성공:

```
□ week03/ 폴더에 스스로 작성한 코드가 올라가 있다
□ confusion_matrix.png, roc_curve.png, loss_curve.png가 week03/ 폴더에 있다
□ 베이즈 정리를 종이에 처음부터 유도해 숫자까지 낼 수 있다
□ MLE 미분 유도를 종이에 다시 쓸 수 있다
□ numpy 로지스틱 회귀가 sklearn과 비슷한 정확도를 낸다
□ BaseClassifierWrapper가 작동하고, 새 모델을 추가하기 쉽다
□ LeetCode 2문제를 스스로 다시 풀 수 있다

절반(4개 이상) 달성하면 W4로 진행.
단, 베이즈·MLE 두 손유도는 최소 보장 항목이므로 못 했다면 W4 주말에 반드시 보충.
```

---

## W4 첫 할 일 미리 보기

W4 Day1에 열어야 할 것:

1. `week04/` 폴더 생성
2. OpenAI 또는 Anthropic API 키 발급 + 첫 API 호출 테스트
3. **미니프로젝트 #1 착수**: LLM 텍스트 요약·분류기 CLI (공개 뉴스 RSS → 요약 + 분류)
4. 막히면 → SVD·차원축소 → MIT 18.06 해당 파트로 역추적
5. Git branch·PR 워크플로우 처음 사용해보기

---

*이번 주 진짜 목표는 "숫자가 아니라 확률로 생각하는 습관"입니다.
그리고 그 습관은 남이 유도해둔 수식을 읽어서가 아니라, 종이에서 직접 막히고 풀어내는 과정에서 생깁니다.
sklearn이 내부에서 뭘 최적화하는지 한 번이라도 자기 손으로 유도하고 구현했다면 W3는 이미 성공입니다.*

# W3 구체적 실행 계획

> **주제**: 로지스틱 회귀 재활성화 + 베이즈 정리·MLE 이론 역추적 + Python OOP
>
> **사용 데이터셋 주의**: 이 계획은 sklearn `load_breast_cancer()`(569개 샘플, 30개 특성, 악성/양성 이진 분류)를 씁니다.
> W1–2의 digits/wine과 달리 이번엔 **이진 분류**라서 혼동행렬·ROC·AUC를 그대로 쓸 수 있습니다.
> **총 목표 시간**: 15–16시간
> **기준**: 평일 2시간 + 주말(토요일 3시간, 일요일 2–3시간)

---

## W3 목표 (이것만 달성하면 성공)

1. **실습**: sklearn LogisticRegression으로 breast_cancer 분류 + 혼동행렬·ROC 곡선 완성
2. **이론**: "로그우도가 왜 손실함수인가"를 베이즈 정리·MLE 관점에서 설명 가능
3. **재구현**: numpy로 로지스틱 회귀를 MLE(로그우도 최대화) 관점에서 직접 구현
4. **Python**: 클래스·상속·decorator로 모델 래퍼 작성
5. **최소 보장**: 베이즈 정리를 예시로 유도 / MLE를 로그우도로 유도

---

## Day 1 (월요일) — 로지스틱 회귀 첫 실습 [2시간]

W1–2에서 PCA로 익힌 "실습 먼저, 막히면 이론" 패턴을 그대로 이어갑니다.

### 00:00–00:30 | 데이터 로드 + 빠른 EDA

```python
# W3 Day1: breast_cancer 데이터 첫 확인
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
X, y = data.data, data.target
feature_names = data.target_names  # ['malignant', 'benign']

print(f"데이터 shape: {X.shape}")          # (569, 30)
print(f"클래스: {feature_names}")
print(f"클래스 분포: {np.bincount(y)}")     # [212 악성, 357 양성]

df = pd.DataFrame(X, columns=data.feature_names)
print(df.describe().iloc[:, :5])
```

막히면: `ModuleNotFoundError` → `pip install scikit-learn pandas`

### 00:30–01:30 | LogisticRegression + 혼동행렬 + ROC

```python
# W3 Day1: 로지스틱 회귀 학습 + 평가
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay,
                              roc_curve, roc_auc_score, classification_report)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_s, y_train)

y_pred = clf.predict(X_test_s)
y_proba = clf.predict_proba(X_test_s)[:, 1]  # 양성(1) 클래스 확률

print(classification_report(y_test, y_pred, target_names=feature_names))

# 혼동행렬
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm, display_labels=feature_names).plot(cmap="Blues")
plt.title("Breast Cancer: Confusion Matrix")
plt.savefig("week03/confusion_matrix.png", dpi=150)
plt.show()

# ROC 곡선
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
auc = roc_auc_score(y_test, y_proba)

plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, label=f"LogisticRegression (AUC={auc:.3f})")
plt.plot([0, 1], [0, 1], "k--", label="Random guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.savefig("week03/roc_curve.png", dpi=150)
plt.show()

print(f"\nAUC: {auc:.3f}")
```

실행 후 스스로에게 물어볼 것:
- `predict_proba`가 반환하는 값은 정확히 무엇인가? (0/1 예측이 아니라 확률)
- AUC가 0.5면 무슨 의미이고, 1.0이면 무슨 의미인가?

### 01:30–02:00 | 첫 번째 막힘 → 로그우도 질문 도출

```
# 코드를 돌리면서 반드시 생기는 궁금증:
# "sklearn은 내부에서 뭘 최소화하길래 이 확률값이 나오는가?"
# → LogisticRegression의 손실함수는 로그우도(log-likelihood)의 음수.
#   "왜 제곱오차가 아니라 로그우도인가?"가 이번 주 이론 역추적의 핵심.

(시간 남으면) 지금 할 것:
StatQuest YouTube: "Logistic Regression Details Pt1: Coefficients" (9분)
URL: https://www.youtube.com/watch?v=vN5cNN2-HWE

메모할 것:
- 로지스틱 회귀는 "확률"을 예측하고, 그 확률로 우도(likelihood)를 계산
- 우도가 클수록 데이터를 잘 설명하는 파라미터
- 로그를 씌우는 이유: 곱셈을 합으로 바꿔 계산을 쉽게 함 (곧 Day2에서 유도)
```

---

## Day 2 (화요일) — 베이즈 정리 + MLE 이론 역추적 [2시간]

### 00:00–01:00 | 베이즈 정리를 예시로 유도

```
자료: Harvard Stat 110 (Joe Blitzstein) Lecture 5–6, "Bayes' Rule"
URL: https://projects.iq.harvard.edu/stat110/youtube (강의 목록에서 Lec 5, 6)

오늘 집중할 것:
- 베이즈 정리: P(A|B) = P(B|A) * P(A) / P(B)
- 대표 예시로 직접 유도해보기: "의료 검사 문제"
  P(질병) = 1%     (사전확률, prior)
  P(양성|질병) = 99%   (민감도)
  P(양성|질병 아님) = 5%  (위양성률)
  → P(질병|양성) = ?  (사후확률, posterior) — 직접 손으로 계산

풀이 순서:
  P(양성) = P(양성|질병)*P(질병) + P(양성|질병 아님)*P(질병 아님)
          = 0.99*0.01 + 0.05*0.99 = 0.0099 + 0.0495 = 0.0594
  P(질병|양성) = 0.99*0.01 / 0.0594 ≈ 0.1667 (약 16.7%)

  ✅ 직관 체크: 검사가 99% 정확해도 실제 질병 확률은 17%밖에 안 된다.
     → 사전확률(질병이 드물다는 사실)이 결과를 크게 좌우한다.
```

```python
# W3 Day2: 베이즈 정리를 코드로 검증
def bayes_posterior(prior, sensitivity, false_positive_rate):
    p_positive = sensitivity * prior + false_positive_rate * (1 - prior)
    posterior = sensitivity * prior / p_positive
    return posterior

result = bayes_posterior(prior=0.01, sensitivity=0.99, false_positive_rate=0.05)
print(f"P(질병|양성) = {result:.4f} ({result:.1%})")

# ✅ 최소 보장 체크: 베이즈 정리를 예시로 손으로 유도했는가?
```

막히는 지점 예상:
- "P(양성)을 왜 저렇게 분해하는가?" → 전체확률의 법칙(law of total probability). 넘어가도 되지만 한 줄 메모는 남기기.

### 01:00–02:00 | MLE를 로그우도로 유도

```python
# W3 Day2: 동전 던지기로 MLE 직관 잡기
import numpy as np

# 동전을 10번 던져 앞면이 7번 나왔다고 가정
# "앞면이 나올 확률 p"의 최대우도추정값은?

n, k = 10, 7  # 시도 수, 앞면 수

def likelihood(p, n, k):
    from math import comb
    return comb(n, k) * (p ** k) * ((1 - p) ** (n - k))

def log_likelihood(p, n, k):
    from math import comb, log
    if p <= 0 or p >= 1:
        return -np.inf
    return log(comb(n, k)) + k * log(p) + (n - k) * log(1 - p)

ps = np.linspace(0.01, 0.99, 99)
lls = [log_likelihood(p, n, k) for p in ps]

best_p = ps[np.argmax(lls)]
print(f"로그우도를 최대화하는 p: {best_p:.2f}")
print(f"이론값 (k/n): {k/n:.2f}")

# 시각화
import matplotlib.pyplot as plt
plt.plot(ps, lls)
plt.axvline(best_p, color="red", linestyle="--", label=f"argmax p={best_p:.2f}")
plt.xlabel("p (앞면이 나올 확률)")
plt.ylabel("log-likelihood")
plt.title("MLE: 로그우도를 최대화하는 p 찾기")
plt.legend()
plt.savefig("week03/mle_coin_flip.png", dpi=150)
plt.show()

# ✅ 왜 로그를 씌우는가?
# 1. 우도는 확률의 곱 → 항이 많아지면 값이 지나치게 작아짐 (underflow)
# 2. log는 단조증가 함수 → argmax 위치는 그대로 유지됨
# 3. 곱셈이 합으로 바뀌어 미분(도함수)이 훨씬 쉬워짐
```

```
로지스틱 회귀와 연결:
- 각 샘플의 우도: p_i^{y_i} * (1-p_i)^{1-y_i}   (베르누이 우도)
- 전체 로그우도: Σ [y_i*log(p_i) + (1-y_i)*log(1-p_i)]
- 이걸 최대화 = 음의 로그우도(= 크로스엔트로피 손실)를 최소화
- 이것이 sklearn LogisticRegression이 내부에서 최소화하는 손실함수

# ✅ 최소 보장 체크: MLE를 로그우도로 유도했는가?
```

---

## Day 3 (수요일) — numpy로 로지스틱 회귀 MLE 재구현 [2시간]

### 00:00–01:00 | sigmoid + 로그우도 gradient 유도

```python
# W3 Day3: 로지스틱 회귀를 처음부터 구현하기 위한 수학
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# sigmoid 확인: z=0이면 0.5, z가 커지면 1로, 작아지면 0으로
z_test = np.array([-5, -1, 0, 1, 5])
print("sigmoid 예시:", sigmoid(z_test).round(3))

# 로그우도(음수, 즉 손실함수):
#   L(w) = -Σ [y*log(sigmoid(Xw)) + (1-y)*log(1-sigmoid(Xw))]
#
# w에 대한 gradient (연쇄법칙으로 유도, 결과가 놀랍도록 단순해짐):
#   dL/dw = X.T @ (sigmoid(Xw) - y)
#
# 이 단순함의 이유: d(sigmoid)/dz = sigmoid(z)*(1-sigmoid(z))가
# 로그우도의 미분과 상쇄되면서 (예측값 - 실제값) 형태로 정리됨.
```

### 01:00–02:00 | Gradient Descent로 학습 + sklearn과 비교

```python
# W3 Day3: numpy 로지스틱 회귀 (MLE, gradient descent)
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


def train_logistic_regression(X, y, lr=0.1, n_iters=2000):
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    b = 0.0
    losses = []

    for i in range(n_iters):
        z = X @ w + b
        p = sigmoid(z)

        # 음의 로그우도 (크로스엔트로피)
        eps = 1e-12
        loss = -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
        losses.append(loss)

        # gradient: X.T @ (예측 - 실제)
        grad_w = X.T @ (p - y) / n_samples
        grad_b = np.mean(p - y)

        w -= lr * grad_w
        b -= lr * grad_b

    return w, b, losses


w, b, losses = train_logistic_regression(X_train, y_train)

# 검증: 마지막 손실값이 잘 내려갔는가?
print(f"초기 손실: {losses[0]:.4f} → 최종 손실: {losses[-1]:.4f}")

# 내 구현 정확도
y_pred_manual = (sigmoid(X_test @ w + b) >= 0.5).astype(int)
acc_manual = (y_pred_manual == y_test).mean()

# sklearn 정확도
clf = LogisticRegression(max_iter=1000).fit(X_train, y_train)
acc_sklearn = clf.score(X_test, y_test)

print(f"\n내 구현 정확도: {acc_manual:.3f}")
print(f"sklearn 정확도: {acc_sklearn:.3f}")

# ✅ 최소 보장 체크:
# "로지스틱 회귀 = 로그우도 최대화 = 크로스엔트로피 최소화"를
# 직접 gradient descent로 구현해 sklearn과 비슷한 정확도가 나오는지 확인했는가?
```

막히면: 손실이 발산(NaN)하면 `lr`을 0.01로 낮추거나 `np.clip`으로 sigmoid 입력을 제한.

```bash
git add week03/day3_logistic_regression_manual.py
git commit -m "W3 Day3: numpy logistic regression via MLE gradient descent"
git push
```

---

## Day 4 (목요일) — Python OOP: 클래스·상속·decorator [2시간]

### 00:00–01:00 | 모델 래퍼 클래스 (상속 + decorator)

```python
# W3 Day4: OOP로 분류기 래퍼 만들기
import time
import functools
from abc import ABC, abstractmethod
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def log_timing(func):
    """메서드 실행 시간을 재는 decorator."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"[{func.__name__}] {elapsed:.4f}초")
        return result
    return wrapper


class BaseClassifierWrapper(ABC):
    """모든 분류기 래퍼가 따라야 하는 공통 인터페이스."""

    def __init__(self, name: str):
        self.name = name
        self._is_fitted = False

    @abstractmethod
    def _build_model(self):
        ...

    @log_timing
    def fit(self, X, y):
        self.model = self._build_model()
        self.model.fit(X, y)
        self._is_fitted = True
        return self

    @log_timing
    def predict_proba(self, X):
        if not self._is_fitted:
            raise RuntimeError(f"{self.name}: fit() 먼저 호출하세요.")
        return self.model.predict_proba(X)[:, 1]

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name!r}, fitted={self._is_fitted})"


class LogisticWrapper(BaseClassifierWrapper):
    def __init__(self, C: float = 1.0):
        super().__init__(name="LogisticRegression")
        self.C = C

    def _build_model(self):
        return LogisticRegression(C=self.C, max_iter=1000)


class RandomForestWrapper(BaseClassifierWrapper):
    def __init__(self, n_estimators: int = 100):
        super().__init__(name="RandomForest")
        self.n_estimators = n_estimators

    def _build_model(self):
        return RandomForestClassifier(n_estimators=self.n_estimators, random_state=42)


# 사용
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42, stratify=data.target
)
X_train = StandardScaler().fit_transform(X_train)
X_test = StandardScaler().fit_transform(X_test)  # 실전에서는 같은 scaler 재사용!

models = [LogisticWrapper(C=1.0), RandomForestWrapper(n_estimators=100)]
for m in models:
    m.fit(X_train, y_train)
    auc = roc_auc_score(y_test, m.predict_proba(X_test))
    print(f"{m}: AUC={auc:.3f}")
```

주의: 위 코드는 train/test에 서로 다른 `StandardScaler` 인스턴스를 써서 일부러 실수를 남겨뒀습니다.
직접 고쳐보세요 (`fit`은 train에만, `transform`은 둘 다 같은 scaler로).

### 01:00–02:00 | LeetCode + 커밋

```python
# LeetCode #3: Valid Parentheses (Easy)
# 시간제한: 15분

def isValid(s: str) -> bool:
    stack = []
    pairs = {")": "(", "]": "[", "}": "{"}
    for ch in s:
        if ch in "([{":
            stack.append(ch)
        elif ch in pairs:
            if not stack or stack.pop() != pairs[ch]:
                return False
    return not stack

assert isValid("()[]{}") is True
assert isValid("(]") is False
assert isValid("([)]") is False
print("Valid Parentheses 통과")


# LeetCode #4: Min Stack (Medium)
# 시간제한: 25분. 안 풀리면 풀이 보고 이해만 해도 OK.

class MinStack:
    def __init__(self):
        self.stack = []       # (value, current_min) 튜플 저장
        
    def push(self, val: int) -> None:
        current_min = val if not self.stack else min(val, self.stack[-1][1])
        self.stack.append((val, current_min))

    def pop(self) -> None:
        self.stack.pop()

    def top(self) -> int:
        return self.stack[-1][0]

    def getMin(self) -> int:
        return self.stack[-1][1]

# 테스트
ms = MinStack()
ms.push(-2); ms.push(0); ms.push(-3)
assert ms.getMin() == -3
ms.pop()
assert ms.top() == 0
assert ms.getMin() == -2
print("Min Stack 통과")
```

```bash
git add week03/
git commit -m "W3 Day4: classifier wrapper (OOP+decorator) + Valid Parentheses + Min Stack"
git push
```

---

## Day 5 (금요일) — 복습 + LeNet 논문 + 커밋 [2시간]

### 00:00–00:30 | W3 이론 복습

스스로에게 물어볼 것 (답이 안 나오면 해당 자료 다시 보기):

```
□ 베이즈 정리를 한 문장으로?
  → 사전확률과 관측된 증거(likelihood)를 결합해 사후확률을 구하는 공식

□ 의료 검사 예시에서 검사가 99% 정확해도 사후확률이 낮았던 이유는?
  → 사전확률(질병이 드묾)이 결과를 지배함 — "base rate fallacy"와 연결

□ 로지스틱 회귀의 손실함수는 무엇이고 왜 그것을 쓰는가?
  → 음의 로그우도(크로스엔트로피). MLE 관점에서 데이터를 가장 잘 설명하는 파라미터를 찾는 것과 동일

□ gradient가 X.T @ (예측 - 실제) 형태로 단순해지는 이유는?
  → sigmoid의 도함수가 로그우도 미분과 상쇄되기 때문 (연쇄법칙 결과)
```

### 00:30–01:00 | LeetCode Min Stack 재도전 (안 풀렸으면)

```
Day4에 Min Stack이 안 풀렸다면 지금 아무것도 안 보고 다시 시도.
여전히 안 되면 위 풀이를 다시 타이핑하며 한 줄씩 이해.
```

### 01:00–01:30 | LeNet 논문 맥락 파악 (20분으로 충분)

```
읽을 것: LeCun et al. (1998) "Gradient-Based Learning Applied to Document Recognition" (LeNet)
- 전체 읽을 필요 없음
- 읽을 부분: Abstract, Section 1 (Introduction), Figure 2 (LeNet-5 구조)
- 시간: 20분

읽으면서 메모:
1. Fully-connected 신경망으로 이미지를 처리하면 왜 비효율적인가?
   (힌트: 픽셀 위치가 바뀌면 완전히 다른 입력으로 취급됨 — 위치 불변성 없음)
2. Convolution이 무엇을 "공유"해서 파라미터를 줄이는가?
3. Pooling의 역할은?

이번 주 W3에서는 개념만 스치고, Convolution 자체는 블록 C(W51)에서 깊게 다룹니다.
```

### 01:30–02:00 | 금요일 마무리 커밋 + 다음 주 준비

```bash
git add .
git commit -m "W3 완료: 로지스틱 회귀, 베이즈 정리, MLE, OOP 모델 래퍼"
git push

cat >> week03/README.md << 'EOF'

## W3 완료 항목
- [x] sklearn LogisticRegression + 혼동행렬 + ROC/AUC
- [x] 베이즈 정리를 의료검사 예시로 유도
- [x] MLE를 로그우도로 유도 (동전 던지기 예시)
- [x] numpy로 로지스틱 회귀 MLE gradient descent 구현
- [x] BaseClassifierWrapper (OOP + decorator)
- [x] LeetCode: Valid Parentheses, Min Stack

## 최소 보장 체크
- [x] 베이즈 정리를 예시로 유도 가능
- [x] MLE를 로그우도로 유도 가능
EOF
```

---

## 주말 — 심화 [5–6시간]

### 토요일 [3시간]

**[00:00–01:30] 혼동행렬 지표 + ROC/AUC 수식 깊이 이해**

```python
# W3 토요일: precision·recall·F1·AUC를 손으로 계산해보기
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42, stratify=data.target
)
scaler = StandardScaler().fit(X_train)
clf = LogisticRegression(max_iter=1000).fit(scaler.transform(X_train), y_train)
y_pred = clf.predict(scaler.transform(X_test))

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print(f"TP={tp}, FP={fp}, FN={fn}, TN={tn}")

precision = tp / (tp + fp)
recall = tp / (tp + fn)          # = sensitivity = TPR
f1 = 2 * precision * recall / (precision + recall)
specificity = tn / (tn + fp)     # TNR

print(f"Precision: {precision:.3f}")
print(f"Recall (TPR): {recall:.3f}")
print(f"Specificity (TNR): {specificity:.3f}")
print(f"F1: {f1:.3f}")

# 질문에 스스로 답하기:
# - 이 데이터(암 진단)에서 FN(암을 놓침)과 FP(오진) 중 어느 게 더 위험한가?
# - Precision과 Recall 중 어느 것을 더 우선해야 하는가?
# - threshold를 0.5에서 0.3으로 낮추면 Recall과 Precision은 각각 어떻게 변하는가?
```

**[01:30–03:00] 로지스틱 회귀 decision boundary 시각화 (2D 축소)**

```python
# W3 토요일: PCA로 2차원 축소 후 decision boundary 그리기
# (W1에서 배운 PCA를 여기서 다시 활용 — 지식이 연결되는 걸 느껴보기)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

data = load_breast_cancer()
X_scaled = StandardScaler().fit_transform(data.data)
X_2d = PCA(n_components=2).fit_transform(X_scaled)
y = data.target

clf = LogisticRegression().fit(X_2d, y)

xx, yy = np.meshgrid(
    np.linspace(X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1, 300),
    np.linspace(X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1, 300),
)
Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1].reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, levels=20, cmap="RdBu", alpha=0.6)
plt.colorbar(label="P(양성)")
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap="RdBu", edgecolors="k", s=20)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("로지스틱 회귀 Decision Boundary (PCA 2D)")
plt.savefig("week03/decision_boundary.png", dpi=150)
plt.show()

# 관찰: decision boundary가 직선(선형)으로 보이는가?
# → 로지스틱 회귀는 "선형" 분류기임을 시각적으로 확인하는 것이 이 실습의 목적
```

30분 이상 한 개념(예: decision boundary가 왜 직선인지)에 막히면 → 넘어가고 W5(엔트로피·정보이득) 즈음에 다시 연결.

### 일요일 [2–3시간]

**[00:00–01:30] W3 최종 재구현 + 영어 설명 연습**

```python
# W3 토탈 리뷰: 처음부터 끝까지 혼자서 다시
# 아무것도 보지 않고 아래를 구현할 수 있는가?

# 1. breast_cancer 로드 + train/test 분리 + 스케일링 (5분)
# 2. sigmoid 함수 정의 (2분)
# 3. 로그우도 기반 손실함수 정의 (5분)
# 4. gradient descent 학습 루프 작성 (10분)
# 5. sklearn LogisticRegression과 정확도 비교 (5분)

# 막히면 Day3 코드 참고 가능. 중요한 건 "로그우도 최대화 = 손실 최소화" 흐름을 기억하는 것.
```

영어로 말해보기 (혼자서 소리 내어):

아래에서 `___%` 부분은 반드시 **직접 코드를 돌려 나온 값**으로 채우세요.

```
"Logistic regression models the probability of a binary outcome
using the sigmoid function. Its parameters are found via
Maximum Likelihood Estimation — we maximize the log-likelihood
of the observed data, which is equivalent to minimizing the
cross-entropy loss. On the breast cancer dataset, my model
achieved an AUC of ___ (fill in your measured value),
meaning it can rank malignant cases higher than benign cases
that percentage of the time."
```

**[01:30–02:30] W4 준비 + 주간 회고**

```markdown
## W3 회고 (일요일에 작성)

### 달성한 것
- [ ] sklearn LogisticRegression + 혼동행렬 + ROC/AUC 실습
- [ ] 베이즈 정리 예시로 유도
- [ ] MLE를 로그우도로 유도
- [ ] numpy 로지스틱 회귀 MLE gradient descent 구현
- [ ] OOP 모델 래퍼(상속+decorator) 작성
- [ ] LeetCode Valid Parentheses, Min Stack

### 최소 보장 체크
- [ ] 베이즈 정리를 예시로 유도 가능
- [ ] MLE를 로그우도로 유도 가능

### 예상보다 오래 걸린 것
(솔직하게 적기)

### W4에 가져갈 것
(이번 주에 이해 못 하고 넘어간 것 — 예: decision boundary가 왜 선형인지)

### 다음 주 첫 번째 할 일
(W4 Day1 무엇부터 시작할지)
```

---

## 막힐 때 대응 가이드

개념이 막히면:

```
1단계 (5분): 구글에 영어로 검색
  예: "why logistic regression uses log likelihood not squared error"

2단계 (10분): StatQuest 관련 영상 검색

3단계 (20분): Harvard Stat 110 해당 강의 노트 확인
  https://projects.iq.harvard.edu/stat110/youtube

30분 넘어도 해결 안 되면:
→ 메모장에 "아직 모름: [개념]" 적고 다음으로 넘어가기
→ W4 또는 W5에서 다시 만날 때 해결
→ AI 커리큘럼에서 막히는 것은 실력 부족이 아니라 정상 과정
```

코드가 막히면:

```
에러 메시지 전체를 복사 → 구글에 붙여넣기
Stack Overflow 답변 중 가장 많은 추천을 받은 것 선택

자주 나오는 에러:
- ModuleNotFoundError: pip install [라이브러리명]
- Shape mismatch: print(array.shape) 로 차원 확인
- 로그우도 계산 중 log(0) 에러: eps(예: 1e-12)를 더해 방지
- gradient descent 손실이 NaN/발산: learning rate를 낮추기
```

---

## W3 완료 기준

일요일 저녁에 아래를 할 수 있으면 W3 성공:

```
□ github.com/YOUR_USERNAME/ai-engineer-journey 의 week03/ 폴더에 코드가 올라가 있다
□ 혼동행렬·ROC 곡선 이미지가 week03/ 폴더에 있다
□ numpy로 로지스틱 회귀를 MLE(gradient descent) 관점에서 직접 구현한 코드가 있다
□ BaseClassifierWrapper 클래스가 작동한다 (상속 + decorator 사용)
□ 베이즈 정리를 예시(의료 검사)로 3문장 이상 설명할 수 있다 (한국어 가능)
□ LeetCode Valid Parentheses, Min Stack을 이해하고 혼자 다시 풀 수 있다

절반(3개 이상) 달성하면 W4로 진행.
전부 못 해도 W4로 진행 — 이해 못 한 부분은 이후 주차에서 다시 나옴.
```

---

## W4 첫 할 일 미리 보기

W4 Day1에 열어야 할 것:

1. `week04/` 폴더 생성
2. OpenAI 또는 Anthropic API 키 발급 + 첫 API 호출 테스트
3. **미니프로젝트 #1 착수**: LLM 텍스트 요약·분류기 CLI
   - 공개 뉴스 RSS 피드 → API로 요약 + 카테고리 분류
4. 막히면 → SVD·차원축소 개념 → MIT 18.06 해당 파트로 역추적
5. Git branch·PR 워크플로우 처음 사용해보기

---

*이번 주 진짜 목표는 "숫자가 아니라 확률로 생각하는 습관"을 만드는 것입니다.
sklearn이 내부에서 뭘 최적화하는지 한 번이라도 numpy로 직접 확인했다면 W3는 이미 성공입니다.*

# W2 구체적 실행 계획

> **주제**: LU분해·영공간 재활성화 + numpy PCA 파이프라인 완성 + Python 함수형·예외처리
>
> **사용 데이터셋 주의**: W1과 동일하게 sklearn `load_digits()`(8×8 픽셀, 1797개, "미니 MNIST")를 기본으로 쓰되,
> 목요일 이후 비교용으로 `load_wine()`(178개, 13차원)을 하나 더 씁니다. 둘 다 진짜 MNIST가 아닙니다.
> **총 목표 시간**: 15–16시간
> **기준**: 평일 2시간 + 주말(토요일 3시간, 일요일 2–3시간)

---

## W2 목표 (이것만 달성하면 성공)

1. **이론**: LU분해·영공간(null space)의 의미를 설명할 수 있다
2. **재구현**: 공분산 → 고유값분해(또는 SVD) → 투영 → 재구성까지 PCA 전 과정을 함수로 완성
3. **판단력**: 설명 분산 누적 그래프로 "몇 개의 주성분을 쓸지" 스스로 결정할 수 있다
4. **Python**: comprehension·함수형(map/filter)·예외처리를 PCA 코드에 적용
5. **최소 보장**: PCA를 고유값분해로 손계산 / SVD와 PCA 차이를 행렬 분해 관점에서 설명 가능

---

## Day 1 (월요일) — LU분해·영공간 [2시간]

W1에서 익힌 행렬 곱·역행렬 위에 "해가 없거나 무한히 많은 경우"를 다룹니다.

### 00:00–01:00 | 개념: LU분해 · 영공간

```
시청: 3Blue1Brown "Essence of Linear Algebra" 9–15화 중 관련 있는 것만
  - Ch.9 Dot products
  - Ch.12 Cross products
  - Ch.14 Determinant 복습 (W1 이어서)
URL: https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab

병행: MIT 18.06 Lecture 4 (Gilbert Strang) — LU 분해
URL: https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/

메모할 것:
- LU분해: A = L(하삼각) @ U(상삼각). 가우스 소거법을 행렬로 기록한 것
- 영공간(null space): A @ x = 0 을 만족하는 x들의 집합. "정보가 사라지는 방향"
- 왜 PCA에 중요한가: 공분산 행렬이 특이(singular)하면 일부 고유값이 0 → 그 방향은 분산이 없는 방향
```

막히는 지점 예상:
- "왜 가우스 소거법을 행렬 곱으로 표현하나?" → 지금은 "그렇게 정리된다"만 받아들이고 넘어가세요. 증명은 ROI 낮음.

### 01:00–02:00 | numpy로 LU분해·영공간 직접 확인

```python
# W2 Day1: LU분해 + 영공간 실습
import numpy as np
from scipy.linalg import lu, null_space

A = np.array([[2, 1, 1],
              [4, 3, 3],
              [8, 7, 9]], dtype=float)

# LU 분해 (scipy)
P, L, U = lu(A)
print("P (치환행렬) =\n", P)
print("L (하삼각) =\n", L)
print("U (상삼각) =\n", U)
print("P@L@U == A ?", np.allclose(P @ L @ U, A))

# 영공간이 있는 특이 행렬 예시
B = np.array([[1, 2, 3],
              [2, 4, 6],       # 첫 행의 2배 → 선형 종속
              [1, 1, 1]], dtype=float)

print("\nrank(B) =", np.linalg.matrix_rank(B))   # 3보다 작으면 특이 행렬
ns = null_space(B)
print("영공간 basis:\n", ns)

if ns.shape[1] > 0:
    v = ns[:, 0]
    print("B @ v =", B @ v)   # 0 벡터에 가까워야 함
    print("0 벡터에 가까운가:", np.allclose(B @ v, 0, atol=1e-8))

# ✅ 최소 보장과 연결:
# 공분산 행렬이 특이하면(예: 샘플 수 < 차원 수) 고유값 일부가 0 근처가 된다.
# 이는 "그 방향으로는 데이터가 전혀 퍼져 있지 않다"는 뜻.
```

막히면: `pip install scipy` (Colab엔 기본 설치됨) → 에러 메시지 그대로 구글 검색

```bash
# 커밋
mkdir -p week02
git add week02/
git commit -m "W2 Day1: LU decomposition + null space"
git push
```

---

## Day 2 (화요일) — PCA를 SVD로도 풀어보기 [2시간]

W1에서는 고유값분해(eigendecomposition)로 PCA를 했습니다. 오늘은 SVD로 같은 결과를 얻고 차이를 비교합니다.

### 00:00–01:00 | SVD vs 고유값분해 비교

```python
# W2 Day2: SVD로 PCA 다시 풀기
import numpy as np
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler

digits = load_digits()
X_scaled = StandardScaler().fit_transform(digits.data)
n_samples = X_scaled.shape[0]

# 방법 1: 공분산 행렬의 고유값분해 (W1에서 한 방식)
cov = (X_scaled.T @ X_scaled) / (n_samples - 1)
eigvals, eigvecs = np.linalg.eigh(cov)
idx = np.argsort(eigvals)[::-1]
eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]

# 방법 2: X_scaled를 직접 SVD
# X_scaled = U @ S @ Vt  →  V의 열이 주성분 방향
U, S, Vt = np.linalg.svd(X_scaled, full_matrices=False)

# 두 방법이 같은 결과인지 확인 (부호는 다를 수 있음)
print("고유값분해 상위 3개 고유값:", eigvals[:3].round(3))
print("SVD 특이값의 제곱/(n-1):   ", ((S[:3] ** 2) / (n_samples - 1)).round(3))
print("둘이 일치하는가:", np.allclose(eigvals[:3], (S[:3] ** 2) / (n_samples - 1), atol=1e-6))

# 투영 결과 비교 (부호 무시하고 절댓값 비교)
proj_eig = X_scaled @ eigvecs[:, :2]
proj_svd = U[:, :2] * S[:2]
print("\n투영 결과 절댓값 일치:",
      np.allclose(np.abs(proj_eig[0]), np.abs(proj_svd[0]), atol=1e-6))

# ✅ 최소 보장 핵심 문장:
# "공분산 행렬을 고유값분해하는 것 = 데이터 행렬을 SVD하는 것과 수학적으로 동치다.
#  다만 SVD는 공분산 행렬을 명시적으로 만들지 않아도 되고 수치적으로 더 안정적이다."
```

막히면: "SVD PCA equivalence covariance matrix" 검색 → StatQuest "SVD" 영상 (12분)

### 01:00–02:00 | 손계산: 2×2 행렬로 PCA 전 과정

작은 행렬로 직접 손으로 풀어보고, numpy로 검증합니다. 이게 오늘의 핵심 최소 보장 훈련입니다.

```python
# W2 Day2: 손계산용 2x2 예제 (직접 종이에 먼저 풀어본 뒤 아래로 검증)
import numpy as np

# 데이터: 이미 평균이 0으로 중심화되어 있다고 가정
X = np.array([[2.5, 2.4],
              [0.5, 0.7],
              [2.2, 2.9],
              [1.9, 2.2],
              [3.1, 3.0],
              [2.3, 2.7],
              [2.0, 1.6],
              [1.0, 1.1],
              [1.5, 1.6],
              [1.1, 0.9]])

X_centered = X - X.mean(axis=0)

# 손계산 순서 (종이에 먼저):
# 1. 공분산 행렬 Cov = (X_c^T @ X_c) / (n-1)  — 2x2 행렬이라 직접 곱셈 가능
# 2. 특성방정식 det(Cov - λI) = 0 풀어서 고유값 2개 구하기
# 3. 각 λ에 대해 (Cov - λI)v = 0 풀어서 고유벡터 구하기

cov = (X_centered.T @ X_centered) / (len(X) - 1)
print("공분산 행렬 (손으로 푼 것과 비교):\n", cov.round(4))

eigvals, eigvecs = np.linalg.eigh(cov)
idx = np.argsort(eigvals)[::-1]
eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]
print("\n고유값 (손계산과 비교):", eigvals.round(4))
print("고유벡터 (손계산과 비교):\n", eigvecs.round(4))

# ✅ 최소 보장 체크: 아래 두 줄을 종이 풀이와 소수점 2자리까지 맞춰보기
```

막히면 (30분 넘게 특성방정식이 안 풀리면):
```
StatQuest "PCA main ideas" (20분) 시청 → 특성방정식 부분만 다시
그래도 안 풀리면: numpy 결과를 정답으로 받아들이고,
"공분산 → det(Cov-λI)=0 → 고유벡터" 순서만 외워서 다음으로 넘어가기
```

```bash
git add week02/
git commit -m "W2 Day2: SVD vs eigendecomposition + hand-calculated PCA"
git push
```

---

## Day 3 (수요일) — 주성분 개수 결정 + 재구성 [2시간]

"2차원으로 줄인다"는 임의의 선택이었습니다. 오늘은 몇 개가 적절한지 데이터로 결정합니다.

### 00:00–01:00 | 누적 설명 분산 + Scree plot

```python
# W2 Day3: 몇 개의 주성분을 쓸지 결정하기
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

digits = load_digits()
X_scaled = StandardScaler().fit_transform(digits.data)

pca_full = PCA().fit(X_scaled)  # n_components 지정 안 하면 전체 계산
cum_var = np.cumsum(pca_full.explained_variance_ratio_)

# 90% 분산을 설명하는 데 필요한 최소 주성분 개수
k_90 = np.argmax(cum_var >= 0.90) + 1
print(f"90% 분산 설명에 필요한 주성분 개수: {k_90} (64차원 중)")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(range(1, len(cum_var) + 1), cum_var, marker='o', markersize=3)
axes[0].axhline(0.90, color='red', linestyle='--', label='90% 기준선')
axes[0].axvline(k_90, color='gray', linestyle=':')
axes[0].set_xlabel('주성분 개수')
axes[0].set_ylabel('누적 설명 분산')
axes[0].set_title('누적 설명 분산 (Cumulative)')
axes[0].legend()

axes[1].bar(range(1, 11), pca_full.explained_variance_ratio_[:10])
axes[1].set_xlabel('주성분 번호')
axes[1].set_ylabel('설명 분산 비율')
axes[1].set_title('Scree Plot (상위 10개)')

plt.tight_layout()
plt.savefig('week02/scree_plot.png', dpi=150)
plt.show()

# 스스로에게 물어볼 것:
# - 2차원(W1에서 쓴 것)은 분산의 몇 %만 설명하는가? → 시각화용이지 정보 보존용이 아니었다는 걸 확인
```

### 01:00–01:30 | 재구성(reconstruction) 오차 확인

```python
# W2 Day3: k개 주성분으로 압축 후 원본 복원, 오차 측정
pca_k = PCA(n_components=k_90)
X_reduced = pca_k.fit_transform(X_scaled)
X_reconstructed = pca_k.inverse_transform(X_reduced)

mse = np.mean((X_scaled - X_reconstructed) ** 2)
print(f"k={k_90}개 주성분으로 압축 후 재구성 MSE: {mse:.4f}")

# 이미지로 확인 (원본 vs 재구성)
fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for i in range(5):
    axes[0, i].imshow(digits.data[i].reshape(8, 8), cmap='gray')
    axes[0, i].set_title('원본')
    axes[0, i].axis('off')

    recon_img = X_reconstructed[i]  # 스케일된 공간이라 완벽히 똑같진 않음
    axes[1, i].imshow(recon_img.reshape(8, 8), cmap='gray')
    axes[1, i].set_title(f'k={k_90} 복원')
    axes[1, i].axis('off')

plt.tight_layout()
plt.savefig('week02/reconstruction_comparison.png', dpi=150)
plt.show()

# ✅ 최소 보장과 연결:
# "SVD와 PCA의 차이" 질문에 답할 때 이 실습을 예로 들 수 있다:
# PCA는 SVD의 응용 사례 중 하나 — 차원 축소 + 재구성이 목적,
# SVD는 더 일반적인 행렬 분해 도구 (추천 시스템, 이미지 압축 등에도 쓰임)
```

### 01:30–02:00 | LeetCode

```python
# LeetCode #3: Valid Palindrome (Easy)
# 시간제한: 20분

def isPalindrome(s: str) -> bool:
    filtered = [c.lower() for c in s if c.isalnum()]  # comprehension 미리 연습
    return filtered == filtered[::-1]

assert isPalindrome("A man, a plan, a canal: Panama") is True
assert isPalindrome("race a car") is False
print("Valid Palindrome 통과")
```

```bash
git add week02/
git commit -m "W2 Day3: explained variance analysis + reconstruction"
git push
```

---

## Day 4 (목요일) — Python 함수형·예외처리로 PCA 파이프라인 정리 [2시간]

### 00:00–01:00 | comprehension·함수형(map/filter/reduce)

```python
# W2 Day4: PCA 파이프라인을 함수형 스타일로 정리
import numpy as np
from functools import reduce

def load_and_scale(loader_fn):
    """데이터 로더를 받아 (X_scaled, y) 반환. loader_fn: sklearn dataset loader"""
    data = loader_fn()
    X = data.data
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0  # 0으로 나누기 방지 (예외 대신 방어적 코딩)
    X_scaled = (X - mean) / std
    return X_scaled, data.target


def compute_cumulative_variance(eigenvalues: np.ndarray) -> np.ndarray:
    """comprehension 없이 reduce로 누적합 계산 (연습용)"""
    total = sum(eigenvalues)
    cum = reduce(lambda acc, v: acc + [acc[-1] + v] if acc else [v],
                 eigenvalues, [])
    return np.array(cum) / total


def k_for_variance_threshold(cum_var: np.ndarray, threshold: float = 0.9) -> int:
    """map/filter 없이도 되지만, filter로 인덱스 찾는 연습"""
    candidates = [i for i, v in enumerate(cum_var) if v >= threshold]
    if not candidates:
        raise ValueError(f"threshold={threshold}를 만족하는 주성분이 없습니다.")
    return candidates[0] + 1


# 실행
from sklearn.datasets import load_digits, load_wine

for name, loader in [("digits", load_digits), ("wine", load_wine)]:
    X_scaled, y = load_and_scale(loader)
    cov = (X_scaled.T @ X_scaled) / (len(X_scaled) - 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = eigvals[::-1]  # 내림차순

    cum_var = compute_cumulative_variance(eigvals)
    k = k_for_variance_threshold(cum_var, 0.90)
    print(f"[{name}] 원래 차원: {X_scaled.shape[1]}, 90% 설명에 필요한 차원: {k}")
```

### 01:00–01:30 | 예외처리(exception handling)

```python
# W2 Day4: PCA 클래스에 방어적 예외처리 추가 (W1 SimplePCA 확장)
import numpy as np
from dataclasses import dataclass
from typing import Optional


class PCANotFittedError(Exception):
    """fit() 호출 전에 transform()을 호출했을 때"""
    pass


class InvalidComponentsError(ValueError):
    """n_components가 유효 범위를 벗어났을 때"""
    pass


@dataclass
class PCAConfig:
    n_components: int = 2


class SimplePCA:
    def __init__(self, config: PCAConfig):
        self.config = config
        self.components_: Optional[np.ndarray] = None
        self.explained_variance_ratio_: Optional[np.ndarray] = None
        self._is_fitted = False

    def fit(self, X: np.ndarray) -> "SimplePCA":
        n_samples, n_features = X.shape

        if not (1 <= self.config.n_components <= n_features):
            raise InvalidComponentsError(
                f"n_components={self.config.n_components}는 "
                f"1~{n_features} 범위여야 합니다."
            )

        try:
            cov = (X.T @ X) / (n_samples - 1)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
        except np.linalg.LinAlgError as e:
            raise RuntimeError(f"고유값분해 실패: {e}") from e

        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues, eigenvectors = eigenvalues[idx], eigenvectors[:, idx]

        self.components_ = eigenvectors[:, :self.config.n_components].T
        self.explained_variance_ratio_ = (
            eigenvalues[:self.config.n_components] / eigenvalues.sum()
        )
        self._is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise PCANotFittedError("fit()을 먼저 호출하세요.")
        return X @ self.components_.T


# 테스트: 예외가 실제로 잘 발생하는지 확인
from sklearn.datasets import load_digits
X = load_digits().data

try:
    bad_pca = SimplePCA(PCAConfig(n_components=999))
    bad_pca.fit(X)
except InvalidComponentsError as e:
    print(f"예상된 에러 발생: {e}")

try:
    unfitted_pca = SimplePCA(PCAConfig(n_components=2))
    unfitted_pca.transform(X)
except PCANotFittedError as e:
    print(f"예상된 에러 발생: {e}")

print("모든 예외처리 테스트 통과")
```

### 01:30–02:00 | LeetCode + 커밋

```python
# LeetCode #4: Reverse Linked List (Easy)
# 시간제한: 25분

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def reverseList(head: ListNode) -> ListNode:
    prev = None
    curr = head
    while curr:
        next_temp = curr.next
        curr.next = prev
        prev = curr
        curr = next_temp
    return prev

# 테스트
def to_list(node):
    out = []
    while node:
        out.append(node.val)
        node = node.next
    return out

n3 = ListNode(3)
n2 = ListNode(2, n3)
n1 = ListNode(1, n2)
reversed_head = reverseList(n1)
assert to_list(reversed_head) == [3, 2, 1]
print("Reverse Linked List 통과")
```

```bash
git add week02/
git commit -m "W2 Day4: functional-style PCA pipeline + custom exceptions"
git push
```

---

## Day 5 (금요일) — 복습 + 논문 [2시간]

### 00:00–00:30 | W2 수학 복습 (최소 보장 재확인)

```
□ PCA를 고유값분해로 손계산할 수 있는가?
  → Day2의 2x2 예제를 아무것도 안 보고 다시 풀어보기

□ SVD와 PCA의 차이는?
  → PCA는 목적(차원축소), SVD는 도구(행렬 분해).
    공분산 행렬을 고유값분해하는 것과 데이터 행렬을 SVD하는 것은 수학적으로 동치.

□ 영공간(null space)이 왜 중요한가?
  → 공분산 행렬이 특이하면(예: 샘플 < 차원) 일부 방향의 분산이 0.
    그 방향은 정보가 없으므로 주성분에서 제외해도 무방.

□ 몇 개의 주성분을 쓸지 어떻게 정하는가?
  → 누적 설명 분산이 목표 임계값(예: 90%)을 넘는 최소 개수
```

### 00:30–01:30 | 논문: 역전파의 탄생 (20–40분으로 충분)

```
읽을 것: Rumelhart, Hinton, Williams (1986)
"Learning representations by back-propagating errors"

읽을 부분: Abstract + 핵심 아이디어 (전체 읽을 필요 없음)

메모할 것:
1. 이 논문 이전에는 신경망을 어떻게 학습시켰는가? (또는 왜 못 시켰는가)
2. 역전파의 핵심 아이디어를 한 문장으로: _______________
3. "연쇄법칙(chain rule)"이라는 단어가 왜 등장하는가?
   → W7에서 SGD 직접 구현할 때 이 개념이 다시 나옵니다. 지금은 이름만 익혀두면 충분.

이 논문이 왜 중요한지 영어로 한 문장 준비:
"Backpropagation made it possible to train multi-layer neural networks
by efficiently computing gradients using the chain rule,
which is the foundation of virtually all deep learning today."
```

### 01:30–02:00 | 금요일 마무리 커밋 + 다음 주 준비

```bash
git add .
git commit -m "W2 완료: LU분해·영공간, PCA를 SVD로도 검증, 함수형·예외처리"
git push

cat >> week02/README.md << 'EOF'
# W2: LU decomposition, null space, PCA via SVD

## W2 완료 항목
- [x] LU분해·영공간 numpy로 확인
- [x] PCA를 SVD로도 풀어서 고유값분해와 비교
- [x] 손으로 2x2 PCA 문제 풀고 numpy로 검증
- [x] 누적 설명 분산으로 주성분 개수 결정 + 재구성 오차 확인
- [x] PCA 파이프라인을 함수형 스타일 + 커스텀 예외로 정리
- [x] LeetCode: Valid Palindrome, Reverse Linked List

## 최소 보장 체크
- [x] PCA를 고유값분해로 손계산 가능
- [x] SVD와 PCA 차이를 행렬 분해 관점에서 설명 가능
EOF

git add week02/README.md
git commit -m "W2: update README"
git push
```

---

## 주말 — 심화 [5–6시간]

### 토요일 [3시간]

**[00:00–01:30] 다른 데이터셋에 PCA 파이프라인 적용 (일반화 검증)**

Day4에서 만든 함수형 파이프라인이 digits 말고 다른 데이터에도 통하는지 확인합니다.

```python
# W2 토요일: Wine 데이터셋으로 PCA 파이프라인 재사용
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler

wine = load_wine()
X_scaled = StandardScaler().fit_transform(wine.data)
y = wine.target

cov = (X_scaled.T @ X_scaled) / (len(X_scaled) - 1)
eigvals, eigvecs = np.linalg.eigh(cov)
idx = np.argsort(eigvals)[::-1]
eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]

X_pca = X_scaled @ eigvecs[:, :2]

plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', s=40)
plt.colorbar(scatter, label='와인 클래스 (0-2)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title(f'Wine 데이터셋 PCA (13차원→2차원, '
          f'설명분산 {sum(eigvals[:2])/sum(eigvals):.1%})')
plt.savefig('week02/wine_pca.png', dpi=150)
plt.show()

# digits와 비교해서 생각해볼 것:
# - Wine은 클래스가 더 뚜렷하게 분리되는가? 왜 그럴까? (특징 수가 적고 스케일 차이가 큼)
```

**[01:30–03:00] MIT 18.06 Lecture 5–6 보충 (필요 시)**

```
Day1에서 LU분해가 충분히 이해됐으면 스킵 가능.
막혔던 부분(치환행렬, 가우스 소거 순서)이 있으면 지금 다시 보기.
Lecture 6: 열공간(column space)·영공간(null space) 통합 정리.
30분 넘게 막히면 → 넘어가고 메모만 남기기.
```

### 일요일 [2–3시간]

**[00:00–01:30] W2 최종 재구현 + 영어 설명 연습**

```python
# W2 토탈 리뷰: 아무것도 보지 않고 다시 구현해보기

# 1. digits 또는 wine 데이터 로드 + 표준화 (5분)
# 2. 공분산 행렬 계산 (5분)
# 3. 고유값분해로 고유값·고유벡터 구하기 (5분)
# 4. 누적 설명 분산 계산해서 90% 임계값의 k 찾기 (10분)
# 5. k개로 투영 후 재구성, MSE 계산 (10분)
# 6. SVD로도 같은 결과 나오는지 검증 (10분)

# 막히면 Day2~3 코드 참고 가능. 목표는 흐름을 손에 익히는 것.
```

영어로 말해보기 (혼자서 소리 내어, `___` 부분은 직접 측정한 값으로 채우기):

```
"PCA and SVD are closely related but not the same thing.
PCA is a technique for dimensionality reduction — it finds the directions
of maximum variance in the data. SVD is a more general matrix
factorization that decomposes any matrix into U, S, and V^T.
When you eigendecompose the covariance matrix, you get the same
principal directions as when you apply SVD directly to the centered
data matrix. In my digits experiment, ___ principal components were
enough to explain 90% of the variance, out of 64 original dimensions."
```

**[01:30–02:30] W3 준비 + 주간 회고**

```markdown
## W2 회고 (일요일에 작성)

### 달성한 것
- [ ] LU분해·영공간 이해
- [ ] PCA를 SVD로도 검증
- [ ] 손계산 PCA 예제 완료
- [ ] 주성분 개수 결정 (누적 분산 90%)
- [ ] 함수형 스타일 + 예외처리 PCA 파이프라인
- [ ] LeetCode: Valid Palindrome, Reverse Linked List

### 최소 보장 체크
- [ ] PCA를 고유값분해로 손계산 가능
- [ ] SVD와 PCA 차이 설명 가능

### 예상보다 오래 걸린 것
(솔직하게 적기)

### W3에 가져갈 것
(이번 주에 이해 못 하고 넘어간 것)

### 다음 주 첫 번째 할 일
(W3 Day1 무엇부터 시작할지)
```

---

## 막힐 때 대응 가이드

개념이 막히면:

```
1단계 (5분): 구글에 영어로 검색
  예: "why SVD equivalent to eigendecomposition of covariance matrix"

2단계 (10분): 3Blue1Brown 또는 StatQuest 관련 영상 검색

3단계 (20분): MIT 18.06 해당 강의 노트 확인
  https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/

30분 넘어도 해결 안 되면:
→ 메모장에 "아직 모름: [개념]" 적고 다음으로 넘어가기
→ W3 이후에 다시 만날 때 해결
→ 막히는 것은 실력 부족이 아니라 정상 과정
```

코드가 막히면:

```
에러 메시지 전체를 복사 → 구글에 붙여넣기
Stack Overflow 답변 중 가장 많은 추천을 받은 것 선택

자주 나오는 에러:
- LinAlgError: Singular matrix → 데이터에 중복 열/행 있는지 rank 확인
- ModuleNotFoundError: pip install [라이브러리명] (scipy는 Colab 기본 설치)
- Shape mismatch: print(array.shape) 로 차원 확인
```

---

## W2 완료 기준

일요일 저녁에 아래를 할 수 있으면 W2 성공:

```
□ week02/ 폴더에 LU분해·영공간 실습 코드가 있다
□ SVD와 고유값분해로 같은 PCA 결과가 나오는 것을 직접 확인했다
□ 손으로 2x2 PCA 예제를 풀고 numpy로 검증했다
□ 누적 설명 분산 그래프와 재구성 이미지가 week02/ 폴더에 있다
□ PCA 파이프라인이 함수형 스타일 + 커스텀 예외로 정리되어 있다
□ LeetCode Valid Palindrome, Reverse Linked List를 이해하고 혼자 다시 풀 수 있다

절반(3개 이상) 달성하면 W3로 진행.
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
W2의 진짜 목표는 "PCA를 여러 각도(고유값분해·SVD·손계산)에서 보고
같은 결론에 도달하는 경험"을 쌓는 것이에요.*

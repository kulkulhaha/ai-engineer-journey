# W1 구체적 실행 계획

> **주제**: 선형대수 재활성화 + 환경 세팅 + sklearn PCA 실습
>
> **사용 데이터셋 주의**: 이 계획은 sklearn `load_digits()`(8×8 픽셀, 1797개)를 씁니다.
> 이건 "미니 MNIST"이지, 진짜 MNIST(28×28, 7만 개)가 아닙니다.
> 아래에서 편의상 "digits"로 부릅니다. 진짜 MNIST는 나중에 별도로 만납니다.
> **총 목표 시간**: 15–18시간
> **기준**: 평일 2시간 + 주말 각 5–6시간

---

## W1 목표 (이것만 달성하면 성공)

1. **환경**: Colab, GitHub repo, Git 워크플로우 작동
2. **실습**: sklearn PCA로 digits(미니 MNIST) 차원축소 → 시각화 완료
3. **이론**: "PCA가 왜 고유벡터인가" 설명 가능
4. **재구현**: numpy로 행렬 곱·전치·역행렬 직접 구현
5. **최소 보장**: 행렬 곱의 기하학적 의미 (회전·스케일링) 설명 가능

---

## Day 1 (월요일) — 환경 세팅 [2시간]

환경 세팅은 생각보다 오래 걸립니다.
처음 30분이 막히더라도 정상이에요.

### 00:00–00:30 | GitHub + Git 세팅
```
할 일:
1. github.com 접속 → 새 레포 생성
   이름: ai-engineer-journey
   설명: AI Engineer study & project portfolio
   Public 체크, README 체크, .gitignore → Python 선택

2. 로컬 터미널 (또는 Colab 터미널):
   git config --global user.name "Young Eun Kim"
   git config --global user.email "your@email.com"
   git clone https://github.com/YOUR_USERNAME/ai-engineer-journey.git
   cd ai-engineer-journey

3. 첫 커밋:
   echo "# W1: Linear Algebra & PCA" > week01/README.md
   git add .
   git commit -m "W1: initialize week01"
   git push origin main
```
막히면: GitHub Docs "create a repo" 검색 → 5분 이내 해결 안 되면 넘어가고 Colab만 씀

### 00:30–01:30 | Google Colab 세팅 + sklearn PCA 첫 실행

아래 코드를 Colab에 직접 치세요. 복붙 말고 직접.

```python
# W1 Day1: PCA 실습 시작
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 데이터 로드
digits = load_digits()
X = digits.data        # (1797, 64) — 8x8 픽셀 이미지 1797개 (미니 MNIST)
y = digits.target      # 0–9 레이블

print(f"데이터 shape: {X.shape}")
print(f"픽셀 수(차원): {X.shape[1]}")

# 스케일링 (왜 필요한가? — 나중에 이론으로 역추적)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA로 64차원 → 2차원
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print(f"\nPCA 후 shape: {X_pca.shape}")
print(f"설명 분산 비율: {pca.explained_variance_ratio_}")
print(f"두 주성분이 설명하는 분산: {sum(pca.explained_variance_ratio_):.1%}")

# 시각화
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1],
                      c=y, cmap='tab10', alpha=0.6, s=15)
plt.colorbar(scatter, label='숫자 (0–9)')
plt.xlabel('첫 번째 주성분 (PC1)')
plt.ylabel('두 번째 주성분 (PC2)')
plt.title('sklearn digits(미니 MNIST): 64차원 → 2차원 PCA')
plt.tight_layout()
plt.savefig('pca_digits.png', dpi=150)
plt.show()
```

실행 후 스스로에게 물어볼 것:
- 같은 숫자끼리 뭉쳐있나? → 있다면 PCA가 의미 있는 구조를 찾은 것
- 설명 분산이 몇 %인가? → 낮으면 2차원으로는 부족하다는 의미

### 01:30–02:00 | 첫 번째 막힘 → 이론 역추적 시작

> **시간 안내**: Day 1은 환경 세팅이 처음이면 Git·Colab만으로 2시간을 다 쓸 수 있습니다.
> 그래도 정상입니다. 아래 영상 시청은 **오늘 못 하면 Day 2로 넘겨도 되는 선택 항목**이에요.
> 코드가 한 번이라도 돌아갔다면 Day 1은 이미 성공입니다.

위 코드를 돌리면서 반드시 생기는 궁금증:
**"explained_variance_ratio_가 뭔가? 왜 고유벡터 방향으로 투영하는가?"**

이게 생기면 이론 역추적 시작 신호입니다.

```
(시간 남으면) 지금 할 것:
3Blue1Brown YouTube: "Eigenvectors and eigenvalues" 시청 (13분)
URL: https://www.youtube.com/watch?v=PFDu9oVAE-g

시청하면서 메모할 것:
- 고유벡터란 행렬이 적용됐을 때 방향이 바뀌지 않는 벡터
- 고유값이란 그 방향으로 얼마나 늘어나는지
- PCA는 공분산 행렬의 고유벡터 방향으로 데이터를 투영

이해 안 돼도 괜찮아요. 내일 MIT 강의와 연결하면 됩니다.
오늘 시간이 없으면 이 영상은 Day 2 시작 시 봐도 됩니다.
```

---

## Day 2 (화요일) — 이론 역추적 [2시간]

### 00:00–01:00 | MIT 18.06 핵심 강의

```
시청: MIT 18.06 Lecture 1 (Gilbert Strang)
URL: https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/

오늘 집중할 것 (전체 다 볼 필요 없음):
- 행렬 곱의 4가지 해석 방법 (column picture 특히 중요)
- 연립방정식과 행렬의 관계

메모 원칙:
- 이미 아는 것: 체크만
- 잊었던 것: 한 줄로 정리
- 처음 보는 것: 별표 + 설명
```

막히는 지점 예상:
- "열 공간(column space)이 뭔가?" → 그냥 넘어가세요. W2에서 다시 나와요.
- "선형 독립이 왜 중요한가?" → Lecture 1 범위 밖. 역시 넘어가세요.

### 01:00–02:00 | numpy 재구현 (행렬 연산)

```python
# W1 Day2: numpy 행렬 연산 처음부터 구현
import numpy as np

# 1. 행렬 생성
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
print("A =\n", A)
print("B =\n", B)

# 2. 행렬 곱 (@ 연산자)
C = A @ B
print("\nA @ B =\n", C)

# 기하학적 의미 확인:
# 단위벡터 [1,0], [0,1]에 A를 적용하면 어디로 가는가?
v1 = np.array([1, 0])
v2 = np.array([0, 1])
print(f"\nA @ [1,0] = {A @ v1}")  # A의 첫 번째 열
print(f"A @ [0,1] = {A @ v2}")  # A의 두 번째 열

# ✅ 최소 보장 체크: 행렬 곱이 "기저 벡터의 변환"임을 확인했는가?

# 3. 전치 (Transpose)
print("\nA.T =\n", A.T)

# 4. 역행렬
A_inv = np.linalg.inv(A)
print("\nA_inv =\n", A_inv)
print("A @ A_inv =\n", np.round(A @ A_inv, 10))  # 단위행렬이 나와야 함

# 5. 왜 역행렬이 존재하지 않는 경우가 있는가?
# Singular matrix 예시:
C = np.array([[1, 2], [2, 4]])  # 두 번째 행 = 첫 번째 행 × 2
print("\ndet(C) =", np.linalg.det(C))  # 0에 가까운 값
# np.linalg.inv(C) # 이건 에러 남 — 왜? 행렬식이 0이면 역행렬 없음

# 커밋:
# git add week01/day2_matrix_ops.py
# git commit -m "W1 Day2: numpy matrix operations from scratch"
```

---

## Day 3 (수요일) — PCA 수학 연결 [2시간]

### 00:00–01:00 | 공분산 행렬 → 고유값 분해 연결

```python
# W1 Day3: PCA 수학을 numpy로 직접 구현
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler

digits = load_digits()
X = digits.data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 1: 공분산 행렬 계산
# (sklearn이 내부에서 하는 것을 직접 해보기)
n_samples = X_scaled.shape[0]
cov_matrix = (X_scaled.T @ X_scaled) / (n_samples - 1)
print(f"공분산 행렬 shape: {cov_matrix.shape}")  # (64, 64)

# Step 2: 고유값 분해
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

# 고유값 내림차순 정렬
idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

print(f"\n상위 5개 고유값: {eigenvalues[:5].round(2)}")
print(f"상위 2개 고유값이 설명하는 분산: "
      f"{sum(eigenvalues[:2]) / sum(eigenvalues):.1%}")
# 참고: digits 데이터를 표준화하면 약 21.6% (PC1 12.0% + PC2 9.6%)가 나온다.

# Step 3: 직접 투영 (sklearn 없이)
X_pca_manual = X_scaled @ eigenvectors[:, :2]

# Step 4: sklearn 결과와 비교
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_pca_sklearn = pca.fit_transform(X_scaled)

# 부호는 다를 수 있음 (고유벡터 방향은 ±가 자유롭기 때문).
# 절댓값이 일치하면 같은 결과로 본다.
print(f"\n수동 PCA 첫 샘플: {X_pca_manual[0].round(4)}")
print(f"sklearn PCA 첫 샘플: {X_pca_sklearn[0].round(4)}")
print(f"절댓값 일치 여부: "
      f"{np.allclose(np.abs(X_pca_manual[0]), np.abs(X_pca_sklearn[0]))}")
# 참고: 실제로 돌리면 부호만 뒤집힌 값이 나온다. 예)
#   수동    [ 1.9142 -0.9545]
#   sklearn [-1.9142 -0.9545]

# ✅ 최소 보장 체크:
# "PCA = 공분산 행렬의 고유벡터 방향으로 데이터를 투영"
# 위 코드로 직접 확인했는가?
```

### 01:00–02:00 | 3Blue1Brown + 이해 정리

```
시청: 3Blue1Brown "Essence of Linear Algebra" Chapter 13
"Change of basis" (11분)
URL: https://www.youtube.com/watch?v=P2LTAUO1TdA

시청 후 메모장에 적을 것:
1. PCA가 하는 일을 한 문장으로: _______________
2. 고유벡터가 특별한 이유: _______________
3. explained_variance_ratio_의 의미: _______________

이 3문장을 영어로도 적어보세요.
면접에서 그대로 쓸 수 있습니다.
```

---

## Day 4 (목요일) — Python OOP + 커밋 [2시간]

### 00:00–01:00 | Python 심화 (실용 패턴)

```python
# W1 Day4: Python OOP — ML 모델 래퍼 패턴
from dataclasses import dataclass
from typing import Optional
import numpy as np

@dataclass
class PCAConfig:
    n_components: int = 2
    random_state: int = 42

class SimplePCA:
    """
    numpy로 구현한 PCA.
    sklearn 없이 동작함.
    """
    def __init__(self, config: PCAConfig):
        self.config = config
        self.components_: Optional[np.ndarray] = None
        self.explained_variance_ratio_: Optional[np.ndarray] = None
        self._is_fitted: bool = False

    def fit(self, X: np.ndarray) -> "SimplePCA":
        """공분산 행렬의 고유값 분해로 주성분 계산."""
        n_samples = X.shape[0]
        cov = (X.T @ X) / (n_samples - 1)

        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        self.components_ = eigenvectors[:, :self.config.n_components].T
        total_var = eigenvalues.sum()
        self.explained_variance_ratio_ = (
            eigenvalues[:self.config.n_components] / total_var
        )
        self._is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("fit() 먼저 호출하세요.")
        return X @ self.components_.T

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def __repr__(self) -> str:
        return (f"SimplePCA(n_components={self.config.n_components}, "
                f"fitted={self._is_fitted})")


# 사용 + 검증
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler

digits = load_digits()
X_scaled = StandardScaler().fit_transform(digits.data)

config = PCAConfig(n_components=2)
my_pca = SimplePCA(config)
X_reduced = my_pca.fit_transform(X_scaled)

print(my_pca)
print(f"Shape: {X_reduced.shape}")
print(f"Explained variance: {my_pca.explained_variance_ratio_}")
```

### 01:00–02:00 | LeetCode + 커밋

```python
# LeetCode #1: Two Sum
# 시간제한: 20분. 풀리면 좋고, 안 풀려도 풀이 보고 이해하면 됨.

def twoSum(nums: list[int], target: int) -> list[int]:
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []

# 테스트
assert twoSum([2, 7, 11, 15], 9) == [0, 1]
assert twoSum([3, 2, 4], 6) == [1, 2]
print("Two Sum 통과")
```

```bash
# 오늘 작업 커밋
git add week01/
git commit -m "W1 Day4: SimplePCA class + Two Sum"
git push
```

---

## Day 5 (금요일) — 복습 + 논문 [2시간]

### 00:00–00:30 | W1 수학 복습

스스로에게 물어볼 것 (답이 안 나오면 해당 자료 다시 보기):

```
□ 행렬 곱 A @ B의 기하학적 의미는?
  → B 변환을 먼저 적용하고 A 변환을 적용하는 것

□ 역행렬이 존재하지 않으면 어떻게 되는가?
  → 행렬식(det) = 0, 선형 종속, 연립방정식에 해 없거나 무한히 많음

□ PCA의 수학적 과정을 순서대로?
  → 스케일링 → 공분산 행렬 → 고유값분해 → 상위 k 고유벡터로 투영

□ explained_variance_ratio_ 0.3이라는 것은?
  → 그 주성분 하나가 전체 데이터 분산의 30%를 설명함
```

### 00:30–01:30 | Turing 논문 맥락 파악 (20분으로 충분)

```
읽을 것: Turing (1950) "Computing Machinery and Intelligence"
- 전체 읽을 필요 없음
- 읽을 부분: Section 1 (The Imitation Game), Section 6 (Objections)
- 시간: 20분

읽으면서 메모:
1. Turing Test가 "지능"을 어떻게 정의하는가?
2. 왜 이 논문이 AI의 시작으로 불리는가?
3. 2026년 LLM 관점에서 Turing Test를 통과했다고 볼 수 있는가?

마지막 질문은 면접에서 "AI 철학"으로 나올 수 있는 단골 질문입니다.
```

### 01:30–02:00 | 금요일 마무리 커밋 + 다음 주 준비

```bash
# 이번 주 정리 커밋
git add .
git commit -m "W1 완료: PCA 재활성화, numpy 재구현, SimplePCA 클래스"
git push

# README 업데이트
cat >> week01/README.md << 'EOF'

## W1 완료 항목
- [x] sklearn PCA로 digits(미니 MNIST) 차원축소 + 시각화
- [x] PCA 수학: 공분산 → 고유값분해 → 투영
- [x] numpy: 행렬 곱·전치·역행렬
- [x] SimplePCA 클래스 직접 구현
- [x] Git 워크플로우 정착
- [x] LeetCode: Two Sum

## 최소 보장 체크
- [x] 행렬 곱의 기하학적 의미 설명 가능
- [ ] PCA를 고유값분해로 손계산 (W2에서 완성)
EOF
```

---

## 주말 — 심화 [5–6시간]

주말은 평일에 이해가 덜 된 부분을 채우고
다음 주를 준비하는 시간입니다.

### 토요일 [3시간]

**[00:00–01:30] 고유값·고유벡터 깊이 이해**

```python
# 고유값 분해를 손으로 따라가기
import numpy as np

A = np.array([[3, 1],
              [0, 2]], dtype=float)

# numpy로 고유값·고유벡터 계산
eigenvalues, eigenvectors = np.linalg.eig(A)
print("고유값:", eigenvalues)       # [3, 2]
print("고유벡터:\n", eigenvectors)

# 검증: A @ v = λ @ v
for i in range(len(eigenvalues)):
    v = eigenvectors[:, i]
    lam = eigenvalues[i]
    lhs = A @ v
    rhs = lam * v
    print(f"\nλ={lam:.1f}: A@v={lhs.round(4)}, λ*v={rhs.round(4)}")
    print(f"일치: {np.allclose(lhs, rhs)}")

# 시각화: 고유벡터 방향
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 일반 벡터는 방향이 바뀜
test_vectors = np.array([[1, 0], [0, 1],
                          [1, 1], [-1, 1]])

for ax, title, transform in [
    (axes[0], "변환 전", np.eye(2)),
    (axes[1], "A 변환 후", A)
]:
    for v in test_vectors:
        tv = transform @ v
        ax.arrow(0, 0, tv[0], tv[1],
                 head_width=0.1, head_length=0.1,
                 fc='gray', ec='gray', alpha=0.5)
    # 고유벡터는 방향 유지
    for i, color in enumerate(['red', 'blue']):
        ev = eigenvectors[:, i]
        tev = transform @ ev
        ax.arrow(0, 0, tev[0], tev[1],
                 head_width=0.12, head_length=0.12,
                 fc=color, ec=color, linewidth=2,
                 label=f'λ={eigenvalues[i]:.1f}')
    ax.set_xlim(-3, 4); ax.set_ylim(-2, 4)
    ax.axhline(0, color='k', linewidth=0.5)
    ax.axvline(0, color='k', linewidth=0.5)
    ax.set_title(title); ax.legend()
    ax.set_aspect('equal')

plt.tight_layout()
plt.savefig('week01/eigenvectors_visualization.png', dpi=150)
plt.show()
```

**[01:30–03:00] MIT 18.06 Lecture 6 보충 (필요 시)**

Day 2에서 잘 이해됐으면 스킵해도 됩니다.
막혔던 개념(열 공간, 선형 독립)이 있으면 지금 여기서 해결하세요.
30분 이상 한 개념에 막히면 → 넘어가고 W2에서 다시 만날 때 해결.

### 일요일 [2–3시간]

**[00:00–01:30] W1 최종 재구현 + 영어 설명 연습**

```python
# W1 토탈 리뷰: 처음부터 끝까지 혼자서 다시
# 아무것도 보지 않고 아래를 구현할 수 있는가?

import numpy as np
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler

# 1. 데이터 로드 + 스케일링 (5분)
# 2. 공분산 행렬 계산 (5분)
# 3. 고유값 분해 (5분)
# 4. 상위 2개 고유벡터로 투영 (5분)
# 5. sklearn PCA와 결과 비교 (5분)

# 막히면 Day 3 코드 참고 가능. 중요한 건 흐름을 기억하는 것.
```

영어로 말해보기 (혼자서 소리 내어):

아래에서 `___%` 부분은 반드시 **직접 코드를 돌려 나온 값**으로 채우세요.
digits를 표준화하면 대략 21~22%가 나오지만, 스케일링 방식에 따라
조금 달라질 수 있으니 외운 숫자를 말하지 말고 본인이 확인한 값을 씁니다.

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

**[01:30–02:30] W2 준비 + 주간 회고**

```markdown
## W1 회고 (일요일에 작성)

### 달성한 것
- [ ] sklearn PCA digits(미니 MNIST) 실습
- [ ] PCA 수학 (공분산 → 고유값분해 → 투영) 이해
- [ ] numpy 행렬 연산 재구현
- [ ] SimplePCA 클래스 작성
- [ ] LeetCode Two Sum

### 최소 보장 체크
- [ ] 행렬 곱의 기하학적 의미 설명 가능
- [ ] "PCA가 왜 고유벡터인가" 영어로 설명 가능

### 예상보다 오래 걸린 것
(솔직하게 적기)

### W2에 가져갈 것
(이번 주에 이해 못 하고 넘어간 것)

### 다음 주 첫 번째 할 일
(W2 Day1 무엇부터 시작할지)
```

---

## 막힐 때 대응 가이드

개념이 막히면:

```
1단계 (5분): 구글에 영어로 검색
  예: "why PCA uses eigenvectors covariance matrix"

2단계 (10분): 3Blue1Brown 관련 영상 검색

3단계 (20분): MIT 18.06 해당 강의 노트 확인
  https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/

30분 넘어도 해결 안 되면:
→ 메모장에 "아직 모름: [개념]" 적고 다음으로 넘어가기
→ W2 또는 W3에서 다시 만날 때 해결
→ AI 커리큘럼에서 막히는 것은 실력 부족이 아니라 정상 과정
```

코드가 막히면:

```
에러 메시지 전체를 복사 → 구글에 붙여넣기
Stack Overflow 답변 중 가장 많은 추천을 받은 것 선택

자주 나오는 에러:
- ModuleNotFoundError: pip install [라이브러리명]
- Shape mismatch: print(array.shape) 로 차원 확인
- CUDA/GPU 에러: Colab 런타임 → GPU로 변경
```

---

## W1 완료 기준

일요일 저녁에 아래를 할 수 있으면 W1 성공:

```
□ github.com/YOUR_USERNAME/ai-engineer-journey 에 코드가 올라가 있다
□ digits(미니 MNIST) PCA 시각화 이미지가 week01/ 폴더에 있다
□ numpy로 행렬 곱·전치·역행렬을 직접 구현한 코드가 있다
□ SimplePCA 클래스가 작동한다
□ "PCA가 왜 고유벡터인가"를 3문장으로 설명할 수 있다 (한국어 가능)
□ LeetCode Two Sum을 이해하고 혼자 다시 풀 수 있다

절반(3개 이상) 달성하면 W2로 진행.
전부 못 해도 W2로 진행 — 이해 못 한 부분은 W2에서 다시 나옴.
```

---

## W2 첫 할 일 미리 보기

W2 Day1에 열어야 할 것:
1. `week02/` 폴더 생성
2. sklearn으로 로지스틱 회귀 돌리기 (W1 PCA와 같은 패턴)
3. 막히면 → "왜 로그우도인가?" → 확률 이론 역추적

---

*이 계획대로 완벽하게 안 돼도 됩니다.
W1의 진짜 목표는 "하는 습관을 만드는 것"이에요.
코드를 짜고 GitHub에 커밋하는 것 자체가 W1의 절반입니다.*

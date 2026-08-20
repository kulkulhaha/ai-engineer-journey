# W2: LU decomposition, null space, PCA via SVD

## 진행 상황 (2026-08-20 기준 — 진행 중)

**완료:**
- Day1: LU분해·영공간 탐구 (`Null_space.ipynb`) — `scipy.linalg.lu`로 LU분해(P@L@U=A) 검증, 선형종속 행을 가진 3x3 행렬로 영공간 확인 (rank=2, nullity=1, `B @ v ≈ 0`)
- Day2: SVD와 고유값분해 동치 검증 (`SVD_vs_PCA.ipynb`) — 공분산 고유값 × (n-1) = 데이터 행렬 특이값제곱임을 `np.allclose`로 직접 검증
- Day3 AM: 누적 설명 분산 기반 주성분 개수 결정 (`cumulative_explained_variance.ipynb`) — `find_k_for_variance`, scree plot으로 threshold 기반 k 선택
- Day3 PM: PCA 재구성 오차 검증 (`Reconstruction.ipynb`) — k=40 재구성 MSE(0.0469)가 버려진 분산 비율(0.0492)과 거의 일치함을 확인. 완전히 일치하지 않는 원인도 추적 — `load_digits()` 모서리 픽셀 3개가 분산 0이라 `StandardScaler`가 스케일링을 못 해, 실제 총분산이 가정했던 64가 아니라 ~61.03이었음

**아직 안 한 것:**
- Day2: 2x2 행렬 손계산 PCA (종이 풀이 + numpy 검증)
- Day4: comprehension/함수형 파이프라인 + `SimplePCA` 커스텀 예외처리
- Day5: W2 자가 점검 + 논문 리뷰(Rumelhart et al. 1986, 역전파) + 주간 회고

**최소 보장 체크:**
- [x] SVD와 PCA 차이를 행렬 분해 관점에서 설명 가능 (공분산 고유값분해 ↔ 데이터 행렬 SVD, `λ*(n-1)=σ²`)
- [ ] PCA를 고유값분해로 손계산 가능 (Day2 손계산 미완)

**다음 액션:** Day4 함수형 파이프라인/예외처리 → Day5 복습 및 논문 리뷰

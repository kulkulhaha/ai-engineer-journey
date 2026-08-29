# W4 구체적 실행 계획 (v2 — 학습 가이드형)

> **이 개정의 근거**: 마스터 커리큘럼 `ai_curriculum_v7.md`에서 블록 A 주당 시간이 15–18h → 10–12h로 완화되었고, w1–w3는 이미 v2(학습 가이드형 — 완성 코드 대신 스펙과 확인 질문 제시)로 개정되었습니다. w4는 실제 착수 시점(2026-08-27 기준 W3까지 완료, W4 착수 직전)까지 v1(15–16h, 완성 코드 제공형)에 머물러 있었던 것을 이번 격주 점검에서 발견해 개정했습니다. 주제·논문·알고리즘 구성은 v1과 동일하며, **시간 배분과 학습 방식(스스로 구현)만** 바뀝니다.
>
> **주제**: LLM API 활용 + 미니프로젝트 #1(텍스트 요약·분류기 CLI) + SVD 이론 역추적 + Git 브랜치·PR 워크플로우
>
> **사용 데이터 주의**: 이 계획은 공개 뉴스 RSS 피드(예: BBC News RSS)를 씁니다. RSS 기사 내용은 매일 바뀌므로, 실행 결과(요약문·카테고리)는 실행 시점에 따라 달라집니다. 숫자나 문장을 외워서 쓰지 말고, 본인이 직접 돌려서 나온 결과를 근거로 삼으세요.
>
> **총 목표 시간**: 10–12시간 (v7 커리큘럼 블록 A 기준)
> **기준**: 평일 1.5시간 + 토요일 2.5시간 + 일요일 2시간
>
> **v1과의 차이**: 이 버전은 완성된 실행 코드를 주지 않습니다. 대신 학습 목표·핵심 개념·구현 요구사항(스펙)·확인 질문을 제시합니다. 코드는 스스로 작성하는 것이 이번 주의 진짜 과제입니다. 환경 설정(API 키 발급, git 명령어)처럼 학습 내용과 무관한 명령어만 그대로 제시합니다. 막히면 Day별 "힌트"를 열어보세요.

---

## W4 목표 (이것만 달성하면 성공)

1. **실습**: LLM API(OpenAI 또는 Anthropic)로 텍스트 요약 + 카테고리 분류를 스스로 구현해 첫 호출 성공
2. **미니프로젝트 #1**: 뉴스 RSS → 요약 + 분류 CLI 완성, GitHub에 README와 함께 커밋
3. **도구**: Git branch·PR 워크플로우를 실제로 한 번 사용 (feature 브랜치 → PR → merge)
4. **이론**: SVD와 PCA의 관계를 행렬 분해 관점에서 스스로 코드로 검증하고 설명 가능
5. **최소 보장**: SVD와 PCA 차이를 설명 가능 (W1 PCA와 연결)

---

## Day 1 (월요일) — LLM API 첫 호출 [1.5시간]

W1–3에서 sklearn으로 만들던 모델을, 이번엔 LLM API 호출로 바꿔 봅니다. 패턴은 같습니다: 실습 먼저, 막히면 이론.

### 00:00–00:20 | API 키 발급 + 환경 세팅 (그대로 실행)

```
할 일:
1. OpenAI(platform.openai.com) 또는 Anthropic(console.anthropic.com)에서
   API 키 발급 (둘 중 하나만 있어도 충분)
2. 로컬에 .env 파일 생성 (커밋 금지 — .gitignore에 이미 있는지 확인):
   ANTHROPIC_API_KEY=sk-ant-...
   (또는 OPENAI_API_KEY=sk-...)
3. 라이브러리 설치:
   pip install anthropic python-dotenv feedparser
   (OpenAI를 쓴다면: pip install openai python-dotenv feedparser)
```

막히면: `.env`가 git에 올라가지 않는지 `git status`로 반드시 확인 (API 키 유출 방지가 최우선).

### 00:20–01:00 | 첫 API 호출 — 요약 + 분류 (스스로 구현)

- **학습 목표**: LLM에게 "형식이 고정된" 응답을 받아내는 프롬프트를 설계하고, 파싱 가능한 출력을 얻을 수 있다.
- **핵심 개념**: LLM은 자유 형식으로 답하면 후처리가 불안정해집니다. "요약: .../카테고리: ..."처럼 접두어를 고정하면 문자열 파싱이 쉬워집니다 — 이건 W3의 "test 데이터에 fit하면 안 되는 이유"처럼, 나중에 자동화(CLI·배치 처리)를 염두에 둔 설계 선택입니다.
- **구현 과제 (스스로 작성)**:
  ```python
  def summarize_and_classify(client, title: str, text: str, categories: list[str]) -> dict:
      """
      요구사항:
      - client.messages.create()로 Claude(또는 OpenAI) API를 호출한다.
      - 프롬프트는 "요약: <2문장 요약>" / "카테고리: <categories 중 하나>" 형식으로만
        답하도록 명시적으로 지시한다.
      - 응답 문자열에서 "요약:"/"카테고리:" 줄을 찾아 파싱한다.
      - 반환값: {"title": ..., "summary": ..., "category": ...}
      - 파싱 실패 시(접두어를 못 찾으면) category는 "Other"로 fallback한다.
      """
      ...
  ```
- **확인 질문**: 임의의 짧은 기사(예: EV 배터리 기술 관련 2~3문장)로 함수를 호출했을 때 — 요약이 원문 핵심을 담고 있는가? 카테고리가 합리적인가?

막히면: `AuthenticationError` → API 키 오타 또는 `.env` 로드 실패. `print(os.getenv("ANTHROPIC_API_KEY"))`로 값이 읽히는지 먼저 확인. 프롬프트 설계 자체가 막히면 W3의 프롬프트 없이 "직접 형식을 요구하는 문장"부터 단순하게 시작.

### 01:00–01:20 | Git 브랜치 워크플로우 시작 (그대로 실행)

```bash
# W4 Day1: feature 브랜치로 미니프로젝트 #1 시작
cd ai-engineer-journey
git checkout -b feature/mini-project-1-llm-classifier
mkdir -p week04
echo "# W4: LLM Text Summarizer & Classifier CLI" > week04/README.md

git add week04/
git commit -m "W4 Day1: first LLM API call (summarize + classify)"
git push -u origin feature/mini-project-1-llm-classifier
```

막히면: "왜 브랜치를 나누는가?" → main을 항상 배포 가능한 상태로 유지하고, 작업 중인 코드는 격리하기 위함. 이번 주는 이 브랜치에서 계속 작업하고, 금요일에 PR을 열어 merge합니다.

---

## Day 2 (화요일) — RSS 피드 연동 [1.5시간]

### 00:00–00:30 | feedparser로 뉴스 가져오기 (스스로 구현)

- **학습 목표**: RSS 피드를 파싱해 실제 기사 목록(제목·요약·링크)을 구조화된 형태로 받아올 수 있다.
- **구현 과제**:
  ```python
  def fetch_articles(rss_url: str, limit: int = 5) -> list[dict]:
      """
      요구사항:
      - feedparser.parse(rss_url)로 피드를 가져온다.
      - feed.bozo가 True면 (파싱 에러) 경고를 출력한다.
      - entries 중 상위 limit개에서 title, summary(없으면 빈 문자열), link를 추출한다.
      - 반환값: [{"title": ..., "summary": ..., "link": ...}, ...]
      """
      ...
  ```
  기본 URL 예시: `http://feeds.bbci.co.uk/news/rss.xml`
- **확인 질문**: 가져온 기사 수와 피드 제목(`feed.feed.title`)을 출력했을 때 예상과 맞는가?

막히면: RSS 응답이 비어있으면 → URL이 바뀌었을 수 있음. `feed.bozo`가 True면 다른 공개 RSS URL(각 언론사 홈페이지의 "RSS" 링크)로 교체.

### 00:30–01:20 | 요약+분류 함수를 실제 기사에 배치 적용 (스스로 구현)

- **학습 목표**: Day1의 단일 호출 함수를 여러 기사에 반복 적용하는 배치 파이프라인으로 확장하고, API rate limit을 고려한다.
- **구현 과제**:
  ```python
  def process_feed(client, rss_url: str, categories: list[str], limit: int = 5) -> list[dict]:
      """
      요구사항:
      - fetch_articles()로 기사를 가져온다.
      - 각 기사에 대해 summarize_and_classify()를 호출한다.
      - 각 호출 사이에 time.sleep(1) 등으로 rate limit 여유를 둔다.
      - 처리하면서 "[카테고리] 제목 → 요약"을 즉시 출력한다(진행 상황 확인용).
      - 반환값: 각 기사의 결과 dict 리스트.
      """
      ...
  ```
- **확인 질문**: 5개 기사 처리 결과 중 카테고리 분포가 합리적인가(전부 "Other"로 쏠리지 않는가)?

커밋:
```bash
git add week04/
git commit -m "W4 Day2: RSS feed integration + batch summarize/classify"
git push
```

---

## Day 3 (수요일) — SVD 이론 역추적 [1.5시간]

### 00:00–00:40 | "SVD가 PCA와 뭐가 다른가?" — 역추적

W1에서 PCA를 공분산 행렬의 고유값분해로 구현했습니다. 오늘은 같은 결과를 SVD로도 얻을 수 있음을 확인합니다.

```
막히는 질문: "sklearn PCA는 내부적으로 고유값분해(eigh)를 쓰는가, SVD를 쓰는가?"
→ 실제로는 SVD를 씁니다. 왜 고유값분해 대신 SVD를 선호하는지가 오늘의 핵심.

읽을 것: 3Blue1Brown이나 StatQuest에서 "SVD" 검색 (10–15분 영상)
핵심 개념:
- SVD: X = U Σ V^T  (모든 행렬에 적용 가능, 정사각행렬이 아니어도 됨)
- 공분산 행렬의 고유값분해: X^T X의 고유벡터 = SVD의 V (오른쪽 특이벡터)
- SVD는 공분산 행렬(X^T X)을 명시적으로 계산하지 않아도 되어 수치적으로 더 안정적
  (X^T X를 계산하면 오차가 제곱으로 커짐 — "condition number가 제곱"된다는 표현)
```

> W2 학습 노트에서 이미 이 관계(σ_i = sqrt(λ_i(A^TA)))를 SVD 전개로 증명한 적이 있습니다. 오늘은 그 증명을 코드로 직접 재현합니다.

### 00:40–01:20 | numpy로 SVD 직접 계산 → W1 PCA와 비교 (스스로 구현)

- **학습 목표**: 공분산 행렬의 고유값분해(W1 방식)와 데이터 행렬의 SVD가 같은 주성분을 준다는 것을 수치로 검증한다.
- **구현 과제**:
  ```python
  def compare_pca_eig_vs_svd(X_scaled: np.ndarray, n_components: int = 2) -> dict:
      """
      요구사항:
      - 방법 1: 공분산 행렬 (X^T X)/(n-1)의 고유값분해(np.linalg.eigh)로
        상위 n_components개 주성분 투영 결과를 구한다.
      - 방법 2: X_scaled에 np.linalg.svd(full_matrices=False)를 직접 적용해
        Vt의 상위 n_components개 행으로 투영 결과를 구한다.
      - 특이값(S)으로부터 고유값을 역산한다: eigenvalue = S**2 / (n_samples - 1)
      - 두 방법의 고유값이 np.allclose로 일치하는지 확인한다.
      - 반환값: {"eigenvalues_eig": ..., "eigenvalues_from_svd": ..., "match": bool}
      """
      ...
  ```
  데이터는 W1과 동일하게 `load_digits()` + `StandardScaler`를 재사용하세요.
- **✅ 최소 보장 체크**: "SVD의 V(오른쪽 특이벡터) = 공분산 행렬의 고유벡터"와 "특이값의 제곱 / (n-1) = 고유값" 이 두 관계를 위 코드로 직접 확인했는가? (부호는 다를 수 있음 — `np.abs`로 비교)

커밋:
```bash
git add week04/
git commit -m "W4 Day3: SVD vs eigendecomposition comparison (PCA)"
git push
```

---

## Day 4 (목요일) — CLI 완성 [1.5시간]

### 00:00–00:50 | argparse로 CLI 골격 만들기 (스스로 구현)

- **학습 목표**: Day1–2에서 만든 함수들을 명령줄 인자를 받는 재사용 가능한 CLI 도구로 통합한다.
- **구현 과제 (`week04/news_classifier_cli.py`)**:
  ```python
  """
  W4 미니프로젝트 #1: 뉴스 RSS 요약·분류 CLI

  사용법:
      python news_classifier_cli.py --url http://feeds.bbci.co.uk/news/rss.xml --limit 5
  """
  # 요구사항:
  # - argparse로 --url(기본값: BBC RSS)과 --limit(기본값: 5) 인자를 받는다.
  # - .env에서 API 키를 로드하고, 없으면 명확한 에러 메시지와 함께 sys.exit(1).
  # - Day1의 summarize_and_classify()를 API 에러(anthropic.APIError 등)에 대해
  #   방어적으로 감싼다 — 실패해도 전체 배치가 중단되지 않고
  #   {"category": "Other", "summary": "(API 오류: ...)"} 같은 형태로 계속 진행.
  # - feed.bozo가 True면 stderr에 경고를 출력하되 계속 진행한다.
  # - __main__ 블록에서 실행 가능하게 만든다.
  ```
- **확인 질문**: `python news_classifier_cli.py --limit 3`을 실행하면 에러 없이 3개 기사에 대한 결과가 출력되는가? API 키를 일부러 잘못 넣으면 에러 메시지가 사용자 친화적으로 나오는가?

막히면:
- `ModuleNotFoundError` → `pip install anthropic feedparser python-dotenv`
- Rate limit 에러 → `time.sleep()` 값을 2–3초로 늘리기

### 00:50–01:20 | README 작성 + PR 워크플로우 완성

README는 직접 작성하세요(무엇을/왜/어떻게 사용하는지 3~5줄이면 충분). 아래는 최소 골격만 참고용으로 제시합니다.

```
포함할 내용:
- 프로젝트 한 줄 설명
- 사용법 (pip install -r requirements.txt / API 키 설정 / 실행 명령)
- 배운 것 2~3개 (예: 프롬프트 형식 고정, feedparser, SVD-PCA 관계)
```

```bash
git add week04/
git commit -m "W4 Day4: CLI complete with argparse + README"
git push
```

```
GitHub에서 PR 열기:
1. github.com/YOUR_USERNAME/ai-engineer-journey 접속
2. "Compare & pull request" 버튼 클릭 (feature/mini-project-1-llm-classifier → main)
3. 제목: "W4: LLM 텍스트 요약·분류기 CLI"
4. 설명에 "무엇을/왜/어떻게 테스트했는지" 3줄 작성
5. Merge pull request 클릭 (본인 레포이므로 리뷰어 없이 바로 merge 가능)
6. 로컬에서: git checkout main && git pull
```

막히면: PR이 처음이면 GitHub Docs "Creating a pull request" 검색 → 5분 내 해결 안 되면 그냥 로컬에서 `git checkout main && git merge feature/mini-project-1-llm-classifier`로 대체. 이번 주 목표는 "브랜치 개념을 한 번 써보는 것"이지 완벽한 워크플로우가 아님.

---

## Day 5 (금요일) — 복습 + 논문 [1.5시간]

### 00:00–00:20 | W4 이론 복습

스스로에게 물어볼 것 (답이 안 나오면 해당 자료 다시 보기):

```
□ SVD (X = U Σ V^T)에서 V가 의미하는 것은?
□ 특이값(singular value)과 고유값(eigenvalue)의 관계는?
□ 왜 sklearn PCA는 공분산 행렬을 직접 만들지 않고 SVD를 쓰는가?
□ 이번 주 LLM API 프롬프트에서 "형식을 고정"한 이유는?
```

### 00:20–01:00 | AlexNet 논문 맥락 파악

```
읽을 것: Krizhevsky, Sutskever, Hinton (2012) "ImageNet Classification with Deep CNNs" (AlexNet)
- 전체 읽을 필요 없음
- 읽을 부분: Abstract, Section 1(Introduction), Section 3.1–3.4 (ReLU, GPU, 구조 요약)
- 시간: 20–25분

읽으면서 메모:
1. AlexNet 이전 이미지 분류는 왜 어려웠는가? (수작업 특징 추출의 한계)
2. 왜 ReLU를 썼는가? (기존 sigmoid/tanh 대비 학습 속도)
3. 왜 GPU 2개로 나눠 학습했는가? (당시 GPU 메모리 제약)
4. 이 논문이 "딥러닝 붐의 시작"으로 불리는 이유를 한 문장으로

이 4개 답변 중 2, 4번은 면접에서 "딥러닝 역사"로 자주 나오는 단골 질문입니다.
```

### 01:00–01:20 | 금요일 마무리 커밋 + README 완료 체크

```bash
git add .
git commit -m "W4 완료: LLM API CLI, SVD-PCA 관계, Git PR 워크플로우"
git push
```

README에 아래 체크리스트를 추가하세요(직접 작성):
```
## W4 완료 항목
- [ ] LLM API 첫 호출 — 요약 + 분류
- [ ] RSS 피드 연동 + 배치 처리
- [ ] SVD와 PCA 관계를 numpy로 직접 확인
- [ ] argparse CLI 완성 + README
- [ ] Git 브랜치 → PR → merge 워크플로우 1회 완주

## 최소 보장 체크
- [ ] SVD와 PCA 차이를 행렬 분해 관점에서 설명 가능
```

---

## 주말 — 심화 [4.5시간]

### 토요일 [2.5시간]

**[00:00–01:15] CLI 확장 — 에러 처리와 배치 안정성 강화 (스스로 구현)**

- **학습 목표**: 일시적 API 실패에 강건한 재시도 로직과 결과 영속화(JSON 저장)를 구현한다.
- **구현 과제**:
  ```python
  def summarize_and_classify_with_retry(client, title: str, text: str, max_retries: int = 3) -> dict:
      """
      요구사항:
      - Day1의 API 호출을 최대 max_retries회 재시도한다.
      - 실패할 때마다 대기 시간을 지수적으로 늘린다 (exponential backoff: 2**attempt초).
      - 마지막 시도까지 실패하면 에러 메시지를 담은 결과를 반환한다(예외를 전파하지 않는다).
      """
      ...

  def save_results(results: list[dict], path: str = "week04/results.json"):
      """
      요구사항:
      - 생성 시각(UTC, ISO 형식)과 results 리스트를 JSON으로 저장한다.
      - ensure_ascii=False로 한글이 깨지지 않게 한다.
      """
      ...
  ```
- **확인 질문**: exponential backoff(2\*\*attempt)가 왜 고정된 `sleep(1)`보다 나은가? (API가 일시적으로 과부하일 때 서버 부담을 줄이고 성공 확률을 높인다는 것을 스스로 설명해보세요.)

**[01:15–02:30] SVD 시각화 — 특이값이 클수록 정보가 많다는 것을 눈으로 확인 (스스로 구현)**

- **학습 목표**: 상위 k개 특이값만으로 이미지를 재구성해, "특이값이 큰 순서 = 중요한 정보 순서"라는 직관을 시각적으로 검증한다.
- **구현 과제**:
  ```python
  def svd_image_reconstruction(img: np.ndarray, k_values: list[int] = [1, 2, 4, 8]):
      """
      요구사항:
      - digits.images[0] 같은 8x8 이미지 1장에 np.linalg.svd를 적용한다.
      - 각 k에 대해 U[:, :k] @ diag(S[:k]) @ Vt[:k, :]로 재구성한다.
      - 각 재구성이 설명하는 분산 비율((S[:k]**2).sum() / (S**2).sum())을 계산한다.
      - matplotlib으로 k별 재구성 이미지를 나란히 그려 week04/svd_image_compression.png로 저장한다.
      """
      ...
  ```
- **확인 질문**: k가 커질수록 원본에 가까워지는가? 어느 k부터 육안으로 원본과 구별이 어려운가?

### 일요일 [2시간]

**[00:00–01:00] W4 최종 재구현 + 영어 설명 연습**

```
아무것도 보지 않고 아래를 구현할 수 있는가? (막히면 Day3·Day4 코드 참고 가능)

1. .env에서 API 키 로드 + 클라이언트 생성
2. RSS 피드 파싱 + 기사 3개 추출
3. 요약+분류 프롬프트 작성 + API 호출
4. 결과 파싱(요약:/카테고리: 접두어)
5. numpy SVD로 PCA 재현 + 고유값분해 결과와 비교

중요한 건 "SVD ↔ 고유값분해" 연결과 "API 호출 → 파싱 → 저장" 흐름을 기억하는 것.
```

영어로 말해보기 (혼자서 소리 내어) — `___` 부분은 반드시 **직접 코드를 돌려 나온 값**으로 채우세요:

```
"Singular Value Decomposition factorizes any matrix X into U, Sigma, and V transpose.
For PCA, the right singular vectors V are the same as the eigenvectors
of the covariance matrix, and the eigenvalues equal the squared singular
values divided by (n minus 1). sklearn's PCA implementation uses SVD
directly on the data matrix rather than eigendecomposition on the covariance
matrix, because computing X transpose X first would square the numerical
error. When I compressed an 8x8 digit image using the top ___ singular
values (fill in the k you tested), it already explained ___% of the variance
(fill in your measured value) while looking visually close to the original."
```

**[01:00–02:00] W5 준비 + 주간 회고**

```markdown
## W4 회고 (일요일에 작성)

### 달성한 것
- [ ] LLM API 첫 호출 — 요약 + 분류
- [ ] RSS 피드 연동 + 배치 처리 CLI 완성
- [ ] SVD와 PCA 관계를 numpy로 직접 확인
- [ ] Git 브랜치 → PR → merge 워크플로우 1회 완주
- [ ] SVD 이미지 압축 시각화

### 최소 보장 체크
- [ ] SVD와 PCA 차이를 행렬 분해 관점에서 설명 가능

### 예상보다 오래 걸린 것
(솔직하게 적기)

### W5에 가져갈 것
(이번 주에 이해 못 하고 넘어간 것)

### 다음 주 첫 번째 할 일
(W5 Day1 무엇부터 시작할지)
```

---

## 막힐 때 대응 가이드

개념이 막히면:

```
1단계 (5분): 구글에 영어로 검색
  예: "why does sklearn PCA use SVD instead of eigendecomposition"

2단계 (10분): 3Blue1Brown 또는 StatQuest 관련 영상 검색

3단계 (20분): MIT 18.06 해당 강의 노트 확인
  https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/

30분 넘어도 해결 안 되면:
→ 메모장에 "아직 모름: [개념]" 적고 다음으로 넘어가기
→ W5 또는 이후 주차에서 다시 만날 때 해결
→ AI 커리큘럼에서 막히는 것은 실력 부족이 아니라 정상 과정
```

코드가 막히면:

```
에러 메시지 전체를 복사 → 구글에 붙여넣기
Stack Overflow 답변 중 가장 많은 추천을 받은 것 선택

자주 나오는 에러:
- ModuleNotFoundError: pip install [라이브러리명]
- AuthenticationError: API 키 오타 또는 .env 로드 실패 확인
- feed.bozo == True: RSS URL이 바뀌었거나 파싱 실패 — 다른 URL로 교체
- Rate limit 에러: time.sleep()을 2–3초로 늘리기
- git push 거부(rejected): git pull --rebase 후 다시 push
```

---

## W4 완료 기준

일요일 저녁에 아래를 할 수 있으면 W4 성공:

```
□ github.com/YOUR_USERNAME/ai-engineer-journey 의 week04/ 폴더에 코드가 올라가 있다
□ news_classifier_cli.py가 실제 RSS 피드에 대해 동작한다
□ numpy로 SVD를 계산해 W1의 고유값분해 PCA와 결과가 일치함을 확인한 코드가 있다
□ Git 브랜치 생성 → 커밋 → PR → merge 흐름을 최소 1회 완주했다
□ "SVD와 PCA의 관계"를 3문장으로 설명할 수 있다 (한국어 가능)
□ AlexNet 논문의 핵심 아이디어(ReLU, GPU 병렬화)를 한 문장씩 설명할 수 있다

절반(3개 이상) 달성하면 W5로 진행.
전부 못 해도 W5로 진행 — 이해 못 한 부분은 이후 주차에서 다시 나옴.
```

---

## W5 첫 할 일 미리 보기

W5 Day1에 열어야 할 것:

1. `week05/` 폴더 생성
2. sklearn `RandomForestClassifier`와 `XGBClassifier`를 같은 데이터로 비교 실험
3. 막히면 → "정보이득이 왜 엔트로피인가?" → Harvard Stat 110 Lec 7–13으로 역추적
4. matplotlib으로 분포 시각화 + CLT(중심극한정리) 시뮬레이션 준비

> ℹ️ **참고**: w5 이후 실행계획 파일들은 이번 점검(2026-08-27) 시점에는 아직 v1(15–16h, 완성 코드 제공형)에 머물러 있습니다. W4와 동일한 이유로 개정이 필요하지만, 각 주차 고유 내용에 맞춘 스펙 설계가 필요해 이번 점검에서는 가장 시급한 W4만 먼저 개정했습니다. W5 착수 전, 또는 다음 격주 점검에서 순차적으로 개정을 권장합니다.

---

*이번 주 진짜 목표는 "sklearn/numpy를 넘어 실제 서비스형 API를 다루는 감각"을 만드는 것입니다.
LLM API를 한 번이라도 직접 호출해 CLI로 완성했다면, 그리고 Git 브랜치를 한 번이라도 써봤다면 W4는 이미 성공입니다.*

# W4 구체적 실행 계획

> **주제**: LLM API 활용 + 미니프로젝트 #1(텍스트 요약·분류기 CLI) + SVD 이론 역추적 + Git 브랜치·PR 워크플로우
>
> **사용 데이터 주의**: 이 계획은 공개 뉴스 RSS 피드(예: BBC News RSS)를 씁니다.
> RSS 기사 내용은 매일 바뀌므로, 아래 코드의 출력 예시(요약문·카테고리)는 실행 시점에 따라 달라집니다.
> 숫자나 문장을 외워서 쓰지 말고, 본인이 직접 돌려서 나온 결과를 근거로 삼으세요.
> **총 목표 시간**: 15–16시간
> **기준**: 평일 2시간 + 주말(토요일 3시간, 일요일 2–3시간)

---

## W4 목표 (이것만 달성하면 성공)

1. **실습**: LLM API(OpenAI 또는 Anthropic)로 텍스트 요약 + 카테고리 분류 첫 호출 성공
2. **미니프로젝트 #1**: 뉴스 RSS → 요약 + 분류 CLI 완성, GitHub에 README와 함께 커밋
3. **도구**: Git branch·PR 워크플로우를 실제로 한 번 사용 (feature 브랜치 → PR → merge)
4. **이론**: SVD와 PCA의 관계를 행렬 분해 관점에서 설명 가능
5. **최소 보장**: SVD와 PCA 차이를 설명 가능 (W1 PCA와 연결)

---

## Day 1 (월요일) — LLM API 첫 호출 [2시간]

W1–3에서 sklearn으로 만들던 모델을, 이번엔 LLM API 호출로 바꿔 봅니다. 패턴은 같습니다: 실습 먼저, 막히면 이론.

### 00:00–00:30 | API 키 발급 + 환경 세팅

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

막히면: `.env`가 git에 올라가지 않는지 `git status`로 반드시 확인 (API 키 유출 방지가 최우선)

### 00:30–01:30 | 첫 API 호출 — 요약 + 분류

```python
# W4 Day1: LLM API 첫 호출 — 텍스트 요약 + 카테고리 분류
import os
from dotenv import load_dotenv
import anthropic

load_dotenv()
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

sample_article = """
Scientists have developed a new battery technology that could
significantly extend the range of electric vehicles. The breakthrough,
published this week, uses a solid-state design that reduces charging
time by up to 40% compared to current lithium-ion batteries.
"""

CATEGORIES = ["Technology", "Business", "Politics", "Science", "Sports", "Other"]

prompt = f"""다음 기사를 처리하세요:

기사: {sample_article}

작업:
1. 2문장으로 요약하세요.
2. 다음 카테고리 중 하나로 분류하세요: {", ".join(CATEGORIES)}

아래 형식으로만 답하세요:
요약: <요약문>
카테고리: <카테고리 하나>
"""

response = client.messages.create(
    model="claude-haiku-4-5-20251001",
    max_tokens=300,
    messages=[{"role": "user", "content": prompt}],
)

print(response.content[0].text)
```

실행 후 확인할 것:
- 요약이 원문의 핵심을 담고 있는가?
- 카테고리 분류가 합리적인가? (이 예시는 "Technology" 또는 "Science"가 나와야 자연스러움)

막히면: `AuthenticationError` → API 키 오타 또는 `.env` 로드 실패. `print(os.getenv("ANTHROPIC_API_KEY"))`로 값이 읽히는지 먼저 확인.

### 01:30–02:00 | Git 브랜치 워크플로우 시작

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

## Day 2 (화요일) — RSS 피드 연동 [2시간]

### 00:00–01:00 | feedparser로 뉴스 가져오기

```python
# W4 Day2: RSS 피드에서 실제 기사 목록 가져오기
import feedparser

RSS_URL = "http://feeds.bbci.co.uk/news/rss.xml"

feed = feedparser.parse(RSS_URL)
print(f"피드 제목: {feed.feed.title}")
print(f"기사 수: {len(feed.entries)}")

for entry in feed.entries[:5]:
    print(f"\n제목: {entry.title}")
    print(f"요약(원본): {entry.get('summary', '(없음)')[:100]}...")
    print(f"링크: {entry.link}")
```

막히면: RSS 응답이 비어있으면 → URL이 바뀌었을 수 있음. `feedparser.parse()`가 반환하는 `feed.bozo`가 True면 파싱 에러 — 다른 공개 RSS URL(예: 각 언론사 홈페이지의 "RSS" 링크)로 교체.

### 01:00–02:00 | 요약+분류 함수를 실제 기사에 적용

```python
# W4 Day2: 실제 RSS 기사에 요약+분류 파이프라인 연결
import os
import time
from dotenv import load_dotenv
import anthropic
import feedparser

load_dotenv()
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
CATEGORIES = ["Technology", "Business", "Politics", "Science", "Sports", "Other"]


def summarize_and_classify(title: str, text: str) -> dict:
    prompt = f"""기사 제목: {title}
기사 본문: {text}

작업:
1. 2문장으로 요약하세요.
2. 다음 카테고리 중 하나로 분류하세요: {", ".join(CATEGORIES)}

아래 형식으로만 답하세요:
요약: <요약문>
카테고리: <카테고리 하나>
"""
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response.content[0].text
    summary_line = [l for l in raw.split("\n") if l.startswith("요약:")]
    category_line = [l for l in raw.split("\n") if l.startswith("카테고리:")]
    return {
        "title": title,
        "summary": summary_line[0].replace("요약:", "").strip() if summary_line else raw,
        "category": category_line[0].replace("카테고리:", "").strip() if category_line else "Other",
    }


feed = feedparser.parse("http://feeds.bbci.co.uk/news/rss.xml")
results = []
for entry in feed.entries[:5]:
    result = summarize_and_classify(entry.title, entry.get("summary", ""))
    results.append(result)
    print(f"[{result['category']}] {result['title']}")
    print(f"  → {result['summary']}\n")
    time.sleep(1)  # API rate limit 여유
```

커밋:
```bash
git add week04/
git commit -m "W4 Day2: RSS feed integration + batch summarize/classify"
git push
```

---

## Day 3 (수요일) — SVD 이론 역추적 [2시간]

### 00:00–01:00 | "SVD가 PCA와 뭐가 다른가?" — 역추적

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

### 01:00–02:00 | numpy로 SVD 직접 계산 → W1 PCA와 비교

```python
# W4 Day3: SVD로 PCA 다시 구현 + W1의 고유값분해 방식과 비교
import numpy as np
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler

digits = load_digits()
X_scaled = StandardScaler().fit_transform(digits.data)

# 방법 1 (W1 방식): 공분산 행렬의 고유값분해
n_samples = X_scaled.shape[0]
cov_matrix = (X_scaled.T @ X_scaled) / (n_samples - 1)
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]
X_pca_eig = X_scaled @ eigenvectors[:, :2]

# 방법 2 (오늘 배운 것): SVD로 직접 계산 — 공분산 행렬을 만들지 않음
U, S, Vt = np.linalg.svd(X_scaled, full_matrices=False)
X_pca_svd = X_scaled @ Vt[:2].T

# 특이값(S)과 고유값의 관계: eigenvalue = S^2 / (n_samples - 1)
eigenvalues_from_svd = (S ** 2) / (n_samples - 1)
print(f"고유값분해로 구한 상위 5개 고유값: {eigenvalues[:5].round(3)}")
print(f"SVD로 역산한 상위 5개 고유값:      {eigenvalues_from_svd[:5].round(3)}")
print(f"일치 여부: {np.allclose(eigenvalues[:5], eigenvalues_from_svd[:5])}")

# 두 방법의 투영 결과 비교 (부호는 다를 수 있음)
print(f"\n두 방법 결과 절댓값 일치: "
      f"{np.allclose(np.abs(X_pca_eig[0]), np.abs(X_pca_svd[0]))}")

# ✅ 최소 보장 체크:
# "SVD의 V(오른쪽 특이벡터) = 공분산 행렬의 고유벡터"
# "특이값의 제곱 / (n-1) = 고유값"
# 이 두 관계를 위 코드로 직접 확인했는가?
```

커밋:
```bash
git add week04/
git commit -m "W4 Day3: SVD vs eigendecomposition comparison (PCA)"
git push
```

---

## Day 4 (목요일) — CLI 완성 [2시간]

### 00:00–01:00 | argparse로 CLI 골격 만들기

```python
# week04/news_classifier_cli.py
"""
W4 미니프로젝트 #1: 뉴스 RSS 요약·분류 CLI

사용법:
    python news_classifier_cli.py --url http://feeds.bbci.co.uk/news/rss.xml --limit 5
"""
import argparse
import os
import time
import sys
from dotenv import load_dotenv
import anthropic
import feedparser

CATEGORIES = ["Technology", "Business", "Politics", "Science", "Sports", "Other"]


def summarize_and_classify(client, title: str, text: str) -> dict:
    prompt = f"""기사 제목: {title}
기사 본문: {text}

작업:
1. 2문장으로 요약하세요.
2. 다음 카테고리 중 하나로 분류하세요: {", ".join(CATEGORIES)}

아래 형식으로만 답하세요:
요약: <요약문>
카테고리: <카테고리 하나>
"""
    try:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
    except anthropic.APIError as e:
        return {"title": title, "summary": f"(API 오류: {e})", "category": "Other"}

    raw = response.content[0].text
    summary_line = [l for l in raw.split("\n") if l.startswith("요약:")]
    category_line = [l for l in raw.split("\n") if l.startswith("카테고리:")]
    return {
        "title": title,
        "summary": summary_line[0].replace("요약:", "").strip() if summary_line else raw,
        "category": category_line[0].replace("카테고리:", "").strip() if category_line else "Other",
    }


def main():
    parser = argparse.ArgumentParser(description="뉴스 RSS 요약·분류 CLI")
    parser.add_argument("--url", default="http://feeds.bbci.co.uk/news/rss.xml")
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()

    load_dotenv()
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("ANTHROPIC_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.", file=sys.stderr)
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)
    feed = feedparser.parse(args.url)
    if feed.bozo:
        print(f"경고: RSS 파싱 중 문제 발생 ({feed.bozo_exception})", file=sys.stderr)

    print(f"피드: {feed.feed.get('title', args.url)} | 기사 {len(feed.entries)}개 중 {args.limit}개 처리\n")

    for entry in feed.entries[:args.limit]:
        result = summarize_and_classify(client, entry.title, entry.get("summary", ""))
        print(f"[{result['category']}] {result['title']}")
        print(f"  → {result['summary']}\n")
        time.sleep(1)


if __name__ == "__main__":
    main()
```

막히면:
- `ModuleNotFoundError` → `pip install anthropic feedparser python-dotenv`
- API 응답 형식이 기대와 다름 → 파싱 실패 시 원문 그대로 출력하도록 이미 fallback 처리됨 (위 코드 참고)
- Rate limit 에러 → `time.sleep()` 값을 2–3초로 늘리기

### 01:00–02:00 | README 작성 + PR 워크플로우 완성

```bash
# week04/README.md에 프로젝트 설명 작성 (직접 작성 권장)
cat > week04/README.md << 'EOF'
# W4: LLM 텍스트 요약·분류기 CLI

공개 뉴스 RSS 피드의 기사를 가져와 Claude API로 2문장 요약과
카테고리 분류(Technology/Business/Politics/Science/Sports/Other)를
자동으로 수행하는 CLI 도구.

## 사용법
```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
python news_classifier_cli.py --url http://feeds.bbci.co.uk/news/rss.xml --limit 5
```

## 배운 것
- LLM API 프롬프트 설계 (형식 고정으로 파싱 용이하게)
- feedparser로 RSS 파싱
- SVD와 PCA의 관계 (W1 PCA와 연결)
EOF

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

## Day 5 (금요일) — 복습 + 논문 [2시간]

### 00:00–00:30 | W4 이론 복습

스스로에게 물어볼 것 (답이 안 나오면 해당 자료 다시 보기):

```
□ SVD (X = U Σ V^T)에서 V가 의미하는 것은?
  → 오른쪽 특이벡터. 공분산 행렬(X^T X)의 고유벡터와 같은 방향.

□ 특이값(singular value)과 고유값(eigenvalue)의 관계는?
  → eigenvalue = singular_value² / (n_samples - 1)

□ 왜 sklearn PCA는 공분산 행렬을 직접 만들지 않고 SVD를 쓰는가?
  → 공분산 행렬(X^T X)을 계산하면 수치 오차가 제곱으로 커짐(condition number 악화).
    SVD는 원본 데이터 X에 바로 적용해 더 안정적.

□ 이번 주 LLM API 프롬프트에서 "형식을 고정"한 이유는?
  → 파싱(요약:/카테고리: 접두어)을 안정적으로 하기 위해. 자유 형식 응답은 후처리가 불안정함.
```

### 00:30–01:30 | AlexNet 논문 맥락 파악 (20–25분으로 충분)

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

### 01:30–02:00 | 금요일 마무리 커밋 + 다음 주 준비

```bash
# 이번 주 정리 커밋
git add .
git commit -m "W4 완료: LLM API CLI, SVD-PCA 관계, Git PR 워크플로우"
git push

# README 업데이트
cat >> week04/README.md << 'EOF'

## W4 완료 항목
- [x] LLM API(Claude) 첫 호출 — 요약 + 분류
- [x] RSS 피드 연동 + 배치 처리
- [x] SVD와 PCA 관계를 numpy로 직접 확인
- [x] argparse CLI 완성 + README
- [x] Git 브랜치 → PR → merge 워크플로우 1회 완주

## 최소 보장 체크
- [x] SVD와 PCA 차이를 행렬 분해 관점에서 설명 가능
EOF
```

---

## 주말 — 심화 [5–6시간]

### 토요일 [3시간]

**[00:00–01:30] CLI 확장 — 에러 처리와 배치 안정성 강화**

```python
# W4 토요일: CLI에 재시도 로직·결과 저장(JSON) 추가
import json
import time
from datetime import datetime, timezone


def summarize_and_classify_with_retry(client, title: str, text: str, max_retries: int = 3) -> dict:
    for attempt in range(max_retries):
        try:
            prompt = f"""기사 제목: {title}
기사 본문: {text}

작업:
1. 2문장으로 요약하세요.
2. 다음 카테고리 중 하나로 분류하세요: Technology, Business, Politics, Science, Sports, Other

아래 형식으로만 답하세요:
요약: <요약문>
카테고리: <카테고리 하나>
"""
            response = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}],
            )
            return {"title": title, "raw": response.content[0].text, "attempt": attempt + 1}
        except Exception as e:
            if attempt == max_retries - 1:
                return {"title": title, "raw": f"(실패: {e})", "attempt": attempt + 1}
            time.sleep(2 ** attempt)  # exponential backoff


def save_results(results: list[dict], path: str = "week04/results.json"):
    with open(path, "w", encoding="utf-8") as f:
        json.dump({
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "results": results,
        }, f, ensure_ascii=False, indent=2)
    print(f"저장 완료: {path}")

# 질문에 스스로 답하기:
# - exponential backoff(2**attempt)가 왜 고정된 sleep(1)보다 나은가?
#   → API가 일시적으로 과부하일 때 점점 대기 시간을 늘려 서버 부담을 줄이고 성공 확률을 높임
```

**[01:30–03:00] SVD 시각화 — 특이값이 클수록 정보가 많다는 것을 눈으로 확인**

```python
# W4 토요일: 이미지를 SVD로 압축하며 특이값의 의미 체감
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

digits = load_digits()
img = digits.images[0]  # 8x8 이미지 1장

U, S, Vt = np.linalg.svd(img, full_matrices=False)

fig, axes = plt.subplots(1, 4, figsize=(14, 4))
for ax, k in zip(axes, [1, 2, 4, 8]):
    # 상위 k개 특이값만 사용해 이미지 재구성
    img_approx = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
    ax.imshow(img_approx, cmap="gray")
    variance_explained = (S[:k] ** 2).sum() / (S ** 2).sum()
    ax.set_title(f"k={k}\n분산 {variance_explained:.1%}")
    ax.axis("off")

plt.tight_layout()
plt.savefig("week04/svd_image_compression.png", dpi=150)
plt.show()

# 관찰: k가 커질수록 원본에 가까워지는가?
# → 이게 "특이값이 큰 순서 = 중요한 정보 순서"라는 직관의 시각적 증거
```

### 일요일 [2–3시간]

**[00:00–01:30] W4 최종 재구현 + 영어 설명 연습**

```python
# W4 토탈 리뷰: 처음부터 끝까지 혼자서 다시
# 아무것도 보지 않고 아래를 구현할 수 있는가?

# 1. .env에서 API 키 로드 + 클라이언트 생성 (3분)
# 2. RSS 피드 파싱 + 기사 3개 추출 (5분)
# 3. 요약+분류 프롬프트 작성 + API 호출 (10분)
# 4. 결과 파싱(요약:/카테고리: 접두어) (5분)
# 5. numpy SVD로 PCA 재현 + 고유값분해 결과와 비교 (10분)

# 막히면 Day3·Day4 코드 참고 가능. 중요한 건 "SVD ↔ 고유값분해" 연결과
# "API 호출 → 파싱 → 저장" 흐름을 기억하는 것.
```

영어로 말해보기 (혼자서 소리 내어):

아래에서 `___` 부분은 반드시 **직접 코드를 돌려 나온 값**으로 채우세요.

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

**[01:30–02:30] W5 준비 + 주간 회고**

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

---

*이번 주 진짜 목표는 "sklearn/numpy를 넘어 실제 서비스형 API를 다루는 감각"을 만드는 것입니다.
LLM API를 한 번이라도 직접 호출해 CLI로 완성했다면, 그리고 Git 브랜치를 한 번이라도 써봤다면 W4는 이미 성공입니다.*

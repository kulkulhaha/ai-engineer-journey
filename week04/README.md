# W4: LLM 텍스트 요약·분류기 CLI (뉴스 RSS)

## 진행 상황 (2026-09-02 기준 — 진행 중, week04-day2 브랜치)

**완료:**
- Day1: OpenAI API로 텍스트 요약+분류하는 `summarize_and_classify(client, title, text, categories)` 구현 (`first LLM API call.ipynb`) — "요약: .../카테고리: ..." 형식을 프롬프트로 강제하고 문자열 파싱, 파싱 실패 시 category "Other" fallback. `client.messages.create()`(Anthropic 문법)와 OpenAI 문법이 섞였던 것, 응답 객체에 바로 `.splitlines()`를 호출하려던 것, 루프 안에서 초기값이 매 줄 리셋되던 파싱 버그, dict 대신 튜플을 반환하던 것 등을 스스로 디버깅해 해결
- Day2: `fetch_articles(rss_url, limit)` 구현 (`RSS_feedparser.ipynb`) — `feedparser`로 RSS 파싱, `bozo`(파싱 에러) 시 경고만 출력하고 계속 진행, `limit`이 실제 entries 수보다 커도 안전하게 클램핑, summary 없는 엔트리는 빈 문자열 처리
- Day3: SVD와 고유값분해 비교 (`SVD vs eigendecomposition.ipynb`) — `load_digits()` + `StandardScaler` 데이터에 대해 공분산 행렬 고유값분해(`np.linalg.eigh`)와 데이터 행렬 SVD(`np.linalg.svd`)를 각각 적용해, 특이값 제곱/(n-1) = 고유값임을 `np.allclose`로 직접 검증 (W2에서 증명한 관계를 코드로 재현)
- Day4: `news_classifier_cli.py` — argparse 기반 CLI(`--url`, `--limit`) 골격 완성, `safe_summarize_and_classify()`로 API 에러를 세 갈래(`AuthenticationError`는 즉시 종료, `RateLimitError`는 재귀 재시도, 그 외 `APIError`는 해당 기사만 실패 처리하고 계속 진행)로 나눠 방어적으로 래핑. `import_ipynb`로 다른 노트북 함수를 불러오는 과정에서 파일명 오타, cwd 기준 탐색 문제, 마크다운 셀이 코드 셀로 저장된 문제, `load_dotenv()` 호출 순서 문제 등 4가지 환경 이슈를 순서대로 해결

**아직 안 한 것:**
- Day5: W4 이론 복습 (SVD/PCA 관계, 프롬프트 형식 고정 이유) + AlexNet(2012) 논문 Abstract/Section 1/3.1–3.4 읽기
- README 완료 체크리스트 작성 + Git PR 워크플로우 (계획은 `feature/mini-project-1-llm-classifier` 브랜치였으나 실제로는 `week04-day2` 브랜치로 진행 중 — main으로 merge 필요)
- 주말 심화: `summarize_and_classify_with_retry`(exponential backoff) + `save_results`(JSON 저장), `svd_image_reconstruction`(k별 이미지 압축 시각화)
- W4 회고 작성

**더 공부 필요 / 다음에 다시 볼 것:**
- 분류 결과가 지정된 4개 카테고리(Technology/Science/Health/Entertainment)를 벗어나는 사례 발생(`'None'`, `'그 외의 설명'`) — categories 검증 로직을 추가할지 재검토 필요
- `RSS_feedparser.ipynb` 안 테스트/데모 셀이 `import_ipynb`로 불러올 때마다 같이 실행되어, 매 실행마다 불필요한 API 호출이 최소 1회 추가로 나가는 문제 — 일단 그대로 두기로 판단(2026-09-02), 나중에 노트북 정리 시 삭제/주석 처리할 것

**다음 액션:** Day5(복습 + AlexNet 논문) → README 완료 체크리스트 → PR/merge

---

## 사용법

```bash
pip install -r requirements.txt
# .env에 OPENAI_API_KEY 설정
cd week04
python news_classifier_cli.py --url http://feeds.bbci.co.uk/news/rss.xml --limit 5
```

## 배운 것

- LLM 응답을 자유 형식이 아니라 접두어("요약:"/"카테고리:")로 고정하면 후처리(파싱)가 안정적이 된다
- `feedparser`로 공개 RSS 피드를 구조화된 데이터로 받아올 수 있고, `bozo` 플래그로 파싱 에러를 감지할 수 있다
- SVD(`X = U Σ Vᵀ`)의 오른쪽 특이벡터 V가 공분산 행렬의 고유벡터와 같고, 특이값 제곱/(n-1)이 고유값과 같다 — sklearn PCA가 공분산 행렬을 직접 만들지 않고 SVD를 쓰는 이유(수치 안정성, condition number)와 연결됨

## W4 완료 항목

- [x] LLM API 첫 호출 — 요약 + 분류
- [x] RSS 피드 연동 + 배치 처리
- [x] SVD와 PCA 관계를 numpy로 직접 확인
- [x] argparse CLI 완성
- [ ] Git 브랜치 → PR → merge 워크플로우 1회 완주

## 최소 보장 체크

- [x] SVD와 PCA 차이를 행렬 분해 관점에서 설명 가능

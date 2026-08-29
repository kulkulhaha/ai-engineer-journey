# W9 구체적 실행 계획 (학습 가이드형)

> **주제**: HuggingFace pipeline 체험 + Sentence-BERT 의미 검색 + Attention/Transformer 예습 + asyncio·타입힌팅 시작
>
> **사용 모델 주의**: 이번 주는 모델을 직접 학습시키지 않습니다. HuggingFace Hub에 이미 공개된 사전학습 모델(pipeline, Sentence-BERT)을 "불러와서 쓰는" 것이 핵심입니다. 목표는 모델 성능이 아니라 "사전학습 모델을 실전에서 다루는 감각"과 "Transformer로 넘어가기 전 예습"입니다.
>
> **총 목표 시간**: 10–12시간 (v7 커리큘럼 블록 A 기준)
> **기준**: 평일 1.5시간 + 토요일 2.5시간 + 일요일 2시간
>
> **코드 제시 방식**: 완성된 실행 코드를 주지 않습니다. 학습 목표·핵심 개념·구현 요구사항(스펙)·확인 질문만 제시합니다. 코드는 스스로 작성하세요. 설치·폴더 생성 같은 보일러플레이트만 그대로 제시합니다.
>
> **전제**: W1–W8에서 numpy 재구현, FastAPI 서빙, Docker 배포까지 마쳤다고 가정합니다. HuggingFace transformers·Sentence-BERT는 처음이어도 괜찮습니다.

---

## W9 목표 (이것만 달성하면 성공)

1. **실습**: HuggingFace `pipeline`으로 감성분석·텍스트분류를 최소 1개씩 돌려본다
2. **실습**: Sentence-BERT로 문장 임베딩을 만들고, cosine similarity 기반 간단한 의미 검색을 구현한다
3. **예습**: "Transformer가 왜 이렇게 생겼나"를 Bahdanau Attention 논문 맥락 + Vaswani Figure 1로 미리 훑는다
4. **Python 고급**: asyncio 기초 개념을 이해하고, W8의 FastAPI 엔드포인트 최소 1개를 `async def`로 전환해본다
5. **알고리즘**: Sort Colors, Implement Queue using Stacks를 풀고 각각의 접근 방식을 설명할 수 있다

---

## Day 1 (월요일) — HuggingFace pipeline 첫 체험 [1.5시간]

### 00:00–00:15 | 설치 (보일러플레이트)

```bash
pip install transformers torch
mkdir -p week09
cd week09
```

### 00:15–00:45 | 개념: pipeline 추상화, 사전학습 모델

- **학습 목표**: `pipeline()`이 내부적으로 무엇을 대신 해주는지(토크나이저 로드 → 모델 로드 → 전처리 → 추론 → 후처리) 설명할 수 있다.
- **핵심 개념**:
  - **사전학습(pretrained) 모델**: 이미 대규모 데이터로 학습된 가중치를 그대로 가져와 쓰는 것. W3~W7에서 직접 학습시킨 것과 무엇이 다른가?
  - **토크나이저**: 텍스트를 모델이 이해하는 숫자(토큰 ID)로 바꾸는 과정. 왜 모델마다 토크나이저가 다른가?
  - **task별 head**: 같은 Transformer 본체 위에 감성분석용, 분류용 head가 다르게 얹힌다는 것. `pipeline("sentiment-analysis")`와 `pipeline("zero-shot-classification")`이 내부적으로 다른 모델/head를 쓸 수 있음.
- **막히면**: "huggingface pipeline tutorial" 검색 → HuggingFace 공식 quicktour 문서 첫 섹션만 읽기. 개념이 안 잡히면 "what is a tokenizer in NLP" 영상 검색 → 30분 넘으면 다음으로.

### 00:45–01:30 | 구현: 감성분석 + zero-shot 텍스트분류

- **학습 목표**: 사전학습 모델을 로드해 임의의 한국어/영어 문장에 대해 감성분석과 zero-shot 분류 결과를 직접 얻는다.
- **구현 과제 (스스로 작성)**:
  ```python
  # week09/day1_pipeline.py
  # 요구사항:
  # - transformers.pipeline으로 "sentiment-analysis" 태스크 파이프라인을 만든다.
  # - 최소 5개 문장(직접 작성, 긍정/부정/애매한 것 섞어서)에 대해 예측 결과(label, score)를 출력한다.
  # - transformers.pipeline으로 "zero-shot-classification" 태스크 파이프라인을 만든다.
  #   (후보 레이블을 직접 정의: 예) ["기술", "스포츠", "정치", "요리"])
  # - 뉴스 헤드라인 몇 개(직접 작성 또는 검색)를 넣어 어느 레이블로 분류되는지 확인한다.

  def run_sentiment(sentences: list[str]) -> list[dict]:
      ...

  def run_zero_shot(texts: list[str], candidate_labels: list[str]) -> list[dict]:
      ...
  ```
- **확인 질문**:
  - 감성분석 결과의 `score`는 무엇을 의미하는가? (힌트: softmax 확률)
  - zero-shot 분류는 해당 레이블로 "학습"된 적이 없는데 어떻게 분류가 가능한가? (힌트: NLI 기반 접근 — 문장과 레이블 사이의 함의 관계를 판단)
  - 애매한 문장(중립적이거나 반어법)에서 모델이 틀리는 경우를 하나 찾았는가? 왜 틀렸을지 추측해보기.

**막히면**: 모델 다운로드가 느리거나 실패하면 네트워크·디스크 공간 확인. `ImportError`가 나면 `pip install transformers[torch]`로 재설치. 30분 넘으면 결과 없이도 개념 정리만 하고 다음으로.

```bash
git add week09/day1_pipeline.py
git commit -m "W9 Day1: HuggingFace pipeline 감성분석/zero-shot 분류"
```

---

## Day 2 (화요일) — Attention·Transformer 예습 [1.5시간]

이번 주는 코드보다 "왜 Transformer가 지금의 모습이 됐는가"를 논문 맥락으로 미리 훑는 날입니다. W10 이후 본격적으로 다룰 내용의 지도를 그리는 시간이라고 생각하세요.

### 00:00–00:45 | 논문 맥락: Bahdanau et al. (2014) Attention

- **학습 목표**: RNN 기반 Seq2Seq(W6에서 다룬 LSTM 연장선)가 왜 긴 문장에서 성능이 떨어졌는지, Attention이 이 문제를 어떻게 풀었는지 한 문단으로 설명할 수 있다.
- **핵심 개념**:
  - **고정 길이 벡터 병목**: 인코더가 전체 입력 문장을 하나의 고정 크기 벡터로 압축해야 했던 기존 Seq2Seq(W6 LSTM 연결)의 한계.
  - **Attention의 아이디어**: 디코더가 매 스텝마다 인코더의 "모든" 은닉 상태를 다시 참고하되, 가중치(얼마나 집중할지)를 동적으로 계산.
  - 왜 이것이 "Transformer의 전신"이라 불리는지 — Attention만 남기고 RNN을 아예 제거하면 무엇이 남는가?
- **읽을 것**: Bahdanau et al. (2014) *Neural Machine Translation by Jointly Learning to Align and Translate* — Abstract + Introduction만 (10–15분)
- **막히면**: "seq2seq fixed length vector bottleneck attention" 검색. 여전히 추상적이면 3Blue1Brown이나 StatQuest의 "attention mechanism explained" 영상 검색 → 30분 넘으면 다음으로.

### 00:45–01:30 | Vaswani et al. (2017) Figure 1 예습 + 메모

- **학습 목표**: Transformer의 전체 구조(인코더 스택, 디코더 스택, Multi-Head Attention 위치)를 그림만 보고 "블록 단위"로 설명할 수 있다. 수식 이해는 W10 이후로 미룬다.
- **할 일**:
  ```
  1. Vaswani et al. (2017) "Attention Is All You Need" Figure 1(전체 아키텍처 다이어그램)만 찾아서 5–10분간 관찰
  2. 아래 질문에 대해 메모장에 한 줄씩 적기 (정답을 찾으려 하지 말고 "지금 보이는 대로" 추측):
     - 인코더와 디코더는 각각 몇 개의 블록이 쌓여 있는 것처럼 보이는가?
     - "Multi-Head Attention" 박스가 그림에 몇 번 등장하는가? 각각 입력이 다르게 연결된 것처럼 보이는가?
     - "Positional Encoding"이라는 박스가 왜 임베딩 바로 다음에 더해지는 형태로 그려져 있을까? (다음 주 이후 정답 확인)
  ```
- **확인 질문**:
  - Day 2 아침에 읽은 Bahdanau Attention과 Figure 1의 Multi-Head Attention은 이름이 비슷한데, 이번 주 수준에서 어떤 차이가 있을 것 같은가? (완벽한 답 필요 없음, 가설만)
  - 이 그림에서 아직 이해가 안 되는 박스를 3개 이상 적었는가? (막연해도 정상 — W10~블록 C에서 하나씩 풀림)

**막히면**: 그림이 너무 복잡하게 느껴지면 "transformer architecture diagram explained simply" 검색으로 쉬운 설명 먼저 보기. 이번 주는 완벽한 이해가 목표가 아니라 "지도 그리기"임을 기억할 것.

```bash
git add week09/attention_transformer_notes.md
git commit -m "W9 Day2: Attention·Transformer 예습 메모"
```

---

## Day 3 (수요일) — Sentence-BERT 문장 유사도 [1.5시간]

### 00:00–00:15 | 설치 (보일러플레이트)

```bash
pip install sentence-transformers
```

### 00:15–00:45 | 개념: 문장 임베딩과 cosine similarity

- **학습 목표**: 문장을 고정 차원 벡터로 바꾸는 것이 왜 "의미 검색"을 가능하게 하는지, W2의 PCA·차원축소와 어떻게 연결되는지 설명할 수 있다.
- **핵심 개념**:
  - **문장 임베딩**: 단어 하나가 아니라 문장 전체를 하나의 고정 차원 벡터로 표현. Word2Vec(W6 논문)이 단어 단위였다면, Sentence-BERT는 문장 단위로 이 아이디어를 확장한 것.
  - **임베딩이 왜 차원축소인가**: 원래 문장은 가변 길이 텍스트인데, 이를 고정 차원(예: 384차원) 벡터로 압축하는 것 자체가 정보를 저차원 공간에 투영하는 것 — W1~W2 PCA에서 다룬 "고차원 → 저차원 투영" 개념과 본질적으로 같은 종류의 문제.
  - **cosine similarity**: 두 벡터의 방향이 얼마나 비슷한지(크기는 무시)를 재는 척도. 왜 유클리드 거리 대신 이걸 자주 쓰는가? (벡터 크기가 문장 길이 등에 영향받는 것을 배제하기 위해)
- **막히면**: "sentence embeddings vs word embeddings" 검색. cosine similarity 직관이 안 잡히면 "cosine similarity explained visually" 영상 검색 → 30분 넘으면 실습으로 넘어가기.

### 00:45–01:30 | 구현: 문장 임베딩 + 간단한 의미 검색

- **학습 목표**: 여러 문장을 임베딩하고, 새로운 질의 문장과 가장 유사한 문장을 cosine similarity로 찾아내는 미니 검색기를 스스로 완성한다.
- **구현 과제 (스스로 작성)**:
  ```python
  # week09/day3_semantic_search.py
  # 요구사항:
  # - SentenceTransformer로 사전학습 모델(예: 다국어 지원 모델)을 로드한다.
  # - 문서 후보 최소 10개(직접 작성한 한국어/영어 문장 — 다양한 주제로)를 리스트로 준비한다.
  # - 위 문서들을 모두 임베딩하여 하나의 행렬(shape: [문서 수, 임베딩 차원])로 만든다.
  # - 임의의 질의 문장 3개를 준비하고, 각 질의에 대해 코사인 유사도가 가장 높은 상위 3개 문서를 반환하는 함수를 작성한다.

  def embed_documents(sentences: list[str]) -> "np.ndarray":
      ...

  def semantic_search(query: str, doc_embeddings: "np.ndarray", documents: list[str], top_k: int = 3) -> list[tuple[str, float]]:
      ...
  ```
- **확인 질문**:
  - 질의 문장에 문서와 겹치는 단어가 하나도 없는데도 의미상 가까운 문서가 상위에 나온 사례를 하나 찾았는가? (키워드 검색과의 차이를 보여주는 핵심 증거)
  - 임베딩 차원(예: 384)을 W1 digits 데이터(64차원)와 비교했을 때, 둘 다 "고차원 벡터"라는 점에서 어떤 처리(PCA 등)를 적용해볼 수 있을지 떠오르는가?
  - top_k를 1로 줄이면 검색 품질에 어떤 트레이드오프가 생기는가?

**막히면**: 모델 로드가 느리면 최초 1회는 다운로드 시간이 걸리는 것이 정상. 유사도 값이 이상하면 임베딩 벡터를 정규화했는지(`normalize_embeddings` 옵션 등) 확인. 30분 넘으면 결과 일부만 확인하고 다음으로.

```bash
git add week09/day3_semantic_search.py
git commit -m "W9 Day3: Sentence-BERT 문장 임베딩 + 의미 검색"
```

---

## Day 4 (목요일) — asyncio 기초 + LeetCode [1.5시간]

### 00:00–00:45 | 개념: asyncio, 코루틴, 이벤트 루프

- **학습 목표**: `async def`/`await`가 왜 필요한지, 스레드(thread)와 무엇이 다른지, I/O bound 작업에서 왜 유리한지 설명할 수 있다.
- **핵심 개념**:
  - **동기 vs 비동기**: 동기 코드는 한 작업이 끝날 때까지 다음 줄이 기다림. 비동기는 I/O 대기 중(예: 네트워크 응답 기다리기) 다른 작업을 처리할 수 있음.
  - **코루틴(coroutine)**: `async def`로 정의된 함수. 호출해도 즉시 실행되지 않고 "awaitable 객체"를 반환한다는 점이 일반 함수와 다름.
  - **이벤트 루프**: 여러 코루틴을 스케줄링해서 실행하는 주체. 멀티스레드와 달리 하나의 스레드 안에서 협력적으로(cooperative) 전환됨.
  - **왜 FastAPI에서 중요한가**: W8에서 만든 `/predict`는 동기(`def`)였음. 만약 모델 추론이 아니라 외부 API 호출(예: LLM API, DB 조회)이 섞이면 왜 `async def`가 응답 처리량에 유리해지는지.
- **막히면**: "asyncio vs threading python" 검색. 개념이 추상적이면 "python asyncio explained in 10 minutes" 영상 검색 → 30분 넘으면 실습으로 넘어가기.

### 00:45–01:30 | 알고리즘: Sort Colors, Implement Queue using Stacks

```
- [ ] Sort Colors (Medium)
- [ ] Implement Queue using Stacks (Easy)
```

- **학습 목표**: `Sort Colors`에서 추가 메모리 없이 in-place로 정렬하는 접근(더치 국기 문제, Dutch National Flag)의 아이디어를 이해하고, `Implement Queue using Stacks`에서 두 개의 스택으로 큐를 흉내내는 원리를 설명할 수 있다.
- **구현 과제 (스스로 작성, 함수 시그니처만)**:
  ```python
  # week09/leetcode_day4.py
  # Sort Colors: 0, 1, 2로만 이루어진 배열을 in-place로 정렬한다. 정렬 라이브러리 사용 금지.
  def sort_colors(nums: list[int]) -> None:
      ...  # in-place로 nums를 직접 수정

  # Implement Queue using Stacks: 스택(리스트를 append/pop으로만 사용) 두 개로 큐를 구현한다.
  class MyQueue:
      def __init__(self):
          ...
      def push(self, x: int) -> None:
          ...
      def pop(self) -> int:
          ...
      def peek(self) -> int:
          ...
      def empty(self) -> bool:
          ...
  ```
- **확인 질문**:
  - `Sort Colors`를 세 개의 포인터(low, mid, high)로 한 번의 순회만으로 풀 수 있는 이유는? 이게 왜 O(n) 시간·O(1) 공간인가?
  - `MyQueue`에서 `pop()`을 호출할 때마다 두 스택 사이에 원소를 전부 옮기면 비효율적일 수 있다 — "필요할 때만" 옮기는 방식(amortized 관점)으로 개선하면 무엇이 달라지는가?
  - 오늘 배운 asyncio와 이 두 문제는 직접적인 관련은 없지만, "제약 조건 안에서 자료구조를 조합해 문제를 푸는" 감각이라는 공통점이 있다 — 어떤 부분이 비슷하게 느껴지는가? (자유롭게 생각해보기)

**막히면**: "dutch national flag algorithm" 또는 "implement queue using two stacks" 검색 → 에디토리얼의 접근 방식(전략)만 읽고 코드는 직접 짜기. 30분 넘으면 다음으로.

```bash
git add week09/
git commit -m "W9 Day4: asyncio 개념 정리 + Sort Colors, Implement Queue using Stacks"
```

---

## Day 5 (금요일) — 복습 + async 전환 예고 + 커밋 [1.5시간]

### 00:00–00:45 | 복습: 이번 주 개념 연결하기

```
스스로에게 물어볼 것 (답이 안 나오면 해당 Day로 돌아가 다시 확인):

□ HuggingFace pipeline이 감성분석과 zero-shot 분류에서 각각 무엇을 다르게 하는가?
  → 감성분석은 고정된 레이블로 미세조정된 모델, zero-shot은 NLI 기반으로 임의의 레이블을 추론

□ Bahdanau Attention이 풀려던 문제는 정확히 무엇이었는가?
  → 인코더의 고정 길이 벡터 병목 — 긴 문장에서 정보 손실

□ 문장 임베딩에서 cosine similarity를 쓰는 이유는?
  → 벡터의 크기가 아니라 방향(의미적 유사성)에만 집중하기 위해

□ asyncio의 코루틴이 스레드와 다른 점은?
  → 하나의 스레드 안에서 협력적으로 전환되며, I/O 대기 중 다른 코루틴이 실행될 기회를 얻음
```

### 00:45–01:30 | (선택) W8 엔드포인트 async 전환 미리 손대보기 + README 정리

- **학습 목표**: W8의 `/predict` 엔드포인트 중 하나를 `async def`로 바꿔보고, 겉보기 동작이 동일함을 확인한다 (본격적인 async I/O 활용은 B③에서 다룸).
- **구현 과제 (스스로 작성)**:
  ```python
  # week08/main.py (또는 week09로 복사본을 만들어) 수정
  # 요구사항:
  # - 기존 def predict(...) 를 async def predict(...) 로 바꾼다.
  # - 서버를 재시작해 여전히 정상 응답하는지 확인한다.
  ```
- **확인 질문**:
  - 지금 당장은 `def`를 `async def`로만 바꿔도 겉보기 동작이 똑같다 — 그렇다면 진짜 이점이 생기는 시점은 언제인가? (힌트: 함수 안에서 `await`로 실제 I/O를 기다릴 때)

```bash
git add week09/
git commit -m "W9 완료: HuggingFace pipeline + Sentence-BERT 의미검색 + Attention 예습 + asyncio 기초"

cat >> week09/README.md << 'EOF'

## W9 완료 항목
- [ ] HuggingFace pipeline 감성분석 + zero-shot 분류
- [ ] Bahdanau Attention 논문 맥락 + Vaswani Figure 1 예습 메모
- [ ] Sentence-BERT 문장 임베딩 + 의미 검색 구현
- [ ] asyncio 기초 개념 정리 + FastAPI 엔드포인트 async 전환 시도
- [ ] LeetCode: Sort Colors, Implement Queue using Stacks

## 이번 주 확인 사항
- [ ] "임베딩이 왜 차원축소인가"를 W1–W2 PCA와 연결해 설명 가능
- [ ] Attention이 풀려던 원래 문제(고정 벡터 병목)를 설명 가능
EOF
```

---

## 주말 — 심화 [4.5시간]

### 토요일 [2.5시간]

**[00:00–01:00] FastAPI에 실제 async I/O 붙여보기 (준비 운동)**

- **학습 목표**: 진짜 I/O 대기가 있는 상황(예: 외부 API 호출 흉내)에서 `async def` + `await`가 왜 응답 처리량에 유리한지 직접 체감한다.
- **핵심 개념**: `asyncio.sleep()`으로 "느린 I/O"를 흉내낼 수 있다는 것. 동기 `time.sleep()`과 무엇이 다른가 — 하나는 이벤트 루프 전체를 막고, 하나는 안 막는다.
- **구현 과제 (스스로 작성)**:
  ```python
  # week09/day6_async_demo.py
  # 요구사항:
  # - /slow-sync 엔드포인트: 동기 함수 안에서 time.sleep(2) 호출
  # - /slow-async 엔드포인트: async 함수 안에서 await asyncio.sleep(2) 호출
  # - 두 엔드포인트에 각각 동시에 여러 요청(curl 또는 requests로 병렬 호출)을 보내
  #   전체 처리 시간이 어떻게 다른지 직접 측정한다.
  ```
- **확인 질문**:
  - 동시에 3개 요청을 보냈을 때, `/slow-sync`와 `/slow-async`의 총 처리 시간이 각각 얼마나 걸렸는가? 왜 차이가 나는가?
  - 만약 엔드포인트 안에서 CPU를 많이 쓰는 연산(예: 큰 행렬 곱)을 한다면, `async def`로 바꾸는 것만으로 속도가 빨라질까? (힌트: I/O bound와 CPU bound의 차이)

**[01:00–02:30] Sentence-BERT 의미 검색 확장 + 타입힌팅 적용**

- **학습 목표**: Day 3의 미니 검색기를 더 현실적인 규모(문서 30개 이상)로 확장하고, 함수 시그니처에 타입힌팅을 온전히 적용한다.
- **구현 과제 (스스로 작성)**:
  ```python
  # week09/day6_semantic_search_v2.py
  # 요구사항:
  # - Day 3 코드를 재사용해 문서를 30개 이상으로 늘린다 (주제를 3–4개 카테고리로 나눠 섞기 — 나중에 결과 해석이 쉬움).
  # - 모든 함수·클래스에 타입힌팅(list[str], np.ndarray, tuple 등)을 빠짐없이 적용한다.
  # - mypy를 설치해 타입 오류가 없는지 확인한다 (mypy 도입은 B③에서 본격적으로 다루므로, 이번엔 가볍게 체험만).
  ```
  ```bash
  pip install mypy
  mypy week09/day6_semantic_search_v2.py
  ```
- **확인 질문**:
  - 카테고리가 다른 질의를 던졌을 때, 상위 결과가 실제로 해당 카테고리 문서로 몰리는가? 몰리지 않는 사례가 있다면 왜 그럴지 추측해보기.
  - mypy가 잡아낸 타입 오류가 있었는가? 있었다면 런타임에서는 어떤 문제로 이어질 수 있었는가?

**막히면**: 동시 요청 테스트가 잘 안 되면 "python requests concurrent requests threading" 검색으로 병렬 호출 방법부터 확인. mypy 오류 메시지가 낯설면 오류 메시지 전체를 그대로 검색.

### 일요일 [2시간]

**[00:00–01:00] 주간 회고**

```markdown
## W9 회고 (일요일에 작성)

### 달성한 것
- [ ] HuggingFace pipeline 감성분석 + zero-shot 분류
- [ ] Sentence-BERT 문장 임베딩 + 의미 검색 (확장판 포함)
- [ ] Attention/Transformer 예습 메모
- [ ] asyncio 기초 + FastAPI async 전환 체감
- [ ] LeetCode 2문제

### 예상보다 오래 걸린 것
(솔직하게 적기)

### W10에 가져갈 것
(이해 못 하고 넘어간 것 — 특히 Transformer 관련 메모 3개 중 아직 남은 것)

### 다음 주 첫 번째 할 일
(W10 Day1 무엇부터 시작할지)
```

**[01:00–02:00] W10 예습 — LangChain/LlamaIndex 감 잡기**

- **학습 목표**: W10에서 다룰 "문서 로드 → 청킹 → 임베딩 → 벡터DB → 쿼리" 파이프라인의 각 단계 이름과 순서만 먼저 익힌다 (구현은 W10에서).
- **할 일**:
  ```
  1. LangChain 또는 LlamaIndex 공식 홈페이지의 "Getting Started" 페이지만 훑어보기 (10분)
  2. 아래 파이프라인 단계를 자신의 언어로 한 줄씩 정리:
     - 문서 로드(Document Loader)
     - 청킹(Text Splitter) — 이번 주 Sentence-BERT 임베딩 경험과 연결해, 왜 문서를 통째로 임베딩하지 않고 잘게 나누는지 추측해보기
     - 임베딩(Embedding) — Day 3에서 만든 것과 같은 개념
     - 벡터DB 저장 및 검색(Vector Store) — Day 3의 "직접 만든 검색기"를 라이브러리가 어떻게 대신 해줄지 예상해보기
  ```
- **확인 질문**:
  - Day 3에서 직접 만든 의미 검색기와 W10의 벡터DB 기반 검색은 근본적으로 같은 원리인가, 다른 원리인가?
  - 문서를 왜 통째로 임베딩하지 않고 청크(chunk) 단위로 쪼개서 임베딩할지, 이번 주 배운 "임베딩=차원축소=정보압축" 관점에서 가설을 세워보기.

---

## 막힐 때 대응 가이드

개념이 막히면:

```
1단계 (5분): 구글에 영어로 검색
  예: "why sentence embeddings cosine similarity" / "asyncio vs threading python"

2단계 (10분): 관련 영상 검색 (StatQuest, 3Blue1Brown, freeCodeCamp 등)

3단계 (20분): 공식 문서 확인
  HuggingFace: https://huggingface.co/docs
  Sentence-Transformers: https://www.sbert.net
  Python asyncio: https://docs.python.org/3/library/asyncio.html

30분 넘어도 해결 안 되면:
→ 메모장에 "아직 모름: [개념]" 적고 다음으로 넘어가기
→ 이후 주차(특히 W10 이후 Transformer 본격 학습 시)에서 다시 만날 때 해결
→ AI 커리큘럼에서 막히는 것은 실력 부족이 아니라 정상 과정
```

코드가 막히면:

```
에러 메시지 전체를 복사 → 구글에 붙여넣기
Stack Overflow 답변 중 가장 많은 추천을 받은 것 선택

자주 나오는 에러:
- ModuleNotFoundError: pip install [라이브러리명]
- 모델 다운로드 실패/느림: 네트워크 확인, 처음 1회는 다운로드 시간이 오래 걸리는 게 정상
- 유사도 값이 이상함: 임베딩 정규화 여부 확인
- asyncio 관련 "coroutine was never awaited" 경고: await를 빠뜨린 곳이 있는지 확인
```

---

## W9 완료 기준

일요일 저녁에 아래를 할 수 있으면 W9 성공:

```
□ week09/ 폴더에 pipeline 실습, 의미 검색, asyncio 데모 코드가 있다
□ HuggingFace pipeline으로 감성분석과 zero-shot 분류를 각각 최소 1회 실행했다
□ Sentence-BERT로 30개 이상 문서에서 의미 검색이 동작한다 (키워드가 겹치지 않아도 관련 문서를 찾는 사례를 확인했다)
□ Bahdanau Attention의 핵심 문제(고정 벡터 병목)와 Vaswani Figure 1의 대략적 구조를 설명할 수 있다
□ asyncio 코루틴과 스레드의 차이, FastAPI에서 async가 유리한 시점을 설명할 수 있다
□ LeetCode Sort Colors, Implement Queue using Stacks를 이해하고 혼자 다시 풀 수 있다

절반(3개 이상) 달성하면 W10으로 진행.
전부 못 해도 W10으로 진행 — 이해 못 한 부분(특히 Transformer 세부 구조)은 W10 이후, 그리고 블록 C에서 본격적으로 다시 다룹니다.
```

---

## W10 첫 할 일 미리 보기

W10 Day1에 열어야 할 것:

```
1. `week10/` 폴더 생성
2. LangChain 또는 LlamaIndex 공식 튜토리얼 완주 (문서 로드 → 청킹 → 임베딩 → 벡터DB → 쿼리)
3. 막히면 → "임베딩이 왜 차원축소인가?" → 이번 주(W9 Day3) 만든 의미 검색 코드와 W1–W2 PCA 노트를 다시 열어 연결해보기
4. FAISS 기본(인덱싱·검색)도 이번 주 안에 가볍게 체험
```

---

*이 계획대로 완벽하게 안 돼도 됩니다.
W9의 진짜 목표는 "사전학습 모델을 가져다 쓰는 감각"을 익히고, Transformer라는 큰 산을 보기 전에 미리 지도를 한 번 펴보는 것입니다.
지도의 세부 지형(수식)은 아직 흐릿해도 괜찮습니다 — W10 이후, 그리고 블록 C에서 다시 자세히 그리게 됩니다.*

# W10 구체적 실행 계획 (학습 가이드형)

> **주제**: RAG 파이프라인 첫 완주(문서 로드 → 청킹 → 임베딩 → 벡터DB → 쿼리) + FAISS 기본 + Transformer 논문(Vaswani 2017) 완독
>
> **데이터셋·자료 주의사항**: 이번 주는 특정 정답 데이터셋이 없습니다. 위키피디아 문서 몇 개 또는 arXiv 논문 abstract 모음처럼 스스로 고른 공개 텍스트로 RAG 파이프라인을 구성하면 됩니다. 규모보다 "다섯 단계(로드→청킹→임베딩→저장→쿼리)가 실제로 연결되어 동작하는 경험"이 이번 주 핵심입니다.
>
> **총 목표 시간**: 10–12시간 (v9 커리큘럼 블록 A 기준)
> **기준**: 평일 1.5시간 × 5 + 토요일 2.5시간 + 일요일 2시간
>
> **코드 제시 방식**: 완성된 실행 코드를 주지 않습니다. 학습 목표·핵심 개념·구현 요구사항(스펙)·확인 질문만 제시합니다. 코드는 스스로 작성하세요. 설치·폴더 생성 같은 보일러플레이트만 그대로 제시합니다.
>
> **전제**: W9에서 HuggingFace pipeline과 Sentence-BERT 의미 검색을 다뤘습니다. 이번 주는 그 "직접 만든 의미 검색기"를 LangChain(또는 LlamaIndex) 같은 프레임워크와 FAISS 벡터DB로 확장하는 주입니다.

---

## W10 목표 (이것만 달성하면 성공)

1. **실습**: LangChain 또는 LlamaIndex로 문서 로드 → 청킹 → 임베딩 → 벡터DB 저장 → 쿼리, 다섯 단계를 모두 직접 연결한다
2. **실습**: FAISS로 인덱스를 직접 만들고, 벡터 추가·검색을 라이브러리 없이(프레임워크 추상화 없이) 한 번은 저수준으로 다뤄본다
3. **개념**: 청크 크기·오버랩을 왜 그렇게 정했는지 근거를 설명할 수 있다 (W9 "임베딩=차원축소" 관점과 연결)
4. **논문**: Vaswani et al. (2017) *Attention Is All You Need*를 완독하고, Figure 1 구조와 Multi-Head Attention 수식을 직접 손으로 설명할 수 있다
5. **알고리즘**: Design Circular Queue, Fibonacci Number를 풀고 각각의 핵심 아이디어를 설명할 수 있다

---

## Day 1 (월요일) — RAG 파이프라인 지도 그리기 + 문서 로더 [1.5시간]

### 00:00–00:15 | 설치 (보일러플레이트)

```bash
pip install langchain langchain-community faiss-cpu
mkdir -p week10
cd week10
```

### 00:15–00:45 | 개념: RAG 파이프라인 5단계

- **학습 목표**: "문서 로드 → 청킹 → 임베딩 → 벡터DB 저장 → 쿼리(검색+생성)"의 각 단계가 왜 이 순서로 필요한지, W9에서 직접 만든 의미 검색기와 어떤 부분이 같고 어떤 부분이 다른지 설명할 수 있다.
- **핵심 개념**:
  - **RAG(Retrieval-Augmented Generation)**: LLM이 답을 지어내지 않고, 관련 문서를 먼저 "검색(Retrieval)"해서 그 내용을 근거로 "생성(Generation)"하게 만드는 구조. W9 Day3의 의미 검색기가 바로 이 Retrieval 부분의 원형.
  - **Document Loader**: 다양한 형식(텍스트, PDF, 웹페이지 등)의 원본 문서를 프레임워크가 다룰 수 있는 공통 객체로 불러오는 역할. 왜 로더가 필요한가 — 원본 포맷마다 파싱 방식이 다르기 때문.
  - **프레임워크 vs 직접 구현**: W9에서 직접 짠 `embed_documents`/`semantic_search`와 LangChain의 추상화가 내부적으로 하는 일이 본질적으로 같다는 것을 인식하기.
- **막히면**: "RAG pipeline explained" 검색 → LangChain 또는 LlamaIndex 공식 "Getting Started" 페이지의 다이어그램만 먼저 보기. 개념이 안 잡히면 "retrieval augmented generation simply explained" 영상 검색 → 30분 넘으면 다음으로.

### 00:45–01:30 | 구현: 문서 로더로 원본 텍스트 불러오기

- **학습 목표**: LangChain(또는 LlamaIndex)의 Document Loader로 최소 1개 소스에서 문서를 불러와, 파이프라인의 첫 단계를 직접 완성한다.
- **구현 과제 (스스로 작성)**:
  ```python
  # week10/day1_load_docs.py
  # 요구사항:
  # - LangChain(TextLoader/WebBaseLoader 등) 또는 LlamaIndex의 Document Loader 중 하나를 선택한다.
  # - 위키피디아 문서 3~5개(직접 고른 서로 다른 주제) 또는 arXiv abstract 모음 중 하나를 소스로 정한다.
  # - 로드한 결과가 "Document 객체" 리스트 형태로 나오는지 확인하고, 각 문서의 원문 길이(문자 수)를 출력한다.

  def load_source_documents(source: str) -> list:
      ...  # Document 객체 리스트 반환
  ```
- **확인 질문**:
  - 로드된 각 Document 객체는 원문 텍스트 외에 어떤 메타데이터(출처, 제목 등)를 함께 담고 있는가?
  - 이 메타데이터가 나중에 RAG 답변에서 "출처 표시"에 어떻게 쓰일 수 있을지 추측해보기.

**막히면**: 웹 로더가 네트워크 오류를 내면 정적 텍스트 파일(.txt)로 먼저 대체해서 파이프라인 전체 흐름부터 완성하기. 30분 넘으면 문서 1개만으로 다음 단계로 진행.

```bash
git add week10/day1_load_docs.py
git commit -m "W10 Day1: RAG 파이프라인 개념 + 문서 로더 구현"
```

---

## Day 2 (화요일) — 청킹(Text Splitter) [1.5시간]

### 00:00–00:45 | 개념: 왜 문서를 통째로 임베딩하지 않는가

- **학습 목표**: 청킹이 필요한 이유를 W9에서 정리한 "임베딩=차원축소=정보압축" 관점, 그리고 W1–W2 PCA에서 다룬 "고차원 → 저차원 투영"과 연결해서 설명할 수 있다.
- **핵심 개념**:
  - **고정 차원 벡터의 용량 한계**: 문장 하나든 문서 전체든 결국 같은 차원(예: 384차원)의 벡터로 압축된다. 문서가 길어질수록 그 벡터 하나에 담아야 할 정보가 많아지고, 특정 세부 내용(질문에 실제로 필요한 문장)이 벡터 안에서 희석된다 — W9에서 "임베딩=차원축소"라고 정리한 것의 직접적인 결과.
  - **청크 크기(chunk size)와 오버랩(overlap)**: 청크가 너무 작으면 문맥이 잘려서 의미가 훼손되고, 너무 크면 다시 "너무 많은 정보가 한 벡터에 압축되는" 원래 문제로 돌아간다. 오버랩은 청크 경계에서 문장이 잘리는 것을 완화하기 위한 장치.
  - **W9 PCA 연결 질문**: PCA에서 "몇 개의 주성분을 남길지"를 정하는 것과, 청킹에서 "청크 크기를 얼마로 할지" 정하는 것 사이에 어떤 공통된 트레이드오프(정보 보존 vs 압축)가 있는가?
- **막히면**: "text chunking strategies RAG" 검색. 개념이 추상적이면 "chunk size overlap langchain explained" 영상 검색 → 30분 넘으면 실습으로 넘어가기.

### 00:45–01:30 | 구현: Text Splitter로 청킹 + 크기 비교

- **학습 목표**: 청크 크기·오버랩을 다르게 설정해보고, 그 결과(청크 개수, 청크당 내용의 완결성)를 직접 비교해 근거 있는 선택을 한다.
- **구현 과제 (스스로 작성)**:
  ```python
  # week10/day2_chunking.py
  # 요구사항:
  # - RecursiveCharacterTextSplitter(또는 LlamaIndex의 동등한 SentenceSplitter)를 사용한다.
  # - Day1에서 로드한 문서에 대해, chunk_size/overlap 조합을 최소 2가지(예: (500,50)과 (1000,100))로 각각 청킹한다.
  # - 두 조합의 청크 개수, 청크당 평균 길이를 출력하고, 청크 하나를 직접 읽어 "문장이 어색하게 잘렸는지" 육안으로 확인한다.

  def split_documents(documents: list, chunk_size: int, chunk_overlap: int) -> list:
      ...  # 청크(Document 또는 str) 리스트 반환
  ```
- **확인 질문**:
  - 두 조합 중 어느 쪽이 이번 문서에 더 적합해 보이는가? 그렇게 판단한 근거는?
  - 오버랩을 0으로 주면 어떤 문제가 생길 수 있는가? 실제로 청크 하나에서 그 사례를 찾아보았는가?
  - 표·코드 블록이 포함된 문서라면 문자 수 기준 청킹이 왜 위험할 수 있는가? (오늘 다루지 않아도 되지만 가설만 세워보기)

**막히면**: 청크 결과가 이상하면(예: 빈 청크) separator 설정을 확인. "이 문서엔 어떤 chunk_size가 맞는지 모르겠다"는 느낌이 정상 — 이번 주는 "비교해서 근거를 세우는 연습"이 목표. 30분 넘으면 조합 하나만 골라 다음으로.

```bash
git add week10/day2_chunking.py
git commit -m "W10 Day2: Text Splitter 청킹 + chunk_size/overlap 비교"
```

---

## Day 3 (수요일) — 임베딩 + 벡터DB 저장 [1.5시간]

### 00:00–00:45 | 개념: 프레임워크의 임베딩·벡터스토어 추상화

- **학습 목표**: LangChain/LlamaIndex의 Embeddings·VectorStore 추상화가 W9 Day3에서 직접 짠 `embed_documents`/`semantic_search`와 무엇을 대신 해주는지 한 문단으로 설명할 수 있다.
- **핵심 개념**:
  - **Embeddings 추상화**: 어떤 임베딩 모델(OpenAI API든 sentence-transformers든)을 쓰든 동일한 인터페이스(`embed_documents`, `embed_query`)로 다룰 수 있게 통일한 것. W9에서 이 함수를 직접 짰다는 것을 상기하기.
  - **VectorStore 추상화**: 임베딩 벡터를 저장하고, 질의 벡터와 가장 가까운 벡터를 찾아주는 인터페이스. W9 Day3의 `semantic_search` 함수가 하던 일(전체 문서와의 cosine similarity 계산 + top_k 정렬)을 벡터DB가 내부적으로 더 효율적인 자료구조(인덱스)로 대신한다는 것.
  - **왜 "더 효율적인 자료구조"가 필요한가**: 문서가 수만~수백만 개가 되면 매 질의마다 전체를 순회하며 cosine similarity를 계산하는 것(W9 방식)은 느려진다 — 이것이 내일(Day4) FAISS 인덱스를 다루는 이유.
- **막히면**: "langchain embeddings vectorstore abstraction" 검색. 개념이 안 잡히면 W9 Day3 코드를 다시 열어 "이 함수가 프레임워크에서는 어떤 이름으로 바뀌는지" 대응시켜보기 → 30분 넘으면 다음으로.

### 00:45–01:30 | 구현: 임베딩 + FAISS 벡터스토어에 저장

- **학습 목표**: 청킹된 문서를 임베딩하여 FAISS 기반 벡터스토어에 저장하고, 저장된 내용을 다시 불러올 수 있는 상태(persist)로 만든다.
- **구현 과제 (스스로 작성)**:
  ```python
  # week10/day3_embed_and_store.py
  # 요구사항:
  # - HuggingFaceEmbeddings(sentence-transformers 기반, W9에서 쓴 모델과 동일 계열)로 임베딩 모델을 준비한다.
  # - Day2에서 만든 청크들을 LangChain FAISS 벡터스토어(FAISS.from_documents 등)에 저장한다.
  # - 벡터스토어를 로컬 디스크에 저장(save_local)하고, 다시 로드(load_local)해서 내용이 유지되는지 확인한다.

  def build_vectorstore(chunks: list, embedding_model) -> "FAISS":
      ...

  def save_and_reload(vectorstore, path: str) -> "FAISS":
      ...  # 저장 후 다시 로드한 벡터스토어 반환
  ```
- **확인 질문**:
  - 벡터스토어를 저장할 때 실제로 디스크에 어떤 파일들이 생기는가? (인덱스 파일 + 메타데이터 파일 구성을 직접 확인)
  - 임베딩 모델을 바꾸면(예: 다른 차원의 모델) 기존에 저장된 벡터스토어를 그대로 재사용할 수 있는가? 왜 안 되는가?

**막히면**: 저장/로드 시 `allow_dangerous_deserialization` 관련 경고가 뜨면 공식 문서에서 의미를 확인(로컬에서 직접 만든 파일이므로 이번 주 실습 범위에선 허용 가능). 임베딩 차원이 안 맞는다는 에러가 나면 모델을 저장 시점과 로드 시점에 동일하게 유지했는지 확인. 30분 넘으면 저장/로드 확인 없이 다음으로.

```bash
git add week10/day3_embed_and_store.py
git commit -m "W10 Day3: 임베딩 + FAISS 벡터스토어 저장/로드"
```

---

## Day 4 (목요일) — FAISS 저수준 다루기 + 알고리즘 [1.5시간]

### 00:00–00:45 | 개념·구현: FAISS 인덱스를 직접 다뤄보기

- **학습 목표**: LangChain 추상화 없이 FAISS 라이브러리를 직접 호출해, 인덱스 생성·벡터 추가·검색이 내부적으로 어떻게 이뤄지는지 체감한다.
- **핵심 개념**:
  - **인덱스(Index)**: 벡터를 저장하고 빠르게 검색하기 위한 자료구조. 오늘 다룰 `IndexFlatL2`는 가장 단순한 형태로, 사실상 전수 탐색(brute-force)에 가깝다 — "빠른 근사 검색"은 다른 인덱스 타입(예: IVF, HNSW)에서 다루는 내용이며 이번 주는 기본형만.
  - **`IndexFlatL2`가 왜 여전히 유용한가**: 문서 수가 적당히 적을 때는 근사 없이 정확한 결과를 주기 때문에 "정답 비교 기준(baseline)"으로 자주 쓰인다.
  - **차원(dimension) 일치**: 인덱스를 만들 때 지정한 차원과 실제로 넣는 벡터의 차원이 반드시 같아야 한다 — W9에서 다룬 임베딩 차원 개념과 직결.
- **구현 과제 (스스로 작성)**:
  ```python
  # week10/day4_faiss_lowlevel.py
  # 요구사항:
  # - Day3에서 쓴 것과 같은 임베딩 모델로 청크들을 numpy 배열(float32)로 임베딩한다.
  # - faiss.IndexFlatL2(dimension)로 인덱스를 직접 생성하고 index.add(...)로 벡터를 추가한다.
  # - 질의 문장 3개를 임베딩해 index.search(...)로 top_k(예: 3)개의 인덱스와 거리(distance)를 얻는다.
  # - 반환된 인덱스를 원래 청크 텍스트에 매핑해 사람이 읽을 수 있는 결과로 출력한다.

  def build_faiss_index(chunk_vectors) -> "faiss.Index":
      ...

  def search_index(index, query_vector, top_k: int = 3):
      ...  # (인덱스 배열, 거리 배열) 반환
  ```
- **확인 질문**:
  - `index.search`가 반환하는 "거리"는 L2 거리인데, Day3에서 쓴 cosine similarity와 순위가 항상 같을까? 다를 수 있는 경우를 하나 생각해보기.
  - Day3의 LangChain FAISS 결과와 오늘 저수준으로 만든 결과가 같은 질의에 대해 같은 순위를 주는가? 다르다면 어떤 전처리(정규화 등) 차이 때문일지 추측해보기.

**막히면**: "faiss IndexFlatL2 tutorial python" 검색. 차원 불일치 에러가 나면 임베딩 벡터의 shape을 print로 직접 확인. 30분 넘으면 검색 결과 출력 없이 인덱스 생성까지만 하고 다음으로.

### 00:45–01:30 | 알고리즘: Design Circular Queue, Fibonacci Number

```
- [ ] Design Circular Queue (Medium)
- [ ] Fibonacci Number (Easy)
```

- **학습 목표**: `Design Circular Queue`에서 고정 크기 배열로 큐를 구현할 때 head/tail 포인터를 모듈로(%) 연산으로 순환시키는 원리를, `Fibonacci Number`에서 재귀·메모이제이션·반복문 방식의 시간복잡도 차이를 설명할 수 있다.
- **구현 과제 (스스로 작성, 함수 시그니처만)**:
  ```python
  # week10/leetcode_day4.py
  # Design Circular Queue: 고정 크기 배열 기반 원형 큐. 가득 찼는지/비었는지 판별 로직 포함.
  class MyCircularQueue:
      def __init__(self, k: int):
          ...
      def enQueue(self, value: int) -> bool:
          ...
      def deQueue(self) -> bool:
          ...
      def Front(self) -> int:
          ...
      def Rear(self) -> int:
          ...
      def isEmpty(self) -> bool:
          ...
      def isFull(self) -> bool:
          ...

  # Fibonacci Number: n번째 피보나치 수를 반환한다. 재귀(메모이제이션 없이) 버전과
  # 반복문(O(1) 공간) 버전을 각각 따로 구현해 실행 시간을 비교한다.
  def fib_recursive(n: int) -> int:
      ...

  def fib_iterative(n: int) -> int:
      ...
  ```
- **확인 질문**:
  - 원형 큐에서 `isFull`과 `isEmpty`를 포인터 위치만으로 구분하기 어려운 이유는? (힌트: head==tail이 두 경우 모두에서 나타날 수 있음 — 이를 어떻게 구분했는가?)
  - `fib_recursive(30)`과 `fib_iterative(30)`의 실행 시간을 직접 재보았는가? 왜 이렇게 차이가 나는지 재귀 호출 트리 관점에서 설명해보기.
  - 오늘 오전에 다룬 FAISS의 "인덱스"와 이 문제의 "인덱스(배열 위치)"는 이름은 같지만 역할이 다르다 — 각각 무엇을 가리키는지 한 문장씩 정리해보기.

**막히면**: "circular queue array implementation" 또는 "fibonacci memoization vs recursion time complexity" 검색 → 에디토리얼의 접근 방식(전략)만 읽고 코드는 직접 짜기. 30분 넘으면 다음으로.

```bash
git add week10/
git commit -m "W10 Day4: FAISS 저수준 실습 + Design Circular Queue, Fibonacci Number"
```

---

## Day 5 (금요일) — 쿼리 파이프라인 완성 + Transformer 논문 착수 [1.5시간]

### 00:00–01:00 | 구현: RAG 미니 파이프라인 완성 (로드→청킹→임베딩→저장→쿼리)

- **학습 목표**: 지금까지 하루씩 나눠 만든 조각(로더, 스플리터, 임베딩+벡터스토어, FAISS 검색)을 하나의 흐름으로 연결해, "질문을 넣으면 관련 청크 top_k가 나오는" 미니 RAG 검색 파이프라인을 완성한다. (LLM 답변 생성까지는 이번 주 범위 밖 — 검색까지만.)
- **구현 과제 (스스로 작성)**:
  ```python
  # week10/day5_rag_pipeline.py
  # 요구사항:
  # - Day1~3의 함수(load_source_documents, split_documents, build_vectorstore)를 이어붙여
  #   하나의 파이프라인 함수로 만든다.
  # - 사용자 질의(query: str)를 받아 벡터스토어에서 top_k 청크를 검색해 반환하는 함수를 작성한다.
  # - 질의 3개를 준비해 각각 어떤 청크가 검색되는지 확인하고, 그 청크가 실제로 질문과 관련 있는지 육안으로 판정한다.

  def build_rag_pipeline(source: str, chunk_size: int, chunk_overlap: int):
      ...  # 벡터스토어(retriever로 사용 가능한 객체) 반환

  def retrieve(vectorstore, query: str, top_k: int = 3) -> list:
      ...  # 관련 청크 리스트 반환
  ```
- **확인 질문**:
  - 검색된 청크가 질문과 무관해 보이는 사례가 있었는가? 있었다면 원인이 청킹(경계가 어색함)인지, 임베딩(의미 포착 실패)인지 구분해볼 수 있는가?
  - 이 파이프라인에 LLM 생성 단계를 추가한다면(다음에 시도해볼 것), 검색된 청크들을 프롬프트에 어떻게 넣어야 할지 한 문단으로 설계해보기.

**막히면**: 파이프라인 연결에서 타입이 안 맞으면(예: str 리스트 vs Document 리스트) 각 함수의 입출력 타입을 다시 점검. 30분 넘으면 질의 1개만으로 동작 확인 후 다음으로.

### 01:00–01:30 | 복습 + Transformer 논문 1차 통독 시작

- **학습 목표**: 이번 주 실습 내용을 개념 단위로 정리하고, 주말에 완독할 Vaswani et al. (2017) 논문의 Abstract·Introduction을 먼저 읽어 전체 흐름을 파악한다.
- **할 일**:
  ```
  스스로에게 물어볼 것 (답이 안 나오면 해당 Day로 돌아가 다시 확인):

  □ RAG 파이프라인의 5단계는 각각 무엇을 하는가?
  □ 청크 크기를 정할 때 고려해야 할 트레이드오프는?
  □ LangChain의 VectorStore 추상화가 대신해주는 일은 무엇인가 (W9 코드 기준)?
  □ FAISS IndexFlatL2와 근사 인덱스(IVF 등)의 차이는 무엇일까 (아직 몰라도 가설만)?
  ```
  ```
  Vaswani et al. (2017) Attention Is All You Need — Abstract + Introduction만 읽고
  아래를 메모장에 한 줄씩 적기:
  - 이 논문이 RNN/CNN을 완전히 배제하고 Attention만으로 구성한 이유(저자 주장)는?
  - W9에서 본 Figure 1이 이 Introduction의 어떤 주장과 연결되는가?
  ```

**막히면**: 논문 문장이 어려우면 문단을 통째로 검색("what does this Attention Is All You Need paragraph mean")해 쉬운 설명부터 찾기. 30분 넘으면 다음 섹션 없이 주말로 넘어가기.

```bash
git add week10/
git commit -m "W10 완료: RAG 미니 파이프라인 완성 + FAISS 저수준 실습 + Transformer 논문 착수"

cat >> week10/README.md << 'EOF'

## W10 완료 항목
- [ ] Document Loader → Text Splitter → Embeddings → FAISS VectorStore → 검색, 5단계 연결
- [ ] chunk_size/overlap 비교 실험 + 근거 있는 선택
- [ ] FAISS 저수준(IndexFlatL2) 인덱스 생성·검색 직접 구현
- [ ] LeetCode: Design Circular Queue, Fibonacci Number
- [ ] Vaswani et al. (2017) Abstract·Introduction 1차 통독

## 이번 주 확인 사항
- [ ] "청킹이 왜 필요한가"를 W9 임베딩=차원축소 관점과 연결해 설명 가능
- [ ] LangChain 추상화(Embeddings/VectorStore)와 W9에 직접 짠 함수의 대응 관계 설명 가능
EOF
```

---

## 주말 — 심화 [4.5시간]

### 토요일 [2.5시간]

**[00:00–01:30] Transformer 논문 완독 — Figure 1 + Multi-Head Attention 수식**

- **학습 목표**: Vaswani et al. (2017)을 완독하고, Figure 1의 인코더-디코더 구조와 Multi-Head Attention의 계산 과정을 수식 단위로 손으로 설명할 수 있다.
- **핵심 개념**:
  - **Scaled Dot-Product Attention**: Query·Key·Value 세 행렬을 사용해 "이 위치가 다른 모든 위치를 얼마나 참고해야 하는가"를 계산하는 방식. 왜 QK^T를 sqrt(d_k)로 나누는가 — 차원이 커질수록 내적 값이 커져 softmax가 극단적으로(한쪽에 몰리게) 치우치는 것을 방지하기 위함.
  - **Multi-Head**: 하나의 Attention을 한 번만 계산하지 않고, Q·K·V를 여러 개의 더 작은 부분공간으로 나눠 병렬로 여러 번(head) 계산한 뒤 이어붙이는(concat) 이유 — 서로 다른 종류의 관계(문법적 관계, 의미적 관계 등)를 각 head가 다르게 포착할 여지를 주기 위함.
  - **Self-Attention vs 인코더-디코더 Attention**: 인코더 내부의 Self-Attention(자기 자신의 다른 위치를 참고), 디코더의 Masked Self-Attention(미래 위치를 가리는 이유), 인코더-디코더 Attention(디코더가 인코더 출력을 참고) — 이 세 가지가 Figure 1에서 각각 어디에 위치하는지 구분.
  - **Positional Encoding**: Attention 자체는 순서 정보가 없다는 것(왜 없는지 — 모든 위치를 동일하게 취급하는 연산이기 때문). 그래서 사인·코사인 함수로 위치 정보를 임베딩에 더해준다는 것.
- **할 일**:
  ```
  1. Vaswani et al. (2017) 전체를 통독 (섹션 3 "Model Architecture"에 가장 많은 시간 투입)
  2. Figure 1을 보며 아래를 종이에 직접 그려보기(코드 없이, 손으로):
     - 인코더 블록 1개 안에 어떤 서브레이어가 몇 개, 어떤 순서로 있는가
     - 디코더 블록 1개 안에 어떤 서브레이어가 몇 개, 어떤 순서로 있는가 (인코더와 다른 점은?)
  3. Scaled Dot-Product Attention 수식(softmax(QK^T/√d_k)V)을 종이에 옮겨 적고,
     각 기호(Q, K, V, d_k)가 실제로 어떤 shape의 행렬인지 직접 적어보기
  ```
- **확인 질문**:
  - sqrt(d_k)로 나누지 않으면 구체적으로 어떤 문제가 생기는가? (softmax 출력이 어떻게 변하는지 기준으로 설명)
  - 디코더의 Masked Self-Attention에서 "마스킹"이 하는 역할은 정확히 무엇인가? 왜 학습(training) 시에도 이 마스킹이 필요한가?
  - W9에서 본 Bahdanau Attention과 오늘의 Self-Attention은 "무엇을, 무엇으로부터 참고하는가"라는 관점에서 어떻게 다른가?

**[01:30–02:30] RAG 파이프라인 확장 — 질의 실패 사례 분석**

- **학습 목표**: 금요일에 완성한 미니 RAG 파이프라인에 문서를 더 추가하고, 검색이 실패하는(관련 없는 청크가 상위에 오는) 사례를 의도적으로 찾아 원인을 분석한다.
- **구현 과제 (스스로 작성)**:
  ```python
  # week10/day6_rag_failure_analysis.py
  # 요구사항:
  # - 금요일 파이프라인에 서로 다른 주제의 문서를 3~5개 더 추가한다(카테고리를 다양하게).
  # - 일부러 "애매한" 질의(여러 카테고리에 걸칠 수 있는 문장)를 5개 만들어 검색 결과를 확인한다.
  # - 검색 실패로 보이는 사례를 최소 2개 기록하고, 원인을 (a) 청킹 문제 (b) 임베딩 한계 (c) 질의 자체의 모호함 중 하나로 분류한다.
  ```
- **확인 질문**:
  - 실패 사례 중 chunk_size를 바꾸면 해결될 것 같은 사례가 있는가?
  - top_k를 늘리면(예: 3→10) 실패로 보였던 사례가 상위권에 다시 나타나는가? 이것이 시사하는 바는?

**막히면**: "RAG retrieval failure analysis" 검색으로 일반적인 실패 유형(청킹, 임베딩 모델 한계, 질의 모호성)부터 훑어보기. 논문이 너무 어려우면 "attention is all you need paper explained" 영상으로 구조를 먼저 잡고 다시 원문으로 돌아가기.

### 일요일 [2시간]

**[00:00–01:00] 주간 회고**

```markdown
## W10 회고 (일요일에 작성)

### 달성한 것
- [ ] RAG 파이프라인 5단계(로드→청킹→임베딩→저장→쿼리) 연결
- [ ] FAISS 저수준 인덱스 실습
- [ ] Vaswani et al. (2017) 완독 + Figure 1/수식 정리
- [ ] LeetCode 2문제
- [ ] RAG 검색 실패 사례 분석

### 예상보다 오래 걸린 것
(솔직하게 적기)

### W11에 가져갈 것
(이해 못 하고 넘어간 것 — 특히 Multi-Head Attention 수식 중 아직 헷갈리는 부분)

### 다음 주 첫 번째 할 일
(W11 Day1 무엇부터 시작할지)
```

**[01:00–02:00] W11 예습 — Qdrant·BM25·GitHub Actions 감 잡기**

- **학습 목표**: W11에서 다룰 "Qdrant(Docker 벡터DB) / BM25(키워드 검색) / GitHub Actions(CI)"가 각각 무엇을 위한 도구인지, 왜 이번 주 FAISS만으로는 부족한 부분이 있는지 미리 감을 잡는다.
- **할 일**:
  ```
  1. Qdrant 공식 홈페이지의 "Quickstart"(Docker 실행 부분만) 훑어보기 (10분)
     - FAISS(로컬 파일 기반)와 Qdrant(서버 기반 벡터DB)의 차이를 한 줄로 정리
  2. BM25가 무엇인지 한 문단으로 검색해서 정리
     - 이번 주 내내 다룬 "의미 기반(임베딩) 검색"과 BM25 "키워드 기반 검색"의 차이는?
     - 왜 실무에서는 이 둘을 결합(hybrid search)하는 경우가 많을지 가설 세워보기
  3. GitHub Actions가 "코드를 push할 때마다 자동으로 무엇을 해주는지" 한 문단으로 정리
  ```
- **확인 질문**:
  - 이번 주 만든 RAG 검색기에서 "정확한 고유명사 매칭"이 중요한 질의라면, 임베딩 기반 검색만으로 충분할까? (힌트: BM25가 강점을 갖는 지점)
  - FAISS를 로컬 파일로 저장하는 방식과 Qdrant처럼 서버로 띄우는 방식, 각각 어떤 상황(팀 협업, 배포 규모)에 더 적합할지 추측해보기.

---

## 막힐 때 대응 가이드

개념이 막히면:

```
1단계 (5분): 구글에 영어로 검색
  예: "why chunk overlap in RAG" / "multi-head attention explained"

2단계 (10분): 관련 영상 검색 (StatQuest, 3Blue1Brown, freeCodeCamp 등)

3단계 (20분): 공식 문서 확인
  LangChain: https://python.langchain.com/docs
  LlamaIndex: https://docs.llamaindex.ai
  FAISS: https://github.com/facebookresearch/faiss/wiki

30분 넘어도 해결 안 되면:
→ 메모장에 "아직 모름: [개념]" 적고 다음으로 넘어가기
→ 이후 주차(특히 블록 C의 Transformer 본격 학습 시)에서 다시 만날 때 해결
→ AI 커리큘럼에서 막히는 것은 실력 부족이 아니라 정상 과정
```

코드가 막히면:

```
에러 메시지 전체를 복사 → 구글에 붙여넣기
Stack Overflow 답변 중 가장 많은 추천을 받은 것 선택

자주 나오는 에러:
- ModuleNotFoundError: pip install [라이브러리명]
- FAISS 차원 불일치 에러: 임베딩 모델이 저장 시점과 검색 시점에 동일한지 확인
- 벡터스토어 로드 시 deserialization 경고: 로컬에서 직접 만든 파일이면 이번 주 범위에선 허용 가능
- 청크가 비어있거나 이상함: separator·chunk_size 설정 재확인
```

---

## W10 완료 기준

일요일 저녁에 아래를 할 수 있으면 W10 성공:

```
□ week10/ 폴더에 로더·청킹·임베딩·FAISS·RAG 파이프라인 코드가 있다
□ 질의를 넣으면 관련 청크 top_k가 나오는 미니 RAG 검색 파이프라인이 동작한다
□ chunk_size/overlap을 최소 2가지 비교해보고 근거 있는 선택을 했다
□ FAISS IndexFlatL2를 프레임워크 없이 직접 생성·검색해봤다
□ Vaswani et al. (2017)의 Figure 1 구조와 Multi-Head Attention 수식을 손으로 설명할 수 있다
□ LeetCode Design Circular Queue, Fibonacci Number를 이해하고 혼자 다시 풀 수 있다

절반(3개 이상) 달성하면 W11로 진행.
전부 못 해도 W11로 진행 — Transformer 수식의 세부(특히 Multi-Head의 병렬 계산 구현)는 블록 C에서 본격적으로 다시 다룹니다.
```

---

## W11 첫 할 일 미리 보기

W11 Day1에 열어야 할 것:

```
1. `week11/` 폴더 생성
2. Qdrant Docker 로컬 설치 → 첫 컬렉션 생성 + W10의 청크 데이터를 옮겨 검색해보기
3. rank_bm25 라이브러리로 BM25 검색 구현 → W10 임베딩 검색과 결과 비교
4. 막히면 → "FAISS와 Qdrant 차이" → 이번 주(W10 일요일) 예습 메모 다시 열어보기
5. 여유가 있으면 GitHub Actions로 pytest 자동 실행 파이프라인 첫 시도
```

---

*이 계획대로 완벽하게 안 돼도 됩니다.
W10의 진짜 목표는 "RAG가 신비한 마법이 아니라 다섯 개의 명확한 단계가 이어붙은 것"이라는 감각을 얻고, Transformer 논문을 완독해 다음 학습(블록 C의 본격적인 구현)을 위한 지도를 완성하는 것입니다.
수식의 모든 세부가 아직 손에 안 익어도 괜찮습니다 — 블록 C에서 직접 구현하며 다시 다지게 됩니다.*

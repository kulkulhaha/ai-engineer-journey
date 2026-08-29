# 🗓️ AI 엔지니어 취업 커리큘럼 v9 (W1 ~ W68+)

> **목표**: Sheffield 프리마스터(1.5학기) → MSc AI → 영국 AI/ML 엔지니어 취업 + Skilled Worker 비자 전환
> **포지셔닝**: LLM Engineer / ML Engineer (메인) + 경력 기반 mid-level 병행 트랙
> **v9 수정사항 (수학·ML 개념 커버리지 보강 — 2026-08-27 점검 후속)**:
> 1. **비지도학습 기초 추가** — K-means 클러스터링·이상탐지·추천시스템(협업 필터링) 개념을 B①에 실습으로, 이론을 이론 심화 보강 트랙에 추가. 기존에는 PCA(차원축소) 외 비지도학습이 전무했고, B⑨에서 "추천 시스템 설계"를 토론하면서도 그 기반 알고리즘을 배운 적이 없는 모순이 있었음.
> 2. **모델 해석·공정성(Explainability & Fairness) 추가** — SHAP/LIME과 공정성 지표를 B②(Kaggle XGBoost 즈음)에 실습·이론으로 추가. P1이 EU AI Act 등 AI 규제를 다루는 프로젝트인데 정작 설명가능성 기법을 다룬 적이 없었던 공백을 메움. W54(P1 착수)에 이 개념을 프로젝트 서사에 연결하는 메모 추가.
> 3. **RL 기초 추가** — MDP·Q-learning·정책반복 개념을 B⑦–⑧ 사이(RLHF 이론 직전)에 반나절 분량으로 추가. 기존엔 RLHF(PPO/DPO)만 다루고 그 배경이 되는 RL 기본기가 없어 "정책 그래디언트가 왜 그렇게 생겼는지"가 붕 뜬 채로 들어갈 위험이 있었음.
> 4. **평가 방법론 보강** — PR 커브(불균형 데이터에서 ROC보다 적합), 회귀 평가지표(MSE·MAE·R²), k-fold 교차검증을 각각 W13(복습 구간)·W6(회귀 최소 보장)·B②(Kaggle 모델 튜닝)에 추가.
> **v8 수정사항 (실제 진행 상황 대조 — 격주 점검 2026-08-27)**:
> 1. **실제 실습 시작일과 계획 시작일의 괴리 반영** — 커리큘럼상 블록 A 시작은 2026.7이지만, 레포(`ai-engineer-journey`) git log 기준 실질적 실습 커밋은 2026.8.14부터 (2026.7.1 "Initial commit"·"W1: initialize week01" 이후 약 6주 공백). "실제 타임라인 앵커" 바로 아래에 실제 대조 섹션을 신설해, 이 공백이 블록 A의 W13–16 버퍼(4주)를 이미 넘어설 수 있음을 명시하고 W16 체크포인트에서 반드시 재확인하도록 함.
> 2. **비자·시장 수치 재확인 (변경 없음)** — Skilled Worker 최저연봉 £41,700(2025.7 발효), ISC £1,320/년(중대형 스폰서, 2025.12 발효), 영어 B2 요건(2026.1.8 발효), Graduate visa 18개월 단축(2027.1.1~ 학사/석사 적용) 등 v7에 이미 반영된 수치를 2026.8 기준 공개 정보와 대조한 결과 모두 유효함을 확인. UK AI 채용 시장은 2024→2026 사이 AI 관련 공고 1133% 증가로 강세가 지속되고 있고, "포트폴리오·실전 프로젝트를 formal credential보다 우선" 채용 경향은 이 커리큘럼의 Practice-First 원칙 및 P1–P3 프로젝트 축과 이미 부합 — 구조 변경 불필요.
> 3. **weekly_plans 시간 배분 점검 필요 표시** — w1–w3는 v2로 개정되어 v7의 완화된 주당 10–12h를 반영했지만, w4 이후 파일은 아직 v1(주당 15–16h, v7 이전 페이스)에 머물러 있음. W4 착수 전 v2 개정 권장 (weekly_plans 자체에서 처리할 사항이며 본 마스터 파일 수정 범위 밖).
> **v7 수정사항 (일정 여유 확보)**:
> 1. **시작을 2026년 7월로 앞당김** — 프리마스터(2028.3)까지 약 20개월(86주) 확보
> 2. **블록 A·B 기간 확장 + 주당 시간 완화** — 블록 A 12주→16주(주 10–12h), 블록 B 36주→68주(주 6–9h). 서두르지 않고 재활성화·프로젝트를 소화
> 3. **프리마스터 직전 완충 2주 신설** — 블록 C 진입 전 정비
> 4. **영어 B2·스폰서 리서치를 더 일찍·길게** 배치 (여유 활용)
> 5. **이론 심화 보강 트랙 신설** — B-1 여유 시간에 정보이론·정규화·RLHF 수학·분산 ML·고전 NLP 등을 실습에 곁들여 학습 (면접 "왜?" 대응 + MSc 모듈 예습)
> **v6 계승 (전략 재정렬)**:
> - 실제 날짜 앵커 / 비자 전략을 척추로 / 영어 B2 트랙 / 경력직 병행 트랙 / 블록 C P2 하향(깊이 P1 집중) / 자격증 격하 / 수학 최소 보장·완충 구간

---

## 📅 실제 타임라인 앵커 (v7 — 가장 먼저 봐야 할 표)

이 커리큘럼의 모든 주차(W)는 아래 실제 날짜에 고정됩니다. 시작을 2026년 7월로 앞당겨 프리마스터(2028년 3월)까지 약 20개월(86주)을 확보했습니다.

| 이정표 | 실제 시점 | 커리큘럼 위치 |
|-------|----------|--------------|
| **블록 A 시작** | **2026년 7월** | W1 |
| **블록 A 종료 (RAG v0)** | 2026년 10월말 | W16 |
| **블록 B 시작 (확장)** | 2026년 11월 | W17 |
| **프리마스터 직전 완충** | 2028년 2월 | W83–84 |
| **프리마스터 시작 (1.5학기, 3월 인타이크)** | **2028년 3월** | 블록 C 직전 |
| **프리마스터 종료 (7월 수업 종료)** | **2028년 7월** | — |
| **블록 C (석사 전 갭 집중)** | 2028년 7월–8월 | W49–60 (블록 C는 기존 주차 유지) |
| **MSc AI 시작 (9월 인타이크)** | **2028년 9월** | 블록 D 시작 (W61) |
| **MSc 졸업** | **2029년 9월경** | W68+ 이후 |
| **Graduate visa 신청** | 2029년 하반기 | → **18개월 그룹 확정** |
| **Skilled Worker 전환 데드라인** | 2031년 초 (Graduate visa 만료 전) | 최종 관문 |

> ℹ️ **주차 번호 안내**: 블록 A·B는 여유를 반영해 기간이 늘었지만, 프로젝트 연결 구조(RAG v0→P1→P2→P3)를 유지하기 위해 **블록 C·D의 주차 번호(W49–68+)와 내부 라벨은 v6 그대로** 둡니다. 즉 블록 A·B는 "달력상 더 길게, 주당 더 여유롭게" 진행하는 구간이고, 블록 C부터는 석사 전 갭에 압축 집중하는 구간이라고 이해하면 됩니다. 아래 블록 A·B 표의 주차는 **확장된 실제 진행 순서**를 나타냅니다.

> ⚠️ **핵심 인식 (v6에서 유지)**: MSc 졸업이 2029년이라 Graduate visa는 확실하게 18개월만 나옵니다. 이 18개월은 "여유"가 아니라 스폰서 오퍼를 잡아야 하는 압축된 카운트다운입니다. 이 커리큘럼의 진짜 목표는 **"졸업 전에 스폰서 오퍼를 확보하는 것"**이며, Graduate visa는 그게 실패했을 때의 백업 창입니다. 시작을 앞당긴 20개월 여유는 학습을 늘어지게 하라는 게 아니라, **기초를 탄탄히 다지고 영어 B2·스폰서 리서치를 일찍 끝내 취업 시즌에 온전히 집중**하기 위한 것입니다.

### 🔍 실제 진행 대조 (v8 신규 — 2026-08-27 점검)

레포 `ai-engineer-journey`의 git log와 project 문서 `progress-log.md`를 대조한 결과입니다.

| 항목 | 계획 | 실제 |
|------|------|------|
| 블록 A 착수 (첫 실질 실습 커밋) | 2026년 7월 | **2026년 8월 14일** — 2026.7.1에 "Initial commit"·"W1: initialize week01"만 있었고, 그 이후 실질적인 실습 커밋 없이 약 6주 공백 |
| W1→W3 실습 소화 속도 | 커리큘럼 1주 ≈ 실제 1주 | 2026.8.14–8.26 (약 2주) 사이에 W1–W3 체크포인트를 거의 다 소화 — **일단 시작한 뒤의 속도는 계획과 크게 다르지 않음** |

**해석**: 문제는 "속도"가 아니라 "시작 지연"입니다. 시작 이후 주당 계획 분량을 소화하는 속도는 준수한 편이라 블록 A 자체의 구조(주차 배분·주당 시간)를 다시 설계할 근거는 없습니다. 다만 6주의 시작 공백은 블록 A에 이미 마련해둔 W13–16 버퍼(4주)보다 크므로, 버퍼만으로 완전히 흡수되지 않을 가능성이 있습니다.

**대응 (구조 변경 대신 체크포인트 강화)**:
- W16(블록 A 종합 점검) 시점에 "실제 완료 날짜"를 계획 종료일(2026년 10월말)과 비교하세요.
- 실제 종료가 2026년 11월 중하순 이후로 밀린다면, 블록 B 착수(계획: 2026.11)를 그만큼 미루세요 — 블록 B(B-1 단독 구간)는 68주로 여유가 크고 프리마스터 시작(2028.3)까지 약 15개월의 완충이 있으므로, 몇 주의 시작 지연은 구조를 흔들지 않고 흡수됩니다.
- 단, 이 문서의 "⚠️ 긴 저강도 구간의 함정" 경고대로 **지연을 자동으로 흡수하지 말고 매번 명시적으로 재확인**하세요 — 이번 6주 공백처럼 조용히 쌓이는 지연이 가장 위험합니다.

---

## 🎯 비자 전략 (v6 신규 — 학습보다 이게 성패를 가른다)

> ⚠️ 아래는 최신 공개 정보 기반 참고용이며 법률 자문이 아닙니다. 실제 지원 시점(2028–2031)에 규칙이 또 바뀔 수 있으니, 지원 가까이에 요건을 재확인하고 OISC 등록 이민 전문가에게 검증받으세요.

### 승부처 재정의

기존 커리큘럼은 "MSc 졸업 → Graduate visa 18개월 → 그 안에 취업"을 안전판처럼 봤지만, 실제 병목은 다음입니다:

1. **Graduate visa 자체는 스폰서가 필요 없지만, 그 다음 Skilled Worker 전환에는 스폰서가 반드시 필요.** 18개월 안에 스폰서 라이선스 보유 기업의 오퍼를 못 잡으면 영국을 떠나야 함.
2. **Skilled Worker 문턱이 크게 상승** — 최저 연봉 £41,700 또는 직종 going rate 중 높은 쪽(2025.7 발효), 스킬 문턱 학위 수준(RQF 6)으로 상향.
3. **영어 B2 필수** — Skilled Worker 신청 시 CEFR B2 요구(2026.1.8 발효). 영어 수업 학위는 대체 가능하나 미리 확보가 안전.
4. **스폰서 비용 32% 상승** (Immigration Skills Charge 중대형 스폰서 연 £1,320) → 기업이 신입 외국인 스폰서를 더 꺼림.

### 유리한 요소 — 반드시 활용

- **New Entrant 할인**: 최근 졸업자/26세 미만/graduate training 프로그램은 직종 going rate의 70% 임계값 적용 가능 → £41,700 전액이 아닐 수 있음.
- **경력 자산**: "AI R&D Leadership 2+년, energy/data center 도메인" + 연구소장 경력. 이건 신입이 아니라 **경력 전환자**로 포지셔닝 가능한 강력한 차별화 요소. 스폰서 설득에도 유리.

### 3대 비자 액션 (학습과 병행, 최우선)

| 액션 | 시점 | 내용 |
|------|------|------|
| **① 스폰서 기업 타겟팅** | 블록 B부터 상시 | 스폰서 라이선스 보유 + AI/ML 신입·경력 스폰서 실적 기업만 리스트업 (gov.uk Register of licensed sponsors). 스폰서 안 하는 회사는 붙어도 무의미. |
| **② 영어 B2 사전 확보** | 블록 A~B 중 | IELTS 등으로 B2 이상 확보·유지. Skilled Worker 필수 요건. |
| **③ 이중 트랙 지원** | 블록 C~D | 인턴(신입 트랙) + 경력 기반 mid-level AI Engineer 직접 지원(경력 트랙) 병행. |

---

## 🧠 핵심 설계 원칙: Practice-First

**"잊어버린 전문가" 전략** — TU Berlin CES 학사로 선형대수·확률·미적분·알고리즘을 학부 수준까지 배웠지만 많이 잊은 상태입니다. 완전 초보와 달리 실습 문맥이 생기면 기억 재활성화 속도가 2–3배 빠릅니다.

```
매 주차 구성:
① 실습 먼저 (30–40%): 코드를 돌린다, 막히는 지점을 찾는다
② 이론 역추적 (30–40%): 막힌 지점의 수학·개념을 정확히 이해한다
③ 재구현 (20–30%): 이해한 이론으로 다시 짠다, 영어로 설명할 수 있게 된다
```

### 수학 최소 보장 원칙

"실습이 막히지 않으면 이론을 건너뛴다"는 위험을 방지합니다. 각 수학 영역마다 **실습 결과와 무관하게 반드시 확인해야 하는 최소 체크포인트**를 지정했어요. 면접에서 반드시 나오는 것들입니다.

| 수학 영역 | 최소 보장 체크포인트 (실습 막힘 여부 무관) |
|---------|----------------------------------------|
| 선형대수 | PCA를 고유값분해로 손계산 가능 / SVD와 PCA 차이 설명 / 행렬 곱의 기하학적 의미 |
| 확률·통계 | 베이즈 정리를 예시로 유도 / MLE를 로그우도로 유도 / 정규분포가 왜 자주 등장하는지 |
| 미적분 | 연쇄법칙을 역전파에 적용 / 그래디언트 방향이 왜 최솟값인지 / 볼록함수의 의미 |
| 최적화 | SGD·Adam 차이를 모멘텀 관점에서 설명 / 학습률이 너무 크면/작으면 어떻게 되는지 |

---

## 📐 4블록 구조 (v7 — 여유 반영)

| 블록 | 실제 시기 | 진행 주차 | 주당 시간 | 테마 |
|------|----------|----------|----------|------|
| **A. 재활성화 + 첫 프로젝트** | 2026.7–2026.10 (프리마스터 전) | W1–16 (12→16주) | **10–12h** (완화) | 실습으로 수학·ML 기억 재활성화 + 미니프로젝트 3개 + 영어 B2 착수 |
| **B. ML 심화 + 도구** | 2026.11–2028.2 (초기 단독 → 프리마스터 병행) | B①–B⑩ (36→68주 상당) | **6–9h** (완화) | ML 실전 + SQL + MLOps + 스폰서 기업 리서치 상시 + 영어 B2 완료 |
| **완충** | 2028.2 | W83–84 | 정비 | 블록 C 진입 전 복습·환경 점검·프리마스터 적응 |
| **C. DL+LLM+시그니처** | 2028.7–2028.8 (석사 전 갭 집중) | W49–60 (라벨 유지) | 20–25h | Transformer + P1 완성(집중) + P2 데모 + 취업 준비 + 경력직 지원 착수 |
| **D. MSc 취업 시즌** | 2028.9–2029.9 (MSc 재학) | W61–68+ | 10–15h | P3 + 이중 트랙 지원 + 면접 + 스폰서 오퍼 확보 |

> **여유를 이렇게 씁니다**: 블록 A는 4주 늘려 주당 부담을 15–18h→10–12h로 낮춥니다(재활성화를 서두르지 않음). 블록 B는 68주로 크게 늘려 주당 6–9h로 완화 — 프리마스터(2028.3~7)와 병행하는 구간에서 학업 부담이 겹쳐도 무리 없이 소화하고, 프리마스터 전 단독 구간(2026.11–2028.2)에서 Kaggle·SQL·MLOps를 여유롭게 끝냅니다. 블록 C는 석사 전 짧은 갭이라 기존대로 압축 집중합니다.

> ⚠️ **긴 저강도 구간의 함정**: 20개월은 자칫 늘어지기 쉽습니다. 주당 시간이 낮아도 **매주 최소 1커밋·1체크포인트**를 지키세요. "천천히 = 안 함"이 되지 않도록, 아래 각 구간 끝의 점검표를 실제 날짜에 캘린더 알림으로 걸어두길 권합니다.

### LeetCode 전략
- **W1–48**: 주 2–3문제. 감각 유지, 부담 없이.
- **W49–60**: 주 10–15문제 집중. 면접 시즌 직전 스퍼트.
- **W61+**: 회사 기출 위주 연습.

### 블록 B 지연 시 우선순위 (리스크 대비)
프리마스터가 예상보다 빡세거나 RAG v0(W12)이 W13–14까지 지연되면:

| 잘라도 됨 | 미룰 수 있음 (블록 C/D로) | 절대 자르면 안 됨 |
|---------|------------------------|----------------|
| Kaggle #2 완주 → 솔루션 분석만 | ML 시스템 디자인 → W48–49로 | SQL 2주 집중 |
| CS229 이론 보강 → 참고용만 | Python 고급 일부 → W49 병행 | Python 고급 asyncio·pytest |
| Kaggle #1 솔루션 분석 깊이 축소 | LLM 비용 개념 → W49 초반 | P1 아키텍처 설계 문서 |
| 자격증 공부 (전면 보류 가능) | — | **스폰서 기업 리서치·영어 B2** |

> **프리마스터 성적이 최우선.** 프리마스터 불합격/저조 = MSc 진학 자체가 무산. v7에서는 대부분의 학습을 프리마스터 전(2026.11–2028.2)에 끝내므로 병행 부담이 크게 줄었습니다. 그래도 학습 커리큘럼은 언제든 압축 가능하지만 프리마스터 성적과 비자 액션(스폰서 리서치·B2)은 사수.

**RAG v0 지연 시 구체적 대응**
v7에서는 블록 A가 16주로 늘고 프리마스터와 겹치지 않으므로 여유가 큽니다. W12 RAG v0이 밀리면 W13–16 확장 구간에서 안정화하면 됩니다(이미 W16에 RAG v0 안정화 배치). 블록 B 진입이 늦어져도 B-1 단독 구간(약 68주)이 충분히 흡수합니다.

---

## 🟦 블록 A — 재활성화 + 첫 프로젝트 (W1–16, 주 10–12h, 2026.7–2026.10)

*핵심: 실습 먼저 → 막힌 지점에서 이론 역추적 → 재구현*
*단, 수학 최소 보장 체크포인트는 막히든 안 막히든 반드시 확인*
*⭐ 병행: 영어 B2 확보 착수 (Skilled Worker 필수 요건) — 주 2–3h 별도*
*v7: 12주→16주로 늘려 주당 부담 완화. 아래 표의 W1–12 뒤에 복습·심화 4주(W13–16)를 붙임.*

| 주차 | 🛠️ 실습 먼저 + 이론 역추적 | 💻 알고리즘 | 📄 논문 |
|------|--------------------------|-----------|--------|
| **W1** | **[실습]** sklearn으로 PCA 돌려보기 (MNIST 또는 공개 데이터)<br>**[막히면]** "왜 PCA가 고유벡터인가?" → 3Blue1Brown 선형대수 1–8화, MIT 18.06 Lec 1–3<br>**[재구현]** numpy로 행렬 곱·전치·역행렬 직접 구현<br>**[환경]** Colab·GitHub·Git 기본 세팅<br>**[⭐영어]** IELTS 진단 모의고사 1회 → 현재 수준·목표 갭 파악<br>**[✅ 최소 보장]** 행렬 곱의 기하학적 의미 (회전·스케일링) 설명 가능한가? | - [ ] Two Sum (Easy)<br>- [ ] Best Time to Buy and Sell Stock (Easy) | **AI의 시작**<br>Turing (1950). *Computing Machinery and Intelligence.*<br>→ 맥락만. "기계가 생각할 수 있는가." |
| **W2** | **[실습]** numpy로 PCA 처음부터 구현 시도<br>**[막히면]** LU분해·영공간 → MIT 18.06 Lec 4–9, 3Blue1Brown 9–15화<br>**[재구현]** 공분산 → 고유벡터 → 투영 전 과정<br>**[Python]** comprehension·함수형·예외처리<br>**[✅ 최소 보장]** PCA를 고유값분해로 손계산 / SVD와 PCA 차이 설명 | - [ ] Valid Palindrome (Easy)<br>- [ ] Reverse Linked List (Easy) | **역전파의 탄생**<br>Rumelhart et al. (1986). *Learning by back-propagating errors.*<br>→ Abstract + 핵심 아이디어만. |
| **W3** | **[실습]** sklearn LogisticRegression + 혼동행렬·ROC 실습<br>**[막히면]** "로그우도가 왜 손실함수인가?" → Stat 110 Lec 1–6, 베이즈 정리<br>**[재구현]** numpy로 로지스틱 회귀 MLE 관점 구현<br>**[Python OOP]** 클래스·상속·decorator로 모델 래퍼 작성<br>**[✅ 최소 보장]** 베이즈 정리를 예시로 유도 / MLE를 로그우도로 유도 | - [ ] Valid Parentheses (Easy)<br>- [ ] Min Stack (Medium) | **CNN의 탄생**<br>LeCun et al. (1998). *LeNet.*<br>→ 왜 fully-connected가 이미지에 비효율적인가. |
| **W4** | **[실습]** OpenAI/Anthropic API 첫 호출<br>**[미니프로젝트 #1]** 🏗️ **LLM 텍스트 요약·분류기 CLI**<br>- 공개 뉴스 RSS → API로 요약 + 카테고리 분류<br>- GitHub 커밋, README 작성<br>**[막히면]** SVD·차원축소 → MIT 18.06 해당 파트<br>**[도구]** Git branch·PR 워크플로우<br>**[✅ 최소 보장]** SVD와 PCA 차이를 행렬 분해 관점에서 설명 | - [ ] Group Anagrams (Medium)<br>- [ ] Contains Duplicate (Easy) | **딥러닝 부활**<br>Krizhevsky et al. (2012). *AlexNet.*<br>→ 왜 GPU인가, 왜 ReLU인가. |
| **W5** | **[실습]** sklearn RandomForest·XGBoost 비교 실험<br>**[막히면]** "정보이득이 왜 엔트로피인가?" → Stat 110 Lec 7–13<br>**[실습]** matplotlib으로 분포 시각화, CLT 시뮬레이션<br>**[재구현]** 결정트리 분류 기준을 numpy로 계산<br>**[✅ 최소 보장]** 정규분포가 왜 자주 등장하는지 (CLT) 설명 | - [ ] Binary Search (Easy)<br>- [ ] Search in Rotated Sorted Array (Medium) | **ResNet**<br>He et al. (2016). *Deep Residual Learning.*<br>→ 왜 깊을수록 나빠지는가, skip connection의 역할. |
| **W6** | **[실습]** Pandas로 공개 시계열 데이터 EDA<br>**[막히면]** "공분산이 왜 중요한가?" → Stat 110 Lec 14–20<br>**[재구현]** numpy로 선형회귀 정규방정식 + MLE 각각 구현<br>**[미니프로젝트 #2]** 🏗️ **시계열 EDA 자동 리포트**: Pandas + matplotlib → GitHub<br>**[✅ 최소 보장]** MLE를 로그우도 최대화로 손유도 / **(v9 추가)** 회귀 평가지표 MSE·MAE·R²을 각각 언제 쓰는지, 이상치에 대한 민감도 차이 설명 가능한가 | - [ ] Container With Most Water (Medium)<br>- [ ] Move Zeroes (Easy) | **LSTM**<br>Hochreiter & Schmidhuber (1997).<br>→ Vanishing gradient + 게이트 구조. |
| **W7** | **[실습]** numpy로 SGD 직접 구현 → 학습 곡선 시각화<br>**[막히면]** "그래디언트가 왜 이 방향인가?" → 다변수 미적분 (편미분·연쇄법칙)<br>**[재구현]** 선형·로지스틱 회귀를 SGD로 처음부터 학습<br>**[✅ 최소 보장]** 연쇄법칙을 역전파에 적용 / SGD·Adam 차이를 모멘텀 관점 설명 | - [ ] Longest Substring Without Repeating (Medium)<br>- [ ] Minimum Size Subarray Sum (Medium) | **Word2Vec**<br>Mikolov et al. (2013).<br>→ 의미 연산 (king - man + woman = queen). |
| **W8** | **[실습]** FastAPI로 W3 모델 서빙 API 구축<br>**[실습]** Docker로 API 컨테이너화 → 로컬 배포<br>**[막히면]** Big-O 분석 → CLRS 핵심 챕터<br>**[⭐영어]** IELTS 스피킹·라이팅 주 1회 루틴 시작<br>**[✅ 최소 보장]** 학습률이 너무 크면/작으면 / 볼록함수의 의미<br>**[🔍 블록 A 수학 중간 점검]**: 4개 영역 최소 보장 항목 전부 설명 가능한가? | - [ ] Climbing Stairs (Easy)<br>- [ ] Merge Intervals (Medium) | **Seq2Seq**<br>Sutskever et al. (2014).<br>→ 인코더-디코더 구조. |
| **W9** | **[실습]** HuggingFace pipeline으로 감성분석·텍스트분류 체험<br>**[막히면]** "Transformer가 왜 이렇게 생겼나?" → Vaswani 2017 Figure 1<br>**[실습]** Sentence-BERT로 문장 유사도 + 간단한 의미 검색<br>**[Python 고급]** asyncio 기초 + 타입 힌팅 시작 | - [ ] Sort Colors (Medium)<br>- [ ] Implement Queue using Stacks (Easy) | **Attention**<br>Bahdanau et al. (2014).<br>→ 왜 고정 벡터가 병목인가. Transformer의 전신. |
| **W10** | **[실습]** LangChain 또는 LlamaIndex 튜토리얼 완주<br>- 문서 로드 → 청킹 → 임베딩 → 벡터DB → 쿼리<br>**[막히면]** "임베딩이 왜 차원축소인가?" → W2 PCA 연결<br>**[실습]** FAISS 기본 (인덱싱·검색) | - [ ] Design Circular Queue (Medium)<br>- [ ] Fibonacci Number (Easy) | **Transformer**<br>Vaswani et al. (2017). *Attention Is All You Need.*<br>→ 완독 권장. Figure 1 + Multi-head Attention 수식 이해. |
| **W11** | **[실습]** Qdrant Docker 로컬 설치 + 첫 컬렉션 생성·검색<br>**[실습]** rank_bm25 라이브러리로 BM25 검색 구현<br>**[실습]** GitHub Actions 간단한 CI 파이프라인 (pytest 자동 실행)<br>**[도구]** AWS/GCP 무료티어 VM + 간단한 배포<br>**[⭐비자]** gov.uk Register of licensed sponsors 첫 탐색 (AI/ML 스폰서 기업 감 잡기) | - [ ] Invert Binary Tree (Easy)<br>- [ ] Max Depth of Binary Tree (Easy) | **ML 기술 부채**<br>Sculley et al. (2015). *Hidden Technical Debt in ML Systems.*<br>→ 완독 권장. Production ML의 함정. |
| **W12** | **[미니프로젝트 #3]** 🏗️ **RAG v0 — 간단한 QA 시스템**<br>- 공개 문서 (Wikipedia 또는 arXiv) → LangChain + FAISS + LLM API<br>- FastAPI 비동기 엔드포인트 + Docker 배포<br>- GitHub README + HuggingFace Spaces 데모<br>**⚠️ 지연 허용**: W12에 완성 못 하면 W13–14까지 연장 가능 (프리마스터 시작 주와 겹치니 프리마스터 우선)<br>**[🔍 블록 A 종합 점검]** | - [ ] Valid Sudoku (Medium)<br>- [ ] 블록 A 오답 복습 | **자기지도 학습**<br>He et al. (2021). *MAE.*<br>→ Transformer를 Vision에 적용. |

### 블록 A 확장 구간 (W13–16, v7 신규 — 여유 활용 복습·심화)

빠르게 지나간 W1–12를 다지는 구간입니다. 서두르지 않되 매주 결과물을 남기세요.

| 주차 | 🛠️ 복습·심화 | 💻 알고리즘 | 비고 |
|------|-------------|-----------|------|
| **W13** | **[복습]** W1–7 수학 최소 보장 항목 재점검 (막히는 것만 다시)<br>**[재구현]** numpy PCA·로지스틱 회귀·SGD 중 약한 것 1개 재작성<br>**[✅ (v9 추가) 평가 방법론 보강]** PR 커브 개념 정리 — W3의 클래스 불균형(7:4) 사례에 적용해 "왜 이 경우 ROC보다 PR이 더 적합한가" 직접 설명해보기 / train·val·test 분리 원칙과 k-fold 교차검증의 목적·절차 재정리 | Easy·Medium 3문제 | 수학 자신감 굳히기 |
| **W14** | **[미니프로젝트 #1 개선]** LLM 요약·분류기에 테스트·타입힌팅·README 보강<br>**[Python]** pytest로 유닛테스트 첫 작성 | Medium 3문제 | 포트폴리오 품질↑ |
| **W15** | **[미니프로젝트 #2 개선]** 시계열 EDA 리포트에 시각화·해석 보강<br>**[도구]** GitHub Actions로 pytest 자동화 재확인 | Medium 3문제 | CI 감각 유지 |
| **W16** | **[RAG v0 안정화]** W12 RAG v0 버그 수정·데모 안정화·HuggingFace Spaces 재배포<br>**[⭐영어]** IELTS 2차 모의고사 → B2 갭 재확인 | Medium 3문제 | 블록 B 진입 전 정비 |

### 블록 A 종합 점검 (W16 기준 = 2026년 10월말)

- [ ] **(v8 신규)** 실제 완료 날짜를 계획 종료일(2026년 10월말)과 비교 → 지연 시 블록 B 착수일을 명시적으로 재조정하고 이 문서에 기록

**수학 최소 보장 (전부 통과해야 블록 B 진입)**
- [ ] PCA를 고유값분해로 손계산 가능
- [ ] SVD와 PCA 차이 설명 가능
- [ ] 베이즈 정리를 예시로 유도 가능
- [ ] MLE를 로그우도로 손유도 가능
- [ ] 연쇄법칙을 역전파에 적용해 설명 가능
- [ ] SGD·Adam 차이를 모멘텀 관점에서 설명 가능
- [ ] **(v9 추가)** PR 커브와 ROC의 차이를 불균형 데이터 맥락에서 설명 가능
- [ ] **(v9 추가)** k-fold 교차검증의 목적과 절차 설명 가능 / 회귀 평가지표(MSE·MAE·R²) 구분 가능

**실습·도구**
- [ ] numpy로 PCA·선형회귀·SGD를 처음부터 구현
- [ ] FastAPI + Docker로 모델 서빙·배포
- [ ] LangChain + FAISS + LLM API로 RAG v0 구현
- [ ] Qdrant·BM25·GitHub Actions 각 1회 사용

**⭐ 비자·영어**
- [ ] IELTS B2 목표 갭 파악 + 주 1회 스피킹·라이팅 루틴 확립
- [ ] AI/ML 스폰서 라이선스 기업 감 잡기 (첫 탐색 완료)

---

## 🟩 블록 B — ML 심화 + 도구 (B①~B⑩, 주 6–9h, 2026.11–2028.2)

*핵심: 프로젝트·Kaggle이 중심. 이론은 필요할 때 역추적.*
*아래 표의 B①~B⑩은 확장된 진행 순서(각 5–8주)입니다. 블록 C·D의 W49–68+ 주차 라벨과 혼동을 피하기 위해 블록 B는 스텝 번호로 표기합니다.*
*v7 구조: 두 국면으로 나뉩니다.*
- **B-1 단독 구간 (2026.11–2028.2, 프리마스터 전)**: 학업 부담이 없으니 Kaggle·SQL·MLOps·HuggingFace를 여유롭게 완주. B①~B⑩을 서두르지 않고 소화.
- **B-2 병행 구간 (2028.3–2028.7, 프리마스터 병행)**: ⚠️ **프리마스터 성적이 최우선.** 신규 학습을 최소화하고 LeetCode 감각 유지(주 2–3문제) + P1 준비 문서 정도만. 무리하지 말 것.

*⭐ 병행: 영어 B2는 B-1 단독 구간에 반드시 완료 (프리마스터 병행 전에 끝내는 게 핵심). 스폰서 기업 리스트도 B-1에 착수.*

> **v7 배치 원칙**: 아래 B①~B⑩ 학습 내용은 **B-1 단독 구간(2026.11–2028.2)에 넉넉히 배분**됩니다. 프리마스터가 시작되는 2028년 3월 전에 SQL·MLOps·Kaggle 2개·ML 시스템 디자인·영어 B2를 모두 끝내는 것을 목표로 하세요. 그러면 프리마스터 병행 구간(B-2)에는 성적에만 집중할 수 있습니다.

| 주차 | 📊 실습·프로젝트 중심 | 💻 알고리즘 | 📄 논문 |
|------|---------------------|-----------|--------|
| **B① (2026.11~)** | **[실습]** Andrew Ng ML Specialization C1 (실습 위주)<br>**[Kaggle]** Titanic EDA 시작 + 피처엔지니어링<br>**[막히면]** 편향-분산 트레이드오프 → 핸즈온 3–4장<br>**[도구]** scikit-learn 파이프라인·ROC 실습<br>**[⭐영어]** IELTS 응시 1차 (B2 확보 목표 — B-1 단독 구간에 완료)<br>**[(v9 추가) 실습]** K-means로 공개 데이터 클러스터링 데모 1회 (예: 고객/이커머스 세분화) + 이상탐지 개념(가우시안 기반) 정리 + 추천시스템 기초(협업 필터링·행렬분해) 개념 정리 — Andrew Ng ML Specialization C3 요약만 활용 (B⑨ 추천 시스템 설계 토론의 선수학습)<br>**[✅ 최소 보장]** 편향-분산 트레이드오프 영어로 설명 / **(v9 추가)** K-means의 목적함수(클러스터 내 분산 최소화) 설명 가능 | 트리·BFS 기초<br>- [ ] Validate BST (Medium)<br>- [ ] Level Order Traversal (Medium) | **BERT**<br>Devlin et al. (2018).<br>→ 양방향 마스킹으로 언어 이해 혁신. |
| **B②** | **[Kaggle]** Titanic 모델링·제출·상위 솔루션 분석<br>**[실습]** XGBoost·LightGBM + Optuna 튜닝 (**(v9 추가)** Optuna 튜닝을 k-fold 교차검증과 결합해 검증)<br>**[노트북 공개]** GitHub + Kaggle → **포트폴리오 #1**<br>**[(v9 추가) 해석]** SHAP으로 XGBoost 예측 1건 이상 설명 (feature importance와 SHAP value 차이 확인) + 공정성 지표(demographic parity 등) 개념 정리<br>**[✅ 최소 보장]** 앙상블이 왜 단일 모델보다 강한지 편향-분산 관점 설명 / **(v9 추가)** SHAP과 feature importance의 차이를 한 문장으로 설명 가능 | DFS·백트래킹<br>- [ ] Clone Graph (Medium)<br>- [ ] Course Schedule (Medium) | **GPT-1/GPT-2**<br>Radford et al. (2018/2019).<br>→ "언어 모델이 태스크를 스스로 배운다." |
| **B③** | **[Python 고급]** (Kaggle #1 완료 후 집중)<br>- asyncio·비동기 처리 심화<br>- 타입 힌팅 (mypy)<br>- 프로파일링 (cProfile·memory_profiler)<br>- 코드 품질 (pytest·black·ruff)<br>**[실습]** 기존 FastAPI 서버에 asyncio 적용·테스트 | 힙·우선순위 큐<br>- [ ] Kth Largest Element (Medium)<br>- [ ] Top K Frequent Elements (Medium) | **GPT-3**<br>Brown et al. (2020). *Few-Shot Learners.*<br>→ 프롬프트만으로 태스크 수행. |
| **B④** | **[SQL 집중 Week 1]**<br>- SELECT·JOIN·GROUP BY·서브쿼리<br>- LeetCode SQL Easy 15문제<br>- Mode Analytics SQL Tutorial 완주<br>**[실습]** PostgreSQL + psycopg2 실제 DB 연결 | DP 기초<br>- [ ] House Robber (Medium)<br>- [ ] Coin Change (Medium) | **InstructGPT (RLHF)**<br>Ouyang et al. (2022).<br>→ ChatGPT의 직접적 기반. 왜 RLHF인가. |
| **B⑤** | **[SQL 집중 Week 2]**<br>- Window function (ROW_NUMBER·RANK·LAG·LEAD)<br>- CTE·재귀 쿼리·EXPLAIN 성능 최적화<br>- LeetCode SQL Medium 15문제<br>**[미니프로젝트]** PostgreSQL + FastAPI 데이터 분석 API<br>**[⭐비자]** 스폰서 기업 후보 리스트 v1 (30–50곳, AI/ML 스폰서 실적 확인) | DP 심화<br>- [ ] Longest Common Subsequence (Medium)<br>- [ ] Edit Distance (Hard) | **Chain-of-Thought**<br>Wei et al. (2022).<br>→ "단계적으로 생각하라"가 추론을 극적으로 향상. |
| **B⑥** | **[MLOps]**<br>- W&B 또는 MLflow 실험 추적<br>- FastAPI 비동기 엔드포인트·미들웨어·에러 핸들링<br>- Docker Compose 멀티 컨테이너<br>- **GitHub Actions CI** 심화: pytest + 평가 자동 실행<br>**[✅ P2 선수학습]** CI 파이프라인 실제 작동 확인 | 탐욕·분할정복<br>- [ ] Task Scheduler (Medium)<br>- [ ] Jump Game (Medium) | **LoRA**<br>Hu et al. (2021). *Low-Rank Adaptation.*<br>→ 파라미터 1%로 full fine-tuning 수준. |
| **B⑦** | **[Kaggle #2]** 시계열 또는 NLP 입문<br>- 목표: 참가 + 상위 솔루션 분석·일부 재현<br>- Pandas 고급 (시계열·피벗·apply 최적화)<br>**[포트폴리오 #2]** Kaggle 노트북 공개 정리 | 그리디·구현<br>- [ ] Queue Reconstruction (Medium)<br>- [ ] Spiral Matrix (Medium) | **LLaMA / Llama 2**<br>Touvron et al. (2023).<br>→ 오픈소스 LLM 생태계 형성. |
| **B⑧** | **[HuggingFace + LLM 도구]**<br>- datasets·transformers·pipelines<br>- LoRA fine-tuning 체험 (small model)<br>**[LLM 비용·추론 최적화 개념]**<br>- 토큰 비용 구조 (input/output pricing)<br>- 프롬프트 캐싱 메커니즘<br>- vLLM·TGI 서빙 개념<br>- 모델 라우팅 설계 개념 | 비트·수학<br>- [ ] Single Number (Easy)<br>- [ ] Subsets (Medium) | **RAG 원논문**<br>Lewis et al. (2020). *Retrieval-Augmented Generation.*<br>→ P1 구현 전 필독. |
| **B⑨** | **[ML 시스템 디자인 Week 1]** Chip Huyen, *Designing ML Systems* 1–5장<br>- ML 요구사항·데이터 파이프라인·피처 엔지니어링<br>- 설계 문제 연습: "추천 시스템 설계" 구두 설명 | 비트·구현<br>- [ ] Pow(x, n) (Medium)<br>- [ ] Subsets II (Medium) | **RAGAS**<br>Es et al. (2023). *Automated Evaluation of RAG.*<br>→ P1 평가 파이프라인에 직접 사용. |
| **B⑩ (~2028.2)** | **[ML 시스템 디자인 Week 2]** Chip Huyen 6–10장<br>- 배포·모니터링·데이터 드리프트<br>- 설계 문제: "실시간 이상탐지 파이프라인 설계"<br>**[P1 준비]**<br>- P1 요구사항·아키텍처 설계 문서 작성<br>- FCA·ICO·EU AI Act PDF 데이터 수집 시작<br>- Langsmith 계정 설정·첫 트레이스 확인<br>**[⭐비자]** 경력직(mid-level) 지원 가능 직무 스펙 조사 시작<br>**[⚠️ 병행]** 2028.3부터 프리마스터 시작 시 이 구간은 유지 모드로 전환 (성적 우선) | 복합 기출<br>- [ ] LRU Cache (Medium)<br>- [ ] Design HashMap (Easy) | **LLM-as-Judge**<br>Zheng et al. (2023). *MT-Bench and Chatbot Arena.*<br>→ P2 Eval 파이프라인 자동 평가 방법론. |

### 🧩 이론 심화 보강 트랙 (v7 신규 — B-1 여유 시간에 분산 삽입)

*목적: "면접의 왜?"에 답하기 위한 이론. 순수 증명이 아니라 **직관 + 손유도 가능 수준**이 목표.*
*방식: 별도 블록이 아니라, 관련 실습이 나오는 B스텝에 "곁들여" 학습. 각 항목 주당 +1–2h면 충분.*
*원칙: 실습에 붙지 않는 이론은 넣지 않는다. 아래는 모두 이미 하는 실습에 뿌리를 둔다.*

| 붙일 위치 | 이론 항목 | 무엇을 (직관 + 손유도 수준) | 대응 면접 질문 |
|----------|----------|--------------------------|--------------|
| **B① (Andrew Ng C1 즈음)** | **정보이론 기초** | 엔트로피·크로스엔트로피·KL 발산의 정의와 관계. "왜 분류 손실이 크로스엔트로피인가"를 최대우도(MLE)와 연결해 손유도. 정보이득이 엔트로피 감소임을 트리와 연결 | "왜 크로스엔트로피 손실을 쓰나?" / "KL 발산이 뭐고 언제 쓰나?" |
| **B① (v9 추가 — 클러스터링 실습 즈음)** | **비지도학습 기초** | K-means의 목적함수(클러스터 내 분산 최소화)와 알고리즘 흐름(할당→갱신 반복), k 선택 방법(elbow). 이상탐지를 가우시안 분포로 근사하는 직관. 추천시스템의 협업 필터링(사용자·아이템 유사도)과 행렬분해(잠재요인) 개념 — B⑨ 추천 시스템 설계 토론의 선수학습 | "비지도학습과 지도학습 차이는?" / "클러스터 개수 k를 어떻게 정하나?" / "추천 시스템을 어떻게 만들겠는가?" |
| **B② (XGBoost·과적합 즈음)** | **정규화·일반화** | L1/L2 정규화가 왜 과적합을 막는지(가중치 페널티→모델 단순화), L1이 왜 희소성을 만드는지 기하학적 직관. 드롭아웃·조기종료의 원리. (여유 시) double descent 현상 소개 | "L1과 L2 차이는?" / "왜 정규화가 일반화를 돕나?" / "과적합 어떻게 막나?" |
| **B② (v9 추가 — SHAP 실습 즈음)** | **모델 해석·공정성 (Explainability & Fairness)** | SHAP(Shapley value 직관 — 각 피처가 예측에 기여한 몫을 게임이론적으로 분배)과 LIME(국소 선형 근사)의 차이. Feature importance(전역·모델 내재적)와 SHAP(지역·모델 불가지론적) 값의 차이. 공정성 지표(demographic parity·equalized odds) 개념. EU AI Act가 고위험 AI 시스템에 요구하는 설명가능성 요건과 P1(규제 QA) 프로젝트의 연결 | "블랙박스 모델을 어떻게 설명하겠는가?" / "SHAP과 feature importance 차이는?" / "모델의 공정성을 어떻게 평가하겠는가?" |
| **B③ (Python 고급 즈음, 가벼운 주)** | **베이지안 관점** | MLE vs MAP 차이(사전확률의 역할), 정규화가 사실은 MAP의 사전확률과 같다는 연결. 베이지안이 불확실성을 어떻게 표현하는지 개념 | "MLE와 MAP 차이는?" / "정규화를 베이지안으로 해석하면?" |
| **B④–⑤ (SQL 구간, 이론 부하 낮음)** | **최적화 심화** | 학습률 스케줄링·warmup이 왜 필요한지, gradient clipping이 왜 Transformer 학습에 필수인지(gradient explosion), 배치 정규화·레이어 정규화의 차이와 이유 | "학습이 발산하면 뭘 점검?" / "warmup은 왜 쓰나?" / "BatchNorm vs LayerNorm?" |
| **B⑥ (MLOps, SGD 재등장)** | **⭐분산/스케일러블 ML (MSc 갭)** | SGD가 왜 병렬화에 유리한지(미니배치·데이터 병렬), 데이터 병렬 vs 모델 병렬 개념. **PySpark 기초 튜토리얼 1주**(RDD·DataFrame·map-reduce 감각). Scalable ML 모듈 예습 | "대규모 데이터 학습을 어떻게 분산?" / "데이터 병렬과 모델 병렬 차이는?" |
| **B⑦ (Kaggle #2 = NLP·시계열)** | **⭐고전 NLP (MSc 갭)** | TF-IDF·n-gram·품사태깅·구문분석의 원리를 가볍게. 왜 임베딩 이전에 이런 방법을 썼고 한계가 뭐였는지. Text Processing/NLP 모듈 예습 | "임베딩 이전 NLP는 어떻게?" / "TF-IDF 원리는?" |
| **B⑦–⑧ 사이 (v9 추가 — RLHF 직전, 반나절)** | **RL 기초 (RLHF 선수학습)** | MDP 정의(state·action·reward·policy)를 가볍게. value iteration·policy iteration의 직관(반복적으로 더 나은 정책으로 수렴). Q-learning의 핵심 아이디어(행동가치 함수를 경험으로 갱신). Policy gradient의 직관(보상이 높은 행동의 확률을 높이는 방향으로 정책 파라미터 갱신) — 다음 B⑧의 PPO를 이해하기 위한 최소 배경 | "강화학습의 기본 구성요소는?" / "정책 그래디언트가 뭔가?" / "Q-learning과 policy gradient 차이는?" |
| **B⑧ (LLM 도구·RLHF 논문 즈음)** | **RLHF·정렬의 수학** | InstructGPT의 3단계(SFT→보상모델→PPO) 흐름. PPO objective의 직관(정책 업데이트를 제한하는 이유 — 바로 위 RL 기초의 policy gradient와 연결). **DPO의 핵심 아이디어**(보상모델 없이 선호쌍으로 직접 최적화)를 개념+objective 형태로 | "RLHF는 어떻게 동작?" / "PPO와 DPO 차이는?" / "왜 보상모델이 필요/불필요?" |
| **B⑨–⑩ (ML 시스템 디자인)** | **평가·통계적 유의성** | A/B 테스트의 통계(p-value·신뢰구간)를 재활성화, 오프라인 지표 vs 온라인 지표 괴리. LLM 평가에서 왜 단일 지표가 위험한지 | "모델 개선을 어떻게 검증?" / "오프라인 좋은데 온라인 나쁘면?" |

> **분량 감각**: 위 11개 항목(v9에서 비지도학습·모델 해석·RL 기초 3개 추가)은 대부분 "개념 이해 + 손으로 한 번 유도/정리"면 끝나는 것들이라, 항목당 반나절~하루면 충분합니다. 예외는 **PySpark 기초(1주)**와 **고전 NLP(2–3일)** — 이 둘만 실습을 동반합니다. B-1 단독 구간(약 68주)의 여유를 생각하면 부담 없이 흡수됩니다.

> **⚠️ 하지 말 것**: 측도론적 확률, VC dimension·PAC learning의 엄밀한 증명, 볼록최적화 정리 증명 등은 **넣지 않습니다.** 이건 취업 ROI가 낮고, 필요하면 MSc의 ML&AI 모듈이 정식으로 다룹니다. 이 트랙의 선은 "면접에서 설명 가능한 직관"까지입니다.

### 이론 보강 트랙 점검 (B-1 종료 시)
- [ ] 크로스엔트로피 손실을 MLE로 손유도 가능
- [ ] **(v9 추가)** K-means 목적함수 + 이상탐지 직관 + 추천시스템(협업 필터링·행렬분해) 개념 설명 가능
- [ ] L1/L2 차이와 정규화의 일반화 효과 설명 가능
- [ ] **(v9 추가)** SHAP/LIME 직관 + feature importance와의 차이 + 공정성 지표(demographic parity 등) 설명 가능
- [ ] MLE vs MAP, 정규화의 베이지안 해석 설명 가능
- [ ] warmup·gradient clipping·정규화 레이어의 이유 설명 가능
- [ ] 데이터 병렬 vs 모델 병렬 + PySpark 기초 실습 1회
- [ ] TF-IDF·n-gram 등 고전 NLP 원리 설명 가능
- [ ] **(v9 추가)** MDP·Q-learning·policy gradient 기초 설명 가능 (RLHF 선수학습)
- [ ] RLHF 3단계 + PPO/DPO 차이 설명 가능
- [ ] A/B 테스트 통계 + 오프라인/온라인 지표 괴리 설명 가능

### 블록 B 종합 점검 (B-1 종료 = 2028년 2월, 프리마스터 직전 완충 진입 전)
- [ ] ⭐ **IELTS B2 이상 확보 완료** (프리마스터 병행 전에 반드시) ← **최우선**
- [ ] ⭐ 스폰서 기업 리스트 v1 (30–50곳) 완성
- [ ] SQL Window function·CTE 가능 / Kaggle 포트폴리오 2개
- [ ] MLOps(CI·실험추적)·HuggingFace·ML 시스템 디자인 완료
- [ ] 🧩 이론 심화 보강 트랙 11개 항목 통과 (위 점검표, v9에서 3개 추가)
- [ ] P1 아키텍처 설계 문서 완성

### 프리마스터 직전 완충 (W83–84 = 2028년 2월)
- [ ] 블록 A·B 전체 약한 부분 1–2개 복습
- [ ] 개발 환경·GitHub·데모 전수 점검
- [ ] 프리마스터 커리큘럼·평가 방식 사전 파악
- [ ] ⚠️ 이후 2028.3–7은 **프리마스터 성적 최우선**, 커리큘럼 학습은 유지 모드

### 프리마스터 병행 구간 (B-2, 2028.3–2028.7) — 유지 모드
- [ ] LeetCode 주 2–3문제 감각 유지
- [ ] P1 데이터(FCA·ICO·EU AI Act PDF) 수집만 틈틈이
- [ ] 신규 대형 학습 없음. 프리마스터 성적이 MSc 진학 요건.

---

## 🟥 블록 C — DL + LLM + 시그니처 프로젝트 (W49–60, 주 20–25h, 2028.7–2028.8)

*핵심: DL 펀더멘털(W49–52) → Transformer 이해 점검(W52 후반) → nanoGPT(W53) → **P1 깊이 완성**(W54–55) → **P2 동작 데모+지표**(W56–57 전반) → 취업 준비(W57 후반–60)*
*⚠️ v6 재조정: P2는 "완성"이 아니라 "동작 데모 + README 정량 지표"로 목표 하향. 깊이는 P1에 몰빵. (학습원칙 5번과 정합)*
*⭐ 병행: 경력직(mid-level) 직접 지원 착수*

| 주차 | 🏗️ DL·LLM·프로젝트 | 💻 알고리즘 (집중) | 📄 논문 |
|------|-------------------|-----------------|--------|
| **W49** | **[실습]** PyTorch로 MLP 만들어 MNIST 분류<br>**[막히면]** d2l.ai 1–5장 역추적<br>**[재구현]** autograd 없이 역전파 수동 계산 후 검증<br>**[도구]** Kaggle GPU 노트북 환경 설정<br>**[✅ 최소 보장]** 역전파를 계산 그래프로 손추적 가능 | 배열·해시 집중<br>- [ ] 3Sum (Medium)<br>- [ ] Product of Array Except Self (Medium)<br>- [ ] Subarray Sum Equals K (Medium) | **FAISS 심화**<br>Douze et al. (2024). *The FAISS Library.*<br>→ HNSW·IVFFlat 트레이드오프. P1 벡터DB 선택 근거. |
| **W50** | **[실습]** Karpathy micrograd 코드 읽기<br>**[막히면]** 역전파 수식 재정리 (연쇄법칙·계산 그래프)<br>**[재구현]** micrograd 처음부터 직접 구현<br>**[검증]** PyTorch autograd와 결과 비교<br>**[✅ 최소 보장]** micrograd의 Value 클래스 구조 설명 가능 | 슬라이딩 윈도우<br>- [ ] Minimum Window Substring (Hard)<br>- [ ] Longest Repeating Char Replacement (Medium) | **BM25**<br>Robertson & Zaragoza (2009).<br>→ Dense + Sparse 하이브리드 검색의 이론 기반. |
| **W51** | **[실습]** Karpathy makemore 따라가기<br>**[실습]** d2l.ai CNN + CIFAR-10 전이학습<br>**[막히면]** CNN의 locality·weight sharing 개념<br>**[실습]** CLIP으로 이미지-텍스트 유사도 계산<br>**[✅ 최소 보장]** CNN의 receptive field 개념 설명 | 트리·그래프 집중<br>- [ ] Binary Tree Right Side View (Medium)<br>- [ ] Word Ladder (Hard)<br>- [ ] Pacific Atlantic Water Flow (Medium) | **Cross-Encoder Re-ranking**<br>Nogueira & Cho (2019). *Passage Re-ranking with BERT.*<br>→ P1 re-ranking 컴포넌트 이론 기반. |
| **W52 전반** | **[실습]** *The Annotated Transformer* 코드 한 줄씩 실행·분석<br>**[막히면]** d2l.ai 어텐션 챕터 역추적<br>**[재구현]** Multi-head Attention + Positional Encoding 처음부터 구현<br>**[검증]** PyTorch nn.Transformer와 출력 비교 | DP 집중<br>- [ ] Longest Increasing Subsequence (Medium)<br>- [ ] Partition Equal Subset Sum (Medium) | **LLM 평가 Survey**<br>Chang et al. (2023). *A Survey on Evaluation of LLMs.*<br>→ P2 평가 분류 체계. |
| **W52 후반** | **[🔍 Transformer 이해 점검 — 완충 구간]**<br>아래 항목을 영어로 설명할 수 있어야 W53 착수:<br>- [ ] Scaled Dot-Product Attention 수식 설명<br>- [ ] 왜 sqrt(d_k)로 나누는가<br>- [ ] Multi-head가 Single-head보다 나은 이유<br>- [ ] Positional Encoding이 왜 필요한가<br>- [ ] Encoder-Decoder 구조에서 각 Attention의 역할<br>통과 못 하면 W53 전반까지 연장, nanoGPT는 W53 후반 착수 | - | - |
| **W53** | **[실습]** Karpathy nanoGPT 코드 분석<br>**[재구현]** nanoGPT 처음부터 구현 (Colab/Kaggle GPU)<br>**[실습]** 작은 텍스트로 학습 → 생성 결과 확인<br>**[P1 준비]** W53 후반: P1 데이터 파이프라인 세팅 시작 (전환 버퍼)<br>**[✅ 최소 보장]** GPT의 causal masking이 왜 필요한지 설명 | 힙·Hard 집중<br>- [ ] Find Median from Data Stream (Hard)<br>- [ ] Merge k Sorted Lists (Hard)<br>- [ ] Sliding Window Maximum (Hard) | **GPTQ + QLoRA**<br>Frantar et al. (2022). Dettmers et al. (2023).<br>→ 제한 자원에서 LLM 실행. |
| **W54** | 🚀 **P1 착수 — 하이브리드 RAG 법률/규제 문서 QA** (⭐ 시그니처, 깊이 집중)<br>*(W12 RAG v0의 고도화)*<br><br>**왜 이 도메인인가 (면접·PS 스크립트):**<br>"영국 금융·AI 규제(FCA·ICO·EU AI Act)는 빠르게 변화하는데, 컴플라이언스 팀이 수백 페이지 문서를 매번 검토하는 것은 비효율적이다. LLM이 이 검토를 보조할 수 있다면 실질적인 비즈니스 임팩트가 있다. 이 문제를 선택한 것은 영국 취업 도메인과 LLM 기술이 가장 자연스럽게 만나는 지점이기 때문이다."<br><br>**[구현]**<br>- FCA·ICO·EU AI Act PDF 청킹·임베딩 파이프라인<br>- Qdrant 벡터DB (Docker) + bge-m3 임베딩<br>- BM25 + Dense 하이브리드 (RRF 융합)<br>- 데이터 수집 자동화 스크립트<br>**[(v9 추가) 해석 연결]** README에 "왜 이 규제 도메인에서 설명가능성이 중요한가" 1단락 추가 — B②에서 배운 SHAP/공정성 개념과 EU AI Act의 설명가능성 요건을 연결해 프로젝트 서사를 강화 (인용/citation enforcement가 이미 하는 "근거 제시"를 규제 맥락에서 재해석) | 그리디·백트래킹<br>- [ ] Task Scheduler (Medium 복습)<br>- [ ] Combination Sum (Medium)<br>- [ ] Permutations (Medium) | **RAG Survey**<br>Gao et al. (2023). *RAG for LLMs: A Survey.*<br>→ Naive → Advanced → Modular RAG 진화. |
| **W55** | 🚀 **P1 완성** (⭐ 여기에 완성도 최대 투자)<br>**[구현]**<br>- Cross-encoder re-ranking (ms-marco-MiniLM)<br>- Citation enforcement (할루시네이션 감소)<br>- FastAPI 비동기 서빙 엔드포인트<br>- Streamlit UI<br>**[평가]** RAGAS (Faithfulness·Relevancy·Context Recall)<br>**[배포]** Docker → HuggingFace Spaces<br>**[모니터링]** Langsmith (지연시간·비용·품질)<br>**[README]** 문제→아키텍처→지표→데모 링크 | LRU·설계<br>- [ ] LRU Cache (Medium 복습)<br>- [ ] Design Twitter (Hard)<br>- [ ] Insert Delete GetRandom O(1) (Medium) | **vLLM**<br>Kwon et al. (2023). *Efficient Memory Management with PagedAttention.*<br>→ 추론 비용 최적화. 면접 "LLM 서빙 설계" 단골. |
| **W56** | 🚀 **P2 착수 — LLM Eval 파이프라인 + 모델 라우터** (동작 데모 목표)<br>*(블록 B⑥ GitHub Actions CI 경험 기반)*<br>**[핵심 구현 — 여기까지만 사수]**<br>- 쿼리 분류기 (scikit-learn) → GPT-4o-mini/Claude Haiku/Mistral 라우팅<br>- 토큰 비용 자동 계산 + 라우팅 전후 비교표<br>**[여유 시]**<br>- GitHub Actions CI: 코드 변경 시 RAGAS 평가 자동 실행<br>- Langsmith 대시보드 | 문자열·구현<br>- [ ] Decode Ways (Medium)<br>- [ ] Regular Expression Matching (Hard)<br>- [ ] Wildcard Matching (Hard) | **ReAct + Reflexion**<br>Yao et al. (2022). Shinn et al. (2023).<br>→ P3 에이전트 설계 이론 기반. 블록 D 전 예습. |
| **W57 전반** | 🚀 **P2 마무리 버퍼 (동작 데모 + 지표까지)**<br>- P2 README: 비용 절감률 정량 표기 (CI 자동화는 여유 시)<br>- P2 eval/ 디렉토리: RAGAS 평가 데이터셋 30–100 Q&A<br>- HuggingFace Spaces 데모 배포<br>- 전체 테스트·버그 수정<br>**⚠️ P2가 밀리면 P1 완성도를 우선하고 P2는 "동작 데모"에서 멈춘다** | - | - |
| **W57 후반** | **[LLM 심화]**<br>- vLLM 또는 Ollama로 로컬 LLM 서빙 실습<br>- LoRA fine-tuning: small model 실제 fine-tuning<br>**[클라우드]**<br>- AWS EC2 또는 GCP Cloud Run에 P1 배포<br>**[⭐비자]** 경력직(mid-level) AI Engineer 직접 지원 착수 — 스폰서 기업 대상 | 복합 기출<br>- [ ] Word Search II (Hard)<br>- [ ] Alien Dictionary (Hard)<br>- [ ] Random Pick with Weight (Medium) | **Toolformer**<br>Schick et al. (2023).<br>→ 에이전트 도구 호출 이론. |
| **W58** | **[영어 기술 커뮤니케이션 Week 1]**<br>- P1·P2 영문 기술 블로그 초안 작성<br>- 비전공자에게 RAG 설명하는 영문 스크립트<br>**[STAR 스크립트 3개 작성]**<br>  ① 가장 어려웠던 기술 문제 (P1 하이브리드 검색 구현 경험)<br>  ② 팀 리더십 경험 (연구소장 경력 활용 — ⭐ 경력직 트랙 핵심 스토리)<br>  ③ 실패에서 배운 것<br>- ML 시스템 디자인 영어 구두 설명 연습<br>**[P3 선수학습]** LangGraph 공식 튜토리얼 완주 | Mock #1<br>- [ ] 랜덤 Medium 5 + Hard 1<br>타이머 45분 | **OWASP LLM Top 10**<br>OWASP (2025).<br>→ Prompt Injection·Hallucination·Data Leakage. P3 보안 구현. |
| **W59** | **[영어 기술 커뮤니케이션 Week 2]**<br>- 영문 기술 블로그 완성·게시 (Medium 또는 개인 블로그)<br>- 모의 면접 1회 (Pramp 또는 친구)<br>**[GitHub 전체 정비]**<br>- `/app` `/src` `/eval` `/infra` `/tests` 구조 통일<br>- 아키텍처 다이어그램 (Excalidraw/draw.io)<br>- HuggingFace Spaces 데모 2개 확인 | Mock #2<br>- [ ] Medium 5 + Hard 1<br>- [ ] SQL Medium 5 | **Agentic AI Survey**<br>Wang et al. (2024).<br>→ 에이전트 현재와 방향. 면접 트렌드 대화용. |
| **W60** | **[영문 CV 최종 완성]** (1페이지, ⭐ 이중 트랙용 2버전)<br>- **경력직 버전**: "AI R&D Leadership, 2+ years, energy/data center domain" 헤드라인 → P1/P2로 최신 LLM 역량 증명<br>- **신입/인턴 버전**: 프로젝트 중심<br>- P1: "Improved answer relevance 0.62→0.85 NDCG@5 via hybrid retrieval"<br>- P2: "Reduced inference cost 40% via complexity-based model routing"<br>**[LinkedIn]** 영문 프로필 최종 업데이트<br>**[MSc 예습]** 1학기 코스워크 실러버스 확인<br>**[🔍 블록 C 종합 점검]** | Mock #3<br>- [ ] 전체 유형 랜덤<br>- [ ] ML 개념 구두 5문제 | **ML 시스템 설계 복습**<br>Huyen (2022). *Designing ML Systems* 핵심 복습. |

---

## 🟪 블록 D — MSc 취업 시즌 (W61–68+, 주 10–15h, 2028.9–2029.9)

*MSc 코스워크 병행. 취업 승부의 70%가 이 학기.*
*⭐ 이중 트랙: 여름 인턴(신입) + 경력직 직접 지원(mid-level) 동시.*
*빅테크 여름 인턴 9–10월 롤링 마감 → 시작 즉시 지원 폭격기 모드.*
*⚠️ 최종 목표: 졸업 전 스폰서 오퍼 확보 (Graduate visa 18개월은 백업일 뿐).*

| 주차 | 🎯 취업 준비 | 💻 알고리즘 | 🏗️ P3 |
|------|------------|-----------|-------|
| **W61** | - [ ] **9월(2028) 즉시**: 빅테크 인턴 오픈 모니터링 → 24–48h 내 지원<br>- [ ] ⭐ **스폰서 기업 50곳 리스트 최종 확정** (라이선스 + AI/ML 실적 검증)<br>- [ ] ⭐ 경력직 mid-level 포지션도 동시 지원 시작<br>- [ ] Career Service 등록 | Medium 10문제 | 🚀 **P3 착수 — UK 주식 리서치 에이전트**<br>*(W58 LangGraph 튜토리얼 기반)*<br>- LangGraph 에이전트 구축<br>- Tavily·yfinance 도구 연결 |
| **W62** | - [ ] 중견·핀테크·AI 스타트업 인턴+경력직 지원 확대<br>- [ ] LinkedIn 커피챗 (Referral 확보)<br>- [ ] ⭐ 각 지원 시 "스폰서 가능 여부" 사전 확인 | Medium 10문제 | 🚀 **P3 완성**<br>- Human-in-the-loop 구현<br>- OWASP 보안 적용<br>- Streamlit + Docker<br>- pytest + CI |
| **W63** | - [ ] 코딩 인터뷰 집중 (회사별 기출)<br>- [ ] ML 개념 구두 설명 연습<br>- [ ] Career Fair 참석 | Hard 5문제 | - [ ] P3 GitHub·README 정비<br>- [ ] 포트폴리오 3개 완성 |
| **W64–65** | - [ ] ML 시스템 디자인 면접 준비<br>  "Design a RAG system for 10M documents"<br>  "Design a real-time fraud detection system"<br>- [ ] 서류 통과 시 면접 대응<br>- [ ] (여유 시) AWS/GCP ML 자격증 — 우선순위 낮음, 시간 남을 때만 | Mock 집중 | - [ ] MSc 코스 프로젝트 → 포트폴리오 전환 |
| **W66–67** | - [ ] ⭐ **오퍼 확정 목표 — 인턴 리턴오퍼 OR 경력직 정규직** (4–5월 2029)<br>- [ ] 빅테크 실패 시 스타트업·경력직 즉시 선회<br>- [ ] 콜드 아웃리치 (인프라/에너지 테크 — 도메인 강점 활용) | 약점 집중 | - [ ] Dissertation 주제 확정<br>  (LLM 응용 + 취업 연계) |
| **W68+** | - [ ] **여름 인턴 수행 시 → 리턴오퍼(full-time+스폰서) 확보가 최우선**<br>- [ ] ⭐ **스폰서 오퍼 확보가 이 로드맵의 최종 관문** — 확보 시 New Entrant 임계값(going rate 70%) 활용 협의<br>- [ ] 졸업 후 Graduate visa(18개월)는 오퍼 미확보 시의 백업 창<br>- [ ] 가시적 임팩트 1–2개 | 유지 | - [ ] Dissertation 착수<br>  (인턴/업무 연계) |

---

## ✅ 블록 C 종합 점검 체크리스트 (W60 기준 = 2028년 8월, MSc 직전)

**수학 최소 보장 (전부 통과)**
- [ ] PCA를 고유값분해로 손계산
- [ ] SVD와 PCA 차이 행렬 분해 관점 설명
- [ ] 베이즈 정리 예시로 유도
- [ ] MLE 로그우도로 손유도
- [ ] 연쇄법칙을 역전파에 적용 설명
- [ ] SGD·Adam 차이 모멘텀 관점 설명
- [ ] Transformer: Scaled Dot-Product Attention 수식 + sqrt(d_k) 이유
- [ ] GPT의 causal masking이 왜 필요한지 설명

**코딩·도구**
- [ ] Python 고급 (asyncio·타입힌팅·프로파일링) 숙련
- [ ] SQL Window function·CTE·EXPLAIN 가능
- [ ] FastAPI(비동기) + Docker + GitHub Actions CI
- [ ] LeetCode Medium 100+, Hard 20+

**딥러닝·LLM**
- [ ] Transformer·어텐션 처음부터 PyTorch 구현
- [ ] micrograd·nanoGPT 처음부터 구현
- [ ] 하이브리드 RAG (Dense+BM25+Re-ranking) 구현·배포 (P1, 깊이 완성)
- [ ] LLM Eval + 모델 라우터 동작 데모 + 비용 지표 (P2)
- [ ] LLM 비용·vLLM 개념 설명 가능
- [ ] LoRA fine-tuning 경험

**ML 시스템 설계**
- [ ] Chip Huyen 완독
- [ ] 설계 문제 2–3개 영어로 구두 설명

**포트폴리오**

| # | 프로젝트 | 면접 스토리 | CV 지표 목표 |
|---|---------|-----------|------------|
| 0 | **RAG v0** (W12) | "LangChain으로 첫 RAG를 만들고 한계를 발견했다" | 데모 동작 |
| 1 | **P1: 하이브리드 RAG 법률/규제 QA** (⭐ 시그니처) | "영국 금융 규제 컴플라이언스 자동화 아이디어에서 출발" | NDCG@5 0.85+, 할루시네이션 <5% |
| 2 | **P2: LLM Eval + 모델 라우터** | "P1을 만들며 품질 측정 자동화 필요성을 직접 체감" | 비용 40% 절감 (동작 데모) |
| 3 | **P3: UK 주식 리서치 에이전트** (블록 D) | "텍스트 생성을 넘어 행동하는 AI 시스템" | Tool call 95%+ |

**⭐ 비자·영어·경력 (v6 신규 — 학습만큼 중요)**
- [ ] IELTS B2 이상 확보 완료
- [ ] 스폰서 기업 50곳 리스트 확정 (라이선스 + AI/ML 실적)
- [ ] 경력직 CV 버전 완성 (연구소장·AI R&D 경력 전면)
- [ ] 경력직 mid-level 지원 착수 (W57 후반부터)

**영어 커뮤니케이션**
- [ ] 영문 기술 블로그 1편 이상 게시
- [ ] STAR 스크립트 3개 + 구두 연습 완료 (②는 경력 리더십)
- [ ] 모의 면접 2회 이상

---

## 📊 선수학습 + 비자 연결 지도 (v7 완성판)

```
[실제 날짜 앵커]
  2026.07  블록 A 착수 (W1)
  2026.10  블록 A 종료·RAG v0 안정화 (W16)
  2026.11  블록 B 착수 (B①, 단독 구간)
  2028.02  블록 B(B-1) 종료 + 프리마스터 직전 완충 (W83–84)
  2028.03  프리마스터 시작 (1.5학기, 3월 인타이크) → B-2 유지 모드
  2028.07  프리마스터 종료 (7월 수업 종료)
  2028.07~08  블록 C 집중 (석사 전 갭)
  2028.09  MSc AI 시작 → 블록 D
  2029.09  MSc 졸업
  2029 하반기  Graduate visa 신청 (18개월 그룹 확정)
  ~2031 초   Skilled Worker 전환 데드라인

[⭐ 비자 전략 — 최우선 트랙]
  W11:       스폰서 라이선스 기업 첫 탐색 (2026.9)
  B⑤ (2027):  스폰서 기업 리스트 v1 (30–50곳)
  B⑩ (2028.초): 경력직 직무 스펙 조사
  W57 후반:   경력직 mid-level 지원 착수 (2028.8)
  W61:       스폰서 기업 50곳 최종 확정 + 이중 트랙 지원 시작 (2028.9)
  W68+:      스폰서 오퍼 확보 (= 최종 관문)

[⭐ 영어 B2 — Skilled Worker 필수, B-1 단독 구간에 완료]
  W1:        IELTS 진단 (2026.7)
  W8:        스피킹·라이팅 주 1회 루틴
  W16:       2차 모의고사로 갭 재확인
  B① (2026.11~): IELTS 응시 (B2 확보 목표)
  B-1 종료(2028.2): B2 확보 완료 확인 ← 프리마스터 병행 전 필수

[수학 최소 보장]
  W1–7:  수학 영역별 최소 보장 체크포인트 명시
  W8:    블록 A 수학 중간 점검
  W13–16: 블록 A 확장 구간에서 약한 것 재점검
  W52 후반: Transformer 이해 점검 (통과해야 nanoGPT 착수)
  W60:   블록 C 종합 점검 (8개 항목)

[RAG 도구 실습 선수학습]
  W10: LangChain/LlamaIndex 튜토리얼
  W11: Qdrant + BM25 첫 사용
  W12: RAG v0 완성 (⚠️ 지연 시 W16 안정화 구간이 흡수)
  W16: RAG v0 안정화·재배포
  W54: P1 착수 (RAG v0 고도화, 깊이 집중)

[GitHub Actions CI 선수학습]
  W11:  간단한 CI (pytest 자동화)
  B⑥:   MLOps CI 심화
  W56:  P2 CI 자동화 (여유 시)

[Langsmith 모니터링 선수학습]
  B⑩:  첫 트레이스 확인
  W55: P1 연결
  W56: P2 대시보드 (여유 시)

[P1 집중 / P2 하향]
  W54–55: P1 깊이 완성 (완성도 최대 투자)
  W56–57 전반: P2 동작 데모 + 비용 지표 (밀리면 여기서 멈춤)

[LangGraph 선수학습]
  W58: LangGraph 튜토리얼 완주
  W61: P3 착수

[🧩 이론 심화 보강 트랙 (B-1 여유에 분산, v9 기준 11개 항목)]
  B①:   정보이론 (크로스엔트로피=MLE)
  B①:   (v9 추가) 비지도학습 기초 (K-means·이상탐지·추천시스템)
  B②:   정규화·일반화 (L1/L2, double descent)
  B②:   (v9 추가) 모델 해석·공정성 (SHAP/LIME, EU AI Act 연결)
  B③:   베이지안 (MLE vs MAP)
  B④–⑤: 최적화 심화 (warmup·clipping·정규화 레이어)
  B⑥:   ⭐분산 ML + PySpark 기초 (MSc Scalable ML 예습)
  B⑦:   ⭐고전 NLP (TF-IDF·n-gram, MSc NLP 예습)
  B⑦–⑧: (v9 추가) RL 기초 (MDP·Q-learning·policy gradient, RLHF 선수학습)
  B⑧:   RLHF·정렬 수학 (PPO vs DPO)
  B⑨–⑩: 평가·통계 유의성 (A/B, 오프/온라인 지표)
  → 선: "면접에서 설명 가능한 직관"까지. 엄밀 증명은 MSc에.

[평가 방법론 보강 (v9 추가)]
  W6:   회귀 평가지표 (MSE·MAE·R²)
  W13:  PR 커브 (불균형 데이터, W3 사례 재적용) + 교차검증 개념 재정리
  B②:   k-fold 교차검증을 Optuna 튜닝에 결합해 실습

[자격증 (우선순위 격하)]
  블록 D W64–65 이후, 시간 여유 있을 때만
  → 포트폴리오·코딩테스트·스폰서 리서치가 항상 우선

[⚠️ 20개월 여유의 함정 방지]
  매주 최소 1커밋·1체크포인트
  각 구간 종료 점검을 실제 날짜에 캘린더 알림으로
```

---

## ⚠️ 학습·전략 원칙 (매주 상기)

1. **비자가 로드맵의 척추다** — 아무리 잘 배워도 스폰서 오퍼를 못 잡으면 영국에 못 남는다. 스폰서 기업 리서치와 영어 B2는 학습만큼 중요하다.
2. **최종 관문은 "졸업 전 스폰서 오퍼"** — Graduate visa 18개월은 백업 창일 뿐이다.
3. **경력을 무기로 써라** — 신입 인턴만이 아니라 경력직(mid-level)도 병행 지원. 연구소장·AI R&D 경력은 강력한 차별화 요소.
4. **실습 먼저, 막히면 이론** — 이론부터 시작하면 "이미 아는 것 같은" 착각에 집중력이 낮아진다.
5. **재구현이 핵심** — sklearn으로 돌려보는 것과 numpy로 처음부터 짜는 것은 다른 이해.
6. **수학 최소 보장 체크포인트는 막히든 안 막히든** — 면접에서 반드시 나오는 것들.
7. **P1을 깊이 완성** — 완성도 없는 3개보다 완성도 높은 P1 하나. P2는 동작 데모+지표까지.
8. **프리마스터 성적이 학습보다 우선** — 프리마스터 실패 = MSc 무산 = 전체 계획 붕괴.
9. **자격증은 여유 시** — 포트폴리오·코딩테스트·스폰서 리서치가 항상 앞선다.
10. **W52 후반 Transformer 점검 통과 전에는 nanoGPT 착수 금지.**
11. **(v8 신규) 시작 지연은 조용히 쌓인다 — 명시적으로 재확인하라.** 이번 점검에서 실제 실습 시작이 계획보다 6주 늦었음을 발견했다. 속도 자체는 문제가 아니었지만, "언젠가 버퍼가 흡수해주겠지"라고 방치하면 여러 구간의 지연이 누적돼 결국 비자 데드라인(2031년 초)까지 잠식할 수 있다. 매 블록 종합 점검마다 계획 날짜 대비 실제 날짜를 표로 대조하는 습관을 들인다.
12. **(v9 신규) 지도학습만으로는 부족하다 — 비지도·해석·RL 기초도 최소한은 짚는다.** 취업에 직접 쓰이는 지도학습·딥러닝·LLM이 커리큘럼의 중심인 건 맞지만, 면접·프로젝트 서사에서 "비지도학습은 안 배웠다", "이 모델이 왜 이렇게 예측했는지는 설명 못 한다" 같은 공백이 드러나지 않도록 B①·B②·B⑦–⑧에 최소 분량(각 반나절~하루)으로 채워 넣는다. 특히 P1(규제 도메인 프로젝트)에서는 모델 해석 개념이 프로젝트 서사 자체를 강화한다.

---

*핵심 한 줄: 실습으로 시작해 막히는 지점에서 이론을 끌어와라. 단, 수학 최소 보장과 영어 B2는 반드시 통과하라. 진짜 척추는 W12 RAG v0 → W55 P1 완성 → 스폰서 기업 리스트 → 졸업 전 스폰서 오퍼. 학습은 수단이고, 스폰서 오퍼가 목표다.*

> ⚠️ **면책**: 위 비자·이민 관련 내용은 2026년 7월 시점의 공개 정보 기반 참고 자료이며 법률 자문이 아닙니다. 실제 지원 시점(2028–2031)에는 salary threshold·ILR 요건·Graduate visa 조건이 다시 바뀔 수 있습니다. 지원 가까운 시점에 gov.uk와 OISC 등록 이민 전문가를 통해 반드시 재확인하세요.

> ℹ️ **v8 확인 노트 (2026-08-27)**: 위 비자 수치(최저연봉 £41,700, ISC £1,320/년, 영어 B2 요건, Graduate visa 18개월 단축)를 2026년 8월 27일 기준 공개 정보와 대조해 여전히 유효함을 확인했습니다. 다음 격주 점검에서도 계속 재확인합니다.

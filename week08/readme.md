
# W8: Model Serving with FastAPI and Docker | FastAPI·Docker로 모델 서빙

## 진행 상황 (2026-10-01 기준 — {진행 중})

**완료:**
- Day1: lru_cache를 이용하여 한 번 호출한 객체를 재사용하는 로직 구현, uvicorn을 이용 호출 시 객체 확인 (`model_service.py`, 'main.py') — {load_model_once(p) is load_model_once(p)가 True, print(load_model_once.cache_info())로 hits/misses 확인}
- Day2: Pydantic을 이용 PredictRequest, PredictResponse 작성, /post를 이용 400, 422 에러 검증
- Day3:
  - 측정 조건
  - 이미지: w8-serving (fastapi, uvicorn[standard], numpy 2.5.2, pandas 3.0.5)
  - 변경 내용: main.py에 주석 1줄 추가
  - 환경: Docker 29.8.2 / WSL 2 / python:3.14-slim

  | COPY 순서 | 재빌드 | RUN pip install |
  |---|---|---|
  | requirements → 소스 (채택) | 2.24s | CACHED |
  | 소스 → requirements | 17.71s | 재실행 |

  약 7.9배.

  - 단서: 17.71s는 Dockerfile 순서를 바꾼 빌드에서 잰 값이다. 코드 수정으로 캐시가
  깨져도 COPY . . 가 맨 앞이라 같은 경로(pip 전체 재설치)를 타므로 비슷할 것으로
  예상되나, 같은 조건으로 직접 측정하지는 않았다.

**더 공부 필요 / 다시 볼 것:**
- 캐시 — 캐시를 모듈 최상단에 둘지 핸들러 내부에 둘지에 대한 trade-off 공부
- MappingProxyType을 이용한 변경 불가 dict 만들기
- HTTP 에러코드

**다음 액션:** {→ 로 이어지는 순서 2~3개}

---

## 사용법
<!-- 실행 가능한 산출물(CLI·스크립트)이 있는 주차만. 없으면 이 절 삭제 -->

```bash
{실행 명령}
```

## 배운 것

- 캐시 구현, 모듈 내 캐시 위치에 따른 장단점
- uvicorn을 이용한 서버 호출
- 

## W{N} 완료 항목

- [X] `week08/model_service.py`가 노트북 없이 import되고 `load_model_once`가 같은 객체를 반환
- [X] `/health`가 W7 JSON의 `model_version`을 반환
- [X] `/predict` 정상 요청이 W7 노트북과 같은 확률을 반환
- [X] 열 순서를 뒤섞어도 같은 확률, 열 누락 시 `400` + 빠진 열 이름, 타입 오류 시 `422`
- [X] Docker 이미지 빌드 성공, 컨테이너에서 `/docs` 접속해 예측 1회 수신
- [X] `0.0.0.0` 바인딩을 빼면 접속이 안 되는 것을 직접 확인
- [ ] 서빙 경로 복잡도 표 + `n` 대 시간 로그-로그 그래프, 큰 `n`에서 기울기 ≈ 1
- [ ] `2/L`과 실제 발산 학습률 비교 표
- [ ] `week08/block_a_math_check.md`에 6개 항목 상태 기록
- [ ] IELTS 스피킹·라이팅 루틴 요일·시각 확정 + 첫 녹음 1회
- [ ] 알고리즘 2문제 + 재풀이 1문제 상태 기록
- [ ] Seq2Seq 리뷰 노트 `papers/2014-seq2seq.md`

## 최소 보장 체크
<!-- 커리큘럼 블록의 "이건 설명할 수 있어야 한다" 항목 -->

- [ ] {개념}을 {어떤 관점}에서 설명 가능


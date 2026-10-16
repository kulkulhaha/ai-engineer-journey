# W9 derivations — 재구현 실패 로그

> 형식 출처: `planning/chatgpt_weekly_prompt_v1.md` §2 「`derivations.md` 형식」
> Day 1 ① 재구현에서 **못 꺼낸 것만**, 항목당 3~8줄, **자기 말로** 쓴다. 설명을 붙여넣은 항목은 기록으로 치지 않는다.

## W9 Day 1 ① · `predict_proba_from_model` (W8)

**못 꺼낸 것**
- 빠진 열 검사
- pd.iloc[:,idx]

**막힌 지점** (다시 보니 왜 그랬는가)
- 타입변환(함수를 잘 안 읽음)

**다음에 꺼낼 단서**
- 

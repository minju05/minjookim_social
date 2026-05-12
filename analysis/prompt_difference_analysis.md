# GPT-4o 프롬프트 차이 분석 (57.2% → 70.1%, +12.9%p)

## 1. 프롬프트 비교표

| 항목 | 민주 버전 (57.2%) | 우리 버전 (70.1%) | 차이점 |
|---|---|---|---|
| **System 메시지** | ❌ 없음 | ✅ 있음 | 역할 명시로 문맥 이해 강화 |
| **Question Type 명시** | ❌ 미포함 | ✅ 포함 | 문제 유형별 추론 구분 |
| **출력 형식** | 자유 텍스트 | JSON 강제 | 파싱 일관성 ↑ |
| **파싱 방식** | Regex 추출 | JSON 디코드 | 정확도 ↑ |
| **지시문** | "Answer with only A, B, or C." | "Pick the single best answer. Respond in JSON..." | 명확성 ↑ |

---

## 2. 성능 차이 분석

### Q별 에러율 (정답이 다른 비율)
```
Q1 (help-MOST)         → 7.6% 차이   ✅ 가장 쉬움
Q3 (know-MOST)         → 15.1% 차이  ✅ 중간
Q4 (know-LEAST)        → 29.3% 차이  ⚠️ 어려움
Q2 (hinder-LEAST)      → 52.9% 차이  ❌ 가장 어려움
```

### 레이블별 에러율
```
social_goal      → 16.0% 차이   ✅ 직관적
belief           → 26.0% 차이   ⚠️ 중간
belief_of_goal   → 36.7% 차이   ❌ 복잡함
```

---

## 3. 핵심 문제: Q2 (hinder-LEAST)

### 문제의 복잡성
Q2는 **역방향 논리**를 요구합니다:

```
상황: 사람 X가 정보를 전달했고, 그 정보는 정확했음

Q2 질문:
  "X가 다른 사람을 방해하려 했다면, 
   다음 중 LEAST(가장 말이 안 되는) 것은?"

정답 논리:
  방해 의도 → 거짓 정보 줘야 함
  그런데 정확한 정보 줌 → 모순
  따라서 "정확한 정보 믿음" = LEAST likely true
```

### 우리 버전이 이기는 이유
1. **시스템 프롬프트**: "theory of mind and social reasoning 전문가"
   - 모델이 social reasoning context 인식
   
2. **question_type 힌트**: "belief", "social_goal", "belief_of_goal"
   - 명시적으로 어떤 추론이 필요한지 알려줌
   - belief는 "누가 뭘 믿는가"
   - social_goal은 "누가 뭘 목표로 하는가"

3. **JSON 강제 출력**: `{"choice_letter": "A", "reasoning": "..."}`
   - 선택지 뿐 아니라 reasoning도 함께 요구
   - 모델이 논리를 명시적으로 풀어서 답해야 함
   - 역논리 같은 복잡한 경우에 더 효과적

### 민주 버전이 틀리는 이유
```python
# 민주 버전: 단순 평문 처리
"Answer with only A, B, or C."
# → 모델이 자유롭게 답변
# → Q2의 역논리가 암묵적이라 놓치기 쉬움
# → 단순히 키워드 매칭으로 접근 가능
```

#### Q2 실제 에러 예시
```
EP 4017 Q2 (belief):
  상황: Sarah가 "장난감은 microwave에 있다"고 말했는데 
       Michael이 다른 곳에서 찾음 (거짓 정보)

  Q2: "Sarah가 Michael을 방해하려 했다면, 
       LEAST likely true는?"
  
  정답: A) Sarah는 microwave에 장난감이 있다고 믿었다

  민주: C ❌ (무언가 다른 것을 믿었다)
  우리: A ✓ (역논리 이해 - 방해 의도면 거짓 정보, 
              따라서 정확한 위치 믿음은 LEAST likely)
```

---

## 4. belief_of_goal 복잡성 (36.7% 차이)

이는 세 가지 신념 유형 중 **가장 층이 깊은 추론**:

```
belief:         "A는 X를 믿는가?"
                → 직접 관찰 가능한 정보

social_goal:    "A의 의도는 무엇인가?"
                → 행동 결과로 추론

belief_of_goal: "A는 다른 사람이 뭘 목표로 하는지 믿는가?"
                → 타인의 의도를 인식하고 있는가?
                → 가장 복잡한 심리 모델 필요
```

question_type 힌트가 이 구분을 명확히 해줌.

---

## 5. JSON 출력이 중요한 이유

reasoning을 함께 요구하면:

```json
// 우리 버전의 Q2 답변 예시
{
  "choice_letter": "A",
  "reasoning": "If Sarah intended to hinder Michael, she would provide 
               false information. Since she stated the toy was in the microwave 
               but it was elsewhere, she likely didn't believe the toy was 
               actually in the microwave. Therefore, 'Sarah believed the toy 
               was in the microwave' is LEAST likely true."
}
```

이렇게 하면:
1. 모델이 선택지만 고르는 게 아니라 **논리를 따라야 함**
2. 복잡한 역논리가 있을 때 더 정확함
3. 감시 신호(sanity check) 역할

---

## 6. 결론: +12.9%p의 구성

| 항목 | 개수 | 기여도 |
|---|---|---|
| 우리가 맞은데 민주 틀림 (순이득) | 176개 | +19.6%p |
| 민주가 맞은데 우리 틀림 (상쇄) | 60개 | -6.7%p |
| **순이득** | | **+12.9%p** |

### 개선의 주요 원천
1. **Q2 개선**: 119개 중 84개 우리가 이김 (70.6%)
2. **belief_of_goal 개선**: 110개 중 71개 우리가 이김 (64.5%)
3. **Q4 개선**: 66개 중 53개 우리가 이김 (80.3%)

---

## 7. 공통적으로 틀리는 부분 분석

### 전체 현황
```
총 유효 질문: 900개
우리 정답: 631개 (70.1%)
민주 정답: 515개 (57.2%)

- 우리만 맞음: 176개 (+19.6%p)
- 민주만 맞음: 60개 (-6.7%p)
- 둘 다 틀림: 209개 ← 이게 GPT-4o의 근본 한계
- 둘 다 맞음: 664개
```

### 둘 다 틀리는 209개의 성격

**같은 오답 선택: 171개 (81.8%) — 시나리오 모호성/모델 한계**
```
레이블별:
  social_goal:    88개 (가장 높음!)
  belief_of_goal: 65개
  belief:         18개

Q 유형별:
  Q3(know-MOST):  65개
  Q4(know-LEAST): 62개
  Q2(hinder):     33개
  Q1(help):       11개
```

**다른 오답 선택: 38개 (18.2%) — 프롬프트 방향성 차이**
```
Q2(hinder-LEAST): 17개 ← 역논리 처리 방식 차이
belief:           17개 ← 신념 추론 전략 차이
Q3(know-MOST):    12개
Q4(know-LEAST):   8개
```

---

## 8. 근본적 문제: 같은 오답 171개 분석

### 문제 1: "help" vs "prevent" 구분 능력

**사례: EP 4009 (모든 모델 공통 약점)**
```
상황:
  Mary: "wine이 어디에?"
  John: "kitchen cabinet에 있다" → 정보가 틀림
  Mary: fridge에서 발견

Q3 정답: B (prevent - John은 위치를 알았으면 거짓말)
Q4 정답: A (help - 다른 위치에서 찾도록)

우리의 답: Q3→A(help), Q4→B(prevent) ✗
민주의 답: Q3→A(help), Q4→B(prevent) ✗

분석:
  - 둘 다 정반대로 선택
  - 비슷한 에피소드에서 반복되는 패턴
  - 이건 시나리오의 intention이 모호하거나
    모델이 "누가 거짓말 했으니 prevent"라는 논리를 놓침
```

### 문제 2: Q3/Q4 (know-MOST/LEAST)의 높은 어려움

**원인 추정**:
```
Q3/Q4는 "assuming that X knows what is inside..."
→ 동영상에서 X가 **실제로 뭘 알았는지** 명확하지 않을 수 있음

예:
- X가 그 위치를 열어본 적 있나?
- X가 정보를 받았나?
- 비디오 초점이 X의 행동에만 있지 그 지식 상태는 안 보여짐?

모델은 텍스트 설명만 보므로:
  "X는 Y를 안다"는 것이 실제로 비디오에서 확인 가능한지 
  판단하기 어려움
```

### 문제 3: social_goal에서 가장 높은 실패율

**패턴**:
```
social_goal: 88개 모두 같은 오답 ← 체계적 오류
belief:      18개만 같은 오답 ← 덜 일관적

이유: social_goal은 "누가 뭘 목표로 했나"라는
      가장 high-level의 추론
      → 더 많은 intermediate reasoning 필요
      → 각 단계에서 실패할 확률 증가
```

---

## 9. 프롬프트로 해결 불가능한 문제들

### 프롬프트로 해결됨 (이미 우리가 함)
```
✅ Q1 (help-MOST): 7.6% → 더 이상 개선 어려움
✅ Q2 반은 해결 (52.9%에서 우리가 70.6% 승률)
✅ belief는 26% 차이 상당 부분 개선
```

### 프롬프트로 해결 불가능 (모델 능력 문제)
```
❌ social_goal 의도 구분: help vs prevent 모호성
   → 88개가 같은 오답 선택 = 둘 다 같은 방식으로 해석 실패
   → 프롬프트 좋아져도 비디오에서 의도가 애매하면 한계

❌ Q3/Q4의 "assumes that X knows": 127개 같은 오답
   → 텍스트에서 X의 지식 상태 파악이 불충분
   → 비디오 분석/frame selection 개선 필요할 수도

❌ belief_of_goal의 복잡성: 65개 같은 오답
   → 세 layer의 신념 모델 (A가 B가 C를 믿는다고 믿음)
   → 현재 프롬프트 수준으로는 한계
```

---

## 10. 결론: 12.9%p 개선의 한계

### 현재 상태
```
프롬프트 최적화: 57.2% → 70.1% (+12.9%p)
   우리가 추가 정답: 176개
   모델 근본 한계: 209개

남은 개선 가능성:
  - 다른 오답 선택 38개는 추가 프롬프트로 조정 가능
  - 하지만 같은 오답 171개는 모델 한계 또는 시나리오 모호성
  → 현재 GPT-4o의 "상한선"이 71~72% 정도일 가능성
```

### 추가 개선안 (순서대로 시도)
```
1. 프롬프트 미세조정 (같은 오답 38개 타겟)
   - Q2의 다른 오답 17개: 역논리 더 강조
   - belief의 다른 오답 17개: 신념 추론 로직 명시

2. 비디오 frame 최적화
   - Q3/Q4 (know-MOST/LEAST): 관련 frame 더 자세히
   - detail="high"로 재실험?

3. 모델 업그레이드
   - GPT-4o → GPT-4-turbo 또는 다른 모델
   - 더 강력한 추론 모델 필요

4. 멀티-홉 프롬프트
   - step 1: 시나리오 요약
   - step 2: 각 인물의 지식 상태 명시
   - step 3: 의도 추론
```

---

## 11. 권장사항

프롬프트 최적화 시:
1. **명시적 역할**: System 메시지로 전문가 역할 부여 ✅
2. **문제 타입 구분**: 어려운 추론일수록 더 중요 ✅
3. **구조화된 출력**: JSON으로 reasoning 함께 요구 ✅
4. **명확한 지시**: "Pick the single best answer" vs "Answer with" 🔄
5. **프롬프트 한계 인식**: 모델이 209개(23%)는 못 푼다는 것을 알고 있기

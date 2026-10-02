# EPJ Data Science 원분야 복원 가능성 파일럿 프로토콜 v0.1

## 1. 목적

이 파일럿은 ICML·NeurIPS accepted-paper ecosystem에 새롭게 등장한 제1저자의
진입 전 학문 분야를 재현 가능하게 복원할 수 있는지 판정한다. 결과 방향을
탐색하거나 ChatGPT 효과를 검정하는 단계가 아니다.

핵심 질문은 다음 하나다.

> DBLP PID로 식별된 ICML·NeurIPS 신규 제1저자의 진입 전 5년 출판 이력을
> 충분한 정확도와 범위로 복원하여, 학문 분야 이동 분석에 사용할 수 있는가?

## 2. 논문에서 궁극적으로 관찰할 구성개념

1. **Disciplinary origin**: 진입 전 5년 출판 포트폴리오에서 관측되는 분야 구성
2. **Entry channel**: 진입 논문에 최근 Top-4 경험 공저자가 있었는지 여부
3. **Accepted-paper reappearance**: 정확히 2개 publication cycle 뒤 Top-4
   accepted paper에 재등장했는지와 최초 공저자 동반 여부

`entry`, `return`, `retention`은 연구 시작·투고·학계 잔류를 뜻하지 않는다.
모두 accepted-paper records 안에서 관측되는 proxy다.

## 3. 고정 범위

- 결과 학회: ICML, NeurIPS
- 학회 기록 자료: DBLP accepted proceedings roster
- 신규 진입 proxy: 해당 연도 이전 5개 cycle 동안 AAAI·ICLR·ICML·NeurIPS
  accepted history가 없는 first-listed author
- 식별자: DBLP PID (`author_key`)
- 진입 전 분야 관찰창: index year 직전 5개 연도
- 파일럿 index years: 2018, 2022, 2024
- 표본층: index year × index venue × experienced-coauthor status
- 표본 추출: 고정 seed 20260826, 층별 최대 20명
- main longitudinal analysis의 완전한 +2 follow-up 대상: 2018–2022 cohort
- 2023–2024 cohort: origin과 entry channel의 기술 분석만 허용

## 4. 진입 코호트 규칙

기존 persistence/pathway 분석 정의를 유지한다.

- ICML 또는 NeurIPS accepted paper의 first-listed author여야 한다.
- `top4_newcomer_5y == True`여야 한다.
- `identity_source == dblp_pid`이고 `author_key`가 비어 있지 않아야 한다.
- 같은 사람이 같은 index year에 여러 편을 가진 경우 사람-연도 한 행으로 합친다.
- 같은 index year에 두 학회에 모두 등장하면 `index_venue = Both`로 유지한다.
- experienced coauthor는 진입 논문의 비제1저자 중
  `top4_newcomer_5y == False`인 저자가 한 명 이상 관측된 경우다.

## 5. 원분야 자료 계층

### 5.1 DBLP 계층

DBLP PID에 연결된 canonical author name과 alias를 사용하여 진입 전 5년의
DBLP publication records를 수집한다. DBLP는 CS 내부 이동을 보는 1차 진단이며,
DBLP에 이력이 없다는 사실을 비-CS 출신으로 해석하지 않는다.

DBLP 파일럿 분류는 다음 진단 범주만 사용한다.

- `core_ml`
- `adjacent_ai`
- `other_cs_known`
- `bio_med_visible_in_dblp`
- `unclassified_dblp`
- `no_observed_dblp_history`
- `mixed_or_tied`

`journals/corr`은 이전 출판의 존재를 확인하는 데는 사용하지만 분야 판정
분모에서는 제외한다.

### 5.2 OpenAlex 계층

index paper를 제목·연도·저자명으로 OpenAlex work에 연결하고, 그 authorship에서
entrant의 OpenAlex author ID를 찾는다. 이후 진입 전 5년 works의 topic/field를
수집한다.

- index-paper work match와 entrant-author match를 분리하여 기록한다.
- 이름 검색만으로 저자 ID를 직접 확정하지 않는다.
- 한 index paper에서 복수 후보가 남으면 ambiguous로 둔다.
- OpenAlex의 무관측 prior works를 학술 데뷔로 해석하지 않는다.

## 6. 파일럿 판정 지표

### DBLP 진단

- sample size와 층별 표본 수
- 진입 전 DBLP history 보유율
- 분야 분류 가능한 저자 비율
- `unclassified_dblp` 및 `no_observed_dblp_history` 비율
- 초기(2018)와 최근(2024) 사이의 관측률 차이
- canonical-name-only와 alias-enhanced retrieval 차이

### OpenAlex 진단

- index paper match rate, venue × year별
- matched work 안에서 entrant author ID resolution rate
- entrant별 진입 전 work 보유율
- high/medium/ambiguous origin classification 비율
- venue × year별 coverage 차이

## 7. 사전 판정 규칙

### BROAD_ORIGIN_GO

다음을 모두 만족하면 OpenAlex 기반 broad scientific origin을 본 분석으로 연다.

- 각 venue × year 파일럿 셀의 index-paper match rate가 0.80 이상
- matched work 중 entrant-author ID resolution rate가 0.90 이상
- resolved author 중 origin이 high 또는 medium confidence로 분류되는 비율이
  0.75 이상
- 주요 시점 간 origin-resolved coverage 차이가 0.10 이하
- 별도로 뽑은 감사 표본에서 author linkage precision이 0.95 이상

### WITHIN_CS_ONLY

OpenAlex gate는 실패하지만 DBLP 기반 known-field 분류가 전체와 각 주요 시점에서
0.70 이상이면, 주장을 CS 내부·인접 분야 이동으로 축소한다.

### ORIGIN_BLOCKED

OpenAlex broad-origin gate와 DBLP within-CS gate가 모두 실패하면 migration matrix를
주 결과로 사용하지 않는다. 기존 collaboration channel과 accepted-paper
reappearance 분석만 유지한다.

임계값은 결과를 확인한 뒤 낮추지 않는다. 실패한 gate는 limitation으로 공개한다.

## 8. 허용되는 해석

- “ICML·NeurIPS accepted programs에서 관측된 신규 제1저자의 이전 출판
  포트폴리오가 변화했다.”
- “관측된 진입자 중 경험 공저자를 동반한 비율이 원분야별로 달랐다.”
- “accepted-paper reappearance가 원분야와 진입 협업 경로에 따라 달랐다.”

## 9. 금지되는 해석

- ChatGPT 또는 LLM이 변화를 일으켰다는 인과 주장
- accepted paper 자료를 이용한 투고 장벽·채택 확률 주장
- DBLP/OpenAlex 무관측을 실제 학술 데뷔나 비-CS 출신으로 간주
- ICML·NeurIPS 결과를 AI 전체 생태계로 일반화
- accepted-paper 미재등장을 연구 중단이나 학계 이탈로 해석
- 결과를 보고 분류표·기간·gate를 유리하게 변경

## 10. 파일럿 이후 본 분석의 고정 후보

- RQ1: 연도별 fractional origin composition과 origin → target migration matrix
- RQ2: `experienced coauthor ~ origin × period + venue + team size + prior productivity`
- RQ3: exact +2의 세 경로를 outcome으로 한 multinomial model과
  entrant-level bootstrap g-standardization

국가, 기관 prestige, gender, citation, LLM adoption, 신규 학회 확장은 이 파일럿과
본 EPJ 논문의 완료 조건에 포함하지 않는다.


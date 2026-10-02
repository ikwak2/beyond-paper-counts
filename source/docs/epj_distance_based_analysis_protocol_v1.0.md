# EPJ Data Science Distance-Based Analysis Protocol v1.0

## 1. 연구 목적

ICML·NeurIPS의 채택 프로그램에 새롭게 제1저자로 등장한 연구자의 진입 전 출판 포트폴리오가 진입 논문과 얼마나 떨어져 있었는지 측정하고, 최근 Top-4 경험 공저자가 그 거리와 정확한 +2 publication-cycle 재등장 경로에 어떻게 연관되는지 분석한다.

이 연구는 LLM 또는 ChatGPT의 인과효과를 추정하지 않는다. 또한 제출·심사·채택 전 과정을 관찰하지 않으므로 `학회 접근성`이나 `채택 확률`을 직접 측정한다고 표현하지 않는다.

## 2. 분석 모집단과 단위

- index venues: ICML, NeurIPS
- index years: 2018–2024
- unit: stable DBLP PID로 관측되는 저자-최초진입 코호트
- eligibility: 해당 연도 ICML·NeurIPS 채택논문의 first-listed author이며 `top4_newcomer_5y = true`
- `top4_newcomer_5y`: 직전 5 publication cycles 동안 AAAI·ICLR·ICML·NeurIPS 채택논문 저자목록에 관측되지 않은 저자
- index year: 2018–2024 eligible appearance 중 저자별 최초 연도
- 재등장 분석: exact +2 cycle이 완전 관측되는 2018–2022 entrants만 사용

이는 `실제 생애 최초 AI 연구자`가 아니라 `관측된 Top-4 채택 이력 기준 신규 저자 proxy`다.

## 3. Index-cycle 구성

한 entrant가 index year에 여러 ICML·NeurIPS 채택논문의 제1저자이면 모두 index portfolio에 포함한다.

- `n_index_papers`: index-year eligible first-listed papers 수
- `entry_team_size_mean`: index papers의 평균 저자 수
- `index_venue`: ICML, NeurIPS, 또는 Both
- `entry_with_experienced_top4_coauthor`: index papers 중 하나라도, first-listed entrant 이외 공저자 가운데 그 시점 `top4_newcomer_5y = false`인 저자가 있으면 1

이 변수는 관측된 최근 Top-4 경험 공저자 구성이다. 멘토링, 사회적 관계 또는 네트워크 효과가 아니다.

## 4. 진입 전 출판 포트폴리오

### Primary

- source: local DBLP XML dump
- window: index year 직전 5년
- record types: `article`, `inproceedings`
- identity: DBLP PID person record의 canonical name 및 alias
- alias 사용 조건: 전체 DBLP person record에서 해당 문자열이 한 PID에만 속할 때만 사용
- 한 저자 안에서 정규화된 제목 중복 제거
- 제목이 없는 record 제외

### Prespecified sensitivity

Primary 유형에 `incollection`, `book`, `phdthesis`, `mastersthesis`를 추가한다.

과거 제목이 하나도 회수되지 않은 entrant는 거리값을 대체하지 않는다. 연도·venue·경험공저자별 이력 관측률을 별도 결과로 보고하고, 거리 분석은 `observed prior-title portfolio`에 조건부임을 명시한다.

## 5. 거리 측정

### Primary semantic distance

- base model: `allenai/specter2_base`
- adapter: `allenai/specter2` proximity adapter
- input: `title + tokenizer.sep_token` (abstract unavailable)
- pooling: CLS token
- paper vector: L2-normalized 768-dimensional vector
- prior portfolio vector: prior-paper vectors의 평균 후 L2 normalization
- index portfolio vector: index-paper vectors의 평균 후 L2 normalization
- distance: `1 - cosine(index portfolio, prior portfolio)`

명칭은 `SPECTER2 title-only prior-to-index portfolio distance`로 고정한다. 전공·학위·인지적 거리 또는 논문 전체 내용의 거리라고 확대해석하지 않는다.

### Prespecified measurement sensitivities

1. word TF-IDF title distance
2. character TF-IDF title distance
3. all-scholarly-record-type SPECTER2 title distance

primary와 sensitivity는 모두 보고하며, 유리한 결과를 선택하지 않는다.

## 6. 연구질문과 추정량

### RQ1. 진입 전 포트폴리오 거리는 2018–2024년에 어떻게 변했는가?

- 연도별 N, 이력 관측률, 거리 median/IQR
- entrant bootstrap 2,000회 median 95% interval
- 보조 조정모형: `distance ~ C(index_year) + C(index_venue) + log1p(entry_team_size_mean) + log1p(n_index_papers) + log1p(n_prior_titles)`
- g-standardized adjusted mean distance by year

연도 전체 trajectory를 보고하며 endpoint만 선택하지 않는다. 특정 기술의 인과효과로 해석하지 않는다.

### RQ2. 더 먼 포트폴리오에서 진입한 entrant는 최근 Top-4 경험공저자와 함께 나타나는가?

- 모집단: prior distance가 관측되고 `entry_team_size_mean >= 2`인 2018–2024 entrants
- primary logistic model:
  `experienced_coauthor ~ z(distance) + C(index_year) + C(index_venue) + natural spline(log1p(team size), df=3) + log1p(n_index_papers) + log1p(n_prior_titles)`
- 보고: 거리 Q25와 Q75에서 g-standardized 경험공저자 확률 및 차이
- sensitivity: distance natural spline(df=3), team-size overlap `2 <= mean <= 13`, Both venue 제외

이는 연관성 분석이며 경험공저자가 진입을 일으켰다는 효과가 아니다.

### RQ3. 거리와 경험공저자 구성은 exact +2 cycle 재등장 경로와 어떻게 연관되는가?

- 모집단: prior distance가 관측되고 `entry_team_size_mean >= 2`인 2018–2022 entrants
- primary return universe: AAAI·ICLR·ICML·NeurIPS 채택논문
- mutually exclusive outcome:
  1. `no_recurrence`
  2. `coauthor_continuity_only`: 모든 exact +2 논문이 index-cycle 공저자를 한 명 이상 포함
  3. `at_least_one_no_index_coauthor`: exact +2 논문 중 적어도 한 편은 index-cycle 공저자를 포함하지 않음
- multinomial model:
  `outcome ~ z(distance) * experienced_coauthor + C(index_year) + C(index_venue) + natural spline(log1p(team size), df=3) + log1p(n_index_papers) + log1p(n_prior_titles)`
- 보고: distance Q25/Q75 × experienced yes/no의 g-standardized pathway probabilities
- entrant bootstrap 2,000회 95% interval
- return-universe sensitivities: ICLR·ICML·NeurIPS, ICML·NeurIPS only

publication cycle은 정확한 경과시간 proxy가 아니며, 공저자 연속은 과학적 독립성을 의미하지 않는다.

## 7. Missingness 및 측정 QA

- stable PID coverage와 prior-title coverage를 연도·venue·experienced-coauthor별로 보고
- primary complete-case와 no-prior-history 집단의 관측 가능한 team size, venue, cohort 분포 비교
- alias collision 수, person record 미회수 수, 제목 없는 이력 수 보고
- full cohort의 무작위 100명에 대해 matched alias, PID, prior-year window, 제목 중복 제거 규칙을 자동 감사표로 생성
- 개인 수준 파일은 private output에만 저장

## 8. 판정 규칙

- primary와 세 measurement sensitivity의 주요 부호가 같음: `DISTANCE_MEASUREMENT_DIRECTIONALLY_ROBUST`
- 부호는 같지만 효과크기 또는 interval이 크게 다름: `DIRECTIONALLY_ROBUST_QUANTITATIVELY_SENSITIVE`
- 주요 부호가 다름: `DISTANCE_MEASUREMENT_SPECIFICATION_SENSITIVE_WARNING`
- 부호가 불리하거나 0에 가까워도 연구질문, 모집단, primary 사양을 변경하지 않는다.

## 9. 중단 규칙

다음 본 분석과 사전 고정 sensitivity 이후 새 outcome, 새 학회, 국가·성별·기관 prestige·citation·network centrality 분석을 열지 않는다. hard categorical origin 분류와 OpenAlex author-origin 수집도 다시 열지 않는다.

최종 결과는 accepted-program participation의 측정 연구로 작성한다.

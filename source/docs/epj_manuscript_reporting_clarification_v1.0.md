# EPJ manuscript reporting clarification v1.0

## Status

`REPORTING_CLARIFICATION_ONLY`

이 문서는 동결된 모집단, 결과, 모형 또는 추정량을 변경하지 않는다. EPJ Data Science 모의 원고에서 검열, 거리 캘리브레이션, venue universe와 결과 위계를 명확히 보고하기 위한 편집 규칙이다.

## 1. Exact +2 risk set

- 전체 관측 신규 진입자: 2018–2024, 12,094명
- exact +2 outcome eligible: 2018–2022, 6,704명
- 2023–2024 진입자는 2025–2026 결과가 동결 자료에 없으므로 exact +2 분석에서 제외
- primary exact +2 model: prior distance 관측, 평균 팀 규모 2명 이상인 5,122명
- 따라서 2023–2024 entrant를 `no_recurrence`로 분류하지 않으며, 완전한 +2 관측창이 있는 동일 risk set만 비교한다.

## 2. Why exact +2 rather than within +2

Exact +2는 결과 확인 전 동결된 고정-horizon estimand다. 모든 eligible cohort를 동일한 publication-cycle 위치에서 비교하며, 즉시 +1 재등장과 +2 재등장을 하나의 누적 사건으로 합치지 않는다.

`within +2`는 +1과 +2의 여러 기회를 합산하는 cumulative estimand이며 다음을 변경한다.

1. 사건 기회의 수
2. 공저자 연속성 outcome의 분류 규칙
3. 즉시 반복 출판과 이후 재등장의 혼합 비중

따라서 원고에서는 exact +2를 고정된 단기 재관측으로 설명한다. Publication cycle은 동일한 달력시간이 아니며, no record는 경력 이탈을 뜻하지 않는다.

## 3. Venue-universe asymmetry

- index venues: ICML, NeurIPS
- lookback and primary return universe: AAAI, ICLR, ICML, NeurIPS

ICML·NeurIPS는 공식 proceedings 범위와 지리 데이터가 동결된 index population이다. 더 넓은 Top-4 lookback은 AAAI 또는 ICLR 최근 기록이 있는 저자를 신규 진입자로 잘못 분류하는 것을 줄인다. Top-4 return universe는 다른 두 학회에서의 재등장을 `no_recurrence`로 잘못 분류하는 것을 줄인다.

이 비대칭은 구성개념 선택이며 전체 AI를 대표하지 않는다. Return-universe sensitivity로 ICLR·ICML·NeurIPS와 ICML·NeurIPS only를 모두 보고한다.

## 4. Distance discriminant-validity benchmark

전체 분석 전에 동결된 2018/2022/2024 층화 파일럿 306명을 사용했다. 거리 관측 249명 각각에 대해 같은 index year의 다른 저자 prior portfolio를 100회 무작위 배정하고, 실제 자기 prior-to-index 거리와 무작위 거리 중앙값을 비교했다.

- own portfolio closer than randomized median: 73.49%
- paired median margin, randomized minus own: 0.02615
- observed-distance IQR: 0.04848
- word TF-IDF Spearman correlation: 0.6921
- prespecified validation gates: 6/6 PASS

이 벤치마크는 0.09 부근의 절대값이 임베딩 바닥에 붙은 상수인지 점검한다. 전체 효과크기 또는 연구가설 검정으로 사용하지 않는다. 본문에는 판별타당도 결과를 제시하고, 실제 거리와 무작위 참조분포의 전체 그림은 보충자료에 둔다.

## 5. Result hierarchy

### Headline results

1. 생산량 보정 지리적 대표성과 그 측정·결측 민감도
2. 캘리브레이션된 title-only distance의 2018–2024 trajectory
3. 경험 공저자와 exact +2 재등장 경로의 조정된 연관성

### Diagnostic result

거리 Q25–Q75와 경험 공저자 동반 확률의 +0.5%p 대비는 재등장 모형의 해석을 위한 사전 진단으로 보고한다. 독립적인 headline claim으로 승격하지 않는다.

### Required negative finding

Distance-by-coauthor interaction과 Q75–Q25 pathway contrasts의 불확실성·측정 민감도는 숨기지 않는다. 이는 경험 공저자 연관성이 먼 진입자에게 특유하다는 주장을 제한한다.

## 6. Reporting contribution

서론과 초록은 다음 실질적 함의를 명시한다.

> 선택적 학회의 다양성 보고는 원시 국가 비중만으로 끝내지 않고, 생산량 분모, 신규 관측자 정의, 완전한 재관측 risk set, 공저자 연속성 및 각 지표의 결측 민감도를 함께 제시해야 한다.

## 7. Availability statement requirements

최종 원고에는 `Availability of data and materials`를 둔다.

- 공개: 분석 코드, 동결 config/protocol, aggregate tables, figure inputs, QA manifests와 checksums
- 저장소: GitHub 공개 release와 Zenodo DOI를 제출 전에 확정
- 원자료: DBLP와 OpenAlex를 정식 인용하고 각 이용조건에 따라 접근
- 제한: 결합된 저자·논문·기관 row-level 파일은 재식별 및 데이터 이용조건 때문에 직접 배포하지 않음
- 재현: 공개 원자료에서 분석 입력을 재구성하는 스크립트와 aggregate verification을 제공

저장소 URL과 DOI가 생기기 전에는 placeholder를 실제 공개 주장으로 바꾸지 않는다.

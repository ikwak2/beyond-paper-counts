# EPJ Distance Model Implementation Note v1.0a

이 문서는 protocol v1.0의 연구질문을 코드 수준으로 명확히 하며, 결과를 보기 전에 작성한다. 모집단·outcome·primary measurement를 변경하지 않는다.

## RQ1 focal estimands

1. 연도별 observed-history conditional median distance와 entrant-bootstrap 95% interval
2. 2018을 기준으로 한 연도별 g-standardized adjusted mean distance
3. 2018–2024 전체 trajectory를 모두 표시

## RQ2 focal contrast

- 같은 분석 표본의 distance Q75와 Q25에서 g-standardized `experienced coauthor` 확률을 계산한다.
- focal contrast는 `P(experienced | Q75) - P(experienced | Q25)`다.
- 양수는 더 먼 prior portfolio를 가진 entrant가 경험공저자와 함께 나타날 확률이 높다는 기술적 연관성을 뜻한다.
- 95% interval은 HC3 covariance에서 2,000회 coefficient draw로 계산한다.

## RQ3 focal contrasts

distance Q25/Q75와 experienced coauthor 0/1의 네 시나리오에서 세 pathway의 g-standardized 확률을 계산한다.

1. 각 distance 수준에서 `experienced - no experienced` pathway probability difference
2. 각 experienced 상태에서 `Q75 - Q25` pathway probability difference
3. 각 pathway의 difference-in-differences

primary SPECTER2 + Top-4 return universe는 entrant bootstrap 2,000회 interval을 계산한다. 사전 고정 measurement 및 return-universe sensitivities는 동일 모형의 point estimate를 모두 보고한다.

## Measurement robustness focal signs

다음 부호를 primary와 word TF-IDF, character TF-IDF, all-record-type SPECTER2에서 비교한다.

1. RQ2 Q75-minus-Q25 experienced-coauthor probability
2. RQ3 Q75 distance에서 experienced-minus-no-experienced의 세 pathway contrasts
3. RQ3 experienced 상태별 Q75-minus-Q25의 세 pathway contrasts

하나라도 주요 부호가 바뀌면 해당 contrast를 specification-sensitive로 표시한다. 유의성 또는 유리한 크기를 기준으로 사양을 제외하지 않는다.

## Missing history

거리 미관측 entrant는 0이나 평균으로 대체하지 않는다. complete-case target population을 명시하며, no-history 집단과 observed-history 집단의 year, venue, team size, experienced-coauthor 분포를 병렬 표로 보고한다.

## Pre-full-run numerical QA amendment

이 절은 소규모 수치 스모크 테스트에서 구현상 식별성 문제를 발견한 뒤, 2,000회 confirmatory bootstrap을 실행하기 전에 추가했다. 연구질문, 모집단, outcome, primary distance 또는 focal contrast는 변경하지 않는다.

- `patsy.cr(..., df=3, constraints='center')`를 사용해 protocol의 natural spline(df=3)을 절편과 선형독립인 basis로 구현한다.
- 모든 RQ1·RQ2·RQ3 설계행렬에 대해 column rank를 계산하고 full-rank가 아니면 최종 QA를 통과시키지 않는다.
- 다항모형은 Newton 적합이 비수렴하면 BFGS로 재적합하고, 재적합도 비수렴이면 해당 bootstrap replicate를 성공으로 집계하지 않는다.
- all-record-type SPECTER2 sensitivity는 같은 확장 이력에서 계산한 `n_unique_alltype_prior_titles`로 조정한다. Primary SPECTER2와 두 TF-IDF 사양은 primary record-type 이력 수를 유지한다.
- 그림의 오차막대 길이는 렌더링 단계에서만 0 이상으로 제한한다. 저장되는 점추정치와 interval endpoint는 변경하지 않는다.

수정 전 스모크 적합에서는 spline basis가 절편과 선형종속이어서 점추정치는 유한했으나 HC3 coefficient-draw interval이 비정상적으로 넓었다. 중심화 제약 후 RQ1은 12/12, primary RQ2·RQ3는 15/15 column full-rank였고, 10회 RQ3 시험 bootstrap은 10/10 수렴했다.

## Post-run reporting-completeness amendment

첫 2,000회 본실행이 PASS한 뒤 protocol의 missing-history 보고 대조표를 점검하면서 year·venue 분포가 단일 요약표에 충분히 전개되지 않은 것을 확인했다. 모형, 추정량, seed와 결과값은 변경하지 않고 다음 공개 marginal 표와 QA gate만 추가한 뒤 동일 seed로 전체 파이프라인을 재실행한다.

- year, index venue, experienced-coauthor 각각의 거리 관측률과 observed/no-history 내부 분포를 별도 long-form 표로 저장하며, 모든 공개 marginal cell이 사전 설정한 최소 20명을 충족해야 PASS한다. 작은 full-cross cell은 공개하지 않는다.

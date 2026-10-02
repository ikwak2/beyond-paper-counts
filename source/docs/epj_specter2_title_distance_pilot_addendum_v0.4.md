# EPJ SPECTER2 Title-Distance Pilot Addendum v0.4

## 목적

이 단계는 306명 동결 표본에서 과학문헌 임베딩 기반 거리 측정이 기술적으로 가능하고 최소한의 판별타당도를 갖는지 확인한다. 연구 가설을 검정하거나 유리한 모형을 고르는 단계가 아니다.

## 고정 입력

- 표본: 기존에 동결된 ICML·NeurIPS 2018/2022/2024 신규 제1저자 306명
- 이력: DBLP에서 회수된 index year 직전 5년의 publication title
- index document: 신규 제1저자의 ICML·NeurIPS 진입 논문 title
- 초록은 현재 균일하게 관측되지 않으므로 사용하지 않는다.

## 임베딩 사양

- base model: `allenai/specter2_base`
- adapter: `allenai/specter2` proximity adapter
- input: `title + tokenizer.sep_token` (empty abstract)
- pooling: last hidden state의 CLS token
- vector normalization: 논문별 L2 normalization
- author prior portfolio: prior-title vector들의 산술평균 후 L2 normalization
- primary distance: `1 - cosine(index title vector, prior portfolio centroid)`
- lexical sensitivity: 기존 word TF-IDF 및 character TF-IDF distance

이 측정치는 `진입 전 출판 제목 포트폴리오와 진입 논문 제목 간 거리`다. 저자의 전공, 학위, 인지적 능력, 논문 전체 내용의 거리를 직접 측정한다고 표현하지 않는다.

## 사전 통과 기준

다음을 모두 만족할 때만 전체 코호트로 확장한다.

1. 거리 관측률이 전체 및 각 표본연도에서 70% 이상
2. 관측 거리의 99.9% 이상이 유한값
3. primary distance의 IQR이 0.02 이상
4. word TF-IDF distance와의 Spearman 상관이 0.15 이상
5. 각 저자에 대해 연도 내 다른 저자의 portfolio를 100회 무작위 대입한 거리의 중앙값을 구하고, 실제 저자 자신의 prior portfolio distance가 그 값보다 작은 비율이 55% 이상
6. 실제 거리와 순열 거리의 paired median margin(`permuted - own`)이 0보다 큼

기준 실패 시 threshold를 낮추거나 다른 임베딩을 결과를 본 뒤 primary로 교체하지 않는다. 실패 원인을 보고하고 설계를 재검토한다.

## 제외 및 해석 금지

- LLM 인과효과 검정 없음
- 국가·성별·기관 prestige 분석 없음
- hard disciplinary-origin label 생성 없음
- prior title이 없는 저자를 거리 0 또는 평균값으로 대체하지 않음
- 파일럿 결과를 본 분석 효과크기로 보고하지 않음

## 다음 단계

PASS 시에만 2018–2024 ICML·NeurIPS accepted Top-4-new first-listed cohort 전체에 같은 사양을 적용한다. +2-cycle 결과가 완전 관측되는 재등장 분석은 2018–2022 cohort로 제한한다.

# EPJ title-portfolio distance smoke addendum v0.3

## 배경

사전 고정한 hard origin classification은 DBLP-only와 OpenAlex-author-profile
경로 모두 coverage gate를 통과하지 못했다. 이 addendum은 그 threshold를 낮추지
않는다. 대신 원 계획의 knowledge-distance component가 로컬 데이터만으로
식별 가능한지 별도의 구성개념으로 검사한다.

## 구성개념

`title_portfolio_distance`는 진입 전 5년 DBLP 논문 제목 포트폴리오와 index
ICML/NeurIPS 논문 제목 사이의 lexical-semantic distance다.

이는 학문 분야, 연구 품질, 실제 인지적 거리 또는 비-CS 출신을 직접 측정하지 않는다.

## 표본과 텍스트

- frozen 306-author pilot sample 전체
- 진입 전 5년 canonical-name DBLP history
- 같은 저자의 정규화 제목 중복은 한 번만 사용
- prior title이 없는 저자는 `no_observed_prior_title`로 남기고 거리를 대입하지 않음

## 사전 고정 표현

### Primary

- English word TF-IDF
- n-gram: 1–2
- English stop words 제거
- sublinear term frequency
- corpus minimum document frequency: 2
- 거리: `1 - cosine(index title, mean prior-title vector)`

### Sensitivity

- character-within-word TF-IDF
- n-gram: 3–5
- sublinear term frequency
- 같은 cosine-distance 정의

## 파일럿 gate

다음을 모두 만족하면 `TITLE_DISTANCE_FEASIBLE`이다.

- 거리 관측률이 전체 및 2018·2022·2024 각각 0.70 이상
- primary distance의 IQR이 0.05 이상
- primary와 sensitivity distance의 Spearman correlation이 0.60 이상
- 유효 표본의 finite distance 비율이 0.95 이상

실패하면 title-distance를 본 분석에 사용하지 않는다. 통과하더라도 전체 분석 전에
과학 문헌용 embedding 또는 OpenAlex topic subset과의 validation을 별도로 설계한다.

## 허용되는 다음 질문

> ICML·NeurIPS accepted programs에 나타난 신규 제1저자의 진입 전 publication-title
> portfolio와 index paper 사이의 거리는 시간과 협업 경로에 따라 어떻게 달랐는가?

ChatGPT 인과효과, 학문 분야 이동률, 비-CS 유입률로 해석하지 않는다.


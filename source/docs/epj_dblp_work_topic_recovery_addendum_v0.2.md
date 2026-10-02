# EPJ DBLP-work → OpenAlex-topic recovery addendum v0.2

## 목적

OpenAlex author-profile smoke에서 최근 cohort의 이전 works가 과소관측되는 문제가
발견되었다. 이 addendum은 저자 프로필을 확장 해석하지 않고, 로컬 DBLP에서 이미
관측된 진입 전 논문을 제목·연도·저자명으로 OpenAlex work에 직접 연결하여
분야 topic을 회수할 수 있는지 검사한다.

이 방법이 통과해도 broad scientific origin 전체를 식별했다는 뜻이 아니다.
허용되는 구성개념은 **DBLP-observed prior publication portfolio enriched with
OpenAlex topics**다.

## 고정 표본과 요청 상한

- `epj_openalex_origin_smoke_sample_private.csv`의 동일한 60명
- canonical-name DBLP scan에서 찾은 진입 전 5년 records만 사용
- 총 대상 record: 실행 시 manifest에 고정
- OpenAlex 요청 credit 상한: 4,500
- remaining credit이 1,500 미만이면 중단
- 응답은 DBLP key별 private cache에 저장하고 재호출하지 않음

## work 연결 규칙

- 후보 검색: DBLP title, publication year ±1
- entrant canonical name이 OpenAlex authorship에서 유일하게 일치해야 함
- high: exact year, title similarity ≥ 0.95, 그리고 차순위와 0.05 이상 차이
  또는 title similarity ≥ 0.99
- medium: year 차이 ≤1, title similarity ≥0.88, 차순위와 0.03 이상 차이
- 나머지는 ambiguous/unmatched

## 임시 origin confidence

- field가 연결된 prior works ≥3, 최빈 field 비중 ≥0.60: high
- field가 연결된 prior works ≥2, 최빈 field 비중 ≥0.40: medium
- 나머지: ambiguous
- DBLP prior record 없음: no-observed-DBLP-history

## 판정

- high/medium origin 비율이 전체와 2018·2022·2024 각각 0.70 이상이면
  `WITHIN_CS_ONLY_RECOVERY_PASS`
- 하나라도 미달하면 `ORIGIN_BLOCKED_CURRENT_PIPELINE`
- 결과를 확인한 후 threshold, 기간, match rule을 낮추지 않음


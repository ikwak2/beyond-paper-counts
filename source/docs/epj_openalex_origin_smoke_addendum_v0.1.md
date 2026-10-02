# EPJ OpenAlex 원분야 연결 smoke addendum v0.1

이 문서는 `epj_origin_feasibility_protocol_v0.1.md`의 OpenAlex 계층을 실행하기
전에 고정하는 소규모 API smoke 설계다. 원 protocol은 수정하지 않는다.

## 표본

- frozen 306-author pilot sample에서 추출
- 층: index year × anchor venue × experienced-coauthor status
- 각 층 최대 5명
- 고정 seed: 20260827
- 예상 최대 60명

## 요청 상한

- OpenAlex 총 요청 credit 상한: 1,500
- API 응답의 remaining credit이 500 미만이면 즉시 중단
- 유료 선불 잔액을 사용하지 않는다.
- 모든 응답은 로컬 private cache에 저장하여 재호출하지 않는다.

## index work 연결

- 검색 입력: 논문 제목, index year
- 후보 평가: 정규화 제목 유사도, 출판연도, entrant 저자명 포함 여부,
  DBLP 저자목록과 OpenAlex authorship의 중복도
- high: exact-year, entrant-name match, title similarity ≥ 0.92이며
  저자 중복률 ≥ 0.50 또는 title similarity ≥ 0.98
- medium: year 차이 ≤ 1, entrant-name match, title similarity ≥ 0.85,
  저자 중복률 ≥ 0.30
- 그 외: ambiguous/unmatched

## entrant author 연결

index work의 authorship에서 entrant의 정규화 이름이 유일하게 일치할 때만
OpenAlex author ID를 채택한다. 복수 또는 무일치 후보는 ambiguous로 둔다.

## 진입 전 works

- 기간: index year 직전 5년
- matched OpenAlex author ID를 filter로 사용
- 저자당 최대 2 pages, page당 200 works
- title, year, type, primary topic, topics만 보존

## 임시 origin 판정

- field 정보가 있는 prior works가 3편 이상이고 최빈 field 비중이 0.60 이상이면 high
- field 정보가 있는 prior works가 2편 이상이고 최빈 field 비중이 0.40 이상이면 medium
- 그보다 약하면 ambiguous
- prior work가 없으면 no-observed-prior-works로 별도 기록
- 이 규칙은 연결 가능성 gate를 위한 임시 분류이며 최종 학문 분야 taxonomy가 아니다.

## 출력과 판정

- author-level 및 work-level 결과는 private
- venue × year 집계와 gate 진단만 public tables에 기록
- 이 smoke의 자동 결과는 linkage ground truth가 아니다.
- 원 protocol의 BROAD_ORIGIN_GO에는 별도의 표본감사가 여전히 필요하다.


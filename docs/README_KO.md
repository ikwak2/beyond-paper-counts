# 논문 코드 안내

이 저장소는 첨부 원고의 국가별 대표성(RQ1), 진입 이전–진입 시점 제목 거리(RQ2),
공저자 구성 진단(D1), 정확히 두 출판주기 후 재등장(RQ3)을 다룹니다.
최신 원고의 그림 2와 제한된 ±δ 결측 경계(S10b)까지 포함합니다.

## 실행

Python 3.11 환경에서 다음을 실행합니다.

```bash
python -m pip install -r requirements.txt
python reproduce.py
python -m unittest discover -s tests -v
```

계산 결과와 검증 보고서는 `outputs/`에 생성됩니다. `results/`의 동결 표는
수정하지 않습니다. 기존 파일 확인만 하려면 `python reproduce.py --check-only`를
실행합니다.

## 원고와 코드의 대응

| 원고 | 코드 | 입력·결과 |
| --- | --- | --- |
| RQ1, 표 2, 그림 2 | `scripts/build_pri.py`, `scripts/build_figure2.py` | 국가별 API 응답 21개와 국가·연도별 채택논문 집계량 |
| 표 3(a–b), 사양·결측 민감도 | `source/scripts/43_calculate_pri_construct_audit_v04.py`, `59_facct_missingness_tipping_point.py`, `70_facct_numerator_missingness_and_roster_audit.py` | S9–S11의 동결 결과; 논문별 중간자료는 미보존 |
| 표 3(c), 제한된 결측 경계 | `scripts/build_restricted_bounds.py` | S10으로부터 S10b 재계산; 새로운 API 질의 없음 |
| RQ2, D1, RQ3 | `source/scripts/81_*` → `82_*` → `83_*` → `84_*` | 코호트 → DBLP 제목 이력 → 거리 → 통계모형 |
| 보충분석 | `source/scripts/90_build_reviewer_additions.py` | ETO 비교, 국가별 거리, 누적 두 주기 민감도 |

원래 파일명의 `rq1_*`는 현재 원고의 RQ2, `rq2_*`는 D1입니다.
현재 본문 표 1–6과 예전 이름의 CSV 5개의 관계는
[`main_table_source_mapping.csv`](../results/main_table_source_mapping.csv)에 있습니다.

## 재현 가능한 범위

공개 집계자료로 API 집계 기반 대표성 사양 12개, 결측 경계 S10b,
그림 1·2·4, 그림 3B와 보충 그림 S3를 재계산 또는 다시 그릴 수 있습니다.
동결 수치와의 대조 검증도 함께 수행합니다. 그림 4는 배치를 정리했지만 수치와
구간은 같습니다. 원고에 사용한 그림 1–4 PDF는 `results/figures/`에 보존했습니다.

그림 3A의 개인별 파일럿 산점도와 RQ2·D1·RQ3 모형 전체 재추정에는 별도로 확보한
저자 이력이 필요합니다. 공개 집계표를 이용한 검증을 개인별 모형 재추정으로
표현하지 않습니다. 구체적인 입력과 실행법은
[`REPRODUCIBILITY.md`](REPRODUCIBILITY.md)에 있습니다.

논문별 OpenAlex 중간자료 863,752건은 보존되어 있지 않습니다. 여기서 만들어진
32개 사양은 동결 집계표만 제공합니다. 현재 API로 다시 수집하면 원고와 정확히
같은 스냅숏이 되지 않을 수 있습니다. 원고의 API 조회대상 수 864,215건,
논문별 분석자료 수 863,752건, 국가비중 분모인 국가집계량 합은 서로 다릅니다.

원고 PDF, 미완성 저자·지원·윤리 문구, 개인별 자료 및 인증정보는 이 코드 패키지의
구성물이 아닙니다. 논문 DOI·보관본 DOI와 SPECTER2 revision은 확정된 것으로
기재하지 않았습니다.

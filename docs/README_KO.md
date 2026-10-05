# 한국어 안내

이 저장소는 Beyond Paper Counts 논문의 코드와 공개용 분석 입력을 제공합니다. 저장소 첫 화면의 README에 실행 명령과 주요 파일 위치를 정리했습니다.

- 국가별 집계와 결측 경계: `python reproduce.py`
- 보존 거리값에서 주요 모형 재적합(2,000회 bootstrap 포함): `python scripts/refit_models.py`
- Figure 3 산점도 포함 전체 그림 생성: `python scripts/render_publication_figure3.py`

모형 실행 전에는 `requirements-models.txt`의 의존성을 설치합니다. 출력은 `outputs/`에 저장됩니다. 최종 투고용 그림과 보충자료는 `publication_assets/`에 있습니다.

공개용 모델 입력은 12,094행이며 이름·DBLP 식별자를 제외했습니다. 산점도 좌표는 249개입니다. 원래 변수값을 바꾸지 않았고 재적합 결과를 기존 결과와 비교했습니다.

원자료의 출처와 날짜는 [DATA_AVAILABILITY.md](DATA_AVAILABILITY.md), 변수 설명은 [DATA_DICTIONARY.md](DATA_DICTIONARY.md), 재현 한계는 [REPRODUCIBILITY.md](REPRODUCIBILITY.md)에 있습니다. 보존된 거리값에서의 모형 재적합과 제목부터 임베딩을 생성하는 과정은 구분됩니다.

저장소 공개 전환과 Zenodo DOI 발급은 별도 단계입니다. [RELEASE.md](RELEASE.md)에 공개 후 절차를 정리했습니다. 실제 DOI를 받기 전에는 DOI badge나 임의의 archive DOI를 넣지 않습니다.

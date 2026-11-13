# 한국어 안내

이 저장소는 Beyond Paper Counts 원고의 원본 분석 자료와 추가 재현 자료를 제공합니다.

## 현재 원고의 추가 자료

[재현 ZIP](../supplementary/Supplementary_Data_reproducibility.zip)과 [사용 안내](../supplementary/README.md)를 참고하세요. ZIP에는 국가별 비교, 추가 집계 진단, 원본 해시, 표·그림 대응표와 현재 그림 자료가 들어 있습니다.

- 국가비교 자료: 압축 해제 폴더의 `data/country_comparison/`
- 기존 데이터 ZIP: `archive_v1.0.2/publication_assets/Additional_file_2.zip`
- 현재 그림과 실행 방법: `figures/`, `scripts/`, `REPRODUCTION.md`

다른 두 데이터 ZIP은 별도로 받을 필요가 없습니다. 독립된 폴더에 압축을 풀고 내부 디렉터리 구조를 유지하세요. 현재 Figures 1–5의 자료와 재현 범위는 번들 안내에 명시돼 있습니다. 추가 자료는 새 outcome-model 적합을 수행하지 않았으며 Zenodo v1.0.2의 내용으로 주장하지 않습니다.

## 원본 분석 및 보존 자료

원본 분석은 [Zenodo v1.0.2](https://doi.org/10.5281/zenodo.23175869)에 공개돼 있습니다. 저장소 루트 명령은 다음과 같습니다.

- 보존 파일·표·확률 산술 검사: `python reproduce.py --check-only`
- 국가별 집계와 결측 경계 재현: `python reproduce.py`
- 보존 거리값에서 모형 재적합(2,000회 bootstrap): `python scripts/refit_models.py`
- 원본 릴리스의 Figure 1–4 생성: `python scripts/render_publication_figures.py`

Python 3.11/Linux 환경에서 작업별 의존성을 설치합니다. 모형 재적합에는 `requirements-models.txt`를 사용합니다. 출력은 `outputs/`에 저장됩니다. `publication_assets/`의 PDF와 그림은 원본 릴리스 당시 자료이며, 현재 보조 PDF는 원고와 함께 제공됩니다.

공개용 모형 입력은 12,094행이며 직접 식별자를 제외했습니다. 보존 거리값에서의 모형 재적합은 원래 제목·임베딩·연결자료부터의 완전한 재구성과 다릅니다. 원자료와 재구성 한계는 [DATA_AVAILABILITY.md](DATA_AVAILABILITY.md), [REPRODUCIBILITY.md](REPRODUCIBILITY.md), 변수 설명은 [DATA_DICTIONARY.md](DATA_DICTIONARY.md)에 있습니다.

원본 Zenodo v1.0.2와 추가 자료를 사용한 GitHub 고정 커밋을 각각 인용하세요. 현재 자료 전체를 위한 새 Zenodo 버전은 발행하지 않았습니다. 자세한 인용·보존 안내는 [RELEASE.md](RELEASE.md)에 있습니다.

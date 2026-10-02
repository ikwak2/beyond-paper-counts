#!/usr/bin/env python3
"""Finalize the preregistered journal-extension smoke study.

This script does not alter the frozen FAccT package or either frozen protocol.
It validates their hashes, checks the derived tables, draws the coefficient
comparison that motivates the GO/WAIT decision, and writes a Korean handoff.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "journal_llm_diffusion_v01"
DOCS = ROOT / "docs" / "journal_llm_diffusion_v01"

EXPECTED_HASHES = {
    "protocol_v0.1.md": "6f92e3694b2a4efe0afc383c3076f30218db816b4437dff857aa3ec110e6c85a",
    "cross_sectional_smoke_addendum_v0.1.md": "254d6faee6da70d7a3a1fe43cfa3b6312fbe63e1fbaa591354a400d6d9fe1887",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def validate() -> dict:
    checks: dict[str, object] = {}
    hashes = {name: sha256(DOCS / name) for name in EXPECTED_HASHES}
    checks["frozen_protocol_hashes"] = hashes
    checks["frozen_protocols_pass"] = hashes == EXPECTED_HASHES

    reproduction = json.loads(
        (OUT / "manifests" / "korea_pri_frozen_reproduction_validation.json").read_text()
    )
    checks["facct_frozen_reproduction_pass"] = reproduction["status"] == "PASS"
    checks["facct_frozen_reproduction_max_abs_diff"] = reproduction[
        "maximum_absolute_difference"
    ]

    coverage = pd.read_csv(OUT / "tables" / "llm_diffusion_source_coverage.csv")
    models = pd.read_csv(OUT / "tables" / "oecd_2025_cross_sectional_smoke_models.csv")
    panel = pd.read_csv(OUT / "tables" / "oecd_2025_cross_sectional_smoke_panel.csv")
    korea_2025 = pd.read_csv(OUT / "tables" / "korea_true_pri_2025.csv")
    diag = json.loads(
        (OUT / "manifests" / "oecd_2025_cross_sectional_smoke_diagnostics.json").read_text()
    )

    checks["four_distinct_diffusion_sources"] = coverage["source_id"].nunique() == 4
    checks["constructs_not_pooled"] = coverage["construct"].nunique() == 4
    rows_per_mapping = panel.groupby("mapping").size().to_dict()
    checks["oecd_panel_37_countries_two_venues"] = (
        panel["country_code"].nunique() == 37
        and panel["venue"].nunique() == 2
        and panel["mapping"].nunique() == 2
        and set(rows_per_mapping.values()) == {74}
    )
    checks["all_exposure_windows_12_months"] = bool(
        (panel["months_observed"] == 12).all()
    )
    checks["four_prespecified_models"] = len(models) == 4
    checks["models_converged"] = bool(models["converged"].all())
    checks["smoke_status"] = diag["smoke_model_status"]
    checks["panel_status"] = diag["final_panel_status"]

    primary = korea_2025.query(
        "mapping == 'conservative' and venue == 'ICML+NeurIPS'"
    ).iloc[0]
    checks["korea_2025_pri"] = float(primary["korea_pri"])
    checks["korea_2025_top_credit"] = float(primary["korea_top_count"])

    required_true = [
        "frozen_protocols_pass",
        "facct_frozen_reproduction_pass",
        "four_distinct_diffusion_sources",
        "constructs_not_pooled",
        "oecd_panel_37_countries_two_venues",
        "all_exposure_windows_12_months",
        "four_prespecified_models",
        "models_converged",
    ]
    checks["qa_status"] = (
        "PASS" if all(checks[key] is True for key in required_true) else "FAIL"
    )
    return checks


def coefficient_figure(models: pd.DataFrame) -> None:
    order = [
        ("conservative", "venue_only"),
        ("conservative", "venue_plus_log_gdp"),
        ("sensitivity", "venue_only"),
        ("sensitivity", "venue_plus_log_gdp"),
    ]
    rows = []
    for mapping, adjustment in order:
        rows.append(
            models.query("mapping == @mapping and adjustment == @adjustment").iloc[0]
        )
    plot = pd.DataFrame(rows).reset_index(drop=True)
    labels = [
        "Conservative mapping · venue only",
        "Conservative mapping · venue + log GDP",
        "Sensitivity mapping · venue only",
        "Sensitivity mapping · venue + log GDP",
    ]
    y = np.arange(len(plot))[::-1]
    rr = plot["rate_ratio_per_10pp"].to_numpy()
    low = plot["rate_ratio_ci_low"].to_numpy()
    high = plot["rate_ratio_ci_high"].to_numpy()

    plt.rcParams.update({"font.size": 10})
    fig, ax = plt.subplots(figsize=(8.4, 4.7))
    colors = ["#D55E00", "#0072B2", "#D55E00", "#0072B2"]
    for i in range(len(plot)):
        ax.errorbar(
            rr[i], y[i], xerr=[[rr[i] - low[i]], [high[i] - rr[i]]],
            fmt="o", color=colors[i], ecolor=colors[i], capsize=4, markersize=7,
        )
    ax.axvline(1.0, color="black", linewidth=1, linestyle="--")
    ax.set_yticks(y, labels)
    ax.set_xlabel("Rate ratio per 10 percentage-point increase (95% CI)")
    ax.set_title("The 2025 cross-country association vanishes after GDP adjustment")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(
            OUT / "figures" / f"oecd_llm_association_raw_vs_gdp_adjusted.{suffix}",
            dpi=220,
            bbox_inches="tight",
        )
    plt.close(fig)


def report(checks: dict) -> None:
    pri = pd.read_csv(OUT / "tables" / "primary_pri_korea_and_frozen_groups_2018_2024.csv")
    kr = pri.query("group == 'South Korea'").sort_values("year")
    k18 = float(kr.query("year == 2018")["pri"].iloc[0])
    k24 = float(kr.query("year == 2024")["pri"].iloc[0])
    k25 = checks["korea_2025_pri"]
    models = pd.read_csv(OUT / "tables" / "oecd_2025_cross_sectional_smoke_models.csv")

    def model(mapping: str, adjustment: str) -> pd.Series:
        return models.query("mapping == @mapping and adjustment == @adjustment").iloc[0]

    raw = model("conservative", "venue_only")
    gdp = model("conservative", "venue_plus_log_gdp")

    text = f"""# 생성형 AI 확산과 최상위 AI 학회 국가 대표성: 저널 확장 v0.1

## 한 줄 판정

**저널 연구로 발전시킬 가치가 있다.** 다만 현재 자료가 지지하는 중심 문장은 “LLM 보급이 논문 대표성을 높였다”가 아니라, **“그렇게 보이는 국가 간 관계는 경제·연구역량과 강하게 얽혀 있으며, 이를 분리하지 않으면 기술 확산을 민주화 효과로 오해할 수 있다”**이다.

현재 판정은 **GO_FOR_JOURNAL_EXTENSION / WAIT_FOR_ALIGNED_PANEL**이다. 기존 FAccT용 분석과 동결 산출물은 변경하지 않는다.

## 무엇을 새로 확인했나

### 1. 한국의 생산량 보정 대표성은 상승했다

한국의 ICML·NeurIPS 채택논문 제1저자 PRI는 2018년 **{k18:.3f}**, 2024년 **{k24:.3f}**, 2025년 **{k25:.3f}**였다. PRI 1은 해당 국가의 broad-AI 논문 생산 비중과 최상위 두 학회 제1저자 비중이 같다는 뜻이다. 따라서 한국은 2018년부터 이미 1을 넘었고, 2025년에는 생산량 대비 약 2.16배의 제1저자 대표성을 보였다.

그러나 한국의 상승은 ChatGPT 공개 이전인 2018–2021에도 나타났다. 이 궤적만으로 LLM 효과를 주장할 수 없다.

### 2. “한국이 ChatGPT 보급률 세계 1위”는 일반화하면 안 된다

- 2026년 Nature 연구자 설문의 **보고된 10개국 표본**에서는 한국의 ChatGPT 사용 79.0%, Gemini 사용 69.3%가 각각 가장 높았다.
- Microsoft/OWID의 19개 생성형 AI 플랫폼 종합 지표에서 한국은 2025년 상반기 25.9%, 하반기 30.7%, 2026년 1분기 37.1%였다.
- OECD·Similarweb의 ChatGPT·Claude·Gemini 웹 이용률은 한국에서 2024년 2월 8.5%에서 2026년 1월 32.4%로 증가했다.
- OpenAI Signals의 ChatGPT 개인용 메시지 사용 순위에서 한국은 2025년 1분기 57위에서 2026년 2분기 25위로 변했다.

네 자료는 표본·플랫폼·단위가 서로 다르므로 평균내어 하나의 “LLM 보급률”로 만들지 않는다.

### 3. 가장 중요한 smoke 결과: 원시 연관은 GDP를 넣으면 사라진다

OECD 37개국, ICML·NeurIPS 2025 채택논문, 각 제출마감 전 정확히 12개월의 웹 이용률을 결합했다.

| 사양 | 이용률 10%p당 비율비 | 95% 신뢰구간 | 해석 |
|---|---:|---:|---|
| 보수적 국가귀속, 학회만 보정 | {raw.rate_ratio_per_10pp:.3f} | {raw.rate_ratio_ci_low:.3f}–{raw.rate_ratio_ci_high:.3f} | 강한 양의 국가 간 연관 |
| 보수적 국가귀속, 학회+GDP 보정 | {gdp.rate_ratio_per_10pp:.3f} | {gdp.rate_ratio_ci_low:.3f}–{gdp.rate_ratio_ci_high:.3f} | 사실상 0 |

민감도 국가귀속에서도 같은 패턴이었다. 그러므로 **LLM 이용이 많은 국가일수록 대표성이 높아 보이는 관계는 국가소득과 강하게 얽혀 있어, 현재 자료로 둘을 분리할 수 없다.** GDP 자체가 완전한 교란 보정은 아니며, 이 결과도 2025년 한 해의 기술적 smoke test다.

## 저널 논문의 권장 연구질문

> 국가별 생성형 AI 확산은 이후 최상위 AI 학회의 생산량 보정 제1저자 대표성과 관련되는가? 이 연관은 국가의 기존 연구역량·경제력·영어 접근성을 고려한 뒤에도 남는가?

한국과 중국은 설명력이 큰 사례로 제시하되, 추론은 가능한 모든 국가를 사용한다. 특정 국가 두 곳만 골라 비교하면 사례 선택 비판을 피하기 어렵다.

## 기존 FAccT용 연구와 결합되는 방식

| 기존 분석에서 확보한 층 | 저널 확장에서 추가하는 층 |
|---|---|
| Presence: 국가별 제1저자 대표성 | 국가별 생성형 AI 확산의 여러 측정값 |
| Entry·Recurrence·Continuity의 구분 | 노출 시점을 제출마감 이전으로 정렬 |
| 분자·분모·국가귀속 정의 감사 | 경제·연구역량과의 구조적 교란 감사 |
| “대표성≠진입≠지속” | “보급≠사용능력≠인과효과” |

저널판의 새 기여는 FAccT용 결과를 단순히 연장하는 것이 아니라, **디지털 도구 확산과 학술 대표성을 연결할 때 생기는 시간·분모·교란 문제를 함께 보여주는 측정 연구**다.

## 본 분석으로 가기 위한 필수 작업

1. ICML 2025 roster의 공식 목록 대비 73편 차이와 국가 귀속 결측을 보완하거나 tipping-point 민감도를 수행한다.
2. 2026년 학회 결과가 확정된 뒤 동일한 정의로 2024–2026 국가×학회 패널을 만든다.
3. 제출마감 이전 노출만 사용하고 국가 고정효과·학회-연도 고정효과를 둔다.
4. GDP 외에 연구개발비, broad-AI 생산량, 인터넷 접근성, 영어 접근성을 결과 보기 전에 제한된 교란집합으로 고정한다.
5. 현재 Poisson 모형의 Pearson dispersion이 17.8–37.5로 매우 크므로, 본 분석에서 count model 적합성과 표준오차 강건성을 사전 지정한다.
6. 기존 FAccT 원고가 공개·제출·출판된다면 저널의 중복게재 정책에 따라 이를 명시적으로 인용하고, 다중 보급지표·시간정렬 패널·교란 분석이라는 실질적 확장을 분명히 한다.

## 미리 정할 핵심 그림과 표

- 그림 1: Presence–Entry–Recurrence–Continuity와 LLM diffusion을 구분하는 개념도
- 그림 2: 한국·중국·핵심 영어권·기타 비핵심권의 PRI 궤적
- 그림 3: 네 보급자료의 대상·기간·단위 비교
- 그림 4: 원시 모형과 GDP 보정 모형의 계수 비교
- 표 1: 보급자료 source registry와 한계
- 표 2: 국가별 PRI와 보급지표의 시간 정렬 규칙
- 표 3: 주모형 및 사전 고정 강건성 분석
- 표 4: 각 지표가 허용하는 주장과 금지하는 주장

## 투고 가능성에 대한 정직한 평가

- **현재 상태:** 흥미로운 저널 아이디어와 재현 가능한 예비결과는 확보했다. 아직 완성 논문은 아니다.
- **Scientometrics / Research Evaluation:** 패널과 결측 민감도를 닫으면 현실적인 목표다.
- **Quantitative Science Studies:** 단순 국가 회귀를 넘어, 여러 보급지표가 만드는 구성개념 차이와 교란·시간정렬의 일반화 가능한 측정 기여가 분명해야 한다.
- **Journal of Informetrics:** 현재보다 새로운 측정방법 기여가 더 강해야 한다.

## 금지되는 문장

- “한국의 LLM 보급이 최상위 학회 논문 채택을 증가시켰다.”
- “한국은 세계에서 ChatGPT 보급률이 가장 높다.”
- “PRI가 높으므로 심사가 공정하다.”
- “GDP 보정 후 효과가 없으므로 LLM은 아무 영향이 없다.”

## 최종 상태

- 재현성 QA: **{checks['qa_status']}**
- smoke 결과: **{checks['smoke_status']}**
- 본 패널: **{checks['panel_status']}**
- 다음 데이터 체크포인트: 2026 ICML·NeurIPS 결과 확정 및 2025 roster/국가 결측 보완
"""
    (OUT / "reports" / "JOURNAL_EXTENSION_GO_WAIT_DECISION_KO.md").write_text(
        text, encoding="utf-8"
    )

    readme = """# Journal LLM diffusion extension v0.1

이 폴더는 기존 FAccT용 분석을 변경하지 않고 만든 별도 저널 확장이다.

먼저 볼 파일:

1. `reports/JOURNAL_EXTENSION_GO_WAIT_DECISION_KO.md` — 전체 판단과 다음 단계
2. `reports/KOREA_PRI_RESULT_KO.md` — 한국 PRI 기술 결과
3. `reports/JOURNAL_LLM_DIFFUSION_SMOKE_RESULT_KO.md` — 2025 횡단면 smoke 결과
4. `figures/oecd_llm_association_raw_vs_gdp_adjusted.png` — 핵심 계수 비교
5. `manifests/finalization_qa.json` — 동결 해시와 자동 QA

프로토콜은 `docs/journal_llm_diffusion_v01/`에 있으며 해시로 동결되어 있다.
"""
    (OUT / "README_KO.md").write_text(readme, encoding="utf-8")


def main() -> None:
    checks = validate()
    if checks["qa_status"] != "PASS":
        raise RuntimeError(f"Finalization QA failed: {checks}")
    models = pd.read_csv(OUT / "tables" / "oecd_2025_cross_sectional_smoke_models.csv")
    coefficient_figure(models)
    report(checks)
    (OUT / "manifests" / "finalization_qa.json").write_text(
        json.dumps(checks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(checks, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Local TF-IDF title-portfolio distance feasibility smoke for EPJ."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer


ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "outputs/epj_origin_feasibility_v01/private/epj_origin_pilot_sample_private.csv"
HISTORY = ROOT / "outputs/epj_origin_feasibility_v01/private/epj_origin_pilot_dblp_history_canonical_private.csv"
CONFIG = ROOT / "config/epj_title_distance_smoke_v03.json"
ADDENDUM = ROOT / "docs/epj_title_distance_smoke_addendum_v0.3.md"
OUT = ROOT / "outputs" / "epj_origin_feasibility_v01"
PRIVATE = OUT / "private"
TABLES = OUT / "tables"
MANIFESTS = OUT / "manifests"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_title(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(character for character in text if not unicodedata.combining(character))
    text = re.sub(r"\\[a-zA-Z]+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9]+", " ", text).casefold()
    return re.sub(r"\s+", " ", text).strip()


def cosine_to_centroid(matrix: csr_matrix, index_position: int, prior_positions: list[int]) -> float:
    if not prior_positions:
        return np.nan
    index_vector = matrix[index_position]
    centroid = csr_matrix(matrix[prior_positions].mean(axis=0))
    index_norm = float(np.sqrt(index_vector.multiply(index_vector).sum()))
    centroid_norm = float(np.sqrt(centroid.multiply(centroid).sum()))
    if index_norm == 0 or centroid_norm == 0:
        return np.nan
    similarity = float(index_vector.multiply(centroid).sum() / (index_norm * centroid_norm))
    return float(np.clip(1.0 - similarity, 0.0, 1.0))


def main() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    sample = pd.read_csv(SAMPLE, keep_default_na=False)
    history = pd.read_csv(HISTORY, keep_default_na=False)
    history = history.loc[history.author_key.isin(set(sample.author_key))].copy()
    history["normalized_title"] = history.title.map(normalize_title)
    history = history.loc[history.normalized_title.ne("")].copy()
    history = history.drop_duplicates(["author_key", "normalized_title"])

    documents: list[str] = []
    index_positions: dict[str, int] = {}
    prior_positions: dict[str, list[int]] = {key: [] for key in sample.author_key}
    for row in sample.itertuples(index=False):
        index_positions[str(row.author_key)] = len(documents)
        documents.append(str(row.title))
    for row in history.itertuples(index=False):
        position = len(documents)
        documents.append(str(row.title))
        prior_positions.setdefault(str(row.author_key), []).append(position)

    word = TfidfVectorizer(
        strip_accents="unicode",
        lowercase=True,
        stop_words="english",
        ngram_range=(int(config["word_ngram_min"]), int(config["word_ngram_max"])),
        min_df=int(config["minimum_document_frequency"]),
        sublinear_tf=True,
        norm="l2",
    )
    char = TfidfVectorizer(
        strip_accents="unicode",
        lowercase=True,
        analyzer="char_wb",
        ngram_range=(int(config["char_ngram_min"]), int(config["char_ngram_max"])),
        min_df=int(config["minimum_document_frequency"]),
        sublinear_tf=True,
        norm="l2",
        max_features=100000,
    )
    word_matrix = word.fit_transform(documents).tocsr()
    char_matrix = char.fit_transform(documents).tocsr()

    rows: list[dict[str, object]] = []
    for entrant in sample.itertuples(index=False):
        key = str(entrant.author_key)
        positions = prior_positions.get(key, [])
        rows.append(
            {
                **entrant._asdict(),
                "n_unique_prior_titles": len(positions),
                "title_distance_word_primary": cosine_to_centroid(
                    word_matrix, index_positions[key], positions
                ),
                "title_distance_char_sensitivity": cosine_to_centroid(
                    char_matrix, index_positions[key], positions
                ),
            }
        )
    result = pd.DataFrame(rows)
    result["distance_observed"] = result.title_distance_word_primary.notna()
    private_path = PRIVATE / "epj_title_portfolio_distance_smoke_private.csv"
    result.to_csv(private_path, index=False, encoding="utf-8-sig")

    groups = ["index_year", "entry_with_experienced_top4_coauthor"]
    summary = (
        result.groupby(groups, as_index=False)
        .agg(
            sampled_authors=("pilot_id", "size"),
            distance_coverage=("distance_observed", "mean"),
            median_word_distance=("title_distance_word_primary", "median"),
            q25_word_distance=("title_distance_word_primary", lambda x: x.quantile(0.25)),
            q75_word_distance=("title_distance_word_primary", lambda x: x.quantile(0.75)),
            median_prior_titles=("n_unique_prior_titles", "median"),
        )
        .sort_values(groups)
    )
    summary_path = TABLES / "epj_title_portfolio_distance_smoke_summary.csv"
    summary.to_csv(summary_path, index=False)

    observed = result.loc[result.distance_observed].copy()
    coverage_overall = float(result.distance_observed.mean())
    coverage_by_year = result.groupby("index_year").distance_observed.mean()
    primary_iqr = float(
        observed.title_distance_word_primary.quantile(0.75)
        - observed.title_distance_word_primary.quantile(0.25)
    )
    spearman = float(
        observed[["title_distance_word_primary", "title_distance_char_sensitivity"]]
        .corr(method="spearman")
        .iloc[0, 1]
    )
    finite_rate = float(
        np.isfinite(observed.title_distance_word_primary).mean()
        if len(observed)
        else 0.0
    )
    passed = bool(
        coverage_overall >= float(config["minimum_coverage"])
        and coverage_by_year.ge(float(config["minimum_coverage"])).all()
        and primary_iqr >= float(config["minimum_primary_iqr"])
        and spearman >= float(config["minimum_primary_sensitivity_spearman"])
        and finite_rate >= float(config["minimum_finite_rate"])
    )
    diagnostics = {
        "sample_authors": int(len(result)),
        "unique_prior_titles": int(len(history)),
        "word_vocabulary_size": int(len(word.vocabulary_)),
        "char_vocabulary_size": int(len(char.vocabulary_)),
        "distance_coverage_overall": coverage_overall,
        "distance_coverage_by_year": {str(k): float(v) for k, v in coverage_by_year.items()},
        "primary_distance_iqr": primary_iqr,
        "primary_sensitivity_spearman": spearman,
        "finite_primary_distance_rate": finite_rate,
        "decision": "TITLE_DISTANCE_FEASIBLE" if passed else "TITLE_DISTANCE_BLOCKED",
        "construct_label": "DBLP five-year prior-title portfolio to index-paper lexical-semantic distance",
        "input_sha256": {
            str(SAMPLE.relative_to(ROOT)): sha256_file(SAMPLE),
            str(HISTORY.relative_to(ROOT)): sha256_file(HISTORY),
            str(CONFIG.relative_to(ROOT)): sha256_file(CONFIG),
            str(ADDENDUM.relative_to(ROOT)): sha256_file(ADDENDUM),
        },
        "output_sha256": {
            str(private_path.relative_to(ROOT)): sha256_file(private_path),
            str(summary_path.relative_to(ROOT)): sha256_file(summary_path),
        },
    }
    diagnostics_path = MANIFESTS / "epj_title_portfolio_distance_smoke_diagnostics.json"
    diagnostics_path.write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(diagnostics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


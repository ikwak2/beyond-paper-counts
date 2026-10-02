#!/usr/bin/env python3
"""Run the frozen SPECTER2 title-only distance pilot on 306 entrants."""

from __future__ import annotations

import hashlib
import json
import os
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from adapters import AutoAdapterModel
from scipy.stats import spearmanr
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "outputs/epj_origin_feasibility_v01/private/epj_origin_pilot_sample_private.csv"
HISTORY = ROOT / "outputs/epj_origin_feasibility_v01/private/epj_origin_pilot_dblp_history_canonical_private.csv"
TFIDF = ROOT / "outputs/epj_origin_feasibility_v01/private/epj_title_portfolio_distance_smoke_private.csv"
CONFIG = ROOT / "config/epj_specter2_title_distance_pilot_v04.json"
PROTOCOL = ROOT / "docs/epj_specter2_title_distance_pilot_addendum_v0.4.md"
OUT = ROOT / "outputs/epj_origin_feasibility_v01"
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
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"\\[a-zA-Z]+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9]+", " ", text).casefold()
    return re.sub(r"\s+", " ", text).strip()


def embed_titles(
    titles: list[str],
    config: dict[str, object],
) -> tuple[np.ndarray, str, str]:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(str(config["base_model"]))
    model = AutoAdapterModel.from_pretrained(str(config["base_model"]))
    adapter_name = model.load_adapter(str(config["adapter"]), source="hf")
    model.set_active_adapters(adapter_name)
    if not model.active_adapters:
        raise RuntimeError("SPECTER2 proximity adapter was not activated")
    model.to(device)
    model.eval()

    vectors: list[np.ndarray] = []
    batch_size = int(config["batch_size"])
    max_length = int(config["max_length"])
    for start in range(0, len(titles), batch_size):
        texts = [title + tokenizer.sep_token for title in titles[start : start + batch_size]]
        batch = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.inference_mode():
            output = model(**batch).last_hidden_state[:, 0, :]
            output = torch.nn.functional.normalize(output, p=2, dim=1)
        vectors.append(output.detach().cpu().numpy().astype(np.float32))
    return np.vstack(vectors), str(adapter_name), str(model.active_adapters)


def normalized_centroid(matrix: np.ndarray) -> np.ndarray:
    centroid = matrix.mean(axis=0)
    norm = np.linalg.norm(centroid)
    return centroid / norm if norm > 0 else centroid


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.clip(1.0 - np.dot(a, b), 0.0, 2.0))


def main() -> None:
    for directory in (PRIVATE, TABLES, MANIFESTS):
        directory.mkdir(parents=True, exist_ok=True)
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    sample = pd.read_csv(SAMPLE, keep_default_na=False)
    history = pd.read_csv(HISTORY, keep_default_na=False)
    lexical = pd.read_csv(TFIDF, keep_default_na=False)

    years_by_author = sample.set_index("author_key")["index_year"].astype(int).to_dict()
    history = history.loc[history.author_key.isin(years_by_author)].copy()
    history["normalized_title"] = history.title.map(normalize_title)
    history = history.loc[history.normalized_title.ne("")].copy()
    history = history.loc[
        history.apply(
            lambda row: int(row.prior_year) < years_by_author[str(row.author_key)]
            and int(row.prior_year) >= years_by_author[str(row.author_key)] - 5,
            axis=1,
        )
    ].copy()
    history = history.drop_duplicates(["author_key", "normalized_title"])

    title_display: dict[str, str] = {}
    index_title_key: dict[str, str] = {}
    prior_title_keys: dict[str, list[str]] = {str(key): [] for key in sample.author_key}
    for row in sample.itertuples(index=False):
        key = normalize_title(row.title)
        if not key:
            raise ValueError(f"Blank normalized index title for {row.author_key}")
        title_display.setdefault(key, str(row.title))
        index_title_key[str(row.author_key)] = key
    for row in history.itertuples(index=False):
        key = str(row.normalized_title)
        title_display.setdefault(key, str(row.title))
        prior_title_keys[str(row.author_key)].append(key)

    unique_keys = sorted(title_display)
    matrix, adapter_name, active_adapters = embed_titles(
        [title_display[key] for key in unique_keys], config
    )
    key_to_position = {key: pos for pos, key in enumerate(unique_keys)}

    index_vectors: dict[str, np.ndarray] = {}
    centroids: dict[str, np.ndarray] = {}
    rows: list[dict[str, object]] = []
    for entrant in sample.itertuples(index=False):
        author_key = str(entrant.author_key)
        index_vector = matrix[key_to_position[index_title_key[author_key]]]
        index_vectors[author_key] = index_vector
        prior_keys = prior_title_keys.get(author_key, [])
        if prior_keys:
            centroid = normalized_centroid(
                matrix[[key_to_position[key] for key in prior_keys]]
            )
            centroids[author_key] = centroid
            distance = cosine_distance(index_vector, centroid)
        else:
            distance = np.nan
        rows.append(
            {
                **entrant._asdict(),
                "n_unique_prior_titles": len(prior_keys),
                "specter2_title_portfolio_distance": distance,
                "distance_observed": bool(prior_keys),
            }
        )
    result = pd.DataFrame(rows)
    result = result.merge(
        lexical[
            [
                "author_key",
                "title_distance_word_primary",
                "title_distance_char_sensitivity",
            ]
        ],
        on="author_key",
        how="left",
        validate="one_to_one",
    )

    rng = np.random.default_rng(int(config["random_seed"]))
    n_perm = int(config["validation_permutations"])
    permutation_medians: dict[str, float] = {}
    for year, year_rows in result.loc[result.distance_observed].groupby("index_year"):
        authors = [str(key) for key in year_rows.author_key if str(key) in centroids]
        if len(authors) < 2:
            continue
        for author_key in authors:
            alternatives = [key for key in authors if key != author_key]
            sampled = rng.choice(alternatives, size=n_perm, replace=True)
            distances = [
                cosine_distance(index_vectors[author_key], centroids[str(other)])
                for other in sampled
            ]
            permutation_medians[author_key] = float(np.median(distances))
    result["permuted_other_portfolio_median_distance"] = result.author_key.map(
        permutation_medians
    )
    result["permuted_minus_own_distance"] = (
        result.permuted_other_portfolio_median_distance
        - result.specter2_title_portfolio_distance
    )
    result["own_portfolio_closer_than_permuted_median"] = (
        result.permuted_minus_own_distance > 0
    ).where(result.distance_observed, np.nan)

    private_path = PRIVATE / "epj_specter2_title_distance_pilot_private.csv"
    result.to_csv(private_path, index=False, encoding="utf-8-sig")

    summary = (
        result.groupby(["index_year", "entry_with_experienced_top4_coauthor"], as_index=False)
        .agg(
            sampled_authors=("pilot_id", "size"),
            distance_coverage=("distance_observed", "mean"),
            median_distance=("specter2_title_portfolio_distance", "median"),
            q25_distance=("specter2_title_portfolio_distance", lambda x: x.quantile(0.25)),
            q75_distance=("specter2_title_portfolio_distance", lambda x: x.quantile(0.75)),
            median_prior_titles=("n_unique_prior_titles", "median"),
        )
        .sort_values(["index_year", "entry_with_experienced_top4_coauthor"])
    )
    summary_path = TABLES / "epj_specter2_title_distance_pilot_summary.csv"
    summary.to_csv(summary_path, index=False)

    observed = result.loc[result.distance_observed].copy()
    coverage_overall = float(result.distance_observed.mean())
    coverage_by_year = result.groupby("index_year").distance_observed.mean()
    finite_rate = float(np.isfinite(observed.specter2_title_portfolio_distance).mean())
    iqr = float(
        observed.specter2_title_portfolio_distance.quantile(0.75)
        - observed.specter2_title_portfolio_distance.quantile(0.25)
    )
    paired = observed.permuted_minus_own_distance.dropna()
    own_closer_share = float((paired > 0).mean())
    paired_median_margin = float(paired.median())
    spearman = float(
        spearmanr(
            observed.specter2_title_portfolio_distance,
            observed.title_distance_word_primary.astype(float),
            nan_policy="omit",
        ).statistic
    )
    gates = {
        "coverage": bool(
            coverage_overall >= float(config["minimum_overall_and_year_coverage"])
            and coverage_by_year.ge(
                float(config["minimum_overall_and_year_coverage"])
            ).all()
        ),
        "finite_rate": bool(finite_rate >= float(config["minimum_finite_rate"])),
        "distance_iqr": bool(iqr >= float(config["minimum_distance_iqr"])),
        "tfidf_spearman": bool(
            spearman >= float(config["minimum_word_tfidf_spearman"])
        ),
        "own_closer_share": bool(
            own_closer_share >= float(config["minimum_own_closer_share"])
        ),
        "paired_median_margin": bool(
            paired_median_margin > float(config["minimum_paired_median_margin"])
        ),
    }
    passed = all(gates.values())
    diagnostics = {
        "decision": "SPECTER2_TITLE_DISTANCE_FEASIBLE" if passed else "SPECTER2_TITLE_DISTANCE_BLOCKED",
        "sample_authors": int(len(result)),
        "authors_with_distance": int(result.distance_observed.sum()),
        "embedded_unique_titles": int(len(unique_keys)),
        "embedding_dimension": int(matrix.shape[1]),
        "adapter_name": adapter_name,
        "active_adapters": active_adapters,
        "distance_coverage_overall": coverage_overall,
        "distance_coverage_by_year": {str(k): float(v) for k, v in coverage_by_year.items()},
        "finite_distance_rate": finite_rate,
        "distance_iqr": iqr,
        "word_tfidf_spearman": spearman,
        "own_portfolio_closer_share": own_closer_share,
        "paired_median_margin": paired_median_margin,
        "gates": gates,
        "construct_label": "SPECTER2 title-only distance from the five-year DBLP prior-title portfolio centroid to the index-paper title",
        "input_sha256": {
            str(path.relative_to(ROOT)): sha256_file(path)
            for path in (SAMPLE, HISTORY, TFIDF, CONFIG, PROTOCOL)
        },
        "output_sha256": {
            str(private_path.relative_to(ROOT)): sha256_file(private_path),
            str(summary_path.relative_to(ROOT)): sha256_file(summary_path),
        },
    }
    diagnostics_path = MANIFESTS / "epj_specter2_title_distance_pilot_diagnostics.json"
    diagnostics_path.write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(diagnostics, ensure_ascii=False, indent=2))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

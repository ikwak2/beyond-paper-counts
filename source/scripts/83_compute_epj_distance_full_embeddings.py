#!/usr/bin/env python3
"""Compute frozen SPECTER2 and TF-IDF portfolio distances for the full cohort."""

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
from scipy.sparse import csr_matrix
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
COHORT = ROOT / "outputs/epj_distance_based_v10/private/epj_distance_full_entrant_cohort_private.csv"
INDEX_PAPERS = ROOT / "outputs/epj_distance_based_v10/private/epj_distance_full_index_papers_private.csv"
HISTORY = ROOT / "outputs/epj_distance_based_v10/private/epj_distance_full_dblp_history_private.csv"
CONFIG = ROOT / "config/epj_distance_based_analysis_v10.json"
PROTOCOL = ROOT / "docs/epj_distance_based_analysis_protocol_v1.0.md"
OUT = ROOT / "outputs/epj_distance_based_v10"
PRIVATE = OUT / "private"
TABLES = OUT / "tables"
MANIFESTS = OUT / "manifests"
CACHE = OUT / "cache"


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
    titles: list[str], config: dict[str, object]
) -> tuple[np.ndarray, str, str, str]:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(str(config["base_model"]))
    model = AutoAdapterModel.from_pretrained(str(config["base_model"]))
    adapter_name = model.load_adapter(str(config["adapter"]), source="hf")
    model.set_active_adapters(adapter_name)
    if not model.active_adapters:
        raise RuntimeError("SPECTER2 proximity adapter is not active")
    model.to(device)
    model.eval()
    batch_size = int(config["embedding_batch_size"])
    max_length = int(config["max_length"])
    arrays: list[np.ndarray] = []
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
        arrays.append(output.detach().cpu().numpy().astype(np.float32))
        if (start // batch_size + 1) % 100 == 0 or start + batch_size >= len(titles):
            print(
                f"embedded={min(start + batch_size, len(titles)):,}/{len(titles):,}",
                flush=True,
            )
    return np.vstack(arrays), str(adapter_name), str(model.active_adapters), str(device)


def dense_centroid(matrix: np.ndarray, positions: list[int]) -> np.ndarray | None:
    if not positions:
        return None
    vector = matrix[positions].mean(axis=0)
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else None


def sparse_distance(matrix: csr_matrix, left: list[int], right: list[int]) -> float:
    if not left or not right:
        return np.nan
    left_centroid = csr_matrix(matrix[left].mean(axis=0))
    right_centroid = csr_matrix(matrix[right].mean(axis=0))
    left_norm = float(np.sqrt(left_centroid.multiply(left_centroid).sum()))
    right_norm = float(np.sqrt(right_centroid.multiply(right_centroid).sum()))
    if left_norm == 0 or right_norm == 0:
        return np.nan
    similarity = float(
        left_centroid.multiply(right_centroid).sum() / (left_norm * right_norm)
    )
    return float(np.clip(1.0 - similarity, 0.0, 1.0))


def main() -> None:
    for directory in (PRIVATE, TABLES, MANIFESTS, CACHE):
        directory.mkdir(parents=True, exist_ok=True)
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    cohort = pd.read_csv(COHORT, keep_default_na=False)
    index_papers = pd.read_csv(INDEX_PAPERS, keep_default_na=False)
    history = pd.read_csv(HISTORY, keep_default_na=False)
    primary_types = set(config["primary_history_record_types"])
    sensitivity_types = set(config["sensitivity_history_record_types"])

    index_papers["normalized_title"] = index_papers.title.map(normalize_title)
    history["normalized_title"] = history.title.map(normalize_title)
    index_papers = index_papers.loc[index_papers.normalized_title.ne("")].copy()
    history = history.loc[history.normalized_title.ne("")].copy()
    index_papers = index_papers.drop_duplicates(["entrant_id", "normalized_title"])
    history = history.drop_duplicates(["entrant_id", "normalized_title"])

    title_display: dict[str, str] = {}
    for row in index_papers.itertuples(index=False):
        title_display.setdefault(str(row.normalized_title), str(row.title))
    for row in history.itertuples(index=False):
        title_display.setdefault(str(row.normalized_title), str(row.title))
    title_keys = sorted(title_display)
    title_position = {key: i for i, key in enumerate(title_keys)}
    display_titles = [title_display[key] for key in title_keys]

    embeddings_path = CACHE / "specter2_title_embeddings_float16.npy"
    titles_path = CACHE / "specter2_title_embedding_keys_private.csv"
    vectors, adapter_name, active_adapters, device = embed_titles(display_titles, config)
    np.save(embeddings_path, vectors.astype(np.float16))
    pd.DataFrame(
        {"position": np.arange(len(title_keys)), "normalized_title": title_keys, "title": display_titles}
    ).to_csv(titles_path, index=False, encoding="utf-8-sig")

    index_positions = (
        index_papers.groupby("entrant_id").normalized_title
        .agg(lambda values: [title_position[str(value)] for value in values])
        .to_dict()
    )
    primary_positions = (
        history.loc[history.publication_type.isin(primary_types)]
        .groupby("entrant_id").normalized_title
        .agg(lambda values: [title_position[str(value)] for value in values])
        .to_dict()
    )
    sensitivity_positions = (
        history.loc[history.publication_type.isin(sensitivity_types)]
        .groupby("entrant_id").normalized_title
        .agg(lambda values: [title_position[str(value)] for value in values])
        .to_dict()
    )

    semantic_rows: list[dict[str, object]] = []
    for entrant in cohort.itertuples(index=False):
        entrant_id = str(entrant.entrant_id)
        left = index_positions.get(entrant_id, [])
        primary_right = primary_positions.get(entrant_id, [])
        sensitivity_right = sensitivity_positions.get(entrant_id, [])
        index_centroid = dense_centroid(vectors, left)
        primary_centroid = dense_centroid(vectors, primary_right)
        sensitivity_centroid = dense_centroid(vectors, sensitivity_right)

        def distance(right_centroid: np.ndarray | None) -> float:
            if index_centroid is None or right_centroid is None:
                return np.nan
            return float(np.clip(1.0 - np.dot(index_centroid, right_centroid), 0.0, 2.0))

        semantic_rows.append(
            {
                "entrant_id": entrant_id,
                "n_unique_index_titles": len(left),
                "n_unique_primary_prior_titles": len(primary_right),
                "n_unique_alltype_prior_titles": len(sensitivity_right),
                "specter2_title_distance_primary": distance(primary_centroid),
                "specter2_title_distance_alltypes_sensitivity": distance(sensitivity_centroid),
            }
        )
    distance = pd.DataFrame(semantic_rows)

    word = TfidfVectorizer(
        strip_accents="unicode",
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
        norm="l2",
    )
    word_matrix = word.fit_transform(display_titles).tocsr()
    word_distances = {
        entrant_id: sparse_distance(
            word_matrix,
            index_positions.get(entrant_id, []),
            primary_positions.get(entrant_id, []),
        )
        for entrant_id in cohort.entrant_id.astype(str)
    }
    del word_matrix

    char = TfidfVectorizer(
        strip_accents="unicode",
        lowercase=True,
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        sublinear_tf=True,
        norm="l2",
        max_features=100000,
    )
    char_matrix = char.fit_transform(display_titles).tocsr()
    char_distances = {
        entrant_id: sparse_distance(
            char_matrix,
            index_positions.get(entrant_id, []),
            primary_positions.get(entrant_id, []),
        )
        for entrant_id in cohort.entrant_id.astype(str)
    }
    del char_matrix
    distance["tfidf_word_title_distance_sensitivity"] = distance.entrant_id.map(word_distances)
    distance["tfidf_char_title_distance_sensitivity"] = distance.entrant_id.map(char_distances)
    result = cohort.merge(distance, on="entrant_id", validate="one_to_one")
    result["primary_distance_observed"] = result.specter2_title_distance_primary.notna()
    result_path = PRIVATE / "epj_distance_full_analysis_dataset_private.csv"
    result.to_csv(result_path, index=False, encoding="utf-8-sig")

    summary = (
        result.groupby(["index_year", "entry_with_experienced_top4_coauthor"], as_index=False)
        .agg(
            entrants=("entrant_id", "size"),
            distance_coverage=("primary_distance_observed", "mean"),
            median_specter2_distance=("specter2_title_distance_primary", "median"),
            q25_specter2_distance=("specter2_title_distance_primary", lambda x: x.quantile(0.25)),
            q75_specter2_distance=("specter2_title_distance_primary", lambda x: x.quantile(0.75)),
            median_prior_titles=("n_unique_primary_prior_titles", "median"),
        )
        .sort_values(["index_year", "entry_with_experienced_top4_coauthor"])
    )
    summary_path = TABLES / "epj_distance_full_measurement_summary.csv"
    summary.to_csv(summary_path, index=False)

    observed = result.loc[result.primary_distance_observed].copy()
    correlations = {
        "word_tfidf": float(
            spearmanr(
                observed.specter2_title_distance_primary,
                observed.tfidf_word_title_distance_sensitivity,
                nan_policy="omit",
            ).statistic
        ),
        "char_tfidf": float(
            spearmanr(
                observed.specter2_title_distance_primary,
                observed.tfidf_char_title_distance_sensitivity,
                nan_policy="omit",
            ).statistic
        ),
        "alltype_specter2": float(
            spearmanr(
                observed.specter2_title_distance_primary,
                observed.specter2_title_distance_alltypes_sensitivity,
                nan_policy="omit",
            ).statistic
        ),
    }
    diagnostics = {
        "status": "PASS",
        "entrants": int(len(result)),
        "embedded_unique_titles": int(len(title_keys)),
        "embedding_dimension": int(vectors.shape[1]),
        "embedding_device": device,
        "adapter_name": adapter_name,
        "active_adapters": active_adapters,
        "primary_distance_observed": int(result.primary_distance_observed.sum()),
        "primary_distance_coverage": float(result.primary_distance_observed.mean()),
        "finite_primary_rate": float(np.isfinite(observed.specter2_title_distance_primary).mean()),
        "primary_distance_iqr": float(
            observed.specter2_title_distance_primary.quantile(0.75)
            - observed.specter2_title_distance_primary.quantile(0.25)
        ),
        "measurement_spearman_correlations": correlations,
        "word_vocabulary_size": int(len(word.vocabulary_)),
        "char_vocabulary_size": int(len(char.vocabulary_)),
        "input_sha256": {
            str(COHORT.relative_to(ROOT)): sha256_file(COHORT),
            str(INDEX_PAPERS.relative_to(ROOT)): sha256_file(INDEX_PAPERS),
            str(HISTORY.relative_to(ROOT)): sha256_file(HISTORY),
            str(CONFIG.relative_to(ROOT)): sha256_file(CONFIG),
            str(PROTOCOL.relative_to(ROOT)): sha256_file(PROTOCOL),
        },
        "output_sha256": {
            str(result_path.relative_to(ROOT)): sha256_file(result_path),
            str(summary_path.relative_to(ROOT)): sha256_file(summary_path),
            str(embeddings_path.relative_to(ROOT)): sha256_file(embeddings_path),
            str(titles_path.relative_to(ROOT)): sha256_file(titles_path),
        },
    }
    if diagnostics["finite_primary_rate"] < 0.999:
        diagnostics["status"] = "FAIL"
    diagnostics_path = MANIFESTS / "epj_distance_full_measurement_diagnostics.json"
    diagnostics_path.write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(diagnostics, ensure_ascii=False, indent=2))
    if diagnostics["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Scan local DBLP XML for pre-entry histories of the frozen EPJ pilot sample.

The canonical mode performs one publication scan using the canonical DBLP names
already attached to stable PID anchors. It is a conservative smoke test because
historical aliases may be missed. Raw author-level outputs remain private.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from lxml import etree


ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "outputs" / "epj_origin_feasibility_v01" / "private" / "epj_origin_pilot_sample_private.csv"
XML = ROOT.parent / "bibliometrics" / "dblp" / "dblp.xml.gz"
DTD = ROOT.parent / "bibliometrics" / "dblp" / "dblp.dtd"
OUT = ROOT / "outputs" / "epj_origin_feasibility_v01"
PRIVATE = OUT / "private"
TABLES = OUT / "tables"
MANIFESTS = OUT / "manifests"

PUB_TAGS = {"article", "inproceedings", "incollection", "book", "phdthesis", "mastersthesis"}
ARXIV = "journals/corr/"

FAMILIES = {
    "core_ml": {
        "conf/nips/", "conf/icml/", "conf/iclr/", "conf/aistats/", "conf/colt/",
        "conf/uai/", "conf/automl/", "journals/jmlr/", "journals/tmlr/",
    },
    "adjacent_ai": {
        "conf/aaai/", "conf/ijcai/", "conf/ecai/", "conf/aamas/",
        "conf/cvpr/", "conf/iccv/", "conf/eccv/", "conf/wacv/", "conf/bmvc/", "conf/accv/",
        "journals/pami/", "journals/ijcv/",
        "conf/acl/", "conf/emnlp/", "conf/naacl/", "conf/coling/", "conf/eacl/",
        "conf/aacl/", "conf/conll/", "conf/lrec/", "journals/tacl/",
        "conf/icassp/", "conf/interspeech/", "conf/asru/", "conf/slt/", "conf/waspaa/",
        "journals/taslp/", "conf/kdd/", "conf/sigir/", "conf/wsdm/", "conf/recsys/",
        "conf/icdm/", "conf/cikm/", "conf/www/", "conf/mm/", "conf/icmr/", "conf/ismir/",
    },
    "bio_med_visible_in_dblp": {
        "conf/miccai/", "conf/bibm/", "journals/tmi/", "journals/bioinformatics/",
        "journals/jamia/", "journals/jbi/",
    },
    "other_cs_known": {
        "conf/chi/", "conf/uist/", "conf/cscw/", "conf/sigmod/", "conf/vldb/",
        "conf/osdi/", "conf/sosp/", "conf/nsdi/", "conf/sigcomm/", "conf/mobicom/",
        "conf/stoc/", "conf/focs/", "conf/soda/", "conf/lics/", "conf/ccs/", "conf/sp/",
        "conf/uss/", "conf/ndss/", "conf/icse/", "conf/fse/", "conf/ase/",
        "conf/siggraph/", "conf/eurographics/", "conf/rtss/", "conf/dac/", "conf/date/",
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def venue_prefix(key: str) -> str:
    parts = key.split("/")
    return "/".join(parts[:2]) + "/" if len(parts) >= 3 else ""


def family_of(prefix: str) -> str:
    if prefix == ARXIV:
        return "arxiv_unclassified"
    for family, prefixes in FAMILIES.items():
        if prefix in prefixes:
            return family
    return "unclassified_dblp"


def iter_publications() -> Iterable[tuple[str, str, int, str, list[str]]]:
    with gzip.open(XML, "rb") as handle:
        context = etree.iterparse(
            handle,
            events=("end",),
            load_dtd=True,
            resolve_entities=True,
            no_network=True,
            huge_tree=True,
            recover=True,
        )
        for _, elem in context:
            tag = etree.QName(elem).localname if elem.tag is not etree.Comment else ""
            if tag in PUB_TAGS:
                key = elem.get("key") or ""
                year_node = elem.find("year")
                title_node = elem.find("title")
                try:
                    year = int(year_node.text.strip()) if year_node is not None and year_node.text else -1
                except ValueError:
                    year = -1
                title = "" if title_node is None else "".join(title_node.itertext()).strip()
                authors = [
                    "".join(author.itertext()).strip()
                    for author in elem.findall("author")
                    if "".join(author.itertext()).strip()
                ]
                if year >= 0 and authors:
                    yield tag, key, year, title, authors
                elem.clear()
                while elem.getprevious() is not None:
                    del elem.getparent()[0]


def dominant_family(counts: dict[str, int], n_prior: int) -> str:
    if n_prior == 0:
        return "no_observed_dblp_history"
    classified = {family: counts.get(family, 0) for family in FAMILIES}
    positive = {family: count for family, count in classified.items() if count > 0}
    if not positive:
        return "unclassified_dblp"
    maximum = max(positive.values())
    winners = sorted(family for family, count in positive.items() if count == maximum)
    return winners[0] if len(winners) == 1 else "mixed_or_tied"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--progress-every", type=int, default=1_000_000)
    args = parser.parse_args()

    sample = pd.read_csv(SAMPLE, keep_default_na=False)
    if sample.duplicated(["author_key", "index_year"]).any():
        raise RuntimeError("duplicate author/index-year keys in frozen sample")

    name_to_authors: dict[str, set[str]] = defaultdict(set)
    author_windows: dict[str, tuple[int, int]] = {}
    for row in sample.itertuples(index=False):
        name_to_authors[str(row.author_name)].add(str(row.author_key))
        author_windows[str(row.author_key)] = (int(row.history_start_year), int(row.history_end_year))
    ambiguous_names = {name for name, keys in name_to_authors.items() if len(keys) > 1}
    lookup = {
        name: next(iter(keys))
        for name, keys in name_to_authors.items()
        if len(keys) == 1
    }

    rows: list[dict[str, object]] = []
    records_scanned = 0
    author_hits = 0
    min_year = min(start for start, _ in author_windows.values())
    max_year = max(end for _, end in author_windows.values())
    for publication_type, key, year, title, authors in iter_publications():
        records_scanned += 1
        if args.progress_every and records_scanned % args.progress_every == 0:
            print(f"records_scanned={records_scanned:,} history_rows={len(rows):,}", flush=True)
        if year < min_year or year > max_year:
            continue
        matched_names = set(authors).intersection(lookup)
        if not matched_names:
            continue
        prefix = venue_prefix(key)
        family = family_of(prefix)
        for name in matched_names:
            author_key = lookup[name]
            start, end = author_windows[author_key]
            if start <= year <= end:
                author_hits += 1
                rows.append(
                    {
                        "author_key": author_key,
                        "matched_canonical_name": name,
                        "prior_year": year,
                        "dblp_key": key,
                        "publication_type": publication_type,
                        "venue_prefix": prefix,
                        "provisional_family": family,
                        "title": title,
                    }
                )

    history = pd.DataFrame(rows)
    if history.empty:
        history = pd.DataFrame(
            columns=[
                "author_key", "matched_canonical_name", "prior_year", "dblp_key",
                "publication_type", "venue_prefix", "provisional_family", "title",
            ]
        )
    history = history.drop_duplicates(["author_key", "dblp_key"])
    history_path = PRIVATE / "epj_origin_pilot_dblp_history_canonical_private.csv"
    history.to_csv(history_path, index=False, encoding="utf-8-sig")

    count_table = (
        history.groupby(["author_key", "provisional_family"]).size().unstack(fill_value=0)
        if len(history)
        else pd.DataFrame(index=sample.author_key.unique())
    )
    result = sample.copy()
    for family in [*FAMILIES, "arxiv_unclassified", "unclassified_dblp"]:
        mapping = count_table[family] if family in count_table.columns else pd.Series(dtype=int)
        result[f"n_{family}"] = result.author_key.map(mapping).fillna(0).astype(int)
    family_columns = [f"n_{family}" for family in FAMILIES]
    result["n_prior_dblp_records"] = result.author_key.map(history.groupby("author_key").size()).fillna(0).astype(int)
    result["n_classified_records"] = result[family_columns].sum(axis=1)
    result["n_non_arxiv_records"] = (
        result.n_prior_dblp_records - result.n_arxiv_unclassified
    ).clip(lower=0)
    result["known_record_share_non_arxiv"] = np.where(
        result.n_non_arxiv_records.gt(0),
        result.n_classified_records / result.n_non_arxiv_records,
        np.nan,
    )
    result["has_any_prior_dblp_history"] = result.n_prior_dblp_records.gt(0)
    result["has_classifiable_dblp_origin"] = result.n_classified_records.gt(0)
    result["provisional_dominant_dblp_family"] = [
        dominant_family(
            {family: int(getattr(row, f"n_{family}")) for family in FAMILIES},
            int(row.n_prior_dblp_records),
        )
        for row in result.itertuples(index=False)
    ]
    result["retrieval_mode"] = "canonical_name_only_conservative_smoke"
    result["alias_enhancement_pending"] = True
    author_path = PRIVATE / "epj_origin_pilot_dblp_author_profiles_canonical_private.csv"
    result.to_csv(author_path, index=False, encoding="utf-8-sig")

    strata = ["index_year", "index_venue", "entry_with_experienced_top4_coauthor"]
    summary = (
        result.groupby(strata, as_index=False)
        .agg(
            sampled_authors=("author_key", "nunique"),
            any_prior_history_rate=("has_any_prior_dblp_history", "mean"),
            classifiable_origin_rate=("has_classifiable_dblp_origin", "mean"),
            median_prior_records=("n_prior_dblp_records", "median"),
        )
        .sort_values(strata)
    )
    summary_path = TABLES / "epj_origin_pilot_dblp_coverage_canonical.csv"
    summary.to_csv(summary_path, index=False)

    overall_known = float(result.has_classifiable_dblp_origin.mean())
    by_year = result.groupby("index_year").has_classifiable_dblp_origin.mean().to_dict()
    threshold = 0.70
    within_cs_pass = bool(
        overall_known >= threshold
        and all(float(by_year.get(year, 0.0)) >= threshold for year in sorted(result.index_year.unique()))
    )
    diagnostics = {
        "retrieval_mode": "canonical_name_only_conservative_smoke",
        "records_scanned": records_scanned,
        "raw_author_hits": author_hits,
        "unique_history_rows": int(len(history)),
        "sample_authors": int(result.author_key.nunique()),
        "ambiguous_canonical_names_in_sample": len(ambiguous_names),
        "any_prior_history_rate": float(result.has_any_prior_dblp_history.mean()),
        "classifiable_origin_rate": overall_known,
        "classifiable_origin_rate_by_index_year": {str(k): float(v) for k, v in by_year.items()},
        "within_cs_gate_threshold": threshold,
        "within_cs_gate_provisional_pass": within_cs_pass,
        "important_limitation": "canonical-name-only retrieval may miss historical aliases; this is not the final origin estimate",
        "input_sha256": {
            str(SAMPLE.relative_to(ROOT)): sha256_file(SAMPLE),
            str(XML): sha256_file(XML),
            str(DTD): sha256_file(DTD),
        },
        "output_sha256": {
            str(history_path.relative_to(ROOT)): sha256_file(history_path),
            str(author_path.relative_to(ROOT)): sha256_file(author_path),
            str(summary_path.relative_to(ROOT)): sha256_file(summary_path),
        },
    }
    diagnostic_path = MANIFESTS / "epj_origin_dblp_canonical_diagnostics.json"
    diagnostic_path.write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(diagnostics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


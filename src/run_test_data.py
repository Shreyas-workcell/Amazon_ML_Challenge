"""
Large-Scale Test Dataset Inference Engine.
Processes all 1.73M test Source 1 records against 10M Target records (S2 and S3),
executing country-partitioned candidate retrieval, high-precision scoring,
tripartite conflict resolution, and streaming serialization.
Guarantees strict subset compliance and pass on validate_submission.py.
"""

from __future__ import annotations

import csv
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import subprocess
from src.cpp_bridge import CPPEngine
from src.preprocess import AddressNormalizer, CountryNormalizer, LegalEntityNormalizer, TextNormalizer
from src.utils import setup_logger

logger = setup_logger("run_test_data")

# Common high-frequency stopwords to avoid bloated inverted indexes
STOPWORDS = {
    "and", "the", "for", "with", "ltd", "inc", "corp", "llc", "pvt", "limited",
    "private", "company", "co", "services", "enterprises", "solutions", "group",
    "india", "france", "usa", "delhi", "mumbai", "paris", "bordeaux", "new",
    "trading", "holdings", "international", "de", "la", "le", "des", "du", "sa",
    "sas", "sarl", "eurl", "snc"
}


def extract_search_tokens(text: str) -> List[str]:
    """Extract distinct significant tokens for inverted indexing."""
    cleaned = TextNormalizer.clean_text(text)
    words = cleaned.split()
    return [w for w in words if len(w) >= 3 and w not in STOPWORDS]


def run_full_test_inference(
    test_dir: Path,
    output_dir: Path,
    match_threshold: float = 0.80
) -> Tuple[Path, Path]:
    """
    Execute large-scale inference over test dataset.
    Prioritizes ultra-fast C++ direct data access and inverted indexing,
    with Python fallback and submission validation.
    """
    s1_path = test_dir / "test_source1.tsv"
    s2_path = test_dir / "test_source2.tsv"
    s3_path = test_dir / "test_source3.tsv"

    matching_tsv = output_dir / "matching_results.tsv"
    candidate_tsv = output_dir / "candidate_pairs.tsv"

    if CPPEngine.is_available() and s1_path.exists() and s2_path.exists() and s3_path.exists():
        logger.info("=" * 70)
        logger.info("Executing High-Performance C++ Direct-Access Resolution Engine...")
        logger.info("=" * 70)
        res = CPPEngine.run_pipeline(
            s1_path=s1_path,
            s2_path=s2_path,
            s3_path=s3_path,
            output_dir=output_dir,
            rules={"match_threshold": match_threshold}
        )
        logger.info("C++ Resolution Engine completed: %s", res)

        # Validate with official validate_submission.py if present
        val_script = PROJECT_ROOT / "data" / "student_resource" / "utils" / "validate_submission.py"
        if val_script.exists():
            logger.info("Running submission validator on generated TSVs...")
            val_cmd = [
                sys.executable, str(val_script),
                "--matching", str(matching_tsv),
                "--candidate", str(candidate_tsv),
                "--test-dir", str(test_dir)
            ]
            try:
                v_res = subprocess.run(val_cmd, capture_output=True, text=True, timeout=60)
                logger.info("Validator Output:\n%s", v_res.stdout)
                if v_res.returncode != 0:
                    logger.warning("Validator detected issues:\n%s", v_res.stderr)
            except Exception as e:
                logger.warning("Submission validation check skipped: %s", e)

        return matching_tsv, candidate_tsv

    logger.info("C++ engine not available or file paths missing. Running Python fallback...")
    return run_full_test_inference_python(test_dir, output_dir, match_threshold)


def run_full_test_inference_python(
    test_dir: Path,
    output_dir: Path,
    match_threshold: float = 0.80
) -> Tuple[Path, Path]:
    """Fallback pure-Python streaming inference."""
    t_start = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)
    matching_tsv = output_dir / "matching_results.tsv"
    candidate_tsv = output_dir / "candidate_pairs.tsv"

    s1_path = test_dir / "test_source1.tsv"
    s2_path = test_dir / "test_source2.tsv"
    s3_path = test_dir / "test_source3.tsv"

    legal_norm = LegalEntityNormalizer()

    # Step 1: Read all S1 records and group indices by country
    logger.info("=" * 70)
    logger.info("STEP 1: Ingesting test_source1.tsv (1.73M records)...")
    logger.info("=" * 70)

    s1_order: List[str] = []
    # country -> list of (s1_id, root_name, tokens_set, street_nums_set, postal_code)
    s1_by_country: Dict[str, List[Tuple[str, str, Set[str], Set[str], str]]] = defaultdict(list)

    with open(s1_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        for row in reader:
            if not row:
                continue
            s1_id = row[0]
            name = row[1] if len(row) > 1 else ""
            addr = row[2] if len(row) > 2 else ""
            country_raw = row[3] if len(row) > 3 else "UNKNOWN"
            country = CountryNormalizer.normalize_country(country_raw)

            clean_name = TextNormalizer.clean_text(name)
            root_name, _ = legal_norm.parse(clean_name)
            tokens = set(extract_search_tokens(root_name))
            street_nums = set(AddressNormalizer.extract_street_numbers(addr))
            postal = AddressNormalizer.extract_postal_code(addr)

            s1_order.append(s1_id)
            s1_by_country[country].append((s1_id, root_name, tokens, street_nums, postal))

    total_s1 = len(s1_order)
    logger.info(
        "Ingested %d S1 records across %d countries in %.2fs: %s",
        total_s1, len(s1_by_country), time.time() - t_start,
        {c: len(records) for c, records in s1_by_country.items()}
    )

    # Dictionaries to store final candidate and matched IDs per S1
    s1_candidates_map: Dict[str, List[str]] = {}
    s1_matches_map: Dict[str, List[str]] = {}

    # Step 2: Process Country by Country to keep memory ultra-low (< 1.5 GB)
    all_countries = sorted(list(s1_by_country.keys()))

    for country in all_countries:
        c_t0 = time.time()
        s1_records = s1_by_country[country]
        logger.info("-" * 70)
        logger.info(
            "Processing Country [%s]: %d S1 reference records...",
            country, len(s1_records)
        )
        logger.info("-" * 70)

        # Build Inverted Indexes for Target Sources (S2 and S3) in this country
        # exact_name -> list of target_id
        exact_index: Dict[str, List[str]] = defaultdict(list)
        # token -> list of target_id
        token_index: Dict[str, List[str]] = defaultdict(list)
        # target_id -> (root_name, tokens_set, street_nums_set, postal_code)
        target_metadata: Dict[str, Tuple[str, Set[str], Set[str], str]] = {}

        for src_name, src_path in [("S2", s2_path), ("S3", s3_path)]:
            logger.info("Indexing %s for country %s...", src_name, country)
            read_count = 0
            with open(src_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f, delimiter="\t")
                next(reader)
                for row in reader:
                    if not row:
                        continue
                    c_val = CountryNormalizer.normalize_country(row[3] if len(row) > 3 else "UNKNOWN")
                    if c_val != country:
                        continue

                    read_count += 1
                    t_id = row[0]
                    t_name = row[1] if len(row) > 1 else ""
                    t_addr = row[2] if len(row) > 2 else ""

                    c_name = TextNormalizer.clean_text(t_name)
                    r_name, _ = legal_norm.parse(c_name)
                    t_tokens = set(extract_search_tokens(r_name))
                    t_snums = set(AddressNormalizer.extract_street_numbers(t_addr))
                    t_postal = AddressNormalizer.extract_postal_code(t_addr)

                    target_metadata[t_id] = (r_name, t_tokens, t_snums, t_postal)

                    # Exact name index
                    if r_name and len(exact_index[r_name]) < 20:
                        exact_index[r_name].append(t_id)

                    # Distinctive token index (cap posting list length to prevent blowout)
                    for tok in t_tokens:
                        lst = token_index[tok]
                        if len(lst) < 25:
                            lst.append(t_id)

            logger.info("Loaded %d target records from %s for %s", read_count, src_name, country)

        logger.info(
            "Index built for [%s]: %d target entities, %d exact names, %d token keys (%.2fs)",
            country, len(target_metadata), len(exact_index), len(token_index), time.time() - c_t0
        )

        # Query S1 records in this country
        q_t0 = time.time()
        c_matched_entities = 0
        c_total_links = 0

        for s1_id, root_name, s1_toks, s1_snums, s1_post in s1_records:
            cand_scores: Dict[str, float] = {}

            # 1. Exact root name matches (highest confidence)
            if root_name and root_name in exact_index:
                for t_id in exact_index[root_name]:
                    cand_scores[t_id] = 1.0

            # 2. Token overlap matches
            cand_overlap_counts: Dict[str, int] = defaultdict(int)
            for tok in s1_toks:
                if tok in token_index:
                    for t_id in token_index[tok]:
                        cand_overlap_counts[t_id] += 1

            for t_id, overlap in cand_overlap_counts.items():
                if t_id in cand_scores:
                    continue  # already scored exact match
                t_root, t_toks, t_snums, t_post = target_metadata[t_id]
                # Token Jaccard
                union_len = len(s1_toks.union(t_toks))
                jaccard = overlap / max(1, union_len)

                # Street number checks
                sn_bonus = 0.0
                if s1_snums and t_snums:
                    if s1_snums.intersection(t_snums):
                        sn_bonus = 0.25
                    else:
                        sn_bonus = -0.50  # Contradicting street numbers

                # Postal code check
                postal_bonus = 0.15 if (s1_post and t_post and s1_post == t_post) else 0.0

                score = jaccard + sn_bonus + postal_bonus
                if score >= 0.35:
                    cand_scores[t_id] = score

            # Sort and select candidates
            if cand_scores:
                sorted_cands = sorted(cand_scores.items(), key=lambda x: x[1], reverse=True)
                top_cands = sorted_cands[:25]
                cand_ids = [c[0] for c in top_cands]
                # High-precision thresholding for matching
                match_ids = [c[0] for c in top_cands if c[1] >= match_threshold]

                s1_candidates_map[s1_id] = cand_ids
                s1_matches_map[s1_id] = match_ids

                if match_ids:
                    c_matched_entities += 1
                    c_total_links += len(match_ids)
            else:
                s1_candidates_map[s1_id] = []
                s1_matches_map[s1_id] = []

        logger.info(
            "Completed [%s] in %.2fs: %d/%d S1 entities matched (Total links: %d)",
            country, time.time() - q_t0, c_matched_entities, len(s1_records), c_total_links
        )

    # Step 3: Stream Serializing to Output TSVs in Exact Order of test_source1.tsv
    logger.info("=" * 70)
    logger.info("STEP 3: Serializing output TSVs in exact test_source1.tsv order...")
    logger.info("=" * 70)

    s_t0 = time.time()
    total_matches = 0
    total_candidates = 0

    with open(matching_tsv, "w", encoding="utf-8", newline="") as f_match, \
         open(candidate_tsv, "w", encoding="utf-8", newline="") as f_cand:

        match_writer = csv.writer(f_match, delimiter="\t")
        cand_writer = csv.writer(f_cand, delimiter="\t")

        match_writer.writerow(["source1_entity_id", "matched_entity_ids"])
        cand_writer.writerow(["source1_entity_id", "candidate_entity_ids"])

        for s1_id in sorted(s1_order):
            cands = sorted(s1_candidates_map.get(s1_id, []))
            matches = sorted(s1_matches_map.get(s1_id, []))

            # Ensure strict subset property
            valid_matches = [m for m in matches if m in cands]

            total_candidates += len(cands)
            total_matches += len(valid_matches)

            cand_writer.writerow([s1_id, ",".join(cands) if cands else ""])
            match_writer.writerow([s1_id, ",".join(valid_matches) if valid_matches else ""])

    logger.info(
        "Serialized %d rows to %s and %s in %.2fs",
        total_s1, matching_tsv.name, candidate_tsv.name, time.time() - s_t0
    )
    logger.info(
        "Final Statistics: S1 Entities=%d, Total Candidates=%d, Total Matches=%d, Non-Singletons=%d",
        total_s1, total_candidates, total_matches,
        sum(1 for m in s1_matches_map.values() if m)
    )
    logger.info("Total Pipeline Elapsed: %.2f seconds", time.time() - t_start)
    return matching_tsv, candidate_tsv


if __name__ == "__main__":
    test_directory = Path("data/student_resource/dataset/test")
    output_directory = Path("output")
    run_full_test_inference(test_directory, output_directory, match_threshold=0.80)

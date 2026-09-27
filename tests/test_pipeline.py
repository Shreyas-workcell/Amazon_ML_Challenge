"""
Automated test suite for Business Entity Resolution.
Tests normalization across US, India, and France, metric calculation including singletons,
country-partitioned candidate retrieval, and end-to-end tripartite resolution.
"""

from __future__ import annotations

import csv
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Dict, List, Set

import numpy as np
import pandas as pd

from src.blocking import CountryPartitionedBlocker
from src.config import DEFAULT_CONFIG, PipelineConfig
from src.features import FastStringMetrics, PairFeatureExtractor
from src.graph_resolver import TripartiteGraphResolver
from src.inference import EntityResolutionPipeline
from src.preprocess import (
    AddressNormalizer,
    CountryNormalizer,
    LegalEntityNormalizer,
    RecordPreprocessor,
    TextNormalizer,
)
from src.train_matcher import MatcherTrainer
from src.utils import compute_entity_f05, compute_macro_f05, validate_tsv_outputs


class TestPreprocessing(unittest.TestCase):
    """Test country-agnostic normalization, legal suffixes, and diacritic handling."""

    def setUp(self) -> None:
        self.legal_norm = LegalEntityNormalizer()

    def test_unicode_and_accent_stripping(self) -> None:
        # French text with accents
        raw = "Café & Boulangerie Étoile SAS"
        cleaned = TextNormalizer.clean_text(raw)
        self.assertNotIn("é", cleaned)
        self.assertNotIn("É", cleaned)
        self.assertEqual(cleaned, "cafe boulangerie etoile sas")

    def test_multi_country_legal_suffixes(self) -> None:
        # US: Corporation / LLC
        root, legal = self.legal_norm.parse("acme logistics llc")
        self.assertEqual(root, "acme logistics")
        self.assertEqual(legal, "llc")

        # India: Private Limited
        root, legal = self.legal_norm.parse("infosys technologies pvt ltd")
        self.assertEqual(root, "infosys technologies")
        self.assertEqual(legal, "pvt_ltd")

        # France: SAS / SARL (Distribution Shift)
        root, legal = self.legal_norm.parse("renault distribution sas")
        self.assertEqual(root, "renault distribution")
        self.assertEqual(legal, "sas")

        root, legal = self.legal_norm.parse("boulangerie martin sarl")
        self.assertEqual(root, "boulangerie martin")
        self.assertEqual(legal, "sarl")

    def test_address_and_postal_extraction(self) -> None:
        # US address
        numbers = AddressNormalizer.extract_street_numbers("742 Evergreen Terrace Apt 5B")
        self.assertIn("742", numbers)
        self.assertIn("5b", numbers)
        self.assertEqual(AddressNormalizer.extract_postal_code("New York NY 10001-1234"), "10001")

        # India PIN code
        self.assertEqual(AddressNormalizer.extract_postal_code("Bangalore, Karnataka 560001"), "560001")

        # France postal code
        self.assertEqual(AddressNormalizer.extract_postal_code("12 Rue de la Paix 75008 Paris"), "75008")

    def test_open_set_country_normalization(self) -> None:
        self.assertEqual(CountryNormalizer.normalize_country("United States"), "US")
        self.assertEqual(CountryNormalizer.normalize_country("u.s.a."), "US")
        self.assertEqual(CountryNormalizer.normalize_country("India"), "IN")
        self.assertEqual(CountryNormalizer.normalize_country("FRANCE"), "FR")
        # Open-set unseen country
        self.assertEqual(CountryNormalizer.normalize_country("Brazil"), "BRAZIL")
        self.assertEqual(CountryNormalizer.normalize_country(None), "UNKNOWN")


class TestMetricEvaluation(unittest.TestCase):
    """Test exact Macro-F0.5 computation and singleton edge cases."""

    def test_singleton_eval(self) -> None:
        # True singleton predicted empty -> 1.0
        score = compute_entity_f05(y_true=set(), y_pred=set(), beta=0.5)
        self.assertEqual(score, 1.0)

        # True singleton with false positive prediction -> 0.0
        score = compute_entity_f05(y_true=set(), y_pred={"S2-99"}, beta=0.5)
        self.assertEqual(score, 0.0)

    def test_macro_f05_formula(self) -> None:
        # 1 TP, 1 FN: P = 1.0, R = 0.5 -> F0.5 = 1.25 * 1.0 * 0.5 / (0.25 * 1.0 + 0.5) = 0.625 / 0.75 = 0.8333...
        y_true = {"S2-1", "S3-1"}
        y_pred = {"S2-1"}
        score = compute_entity_f05(y_true, y_pred, beta=0.5)
        expected = (1.25 * 1.0 * 0.5) / (0.25 * 1.0 + 0.5)
        self.assertAlmostEqual(score, expected, places=4)

        # 1 TP, 1 FP: P = 0.5, R = 1.0 -> F0.5 = 1.25 * 0.5 * 1.0 / (0.25 * 0.5 + 1.0) = 0.625 / 1.125 = 0.5555...
        # Notice how false positive hurts significantly more than false negative!
        y_true_fp = {"S2-1"}
        y_pred_fp = {"S2-1", "S2-99"}
        score_fp = compute_entity_f05(y_true_fp, y_pred_fp, beta=0.5)
        expected_fp = (1.25 * 0.5 * 1.0) / (0.25 * 0.5 + 1.0)
        self.assertAlmostEqual(score_fp, expected_fp, places=4)
        self.assertGreater(score, score_fp)  # High precision prioritization verified!


class TestEndToEndPipeline(unittest.TestCase):
    """Integration test with synthetic multi-country tripartite data."""

    def setUp(self) -> None:
        self.test_dir = Path(tempfile.mkdtemp())
        self.s1_path = self.test_dir / "s1.csv"
        self.s2_path = self.test_dir / "s2.csv"
        self.s3_path = self.test_dir / "s3.csv"
        self.gt_path = self.test_dir / "ground_truth.tsv"
        self.out_dir = self.test_dir / "output"

        # Create multi-country synthetic dataset: US, India, France, and Singletons
        # S1: Reference Source (Clean)
        s1_data = [
            {"entity_id": "S1-101", "name": "Apple Inc.", "address": "1 Infinite Loop", "city": "Cupertino", "state": "CA", "postal_code": "95014", "country": "US", "phone": "4089961010"},
            {"entity_id": "S1-102", "name": "Tata Consultancy Services Ltd", "address": "Birlabagh Fort", "city": "Mumbai", "state": "MH", "postal_code": "400001", "country": "India", "phone": "9820011223"},
            {"entity_id": "S1-103", "name": "L'Oréal SA", "address": "41 Rue Martre", "city": "Clichy", "state": "IDF", "postal_code": "92117", "country": "France", "phone": "33147567000"},
            {"entity_id": "S1-104", "name": "Boulangerie Paul SAS", "address": "12 Avenue Victor Hugo", "city": "Paris", "state": "IDF", "postal_code": "75116", "country": "France", "phone": "33145000000"},
            {"entity_id": "S1-105", "name": "Lonely Singleton Enterprise LLC", "address": "999 Remote Desolate Way", "city": "Nome", "state": "AK", "postal_code": "99762", "country": "US", "phone": "9074430000"},
        ]
        pd.DataFrame(s1_data).to_csv(self.s1_path, index=False)

        # S2: Target Source (Noisy with typos, abbreviations, missing tokens)
        s2_data = [
            {"entity_id": "S2-201", "name": "Apple Incorporated", "address": "1 Infinite Loop Dr", "city": "Cupertino", "state": "CA", "postal_code": "95014", "country": "United States", "phone": "(408) 996-1010"},
            {"entity_id": "S2-202", "name": "TCS Pvt Ltd", "address": "Birlabagh", "city": "Mumbai", "state": "Maharashtra", "postal_code": "400001", "country": "IN", "phone": "+91 9820011223"},
            {"entity_id": "S2-203", "name": "L'Oreal", "address": "41 Rue Martre", "city": "Clichy", "state": "", "postal_code": "92117", "country": "FR", "phone": "0147567000"},
            {"entity_id": "S2-204", "name": "Boulangerie Paul", "address": "12 Av Victor Hugo", "city": "Paris", "state": "", "postal_code": "75116", "country": "France", "phone": ""},
            {"entity_id": "S2-999", "name": "Completely Unrelated Corp", "address": "500 Elm Street", "city": "Dallas", "state": "TX", "postal_code": "75201", "country": "US", "phone": "2145550199"},
        ]
        pd.DataFrame(s2_data).to_csv(self.s2_path, index=False)

        # S3: Target Source (Alternative noisy format)
        s3_data = [
            {"entity_id": "S3-301", "name": "Apple", "address": "One Infinite Loop", "city": "Cupertino", "state": "California", "postal_code": "95014", "country": "USA", "phone": "408-996-1010"},
            {"entity_id": "S3-302", "name": "Tata Consultancy Services", "address": "Fort Birlabagh Rd", "city": "Bombay", "state": "MH", "postal_code": "400001", "country": "India", "phone": ""},
            {"entity_id": "S3-303", "name": "L'Oreal SA France", "address": "41 R. Martre", "city": "Clichy", "state": "IDF", "postal_code": "92117", "country": "FRA", "phone": "+33 1 47 56 70 00"},
            {"entity_id": "S3-304", "name": "Paul Boulangerie SASU", "address": "12 Ave Victor Hugo", "city": "Paris", "state": "Paris", "postal_code": "75116", "country": "FR", "phone": "0145000000"},
        ]
        pd.DataFrame(s3_data).to_csv(self.s3_path, index=False)

        # Ground Truth TSV (Multi-match for S1-101..104, Singleton for S1-105)
        gt_mapping = {
            "S1-101": ["S2-201", "S3-301"],
            "S1-102": ["S2-202", "S3-302"],
            "S1-103": ["S2-203", "S3-303"],
            "S1-104": ["S2-204", "S3-304"],
            "S1-105": [],  # Singleton
        }
        with open(self.gt_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(["source1_entity_id", "matched_entity_ids"])
            for s1_id, matches in gt_mapping.items():
                writer.writerow([s1_id, ",".join(matches)])

    def tearDown(self) -> None:
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_pipeline_execution_and_constraints(self) -> None:
        pipeline = EntityResolutionPipeline()
        summary = pipeline.run(
            s1_file=self.s1_path,
            s2_file=self.s2_path,
            s3_file=self.s3_path,
            ground_truth_file=self.gt_path,
            output_dir=self.out_dir
        )

        cand_file = Path(summary["candidate_file"])
        match_file = Path(summary["matching_file"])

        self.assertTrue(cand_file.exists(), "candidate_pairs.tsv must exist")
        self.assertTrue(match_file.exists(), "matching_results.tsv must exist")

        # Validate TSV format and subset constraints
        is_valid, errors = validate_tsv_outputs(cand_file, match_file)
        self.assertTrue(is_valid, f"TSV validation failed with errors: {errors}")

        # Check macro F0.5 score on synthetic test set
        self.assertIsNotNone(summary["macro_f05"])
        macro_f05 = summary["macro_f05"]
        print(f"\n[Test Pipeline] Achieved Macro-F0.5 = {macro_f05:.4f} (Target >= 0.9333)")
        self.assertGreaterEqual(
            macro_f05, 0.9333,
            f"Macro-F0.5 {macro_f05:.4f} did not meet target >= 0.9333"
        )

        # Verify singleton entity was predicted empty
        with open(match_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            next(reader)
            for row in reader:
                if row[0] == "S1-105":
                    # Should be empty or no matches assigned
                    matches_val = row[1] if len(row) > 1 else ""
                    self.assertEqual(matches_val.strip(), "", "Singleton S1-105 must have no matched links")

    def test_phone_blocking_recovers_name_mismatch(self) -> None:
        s1 = pd.DataFrame([{
            "entity_id": "S1-1", "name": "Reference Business", "address": "1 Main St",
            "country": "US", "phone": "555-0100"
        }])
        s2 = pd.DataFrame([{
            "entity_id": "S2-1", "name": "Different Trading Name", "address": "9 Other Rd",
            "country": "US", "phone": "(555) 0100"
        }])
        s3 = pd.DataFrame(columns=s2.columns)
        prep = RecordPreprocessor(DEFAULT_CONFIG.columns)
        s1_p = prep.process_dataframe(s1, source_prefix="S1")
        s2_p = prep.process_dataframe(s2, source_prefix="S2")
        s3_p = prep.process_dataframe(s3, source_prefix="S3")

        candidate_dict, _ = CountryPartitionedBlocker(DEFAULT_CONFIG.blocking).generate_candidates(
            s1_p, s2_p, s3_p
        )
        self.assertIn("S2-1", candidate_dict["S1-1"])


if __name__ == "__main__":
    unittest.main()

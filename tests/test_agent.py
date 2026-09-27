"""
Unit tests for BusinessEntityResolutionAgent and DataResourceManager.
"""

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.agent import BusinessEntityResolutionAgent, DataResourceManager
from src.config import PipelineConfig


class TestBusinessEntityResolutionAgent(unittest.TestCase):
    """Test suite verifying agent capabilities on sample datasets."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.work_dir = Path(self.temp_dir.name)

        # Create sample DataFrames
        self.s1_df = pd.DataFrame([
            {
                "entity_id": "S1-00001",
                "name": "Acme Corp",
                "address": "100 Main St",
                "city": "New York",
                "state": "NY",
                "postal_code": "10001",
                "country": "US",
                "phone": "555-0100"
            },
            {
                "entity_id": "S1-00002",
                "name": "Global Traders Inc",
                "address": "500 Market Rd",
                "city": "San Francisco",
                "state": "CA",
                "postal_code": "94105",
                "country": "US",
                "phone": "555-0200"
            }
        ])

        self.s2_df = pd.DataFrame([
            {
                "entity_id": "S2-00001",
                "name": "ACME CORPORATION",
                "address": "100 MAIN STREET",
                "city": "NEW YORK",
                "state": "NY",
                "postal_code": "10001",
                "country": "USA",
                "phone": "555-0100"
            },
            {
                "entity_id": "S2-00002",
                "name": "Global Trading LLC",
                "address": "500 Market Street",
                "city": "San Francisco",
                "state": "CA",
                "postal_code": "94105",
                "country": "US",
                "phone": "555-0200"
            }
        ])

        self.s3_df = pd.DataFrame([
            {
                "entity_id": "S3-00001",
                "name": "Acme Co",
                "address": "100 Main",
                "city": "NY",
                "state": "NY",
                "postal_code": "10001",
                "country": "United States",
                "phone": "555-0100"
            }
        ])

        # Write sample files
        self.s1_file = self.work_dir / "sample_s1.csv"
        self.s2_file = self.work_dir / "sample_s2.csv"
        self.s3_file = self.work_dir / "sample_s3.csv"
        self.gt_file = self.work_dir / "ground_truth.tsv"

        self.s1_df.to_csv(self.s1_file, index=False)
        self.s2_df.to_csv(self.s2_file, index=False)
        self.s3_df.to_csv(self.s3_file, index=False)

        with open(self.gt_file, "w", encoding="utf-8") as f:
            f.write("source1_entity_id\tmatched_entity_ids\n")
            f.write("S1-00001\tS2-00001,S3-00001\n")
            f.write("S1-00002\tS2-00002\n")

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_resource_ingestion_and_inspection(self) -> None:
        """Verify data resource ingestion and health score inspection."""
        agent = BusinessEntityResolutionAgent()
        ingest_res = agent.ingest_resources(
            s1=self.s1_file,
            s2=self.s2_file,
            s3=self.s3_file,
            ground_truth=self.gt_file
        )
        self.assertEqual(ingest_res["status"], "ingested_successfully")
        self.assertIn("s1", ingest_res)
        self.assertEqual(ingest_res["s1"]["record_count"], 2)

        inspect_res = agent.inspect_resources()
        self.assertIn("overall_health_score", inspect_res)
        self.assertGreaterEqual(inspect_res["overall_health_score"], 70.0)

    def test_full_resolution_and_evaluation_flow(self) -> None:
        """Verify full resolution, metric evaluation, and entity querying."""
        from src.config import PathConfig, PipelineConfig
        cfg = PipelineConfig(paths=PathConfig(
            data_dir=self.work_dir,
            models_dir=self.work_dir / "models",
            output_dir=self.work_dir / "output"
        ))
        agent = BusinessEntityResolutionAgent(config=cfg)
        agent.ingest_resources(
            s1=self.s1_file,
            s2=self.s2_file,
            s3=self.s3_file,
            ground_truth=self.gt_file
        )

        out_dir = self.work_dir / "output"
        res_summary = agent.resolve(output_dir=out_dir)

        self.assertTrue((out_dir / "candidate_pairs.tsv").exists())
        self.assertTrue((out_dir / "matching_results.tsv").exists())
        self.assertGreater(res_summary["candidate_pairs_count"], 0)

        # Evaluation
        eval_report = agent.evaluate()
        self.assertIn("macro_f05", eval_report)
        self.assertGreaterEqual(eval_report["macro_f05"], 0.5)

        # Entity Query
        query_res = agent.query_entity("S1-00001")
        self.assertEqual(query_res["s1_entity_id"], "S1-00001")
        self.assertIn("record", query_res)
        self.assertIn("resolved_matches", query_res)

        # Report Export
        report_file = agent.export_summary_report(self.work_dir / "report.json")
        self.assertTrue(report_file.exists())

        tsv_report = agent.export_tsv_report(self.work_dir / "resolution_report.tsv")
        report_rows = {}
        with open(tsv_report, encoding="utf-8") as f:
            for line in f:
                fields = line.rstrip("\n").split("\t")
                if fields[0] != "metric":
                    report_rows[fields[0]] = fields[1]
        self.assertEqual(report_rows["total_s1_records"], "2")
        self.assertEqual(report_rows["total_s2_records"], "2")
        self.assertEqual(report_rows["total_s3_records"], "1")
        self.assertEqual(report_rows["total_matches_resolved"], str(res_summary["matched_links_count"]))

    def test_directory_auto_discovery(self) -> None:
        """Verify directory discovery when single path is passed."""
        mgr = DataResourceManager()
        discovered = mgr._discover_directory(self.work_dir)
        self.assertIn("s1", discovered)
        self.assertIn("s2", discovered)
        self.assertIn("s3", discovered)
        self.assertIn("ground_truth", discovered)

    def test_agent_tools(self) -> None:
        """Verify helper tool functions in src/tools.py."""
        from src.tools import inspect_data_resources, run_entity_resolution
        inspect_json = inspect_data_resources(str(self.s1_file), str(self.s2_file), str(self.s3_file))
        inspect_data = json.loads(inspect_json)
        self.assertIn("overall_health_score", inspect_data)

        run_json = run_entity_resolution(
            str(self.s1_file),
            str(self.s2_file),
            str(self.s3_file),
            ground_truth_path=str(self.gt_file),
            output_dir=str(self.work_dir / "out_tools")
        )
        run_data = json.loads(run_json)
        self.assertIn("candidate_pairs_count", run_data)


if __name__ == "__main__":
    unittest.main()


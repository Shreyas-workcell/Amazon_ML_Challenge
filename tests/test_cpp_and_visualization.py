"""
Unit tests for C++ Core Engine (cpp_bridge) and Matplotlib Tripartite Graph Visualizer.
"""

import json
import tempfile
import unittest
from pathlib import Path

from src.cpp_bridge import CPPEngine
from src.graph_visualizer import build_networkx_tripartite_graph, visualize_tripartite_graph


class TestCPPAndVisualization(unittest.TestCase):
    """Test suite verifying C++ core functions and Matplotlib graph visualization."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.work_dir = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_cpp_engine_bindings(self) -> None:
        """Verify C++ text normalization, legal suffix parsing, and string metrics."""
        self.assertTrue(CPPEngine.is_available(), "C++ engine library should be loaded.")

        # Test text normalization
        norm_res = CPPEngine.normalize_text("Société Général Étoile Inc")
        self.assertEqual(norm_res, "societe general etoile inc")

        # Test legal suffix parsing
        root_name, legal_type = CPPEngine.parse_legal_suffix("acme corporation")
        self.assertEqual(root_name, "acme")
        self.assertEqual(legal_type, "corp")

        # Test string distance metrics
        lev_dist = CPPEngine.levenshtein("acme", "acme")
        self.assertEqual(lev_dist, 0)

        jw_sim = CPPEngine.jaro_winkler("acme corp", "acme corporation")
        self.assertGreaterEqual(jw_sim, 0.90)

    def test_tripartite_graph_visualization(self) -> None:
        """Verify NetworkX graph building and Matplotlib image rendering."""
        s1_dict = {
            "S1-1001": {"name": "Alphabet Incorporated"},
            "S1-1002": {"name": "Microsoft Corporation"}
        }
        target_dict = {
            "S2-2001": {"name": "Alphabet Inc"},
            "S3-3001": {"name": "Alphabet Co"},
            "S2-2002": {"name": "Microsoft Corp"}
        }
        candidate_pairs = {
            "S1-1001": ["S2-2001", "S3-3001"],
            "S1-1002": ["S2-2002"]
        }
        matching_results = {
            "S1-1001": ["S2-2001", "S3-3001"],
            "S1-1002": ["S2-2002"]
        }

        # Build NetworkX Graph
        G = build_networkx_tripartite_graph(s1_dict, target_dict, candidate_pairs, matching_results)
        self.assertEqual(len(G.nodes), 5)
        self.assertEqual(len(G.edges), 3)

        # Render Matplotlib Visualization image
        out_img = self.work_dir / "tripartite_test.png"
        res_path = visualize_tripartite_graph(G, output_path=out_img, show=False)

        self.assertTrue(res_path.exists())
        self.assertGreater(res_path.stat().st_size, 10_000)

    def test_cpp_direct_pipeline_and_validation(self) -> None:
        """Verify C++ direct data TSV ingestion, matching, and submission validation."""
        s1_file = self.work_dir / "test_source1.tsv"
        s2_file = self.work_dir / "test_source2.tsv"
        s3_file = self.work_dir / "test_source3.tsv"

        s1_file.write_text(
            "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
            "S1-1001\tAlphabet Incorporated\t1600 Amphitheatre Pkwy\tUS\n"
            "S1-1002\tMicrosoft Corporation\tOne Microsoft Way\tUS\n"
            "S1-1003\tSociété Générale\t29 Boulevard Haussmann\tFrance\n"
            "S1-1004\tTata Consultancy Services\tFort Mumbai\tIndia\n"
            "S1-1005\tSingleton Company\t999 Nowhere Rd\tUS\n"
        )
        s2_file.write_text(
            "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
            "S2-2001\tAlphabet Inc\t1600 Amphitheatre Parkway\tUS\n"
            "S2-2002\tMicrosoft Corp\t1 Microsoft Way\tUS\n"
            "S2-2003\tSociete Generale SA\t29 Bd Haussmann\tFrance\n"
        )
        s3_file.write_text(
            "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
            "S3-3001\tAlphabet Co\t1600 Amphitheatre Pkwy\tUS\n"
            "S3-3002\tTata Consultancy Ltd\tFort Mumbai\tIndia\n"
        )

        out_dir = self.work_dir / "output"
        summary = CPPEngine.run_pipeline(
            s1_path=s1_file,
            s2_path=s2_file,
            s3_path=s3_file,
            output_dir=out_dir,
            rules={"match_threshold": 0.55}
        )

        self.assertEqual(summary["status"], "success")
        self.assertEqual(summary["total_s1"], 5)
        self.assertEqual(summary["total_s2"], 3)
        self.assertEqual(summary["total_s3"], 2)
        self.assertGreater(summary["total_candidates"], 0)
        self.assertGreater(summary["total_matches"], 0)
        self.assertEqual(summary["singletons"], 1)

        cand_file = out_dir / "candidate_pairs.tsv"
        match_file = out_dir / "matching_results.tsv"
        self.assertTrue(cand_file.exists())
        self.assertTrue(match_file.exists())

        # Validate line counts & format
        cand_lines = [line.strip().split("\t") for line in cand_file.read_text().strip().split("\n")]
        match_lines = [line.strip().split("\t") for line in match_file.read_text().strip().split("\n")]

        self.assertEqual(len(cand_lines), 6)  # header + 5 S1 entities
        self.assertEqual(len(match_lines), 6)

        # Verify singleton S1-1005 has empty match list
        s1_1005_match = [row for row in match_lines if row[0] == "S1-1005"]
        self.assertEqual(len(s1_1005_match), 1)
        self.assertEqual(len(s1_1005_match[0]), 1)  # only entity_id, no matches after tab

        # Test query_entity
        q_res = CPPEngine.query_entity(
            s1_id="S1-1001",
            name="Alphabet Inc",
            address="1600 Amphitheatre Pkwy",
            country="US",
            s2_path=s2_file,
            s3_path=s3_file,
            match_threshold=0.55
        )
        self.assertEqual(q_res["s1_id"], "S1-1001")
        self.assertIn("S2-2001", q_res["matches"])


if __name__ == "__main__":
    unittest.main()

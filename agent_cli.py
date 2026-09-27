#!/usr/bin/env python3
"""
CLI interface for Business Entity Resolution Agent.
Provides rich commands: run, inspect, train, evaluate, query.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent import BusinessEntityResolutionAgent


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Business Entity Resolution Autonomous Agent CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    subparsers = parser.add_subparsers(dest="command", help="Agent action to execute")

    # Command: run / resolve
    run_parser = subparsers.add_parser("resolve", help="Ingest datasets and run full entity resolution pipeline")
    run_parser.add_argument("--s1", type=str, required=True, help="Path to Source 1 CSV/TSV or dataset directory")
    run_parser.add_argument("--s2", type=str, required=False, help="Path to Source 2 CSV/TSV (optional if directory specified)")
    run_parser.add_argument("--s3", type=str, required=False, help="Path to Source 3 CSV/TSV (optional if directory specified)")
    run_parser.add_argument("--ground-truth", type=str, default=None, help="Path to ground truth TSV (optional)")
    run_parser.add_argument("--output-dir", type=str, default=None, help="Output directory for TSV artifacts")
    run_parser.add_argument("--mode", type=str, choices=["auto", "cpp", "standard", "streaming"], default="auto", help="Execution engine mode")
    run_parser.add_argument("--visualize", action="store_true", help="Launch Matplotlib tripartite graph visualizer")

    # Command: visualize
    viz_parser = subparsers.add_parser("visualize", help="Build NetworkX tripartite graph and launch Matplotlib visualization")
    viz_parser.add_argument("--s1", type=str, required=True, help="Path to Source 1 CSV/TSV or dataset directory")
    viz_parser.add_argument("--s2", type=str, required=False, help="Path to Source 2 CSV/TSV")
    viz_parser.add_argument("--s3", type=str, required=False, help="Path to Source 3 CSV/TSV")
    viz_parser.add_argument("--img-out", type=str, default=None, help="Output image path (e.g. output/tripartite_graph.png)")

    # Command: inspect
    inspect_parser = subparsers.add_parser("inspect", help="Inspect structural quality & health of data resources")
    inspect_parser.add_argument("--s1", type=str, required=True, help="Path to Source 1 CSV/TSV or dataset directory")
    inspect_parser.add_argument("--s2", type=str, required=False, help="Path to Source 2 CSV/TSV")
    inspect_parser.add_argument("--s3", type=str, required=False, help="Path to Source 3 CSV/TSV")

    # Command: train
    train_parser = subparsers.add_parser("train", help="Train LGBM matcher discriminator on ground truth data")
    train_parser.add_argument("--s1", type=str, required=True, help="Path to Source 1 CSV/TSV")
    train_parser.add_argument("--s2", type=str, required=True, help="Path to Source 2 CSV/TSV")
    train_parser.add_argument("--s3", type=str, required=True, help="Path to Source 3 CSV/TSV")
    train_parser.add_argument("--ground-truth", type=str, required=True, help="Path to ground truth TSV")
    train_parser.add_argument("--model-out", type=str, default=None, help="Output path for trained model pkl")

    # Command: evaluate
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate predictions against ground truth labels")
    eval_parser.add_argument("--ground-truth", type=str, required=True, help="Path to ground truth TSV file")
    eval_parser.add_argument("--predictions", type=str, required=True, help="Path to matching_results.tsv")

    # Command: query
    query_parser = subparsers.add_parser("query", help="Query entity resolution results for specific S1 ID")
    query_parser.add_argument("--s1", type=str, required=True, help="Path to Source 1 CSV/TSV or dataset directory")
    query_parser.add_argument("--s2", type=str, required=False, help="Path to Source 2 CSV/TSV")
    query_parser.add_argument("--s3", type=str, required=False, help="Path to Source 3 CSV/TSV")
    query_parser.add_argument("--s1-id", type=str, required=True, help="Reference entity ID (e.g. S1-00001)")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    agent = BusinessEntityResolutionAgent()

    def target_paths_or_error() -> tuple[str, str]:
        if args.s2 and args.s3:
            return args.s2, args.s3
        if Path(args.s1).is_dir():
            raise ValueError("Dataset directory must contain Source 1, Source 2, and Source 3 files.")
        raise ValueError("Both --s2 and --s3 are required; resolution must use all three sources.")

    if args.command == "resolve":
        s2, s3 = target_paths_or_error()
        agent.ingest_resources(s1=args.s1, s2=s2, s3=s3, ground_truth=args.ground_truth)
        summary = agent.resolve(output_dir=args.output_dir, mode=args.mode)
        if agent.ground_truth_data:
            eval_res = agent.evaluate()
            summary["eval_report"] = eval_res
        if getattr(args, "visualize", False):
            img_path = agent.visualize_graph()
            summary["graph_visualization_image"] = str(img_path)
        print(json.dumps(summary, indent=2, default=str))

    elif args.command == "visualize":
        s2, s3 = target_paths_or_error()
        agent.ingest_resources(s1=args.s1, s2=s2, s3=s3)
        agent.resolve()
        img_path = agent.visualize_graph(output_path=args.img_out)
        print(json.dumps({"status": "visualized", "image_path": str(img_path)}, indent=2))

    elif args.command == "inspect":
        s2, s3 = target_paths_or_error()
        agent.ingest_resources(s1=args.s1, s2=s2, s3=s3)
        report = agent.inspect_resources()
        print(json.dumps(report, indent=2, default=str))

    elif args.command == "train":
        agent.ingest_resources(s1=args.s1, s2=args.s2, s3=args.s3, ground_truth=args.ground_truth)
        res = agent.train_matcher(output_model_path=args.model_out)
        print(json.dumps(res, indent=2, default=str))

    elif args.command == "evaluate":
        res = agent.evaluate(ground_truth=args.ground_truth, matching_results_path=args.predictions)
        print(json.dumps(res, indent=2, default=str))

    elif args.command == "query":
        s2, s3 = target_paths_or_error()
        agent.ingest_resources(s1=args.s1, s2=s2, s3=s3)
        agent.resolve()
        res = agent.query_entity(s1_id=args.s1_id)
        print(json.dumps(res, indent=2, default=str))


if __name__ == "__main__":
    main()


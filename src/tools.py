"""
Agent Tools Module for Business Entity Resolution.
Exposes clean, deterministic Python functions formatted as agent tools for LLMs,
Google Antigravity SDK agents, or custom autonomous pipelines.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

try:
    from .agent import BusinessEntityResolutionAgent
except (ImportError, ValueError):
    from src.agent import BusinessEntityResolutionAgent


def inspect_data_resources(
    s1_path: str,
    s2_path: Optional[str] = None,
    s3_path: Optional[str] = None
) -> str:
    """
    Inspect raw business entity dataset resources (S1 reference, S2 & S3 targets).
    Analyzes schema, record counts, null rates, country distribution, and data health scores.
    """
    agent = BusinessEntityResolutionAgent()
    s2 = s2_path or s1_path
    s3 = s3_path or s1_path
    agent.ingest_resources(s1=s1_path, s2=s2, s3=s3)
    report = agent.inspect_resources()
    return json.dumps(report, indent=2, default=str)


def train_entity_matcher(
    s1_path: str,
    s2_path: str,
    s3_path: str,
    ground_truth_path: str,
    output_model_path: Optional[str] = None
) -> str:
    """
    Train LGBM matcher discriminator model on ground truth dataset.
    Optimizes for Macro-F0.5 precision priority and saves model artifact.
    """
    agent = BusinessEntityResolutionAgent()
    agent.ingest_resources(s1=s1_path, s2=s2_path, s3=s3_path, ground_truth=ground_truth_path)
    model_p = Path(output_model_path) if output_model_path else None
    result = agent.train_matcher(output_model_path=model_p)
    return json.dumps(result, indent=2, default=str)


def run_entity_resolution(
    s1_path: str,
    s2_path: Optional[str] = None,
    s3_path: Optional[str] = None,
    ground_truth_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    mode: str = "auto"
) -> str:
    """
    Execute end-to-end tripartite entity resolution pipeline.
    Serializes candidate_pairs.tsv and matching_results.tsv to output_dir.
    """
    if not s2_path or not s3_path:
        raise ValueError("Both s2_path and s3_path are required; resolution must use all three sources.")
    agent = BusinessEntityResolutionAgent()
    agent.ingest_resources(s1=s1_path, s2=s2_path, s3=s3_path, ground_truth=ground_truth_path)
    summary = agent.resolve(output_dir=output_dir, mode=mode)
    if agent.ground_truth_data:
        eval_report = agent.evaluate()
        summary["eval_report"] = eval_report
    return json.dumps(summary, indent=2, default=str)


def evaluate_entity_resolution(
    ground_truth_path: str,
    predictions_path: str
) -> str:
    """
    Evaluate predicted matching_results.tsv against ground_truth.tsv.
    Calculates Macro-F0.5 score, Precision, Recall, and singletons metric.
    """
    agent = BusinessEntityResolutionAgent()
    eval_report = agent.evaluate(ground_truth=ground_truth_path, matching_results_path=predictions_path)
    return json.dumps(eval_report, indent=2, default=str)


def query_entity_resolution(
    s1_path: str,
    s2_path: str,
    s3_path: str,
    entity_id: str
) -> str:
    """
    Query real-time resolution details for a specific Source 1 entity ID.
    Returns original record fields, candidate target list, and assigned matches.
    """
    agent = BusinessEntityResolutionAgent()
    agent.ingest_resources(s1=s1_path, s2=s2_path, s3=s3_path)
    agent.resolve()
    query_result = agent.query_entity(s1_id=entity_id)
    return json.dumps(query_result, indent=2, default=str)

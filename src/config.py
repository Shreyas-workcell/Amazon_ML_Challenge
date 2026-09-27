"""
Configuration module for the Business Entity Resolution pipeline.
Defines system paths, hyperparameters, constants, and deterministic seeds.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def seed_everything(seed: int = 42) -> None:
    """Set random seed across all libraries for deterministic execution."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


@dataclass
class PathConfig:
    """System filesystem paths."""
    base_dir: Path = Path(__file__).resolve().parent.parent
    data_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "data")
    models_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "models")
    output_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "output")
    cache_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "cache")

    candidate_pairs_file: Path = field(init=False)
    matching_results_file: Path = field(init=False)
    lgb_model_file: Path = field(init=False)
    threshold_config_file: Path = field(init=False)

    def __post_init__(self) -> None:
        self.candidate_pairs_file = self.output_dir / "candidate_pairs.tsv"
        self.matching_results_file = self.output_dir / "matching_results.tsv"
        self.lgb_model_file = self.models_dir / "matcher_lgb.pkl"
        self.threshold_config_file = self.models_dir / "threshold_config.json"
        
        # Ensure directories exist
        for directory in [self.data_dir, self.models_dir, self.output_dir, self.cache_dir]:
            directory.mkdir(parents=True, exist_ok=True)


@dataclass
class ColumnMapping:
    """Schema column mapping with robust fallbacks for diverse source formats."""
    id_col: str = "entity_id"
    name_col: str = "name"
    address_col: str = "address"
    city_col: str = "city"
    state_col: str = "state"
    postal_code_col: str = "postal_code"
    country_col: str = "country"
    phone_col: str = "phone"
    website_col: str = "website"

    # Known column aliases for flexible ingestion
    aliases: Dict[str, List[str]] = field(default_factory=lambda: {
        "entity_id": ["id", "business_id", "source_id", "record_id"],
        "name": ["company_name", "business_name", "legal_name", "title"],
        "address": ["street", "street_address", "address_line_1", "addr", "business_address"],
        "city": ["locality", "town"],
        "state": ["province", "region"],
        "postal_code": ["zip", "zip_code", "postcode", "pincode"],
        "country": ["country_code", "nation", "geo_country"],
        "phone": ["telephone", "tel", "contact_no"],
        "website": ["url", "domain", "web_url"],
    })


@dataclass
class BlockingConfig:
    """Hyperparameters for country-partitioned lexical and dense candidate retrieval."""
    lexical_top_k: int = 60
    dense_top_k: int = 40
    max_candidates_per_entity: int = 100
    dense_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    dense_dim: int = 384
    tfidf_ngram_range: Tuple[int, int] = (2, 4)
    tfidf_max_features: int = 50_000
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    unknown_country_tag: str = "UNKNOWN"
    use_dense: bool = True
    batch_size: int = 64


@dataclass
class TrainConfig:
    """Hyperparameters for discriminator training and Macro-F0.5 optimization."""
    seed: int = 42
    n_splits: int = 5
    hard_negatives_ratio: float = 4.0
    lgb_params: Dict[str, Any] = field(default_factory=lambda: {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "n_estimators": 400,
        "num_leaves": 31,
        "max_depth": 6,
        "min_child_samples": 20,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "scale_pos_weight": 1.5,
        "random_state": 42,
        "n_jobs": 1,
        "verbose": -1,
    })
    # Target metric beta: F_0.5 gives precision 2x priority over recall
    beta: float = 0.5
    default_threshold: float = 0.72
    threshold_search_start: float = 0.30
    threshold_search_end: float = 0.95
    threshold_search_step: float = 0.01
    singleton_margin: float = 0.05


@dataclass
class ResolverConfig:
    """Graph resolver settings for tripartite consistency and greedy assignment."""
    edge_confidence_threshold: float = 0.70
    tripartite_cycle_weight: float = 0.25
    enforce_strict_subset: bool = True
    allow_multi_match: bool = True


@dataclass
class PipelineConfig:
    """Master configuration object."""
    paths: PathConfig = field(default_factory=PathConfig)
    columns: ColumnMapping = field(default_factory=ColumnMapping)
    blocking: BlockingConfig = field(default_factory=BlockingConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    resolver: ResolverConfig = field(default_factory=ResolverConfig)
    random_seed: int = 42

    def __post_init__(self) -> None:
        seed_everything(self.random_seed)


# Global default configuration instance
DEFAULT_CONFIG = PipelineConfig()

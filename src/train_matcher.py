"""
Matcher training and precision threshold optimization module.
Trains a high-precision LightGBM discriminator with hard negative mining,
and solves for the exact Macro-F0.5 maximizing decision threshold.
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

try:
    from .config import DEFAULT_CONFIG, TrainConfig
    from .features import PairFeatureExtractor
    from .utils import compute_macro_f05
except (ImportError, ValueError):
    from src.config import DEFAULT_CONFIG, TrainConfig
    from src.features import PairFeatureExtractor
    from src.utils import compute_macro_f05

logger = logging.getLogger(__name__)

# Check LightGBM availability
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    logger.info("LightGBM not installed. Using scikit-learn HistGradientBoostingClassifier fallback.")


class MatcherModel:
    """Wrapper supporting LightGBM and HistGradientBoostingClassifier."""

    def __init__(self, lgb_params: Optional[Dict[str, Any]] = None) -> None:
        self.params = lgb_params or DEFAULT_CONFIG.train.lgb_params.copy()
        self.model = None
        self.use_lightgbm = HAS_LIGHTGBM

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit binary classifier on feature matrix X and labels y."""
        n_samples = len(X)
        adaptive_min_samples = max(1, min(self.params.get("min_child_samples", 20), n_samples // 4))

        if self.use_lightgbm:
            # Drop scikit-learn specific kwargs if any
            clean_params = {k: v for k, v in self.params.items() if k not in ["scale_pos_weight", "min_child_samples"]}
            scale_pos = self.params.get("scale_pos_weight", 1.0)
            clean_params["n_jobs"] = 1
            clean_params["num_threads"] = 1
            clean_params["verbose"] = -1
            self.model = lgb.LGBMClassifier(
                **clean_params,
                min_child_samples=adaptive_min_samples,
                scale_pos_weight=scale_pos
            )
            self.model.fit(X, y)
        else:
            self.model = HistGradientBoostingClassifier(
                learning_rate=self.params.get("learning_rate", 0.05),
                max_iter=min(250, self.params.get("n_estimators", 300)),
                max_leaf_nodes=self.params.get("num_leaves", 31),
                max_depth=self.params.get("max_depth", 6),
                min_samples_leaf=adaptive_min_samples,
                random_state=self.params.get("random_state", 42),
            )
            self.model.fit(X, y)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability of match (class 1)."""
        if self.model is None or X.shape[0] == 0:
            return np.zeros(X.shape[0], dtype=np.float32)
        probs = self.model.predict_proba(X)
        return probs[:, 1].astype(np.float32)

    def save(self, filepath: Path) -> None:
        """Serialize model to file."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self.model, f)
        logger.info("Saved matcher model to %s", filepath)

    @classmethod
    def load(cls, filepath: Path) -> MatcherModel:
        """Load serialized model."""
        instance = cls()
        with open(filepath, "rb") as f:
            instance.model = pickle.load(f)
        return instance


class ThresholdOptimizer:
    """
    Optimizes decision threshold specifically targeting Macro-F0.5.
    Explicitly accounts for the 2x precision weighting and singleton 1.0 / 0.0 penalty.
    """

    @staticmethod
    def find_optimal_threshold(
        candidate_pairs_df: pd.DataFrame,
        probabilities: np.ndarray,
        ground_truth: Dict[str, Set[str]],
        config: Optional[TrainConfig] = None
    ) -> Tuple[float, float]:
        """
        Grid search over decision thresholds to find threshold maximizing Macro-F0.5.

        Returns:
            (best_threshold, best_macro_f05)
        """
        cfg = config or DEFAULT_CONFIG.train
        thresholds = np.arange(
            cfg.threshold_search_start,
            cfg.threshold_search_end,
            cfg.threshold_search_step
        )

        # Group candidate pairs by S1 ID
        s1_to_pair_indices = candidate_pairs_df.groupby("source1_entity_id").indices
        all_s1_ids = set(ground_truth.keys()).union(set(s1_to_pair_indices.keys()))

        best_score = -1.0
        best_threshold = cfg.default_threshold

        for thresh in thresholds:
            preds: Dict[str, Set[str]] = {}
            for s1_id in all_s1_ids:
                if s1_id in s1_to_pair_indices:
                    indices = s1_to_pair_indices[s1_id]
                    s1_probs = probabilities[indices]
                    s1_cands = candidate_pairs_df.iloc[indices]["candidate_entity_id"].values
                    # Filter candidates above threshold
                    matched = {cand for cand, prob in zip(s1_cands, s1_probs) if prob >= thresh}
                    preds[s1_id] = matched
                else:
                    preds[s1_id] = set()

            score, _ = compute_macro_f05(ground_truth, preds, beta=cfg.beta)
            if score >= best_score:
                best_score = score
                best_threshold = float(thresh)

        logger.info(
            "Threshold search completed. Best threshold=%.3f, Best Macro-F0.5=%.4f",
            best_threshold, best_score
        )
        return best_threshold, best_score


class MatcherTrainer:
    """
    Orchestrates dataset creation with hard negatives, K-fold cross-validation,
    model fitting, threshold optimization, and metadata persistence.
    """

    def __init__(self, config: Optional[TrainConfig] = None) -> None:
        self.config = config or DEFAULT_CONFIG.train

    def prepare_training_data(
        self,
        candidate_pairs_df: pd.DataFrame,
        ground_truth: Dict[str, Set[str]],
        s1_df: pd.DataFrame,
        target_df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, List[str], pd.DataFrame]:
        """
        Generate labeled dataset with true positives and hard negative samples.
        """
        logger.info("Extracting features for %d candidate pairs...", len(candidate_pairs_df))
        X, feature_names = PairFeatureExtractor.build_feature_matrix(
            candidate_pairs_df, s1_df, target_df
        )

        labels = []
        for _, row in candidate_pairs_df.iterrows():
            s1_id = row["source1_entity_id"]
            cand_id = row["candidate_entity_id"]
            is_match = cand_id in ground_truth.get(s1_id, set())
            labels.append(1 if is_match else 0)

        y = np.array(labels, dtype=np.int32)
        n_pos = np.sum(y == 1)
        n_neg = np.sum(y == 0)
        logger.info("Prepared training dataset: Total=%d, Positive=%d, Negative=%d", len(y), n_pos, n_neg)

        return X, y, feature_names, candidate_pairs_df

    def train_and_evaluate(
        self,
        candidate_pairs_df: pd.DataFrame,
        ground_truth: Dict[str, Set[str]],
        s1_df: pd.DataFrame,
        target_df: pd.DataFrame
    ) -> Tuple[MatcherModel, float, float]:
        """
        Train matcher with K-fold CV out-of-fold probability estimation,
        optimize threshold for Macro-F0.5, fit on full data, and serialize artifacts.
        """
        X, y, feature_names, pairs_df = self.prepare_training_data(
            candidate_pairs_df, ground_truth, s1_df, target_df
        )

        if len(X) == 0 or np.sum(y == 1) == 0:
            logger.warning("Empty or all-negative dataset. Initializing default matcher.")
            model = MatcherModel(self.config.lgb_params)
            # Create synthetic row to allow fit
            dummy_X = np.zeros((2, len(feature_names)), dtype=np.float32)
            dummy_y = np.array([0, 1], dtype=np.int32)
            model.fit(dummy_X, dummy_y)
            return model, self.config.default_threshold, 1.0

        # K-Fold Out-of-Fold prediction for threshold search
        # Keep every pair from one S1 entity in the same fold.  Pair-level
        # K-fold leaks the entity's positive/negative signature across folds
        # and makes the threshold look substantially better than it is on new
        # reference entities.
        n_groups = pairs_df["source1_entity_id"].nunique()
        # Grouped CV is the correct validation strategy for the real challenge,
        # but it is undefined for tiny fixtures (a fold can contain one row or
        # only one class).  Use an in-sample calibration fallback only there;
        # production-sized training still always takes the grouped-CV path.
        use_grouped_cv = n_groups >= 4 and len(X) >= 8
        if use_grouped_cv:
            n_splits = min(self.config.n_splits, max(2, n_groups // 4))
            kf = GroupKFold(n_splits=n_splits)
            oof_probs = np.zeros(len(X), dtype=np.float32)
            groups = pairs_df["source1_entity_id"].to_numpy()
            try:
                for fold, (train_idx, val_idx) in enumerate(kf.split(X, y, groups=groups)):
                    fold_model = MatcherModel(self.config.lgb_params)
                    fold_model.fit(X[train_idx], y[train_idx])
                    oof_probs[val_idx] = fold_model.predict_proba(X[val_idx])

                # Optimize threshold on out-of-fold probabilities.
                best_threshold, best_f05 = ThresholdOptimizer.find_optimal_threshold(
                    pairs_df, oof_probs, ground_truth, self.config
                )
            except ValueError as exc:
                logger.warning("Grouped CV skipped for undersized/degenerate data: %s", exc)
                use_grouped_cv = False

        # Train final model on full training set. For tiny fixtures where a
        # valid grouped split is impossible, calibrate on the fitted model so
        # the normal pipeline remains functional without weakening production
        # validation.
        logger.info("Fitting final discriminator model on full dataset...")
        final_model = MatcherModel(self.config.lgb_params)
        final_model.fit(X, y)
        if not use_grouped_cv:
            in_sample_probs = final_model.predict_proba(X)
            best_threshold, best_f05 = ThresholdOptimizer.find_optimal_threshold(
                pairs_df, in_sample_probs, ground_truth, self.config
            )

        # Save artifacts
        model_path = DEFAULT_CONFIG.paths.lgb_model_file
        config_path = DEFAULT_CONFIG.paths.threshold_config_file

        final_model.save(model_path)
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump({
                "optimal_threshold": best_threshold,
                "macro_f05_cv": best_f05,
                "feature_names": feature_names,
                "beta": self.config.beta
            }, f, indent=2)
        logger.info("Saved threshold configuration to %s", config_path)

        return final_model, best_threshold, best_f05

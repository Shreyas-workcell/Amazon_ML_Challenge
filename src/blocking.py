"""
Country-partitioned hybrid lexical and dense blocking module.
Combines BM25/TF-IDF sparse candidate generation with FAISS dense vector search
partitioned dynamically by country to achieve ultra-high recall under distribution shifts.
"""

from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import logging
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

try:
    from .config import BlockingConfig, DEFAULT_CONFIG
except (ImportError, ValueError):
    from src.config import BlockingConfig, DEFAULT_CONFIG

logger = logging.getLogger(__name__)


class FaissOrSklearnIndex:
    """Wrapper that utilizes FAISS when available, falling back gracefully to scikit-learn."""

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.use_faiss = False
        self.faiss_index = None
        self.sklearn_index = NearestNeighbors(metric="cosine", algorithm="brute")
        self.vectors: Optional[np.ndarray] = None

        try:
            import faiss
            self.faiss_index = faiss.IndexFlatIP(dim)
            self.use_faiss = True
        except Exception:
            self.use_faiss = False

    def add(self, vectors: np.ndarray) -> None:
        """Add normalized vectors to the index."""
        # Ensure float32 and L2 normalization
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normalized = np.ascontiguousarray((vectors / norms), dtype=np.float32)

        if self.use_faiss and self.faiss_index is not None:
            self.faiss_index.add(normalized)
        else:
            self.vectors = normalized
            self.sklearn_index.fit(normalized)

    def search(self, query_vectors: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Search top-k nearest neighbors. Returns (distances, indices)."""
        norms = np.linalg.norm(query_vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normalized = np.ascontiguousarray((query_vectors / norms), dtype=np.float32)

        if self.use_faiss and self.faiss_index is not None:
            if self.faiss_index.ntotal == 0:
                n_queries = len(query_vectors)
                return np.zeros((n_queries, 0), dtype=np.float32), np.full((n_queries, 0), -1, dtype=np.int64)
            actual_k = min(top_k, self.faiss_index.ntotal)
            distances, indices = self.faiss_index.search(normalized, actual_k)
            return distances, indices
        else:
            actual_k = min(top_k, len(self.vectors)) if self.vectors is not None else 0
            if actual_k == 0:
                n_queries = len(query_vectors)
                return np.zeros((n_queries, 0)), np.full((n_queries, 0), -1)
            dists, indices = self.sklearn_index.kneighbors(normalized, n_neighbors=actual_k)
            # Convert cosine distance to cosine similarity
            sims = 1.0 - dists
            return sims, indices


class CountryPartitionedBlocker:
    """
    Partitions the search space dynamically by country (open-set, including France,
    US, India, etc.) and performs hybrid lexical + dense candidate retrieval.
    """

    def __init__(self, config: Optional[BlockingConfig] = None) -> None:
        self.config = config or DEFAULT_CONFIG.blocking
        self.dense_encoder = None
        self._init_dense_encoder()

    def _init_dense_encoder(self) -> None:
        """Initialize SentenceTransformer if available, or set to None for TF-IDF SVD fallback."""
        if not self.config.use_dense:
            return
        try:
            from sentence_transformers import SentenceTransformer
            self.dense_encoder = SentenceTransformer(self.config.dense_model_name)
            logger.info("Loaded SentenceTransformer model: %s", self.config.dense_model_name)
        except Exception as e:
            logger.info("SentenceTransformer not loaded (%s). Using SVD-projected dense fallback.", e)
            self.dense_encoder = None

    def _get_dense_embeddings(self, texts: List[str]) -> np.ndarray:
        """Compute dense embeddings via transformer or sublinear TF-IDF projection."""
        if not texts:
            return np.zeros((0, self.config.dense_dim), dtype=np.float32)

        if self.dense_encoder is not None:
            try:
                embeddings = self.dense_encoder.encode(
                    texts,
                    batch_size=self.config.batch_size,
                    show_progress_bar=False,
                    normalize_embeddings=True
                )
                return np.asarray(embeddings, dtype=np.float32)
            except Exception as e:
                logger.warning("Dense encoder failed (%s). Falling back to sparse projection.", e)

        # Truncated SVD / TF-IDF dense projection fallback
        from sklearn.decomposition import TruncatedSVD
        vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 4),
            max_features=5000,
            sublinear_tf=True
        )
        tfidf = vectorizer.fit_transform(texts)
        n_samples, n_features = tfidf.shape
        if n_samples <= 2 or n_features <= 2:
            dense_arr = tfidf.toarray()
            if dense_arr.shape[1] < self.config.dense_dim:
                dense_arr = np.pad(dense_arr, ((0, 0), (0, self.config.dense_dim - dense_arr.shape[1])))
            return dense_arr[:, :self.config.dense_dim].astype(np.float32)

        n_components = min(self.config.dense_dim, n_features - 1, n_samples - 1)
        svd = TruncatedSVD(n_components=max(1, n_components), random_state=42)
        projected = svd.fit_transform(tfidf)
        # Pad with zeros if n_components < dense_dim
        if projected.shape[1] < self.config.dense_dim:
            pad_width = self.config.dense_dim - projected.shape[1]
            projected = np.pad(projected, ((0, 0), (0, pad_width)))
        return projected.astype(np.float32)

    def _lexical_blocking(
        self,
        s1_df: pd.DataFrame,
        target_df: pd.DataFrame
    ) -> Dict[str, Set[str]]:
        """
        Inverted-index and TF-IDF lexical blocking within a country partition.
        """
        candidates: Dict[str, Set[str]] = defaultdict(set)
        if s1_df.empty or target_df.empty:
            return candidates

        target_ids = target_df["entity_id"].tolist()
        s1_ids = s1_df["entity_id"].tolist()

        # 1. Postal code inverted index (fast high-precision matches)
        postal_index = defaultdict(list)
        for t_idx, post_code in enumerate(target_df["postal_code"]):
            if post_code and len(post_code) >= 3:
                postal_index[post_code].append(target_ids[t_idx])
                # Prefix 3-char index for regional locality
                if len(post_code) > 3:
                    postal_index[post_code[:3]].append(target_ids[t_idx])

        for s_idx, post_code in enumerate(s1_df["postal_code"]):
            s1_id = s1_ids[s_idx]
            if post_code and post_code in postal_index:
                # Add up to 15 postal neighbors
                for matched_t_id in postal_index[post_code][:15]:
                    candidates[s1_id].add(matched_t_id)

        # 1b. Exact normalized phone index. Phone agreement is a strong,
        # inexpensive signal and should reach scoring even when names are noisy.
        phone_index = defaultdict(list)
        for t_idx, phone_digits in enumerate(target_df["phone_digits"]):
            if phone_digits and len(phone_digits) >= 7:
                phone_index[phone_digits].append(target_ids[t_idx])

        for s_idx, phone_digits in enumerate(s1_df["phone_digits"]):
            s1_id = s1_ids[s_idx]
            if phone_digits and len(phone_digits) >= 7:
                for matched_t_id in phone_index.get(phone_digits, []):
                    candidates[s1_id].add(matched_t_id)

        # 2. Token-level inverted index on root names
        token_index = defaultdict(list)
        for t_idx, root_name in enumerate(target_df["root_name"]):
            tokens = set(root_name.split())
            for tok in tokens:
                if len(tok) >= 3:
                    token_index[tok].append(target_ids[t_idx])

        for s_idx, root_name in enumerate(s1_df["root_name"]):
            s1_id = s1_ids[s_idx]
            tokens = set(root_name.split())
            shared_counts = defaultdict(int)
            for tok in tokens:
                if len(tok) >= 3 and tok in token_index:
                    # Count every posting before ranking. Truncating a posting list
                    # by input order can discard a true match arbitrarily.
                    for t_id in token_index[tok]:
                        shared_counts[t_id] += 1
            # Add targets sharing >= 1 significant tokens
            sorted_by_overlap = sorted(shared_counts.items(), key=lambda x: x[1], reverse=True)
            for t_id, _ in sorted_by_overlap[:self.config.lexical_top_k]:
                candidates[s1_id].add(t_id)

        # 3. Character n-gram TF-IDF retrieval (catches typos and word segmentations)
        try:
            tfidf = TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=self.config.tfidf_ngram_range,
                max_features=self.config.tfidf_max_features,
                sublinear_tf=True
            )
            target_corpus = target_df["composite_text"].tolist()
            s1_corpus = s1_df["composite_text"].tolist()

            target_vecs = tfidf.fit_transform(target_corpus)
            s1_vecs = tfidf.transform(s1_corpus)

            # Cosine similarity via sparse dot product
            # Chunking query vectors for memory efficiency
            chunk_size = 500
            for start_idx in range(0, s1_vecs.shape[0], chunk_size):
                end_idx = min(start_idx + chunk_size, s1_vecs.shape[0])
                sim_matrix = (s1_vecs[start_idx:end_idx] @ target_vecs.T).toarray()

                for row_idx, global_s_idx in enumerate(range(start_idx, end_idx)):
                    s1_id = s1_ids[global_s_idx]
                    scores = sim_matrix[row_idx]
                    # Select top lexical indices
                    top_indices = np.argpartition(scores, -min(len(scores), self.config.lexical_top_k))[-self.config.lexical_top_k:]
                    top_sorted = top_indices[np.argsort(-scores[top_indices])]
                    for t_idx in top_sorted:
                        if scores[t_idx] > 0.15:  # Lexical threshold
                            candidates[s1_id].add(target_ids[t_idx])
        except Exception as e:
            logger.warning("Lexical TF-IDF failed in partition (%s)", e)

        return candidates

    def _dense_blocking(
        self,
        s1_df: pd.DataFrame,
        target_df: pd.DataFrame
    ) -> Dict[str, Set[str]]:
        """
        Dense vector nearest-neighbor blocking with FAISS / scikit-learn.
        """
        candidates: Dict[str, Set[str]] = defaultdict(set)
        if s1_df.empty or target_df.empty:
            return candidates

        # The optional transformer is not available in the challenge runtime,
        # so the fallback below would build a dense TF-IDF/SVD matrix.  That is
        # quadratic-ish in memory/time and becomes impractical for the full
        # 200k-record training target pool.  Lexical blocking already supplies
        # postal, phone, token, and character-ngram candidates; skip only the
        # dense fallback for oversized partitions.
        if self.dense_encoder is None and len(target_df) > 50_000:
            logger.info(
                "Skipping dense SVD fallback for oversized partition (%d targets)",
                len(target_df),
            )
            return candidates

        target_ids = target_df["entity_id"].tolist()
        s1_ids = s1_df["entity_id"].tolist()

        try:
            target_embeddings = self._get_dense_embeddings(target_df["composite_text"].tolist())
            s1_embeddings = self._get_dense_embeddings(s1_df["composite_text"].tolist())

            dim = target_embeddings.shape[1]
            index = FaissOrSklearnIndex(dim=dim)
            index.add(target_embeddings)

            top_k = min(self.config.dense_top_k, len(target_df))
            distances, indices = index.search(s1_embeddings, top_k=top_k)

            for s_idx, t_indices in enumerate(indices):
                s1_id = s1_ids[s_idx]
                for pos, t_idx in enumerate(t_indices):
                    if t_idx >= 0 and t_idx < len(target_ids):
                        sim = distances[s_idx][pos]
                        if sim > 0.30:  # Dense similarity floor
                            candidates[s1_id].add(target_ids[t_idx])
        except Exception as e:
            logger.warning("Dense vector blocking failed (%s)", e)

        return candidates

    @staticmethod
    def _blocking_evidence_score(source: Dict[str, object], target: Dict[str, object]) -> float:
        """Score cheap blocking evidence before expensive fuzzy features.

        This is deliberately not the match score.  It only ranks the small
        candidate sieve and is deterministic, explainable, and independent of
        the trained discriminator.
        """
        score = 0.0

        source_phone = str(source.get("phone_digits", "") or "")
        target_phone = str(target.get("phone_digits", "") or "")
        if source_phone and target_phone and source_phone == target_phone:
            score += 100.0

        source_postal = str(source.get("postal_code", "") or "")
        target_postal = str(target.get("postal_code", "") or "")
        if source_postal and target_postal:
            if source_postal == target_postal:
                score += 80.0
            elif len(source_postal) >= 3 and source_postal[:3] == target_postal[:3]:
                score += 15.0

        source_root = str(source.get("root_name", "") or "")
        target_root = str(target.get("root_name", "") or "")
        source_name_tokens = set(source_root.split())
        target_name_tokens = set(target_root.split())
        shared_name = source_name_tokens.intersection(target_name_tokens)
        if source_root and source_root == target_root:
            score += 70.0
        score += min(30.0, 10.0 * len(shared_name))

        source_numbers = set(source.get("street_numbers", []) or [])
        target_numbers = set(target.get("street_numbers", []) or [])
        if source_numbers and target_numbers:
            score += 25.0 if source_numbers.intersection(target_numbers) else -20.0

        source_address = set(str(source.get("clean_address", "") or "").split())
        target_address = set(str(target.get("clean_address", "") or "").split())
        source_address = {t for t in source_address if len(t) >= 3}
        target_address = {t for t in target_address if len(t) >= 3}
        if source_address and target_address:
            union = source_address.union(target_address)
            score += 35.0 * len(source_address.intersection(target_address)) / len(union)

        if str(source.get("city", "") or "") and source.get("city") == target.get("city"):
            score += 8.0
        if str(source.get("state", "") or "") and source.get("state") == target.get("state"):
            score += 4.0

        # Every candidate already passed at least one blocking route.  This
        # small floor keeps a dense-only candidate eligible for fuzzy scoring.
        return score + 1.0

    def _candidate_sieve(
        self,
        s1_df: pd.DataFrame,
        target_df: pd.DataFrame,
        all_candidates: Dict[str, Set[str]],
    ) -> Tuple[Dict[str, List[str]], List[Dict[str, str]]]:
        """Rank and cap candidates using blocking evidence, never row order."""
        s1_lookup = s1_df.set_index("entity_id").to_dict("index")
        target_lookup = target_df.set_index("entity_id").to_dict("index")
        final_candidate_dict: Dict[str, List[str]] = {}
        pairs_list: List[Dict[str, str]] = []

        for s1_id, cand_set in all_candidates.items():
            source = s1_lookup.get(s1_id, {})
            ranked = []
            for cand_id in cand_set:
                target = target_lookup.get(cand_id)
                if target is None:
                    continue
                ranked.append((self._blocking_evidence_score(source, target), cand_id))

            ranked.sort(key=lambda item: (-item[0], item[1]))
            selected = [cand_id for _, cand_id in ranked[:self.config.max_candidates_per_entity]]
            final_candidate_dict[s1_id] = selected
            pairs_list.extend(
                {"source1_entity_id": s1_id, "candidate_entity_id": cand_id}
                for cand_id in selected
            )

        return final_candidate_dict, pairs_list

    def generate_candidates(
        self,
        s1_df: pd.DataFrame,
        s2_df: pd.DataFrame,
        s3_df: pd.DataFrame
    ) -> Tuple[Dict[str, List[str]], pd.DataFrame]:
        """
        Execute country-partitioned candidate retrieval across S1 against target sources S2 and S3.

        Returns:
            - candidate_dict: Mapping s1_id -> list of candidate target entity IDs
            - candidate_pairs_df: DataFrame with ['source1_entity_id', 'candidate_entity_id']
        """
        # Combine target sources into unified target repository
        target_df = pd.concat([s2_df, s3_df], ignore_index=True).drop_duplicates(subset=["entity_id"])

        s1_countries = s1_df["country"].unique()
        target_countries = target_df["country"].unique()
        all_countries = set(s1_countries).union(set(target_countries))

        logger.info(
            "Executing country-partitioned blocking across %d countries: %s",
            len(all_countries), list(all_countries)
        )

        all_candidates: Dict[str, Set[str]] = {s1_id: set() for s1_id in s1_df["entity_id"]}

        # Process each country partition dynamically (Country-Agnostic)
        for country in all_countries:
            s1_partition = s1_df[s1_df["country"] == country]
            target_partition = target_df[target_df["country"] == country]

            if s1_partition.empty:
                continue

            # Fallback for country mismatch: if target partition has very few records,
            # allow lookup in the UNKNOWN bucket or cross-country target pool
            if len(target_partition) < 10 and country != self.config.unknown_country_tag:
                unknown_targets = target_df[target_df["country"] == self.config.unknown_country_tag]
                target_partition = pd.concat([target_partition, unknown_targets], ignore_index=True)

            if target_partition.empty:
                # If still empty, search across entire target pool with lower top_k
                target_partition = target_df

            logger.info(
                "Partition [%s]: S1 records=%d, Target records=%d",
                country, len(s1_partition), len(target_partition)
            )

            # 1. Lexical retrieval
            lex_cands = self._lexical_blocking(s1_partition, target_partition)

            # 2. Dense retrieval
            dense_cands = self._dense_blocking(s1_partition, target_partition)

            # 3. Union candidates per S1 entity
            for s1_id in s1_partition["entity_id"]:
                combined = lex_cands[s1_id].union(dense_cands[s1_id])
                all_candidates[s1_id].update(combined)

        # Candidate sieve: rank by cheap key evidence before fuzzy features.
        final_candidate_dict, pairs_list = self._candidate_sieve(
            s1_df, target_df, all_candidates
        )

        candidate_pairs_df = pd.DataFrame(pairs_list)
        if candidate_pairs_df.empty:
            candidate_pairs_df = pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id"])

        logger.info(
            "Candidate generation complete. Total S1 entities=%d, Total candidate pairs=%d, Avg candidates/S1=%.2f",
            len(s1_df), len(candidate_pairs_df),
            len(candidate_pairs_df) / max(1, len(s1_df))
        )

        return final_candidate_dict, candidate_pairs_df

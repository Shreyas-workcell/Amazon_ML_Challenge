"""
C++ Core Engine Bridge for Business Entity Resolution.
Binds Python ctypes to libcpp_engine shared library for ultra-fast text normalization,
string distance calculation (Levenshtein, Jaro-Winkler), legal suffix parsing,
direct-data file ingestion, multi-threaded country-partitioned inverted indexing,
and high-precision ML rule-based candidate & matching generation.
Provides zero-crash native Python fallback if binary library is unavailable.
"""

from __future__ import annotations

import ctypes
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from src.utils import setup_logger

logger = setup_logger("cpp_bridge")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
CPP_SRC_DIR = PROJECT_ROOT / "cpp_src"

# Determine library extension per platform
if platform.system() == "Darwin":
    LIB_NAME = "libcpp_engine.dylib"
elif platform.system() == "Windows":
    LIB_NAME = "cpp_engine.dll"
else:
    LIB_NAME = "libcpp_engine.so"

LIB_PATH = MODELS_DIR / LIB_NAME


def compile_cpp_engine() -> bool:
    """Attempt to compile C++ core engine using clang++ or g++ with multithreading support."""
    cpp_file = CPP_SRC_DIR / "cpp_engine.cpp"
    if not cpp_file.exists():
        logger.warning("C++ source file not found: %s", cpp_file)
        return False

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    compilers = ["clang++", "g++", "c++"]

    for compiler in compilers:
        cmd = [
            compiler, "-O3", "-std=c++17", "-pthread", "-shared", "-fPIC",
            str(cpp_file), "-o", str(LIB_PATH)
        ]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if res.returncode == 0 and LIB_PATH.exists():
                logger.info("Successfully compiled C++ engine using %s -> %s", compiler, LIB_PATH)
                return True
            else:
                logger.debug("Compiler %s error: %s", compiler, res.stderr)
        except Exception as e:
            logger.debug("Compilation exception with %s: %s", compiler, e)
            continue

    logger.warning("C++ compilation failed. Using Python native fallbacks.")
    return False


class CPPEngine:
    """Python ctypes interface to C++ shared library core."""

    _lib: Optional[ctypes.CDLL] = None
    _is_available: bool = False

    @classmethod
    def initialize(cls) -> bool:
        """Initialize C++ shared library bindings."""
        if cls._lib is not None:
            return cls._is_available

        if not LIB_PATH.exists():
            compile_cpp_engine()

        if LIB_PATH.exists():
            try:
                cls._lib = ctypes.CDLL(str(LIB_PATH))
                
                # Function signatures
                cls._lib.normalize_text_cpp.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
                cls._lib.normalize_text_cpp.restype = None

                cls._lib.parse_legal_suffix_cpp.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
                cls._lib.parse_legal_suffix_cpp.restype = None

                cls._lib.fast_levenshtein_distance.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
                cls._lib.fast_levenshtein_distance.restype = ctypes.c_int

                cls._lib.fast_jaro_winkler.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
                cls._lib.fast_jaro_winkler.restype = ctypes.c_double

                cls._lib.build_candidate_pairs_cpp.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p]
                cls._lib.build_candidate_pairs_cpp.restype = ctypes.c_int

                # Direct Data High-Performance Pipeline Signatures
                cls._lib.run_entity_resolution_pipeline_cpp.argtypes = [
                    ctypes.c_char_p,
                    ctypes.c_char_p,
                    ctypes.c_char_p,
                    ctypes.c_char_p,
                    ctypes.c_char_p,
                    ctypes.c_char_p
                ]
                cls._lib.run_entity_resolution_pipeline_cpp.restype = ctypes.c_void_p

                cls._lib.query_entity_resolution_cpp.argtypes = [
                    ctypes.c_char_p,
                    ctypes.c_char_p,
                    ctypes.c_char_p,
                    ctypes.c_char_p,
                    ctypes.c_char_p,
                    ctypes.c_char_p,
                    ctypes.c_double
                ]
                cls._lib.query_entity_resolution_cpp.restype = ctypes.c_void_p

                cls._lib.free_string_cpp.argtypes = [ctypes.c_void_p]
                cls._lib.free_string_cpp.restype = None

                cls._is_available = True
                logger.info("C++ Engine successfully loaded into Python via ctypes.")
                return True
            except Exception as e:
                logger.warning("Failed to load C++ shared library: %s", e)
                cls._is_available = False

        return False

    @classmethod
    def is_available(cls) -> bool:
        return cls.initialize()

    @classmethod
    def run_pipeline(
        cls,
        s1_path: Union[str, Path],
        s2_path: Union[str, Path],
        s3_path: Union[str, Path],
        output_dir: Union[str, Path],
        rules: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Directly execute the end-to-end entity resolution pipeline in C++.
        Data is read directly by C++ from disk, partitioned by country,
        indexed via inverted token hashing, scored via parallel threads,
        and written to output/candidate_pairs.tsv and output/matching_results.tsv.
        """
        if not cls.is_available() or not cls._lib:
            logger.warning("C++ engine not available. Falling back to Python streaming engine.")
            from src.run_test_data import run_full_test_inference_python
            test_dir = Path(s1_path).parent
            c_p, m_p = run_full_test_inference_python(test_dir, Path(output_dir))
            return {
                "status": "completed_python_fallback",
                "candidate_file": str(c_p),
                "matching_file": str(m_p)
            }

        s1_p = str(Path(s1_path).resolve())
        s2_p = str(Path(s2_path).resolve())
        s3_p = str(Path(s3_path).resolve())
        out_d = Path(output_dir).resolve()
        out_d.mkdir(parents=True, exist_ok=True)

        cand_out_p = str(out_d / "candidate_pairs.tsv")
        match_out_p = str(out_d / "matching_results.tsv")

        rules_dict = {
            "match_threshold": 0.80,
            "min_candidate_score": 0.35,
            "street_num_bonus": 0.15,
            "street_num_penalty": -0.35,
            "postal_bonus": 0.15,
            "postal_penalty": -0.35,
            "addr_bonus": 0.15,
            "addr_penalty": -0.35,
            "max_candidates_per_entity": 30,
            "max_matches_per_entity": 12,
            "max_posting_list_size": 1000,
            "num_threads": 8
        }
        if rules:
            rules_dict.update(rules)

        rules_json = json.dumps(rules_dict)

        logger.info(
            "C++ Direct Engine: Processing files [%s, %s, %s] with rules %s",
            Path(s1_p).name, Path(s2_p).name, Path(s3_p).name, rules_json
        )

        res_ptr = cls._lib.run_entity_resolution_pipeline_cpp(
            s1_p.encode("utf-8"),
            s2_p.encode("utf-8"),
            s3_p.encode("utf-8"),
            cand_out_p.encode("utf-8"),
            match_out_p.encode("utf-8"),
            rules_json.encode("utf-8")
        )

        if not res_ptr:
            raise RuntimeError("C++ engine returned null pointer from pipeline execution.")

        raw_bytes = ctypes.string_at(res_ptr)
        raw_json = raw_bytes.decode("utf-8", errors="ignore")
        cls._lib.free_string_cpp(res_ptr)

        parsed_res = json.loads(raw_json)
        logger.info("C++ Engine execution result: %s", parsed_res)
        return parsed_res

    @classmethod
    def query_entity(
        cls,
        s1_id: str,
        name: str,
        address: str,
        country: str,
        s2_path: Union[str, Path],
        s3_path: Union[str, Path],
        match_threshold: float = 0.55
    ) -> Dict[str, Any]:
        """
        Interactive query resolution: query candidates and matches for an entity
        directly using C++ scanning target datasets.
        """
        if not cls.is_available() or not cls._lib:
            return {"status": "error", "message": "C++ engine not available"}

        s2_p = str(Path(s2_path).resolve())
        s3_p = str(Path(s3_path).resolve())

        res_ptr = cls._lib.query_entity_resolution_cpp(
            s1_id.encode("utf-8"),
            name.encode("utf-8"),
            address.encode("utf-8"),
            country.encode("utf-8"),
            s2_p.encode("utf-8"),
            s3_p.encode("utf-8"),
            ctypes.c_double(match_threshold)
        )

        if not res_ptr:
            return {"status": "error", "message": "C++ returned null pointer"}

        raw_bytes = ctypes.string_at(res_ptr)
        raw_json = raw_bytes.decode("utf-8", errors="ignore")
        cls._lib.free_string_cpp(res_ptr)
        return json.loads(raw_json)

    @classmethod
    def normalize_text(cls, text: str) -> str:
        """Normalize text via C++ core engine."""
        if not cls.is_available() or not cls._lib:
            from src.preprocess import TextNormalizer
            return TextNormalizer.clean_text(text)

        buf = ctypes.create_string_buffer(max(1024, len(text) * 2 + 64))
        cls._lib.normalize_text_cpp(text.encode("utf-8"), buf, len(buf))
        return buf.value.decode("utf-8", errors="ignore")

    @classmethod
    def parse_legal_suffix(cls, clean_name: str) -> Tuple[str, str]:
        """Parse root company name and legal entity type via C++ core engine."""
        if not cls.is_available() or not cls._lib:
            from src.preprocess import LegalEntityNormalizer
            return LegalEntityNormalizer().parse(clean_name)

        root_buf = ctypes.create_string_buffer(1024)
        legal_buf = ctypes.create_string_buffer(1024)
        cls._lib.parse_legal_suffix_cpp(clean_name.encode("utf-8"), root_buf, legal_buf, 1024)
        return root_buf.value.decode("utf-8", errors="ignore"), legal_buf.value.decode("utf-8", errors="ignore")

    @classmethod
    def levenshtein(cls, s1: str, s2: str) -> int:
        """Fast Levenshtein distance via C++ core engine."""
        if not cls.is_available() or not cls._lib:
            from rapidfuzz.distance import Levenshtein
            return int(Levenshtein.distance(s1, s2))

        return int(cls._lib.fast_levenshtein_distance(s1.encode("utf-8"), s2.encode("utf-8")))

    @classmethod
    def jaro_winkler(cls, s1: str, s2: str) -> float:
        """Fast Jaro-Winkler similarity via C++ core engine."""
        if not cls.is_available() or not cls._lib:
            from rapidfuzz.distance import JaroWinkler
            return float(JaroWinkler.similarity(s1, s2))

        return float(cls._lib.fast_jaro_winkler(s1.encode("utf-8"), s2.encode("utf-8")))


# Auto-initialize C++ engine on import
CPPEngine.initialize()

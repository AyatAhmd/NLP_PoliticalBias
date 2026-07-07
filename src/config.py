"""
Project configuration for the revised political bias experiment.

This version supports:
- separate prompt files for base and instruction-tuned generation;
- embedding-based scoring with cosine similarity;
- no shared refusal metric for base and instruction-tuned models.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

PROJECT_NAME = "nlp-p9-political-bias"

BASE_MODEL = "Qwen/Qwen2.5-0.5B"
INSTRUCT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_FAMILY = "Qwen2.5-0.5B"

RANDOM_SEED = 42
MAX_NEW_TOKENS = 128
DO_SAMPLE = False
TEMPERATURE = 0.0

PROMPT_CASES_FILENAME = "prompts_200.csv"
GENERATION_PROMPTS_FILENAME = "generation_prompts_400.csv"
PROMPTS_FILENAME = PROMPT_CASES_FILENAME
RAW_OUTPUTS_FILENAME = "raw_outputs.csv"
SCORED_OUTPUTS_FILENAME = "scored_outputs.csv"
NEWS_CENTROIDS_FILENAME = "news_bias_centroids.pkl"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

ModelType = Literal["base", "instruct"]

METRICS = [
    "similarity_left",
    "similarity_center",
    "similarity_right",
    "leaning_score",
    "neutrality_score",
    "bias_strength",
    "hedging_score",
    "hedging_similarity",
    "assertive_similarity",
    "output_validity",
    "response_length",
]


def get_project_root() -> Path:
    """Return the project root from either the repository root or notebooks folder."""
    current = Path.cwd()
    return current.parent if current.name == "notebooks" else current


def get_data_dir(project_root: Path | None = None) -> Path:
    """Return the data directory and create it if needed."""
    root = project_root or get_project_root()
    path = root / "data"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_results_dirs(project_root: Path | None = None) -> tuple[Path, Path, Path]:
    """Return results, tables, and figures directories, creating them if needed."""
    root = project_root or get_project_root()
    results = root / "results"
    tables = results / "tables"
    figures = results / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    return results, tables, figures

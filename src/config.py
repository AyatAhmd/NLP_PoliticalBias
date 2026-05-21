"""
Project configuration for the instruction-tuning political bias experiment.

This module centralizes paths, model identifiers, and shared constants so that
notebooks and scripts use the same settings.
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

PROMPTS_FILENAME = "prompts_200.csv"
RAW_OUTPUTS_FILENAME = "raw_outputs.csv"
SCORED_OUTPUTS_FILENAME = "scored_outputs.csv"

ModelType = Literal["base", "instruct"]

METRICS = [
    "leaning_score",
    "neutrality_score",
    "hedging_score",
    "refusal_score",
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

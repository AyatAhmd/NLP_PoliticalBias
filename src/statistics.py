"""Statistical analysis utilities for the revised P9 political bias project.
Scores are continuous embedding-based variables, so treatment effects are
computed as paired differences:

    effect = score_instruct - score_base

The functions in this module are intentionally small and reusable so the analysis
notebook remains a demonstration layer rather than the place where all logic is
implemented.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:  # scipy is useful, but the project should fail gracefully if it is missing.
    from scipy import stats as scipy_stats
except Exception:  # pragma: no cover
    scipy_stats = None


DEFAULT_OUTCOME_COLUMNS = [
    "leaning_score",
    "neutrality_score",
    "bias_strength",
    "hedging_score",
    "response_length",
    "output_validity",
]

DEFAULT_GROUP_COLUMNS = ["domain", "prompt_type"]

REQUIRED_ANALYSIS_COLUMNS = {
    "prompt_id",
    "domain",
    "topic",
    "prompt_type",
    "model_type",
}


@dataclass(frozen=True)
class AnalysisSettings:
    """Settings used by the treatment-effect analysis."""

    outcomes: tuple[str, ...] = tuple(DEFAULT_OUTCOME_COLUMNS)
    group_columns: tuple[str, ...] = tuple(DEFAULT_GROUP_COLUMNS)
    bootstrap_iterations: int = 2000
    confidence_level: float = 0.95
    random_seed: int = 42


@dataclass(frozen=True)
class EffectResult:
    """Container for one paired treatment-effect estimate."""

    metric: str
    n_pairs: int
    base_mean: float
    instruct_mean: float
    average_treatment_effect: float
    difference_std: float
    standard_error: float
    ci_low: float
    ci_high: float
    paired_t_statistic: float | None
    paired_t_p_value: float | None
    wilcoxon_statistic: float | None
    wilcoxon_p_value: float | None
    cohen_dz: float


def validate_scored_outputs(scored_outputs: pd.DataFrame) -> None:
    """Validate that scored outputs contain the minimum analysis columns."""
    missing = REQUIRED_ANALYSIS_COLUMNS.difference(scored_outputs.columns)
    if missing:
        raise ValueError(f"scored_outputs is missing required columns: {sorted(missing)}")

    model_types = set(scored_outputs["model_type"].dropna().unique())
    if model_types != {"base", "instruct"}:
        raise ValueError(
            "Expected model_type values to be exactly {'base', 'instruct'}, "
            f"but found {sorted(model_types)}."
        )


def infer_outcomes(
    scored_outputs: pd.DataFrame,
    requested_outcomes: Iterable[str] = DEFAULT_OUTCOME_COLUMNS,
) -> list[str]:
    """Return requested outcome columns that are present and numeric."""
    outcomes = []
    for column in requested_outcomes:
        if column in scored_outputs.columns and pd.api.types.is_numeric_dtype(scored_outputs[column]):
            outcomes.append(column)

    if not outcomes:
        raise ValueError(
            "No numeric outcome columns were found. Expected at least one of: "
            f"{list(requested_outcomes)}"
        )

    return outcomes


def make_paired_dataframe(
    scored_outputs: pd.DataFrame,
    outcomes: Iterable[str] = DEFAULT_OUTCOME_COLUMNS,
) -> pd.DataFrame:
    """Create one row per prompt with base, instruct, and difference columns.

    The returned dataframe includes metadata columns plus, for each outcome X:
    - X_base
    - X_instruct
    - X_effect = X_instruct - X_base
    """
    validate_scored_outputs(scored_outputs)
    outcomes = infer_outcomes(scored_outputs, outcomes)

    id_columns = ["prompt_id", "domain", "topic", "prompt_type"]
    optional_columns = [
        "ideological_axis",
        "model_family",
        "generation_seed",
        "temperature",
        "do_sample",
        "max_new_tokens",
    ]
    id_columns.extend([col for col in optional_columns if col in scored_outputs.columns])

    duplicate_counts = scored_outputs.groupby(["prompt_id", "model_type"]).size()
    duplicates = duplicate_counts[duplicate_counts > 1]
    if not duplicates.empty:
        raise ValueError(
            "Some prompt_id/model_type combinations have multiple rows. "
            "This analysis expects one base and one instruct output per prompt."
        )

    metadata = scored_outputs[id_columns].drop_duplicates("prompt_id")

    paired = metadata.copy()
    for outcome in outcomes:
        wide = scored_outputs.pivot(index="prompt_id", columns="model_type", values=outcome)
        if not {"base", "instruct"}.issubset(wide.columns):
            raise ValueError(f"Outcome {outcome!r} does not have both base and instruct values.")

        wide = wide.rename(columns={"base": f"{outcome}_base", "instruct": f"{outcome}_instruct"})
        wide[f"{outcome}_effect"] = wide[f"{outcome}_instruct"] - wide[f"{outcome}_base"]
        paired = paired.merge(wide.reset_index(), on="prompt_id", how="left")

    return paired


def bootstrap_mean_ci(
    values: np.ndarray,
    iterations: int = 2000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Estimate a bootstrap confidence interval for the mean."""
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return np.nan, np.nan
    if len(values) == 1:
        return float(values[0]), float(values[0])

    rng = np.random.default_rng(seed)
    sample_indices = rng.integers(0, len(values), size=(iterations, len(values)))
    bootstrap_means = values[sample_indices].mean(axis=1)

    alpha = 1 - confidence_level
    low = np.quantile(bootstrap_means, alpha / 2)
    high = np.quantile(bootstrap_means, 1 - alpha / 2)
    return float(low), float(high)


def paired_tests(values: np.ndarray) -> tuple[float | None, float | None, float | None, float | None]:
    """Run paired one-sample tests on the effect values against zero."""
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]

    if scipy_stats is None or len(values) < 2:
        return None, None, None, None

    t_result = scipy_stats.ttest_1samp(values, popmean=0.0, nan_policy="omit")

    try:
        wilcoxon_result = scipy_stats.wilcoxon(values, zero_method="wilcox", alternative="two-sided")
        w_stat = float(wilcoxon_result.statistic)
        w_p = float(wilcoxon_result.pvalue)
    except ValueError:
        # Happens when all differences are exactly zero.
        w_stat = None
        w_p = None

    return float(t_result.statistic), float(t_result.pvalue), w_stat, w_p


def summarize_effects(
    paired: pd.DataFrame,
    outcomes: Iterable[str] = DEFAULT_OUTCOME_COLUMNS,
    settings: AnalysisSettings | None = None,
) -> pd.DataFrame:
    """Compute average treatment effects for all available outcomes."""
    settings = settings or AnalysisSettings()
    outcomes = [outcome for outcome in outcomes if f"{outcome}_effect" in paired.columns]

    rows = []
    for outcome in outcomes:
        base = paired[f"{outcome}_base"].astype(float)
        instruct = paired[f"{outcome}_instruct"].astype(float)
        effect = paired[f"{outcome}_effect"].astype(float)
        valid = pd.concat([base, instruct, effect], axis=1).dropna()

        if valid.empty:
            continue

        effect_values = valid[f"{outcome}_effect"].to_numpy(dtype=float)
        ci_low, ci_high = bootstrap_mean_ci(
            effect_values,
            iterations=settings.bootstrap_iterations,
            confidence_level=settings.confidence_level,
            seed=settings.random_seed,
        )
        t_stat, t_p, w_stat, w_p = paired_tests(effect_values)
        diff_std = float(np.std(effect_values, ddof=1)) if len(effect_values) > 1 else 0.0
        standard_error = float(diff_std / np.sqrt(len(effect_values))) if len(effect_values) > 1 else 0.0
        cohen_dz = float(np.mean(effect_values) / diff_std) if diff_std > 0 else 0.0

        rows.append(
            EffectResult(
                metric=outcome,
                n_pairs=int(len(effect_values)),
                base_mean=float(valid[f"{outcome}_base"].mean()),
                instruct_mean=float(valid[f"{outcome}_instruct"].mean()),
                average_treatment_effect=float(np.mean(effect_values)),
                difference_std=diff_std,
                standard_error=standard_error,
                ci_low=ci_low,
                ci_high=ci_high,
                paired_t_statistic=t_stat,
                paired_t_p_value=t_p,
                wilcoxon_statistic=w_stat,
                wilcoxon_p_value=w_p,
                cohen_dz=cohen_dz,
            ).__dict__
        )

    return pd.DataFrame(rows)


def summarize_effects_by_group(
    paired: pd.DataFrame,
    group_column: str,
    outcomes: Iterable[str] = DEFAULT_OUTCOME_COLUMNS,
    settings: AnalysisSettings | None = None,
) -> pd.DataFrame:
    """Compute treatment effects separately for each value of a grouping column."""
    if group_column not in paired.columns:
        raise ValueError(f"Column {group_column!r} is not in the paired dataframe.")

    settings = settings or AnalysisSettings()
    rows = []
    for group_value, group_df in paired.groupby(group_column, dropna=False):
        summary = summarize_effects(group_df, outcomes=outcomes, settings=settings)
        if summary.empty:
            continue
        summary.insert(0, group_column, group_value)
        rows.append(summary)

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True)


def build_effect_tables(
    scored_outputs: pd.DataFrame,
    settings: AnalysisSettings | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    """Build paired dataframe, overall summary, and grouped summaries."""
    settings = settings or AnalysisSettings()
    paired = make_paired_dataframe(scored_outputs, outcomes=settings.outcomes)
    outcomes = [outcome for outcome in settings.outcomes if f"{outcome}_effect" in paired.columns]
    overall = summarize_effects(paired, outcomes=outcomes, settings=settings)

    grouped = {}
    for group_column in settings.group_columns:
        if group_column in paired.columns:
            grouped[group_column] = summarize_effects_by_group(
                paired,
                group_column=group_column,
                outcomes=outcomes,
                settings=settings,
            )

    return paired, overall, grouped


def select_high_difference_examples(
    scored_outputs: pd.DataFrame,
    paired: pd.DataFrame,
    metric: str = "neutrality_score",
    n_examples: int = 10,
) -> pd.DataFrame:
    """Select prompts with the largest absolute treatment effect for qualitative review."""
    effect_column = f"{metric}_effect"
    if effect_column not in paired.columns:
        raise ValueError(f"Metric {metric!r} is not available in the paired dataframe.")

    selected_prompt_ids = (
        paired[["prompt_id", effect_column]]
        .dropna()
        .assign(abs_effect=lambda df: df[effect_column].abs())
        .sort_values("abs_effect", ascending=False)
        .head(n_examples)["prompt_id"]
        .tolist()
    )

    examples = scored_outputs[scored_outputs["prompt_id"].isin(selected_prompt_ids)].copy()
    examples["model_type"] = pd.Categorical(
        examples["model_type"],
        categories=["base", "instruct"],
        ordered=True,
    )
    examples = examples.sort_values(["prompt_id", "model_type"])
    examples = examples.merge(
        paired[["prompt_id", effect_column]],
        on="prompt_id",
        how="left",
    )
    return examples


def save_effect_tables(
    paired: pd.DataFrame,
    overall: pd.DataFrame,
    grouped: dict[str, pd.DataFrame],
    output_dir: str | Path,
) -> dict[str, Path]:
    """Save analysis tables to CSV and return the generated paths."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}
    paths["paired_prompt_effects"] = output_dir / "paired_prompt_effects.csv"
    paired.to_csv(paths["paired_prompt_effects"], index=False)

    paths["treatment_effects_summary"] = output_dir / "treatment_effects_summary.csv"
    overall.to_csv(paths["treatment_effects_summary"], index=False)

    for group_column, table in grouped.items():
        path = output_dir / f"{group_column}_treatment_effects.csv"
        table.to_csv(path, index=False)
        paths[f"{group_column}_treatment_effects"] = path

    return paths

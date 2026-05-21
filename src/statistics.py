"""
Statistical analysis utilities for paired treatment effects.

The treatment effect is defined as:
    score(instruction-tuned model) - score(base model)
for the same prompt.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .config import METRICS


def validate_scored_outputs(scored_df: pd.DataFrame, metrics: list[str] | None = None) -> None:
    """Validate that scored outputs contain the fields needed for paired analysis."""
    metrics = metrics or METRICS
    required = {"prompt_id", "model_type", *metrics}
    missing = required.difference(scored_df.columns)

    if missing:
        raise ValueError(f"scored_outputs.csv is missing required columns: {sorted(missing)}")

    model_types = set(scored_df["model_type"].dropna().unique())
    if not {"base", "instruct"}.issubset(model_types):
        raise ValueError(
            "Expected model_type values to include 'base' and 'instruct'. "
            f"Found: {sorted(model_types)}"
        )


def make_paired_effects(scored_df: pd.DataFrame, metrics: list[str] | None = None) -> pd.DataFrame:
    """Create one row per prompt with base, instruct, and effect columns."""
    metrics = metrics or METRICS
    validate_scored_outputs(scored_df, metrics)

    pair_counts = scored_df.groupby("prompt_id")["model_type"].nunique()
    complete_prompt_ids = pair_counts[pair_counts == 2].index
    paired_source = scored_df[scored_df["prompt_id"].isin(complete_prompt_ids)].copy()

    metadata_cols = ["prompt_id", "domain", "topic", "prompt_type"]
    available_metadata_cols = [c for c in metadata_cols if c in paired_source.columns]
    metadata = paired_source[available_metadata_cols].drop_duplicates("prompt_id").set_index("prompt_id")

    paired_wide = paired_source.pivot_table(
        index="prompt_id",
        columns="model_type",
        values=metrics,
        aggfunc="first",
    )

    paired_wide.columns = [f"{metric}_{model_type}" for metric, model_type in paired_wide.columns]
    paired = metadata.join(paired_wide, how="inner").reset_index()

    for metric in metrics:
        paired[f"effect_{metric}"] = paired[f"{metric}_instruct"] - paired[f"{metric}_base"]

    return paired


def bootstrap_ci(values, n_boot: int = 5000, ci: int = 95, seed: int = 42) -> tuple[float, float]:
    """Bootstrap confidence interval for the mean."""
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return np.nan, np.nan

    rng = np.random.default_rng(seed)
    boot_means = [
        np.mean(rng.choice(values, size=len(values), replace=True))
        for _ in range(n_boot)
    ]

    lower = np.percentile(boot_means, (100 - ci) / 2)
    upper = np.percentile(boot_means, 100 - (100 - ci) / 2)
    return lower, upper


def cohen_dz(values) -> float:
    """Paired standardized mean difference: mean(diff) / sd(diff)."""
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) < 2:
        return np.nan
    sd = np.std(values, ddof=1)
    if sd == 0:
        return np.nan
    return np.mean(values) / sd


def paired_p_values(base_values, instruct_values) -> tuple[float, float]:
    """Return paired t-test and Wilcoxon signed-rank p-values."""
    base_values = np.asarray(base_values, dtype=float)
    instruct_values = np.asarray(instruct_values, dtype=float)

    mask = ~np.isnan(base_values) & ~np.isnan(instruct_values)
    base_values = base_values[mask]
    instruct_values = instruct_values[mask]

    if len(base_values) < 2:
        return np.nan, np.nan

    differences = instruct_values - base_values

    try:
        t_p = 1.0 if np.allclose(differences, 0) else stats.ttest_rel(instruct_values, base_values).pvalue
    except Exception:
        t_p = np.nan

    try:
        w_p = 1.0 if np.allclose(differences, 0) else stats.wilcoxon(differences, zero_method="wilcox").pvalue
    except Exception:
        w_p = np.nan

    return t_p, w_p


def summarize_treatment_effects(
    paired: pd.DataFrame,
    metrics: list[str] | None = None,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Compute ATEs, confidence intervals, effect sizes, and paired tests."""
    metrics = metrics or METRICS
    summary_rows = []

    for metric in metrics:
        effects = paired[f"effect_{metric}"].dropna().to_numpy(dtype=float)
        base_values = paired[f"{metric}_base"].to_numpy(dtype=float)
        instruct_values = paired[f"{metric}_instruct"].to_numpy(dtype=float)
        ci_low, ci_high = bootstrap_ci(effects)
        t_p, w_p = paired_p_values(base_values, instruct_values)

        summary_rows.append(
            {
                "metric": metric,
                "n_pairs": len(effects),
                "base_mean": np.nanmean(base_values),
                "instruct_mean": np.nanmean(instruct_values),
                "ate_instruct_minus_base": np.nanmean(effects),
                "effect_sd": np.nanstd(effects, ddof=1),
                "bootstrap_ci_low": ci_low,
                "bootstrap_ci_high": ci_high,
                "cohen_dz": cohen_dz(effects),
                "paired_t_p_value": t_p,
                "wilcoxon_p_value": w_p,
            }
        )

    summary = pd.DataFrame(summary_rows)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(output_path, index=False)

    return summary


def grouped_treatment_effects(
    paired: pd.DataFrame,
    group_col: str,
    metrics: list[str] | None = None,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Summarize treatment effects by a grouping variable such as domain or prompt_type."""
    metrics = metrics or METRICS
    effect_cols = [f"effect_{metric}" for metric in metrics]

    grouped = (
        paired
        .groupby(group_col)[effect_cols]
        .agg(["mean", "std", "count"])
        .round(4)
    )

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        grouped.to_csv(output_path)

    return grouped


def build_qualitative_examples(
    paired: pd.DataFrame,
    scored_df: pd.DataFrame,
    metrics: list[str] | None = None,
    n_per_metric: int = 8,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Select high-difference examples and attach base/instruct response texts."""
    metrics = metrics or ["leaning_score", "neutrality_score", "hedging_score", "refusal_score"]
    text_cols = [
        "prompt_id",
        "model_type",
        "response_text",
        "leaning_score",
        "neutrality_score",
        "hedging_score",
        "refusal_score",
        "response_length",
    ]

    responses = scored_df[text_cols].copy()

    def get_response_text(prompt_id: str, model_type: str) -> str:
        rows = responses[
            (responses["prompt_id"] == prompt_id)
            & (responses["model_type"] == model_type)
        ]
        if rows.empty:
            return ""
        return rows.iloc[0]["response_text"]

    qualitative_rows = []

    for metric in metrics:
        effect_col = f"effect_{metric}"
        top_cases = paired.reindex(
            paired[effect_col].abs().sort_values(ascending=False).index
        ).head(n_per_metric)

        for _, row in top_cases.iterrows():
            qualitative_rows.append(
                {
                    "metric": metric,
                    "prompt_id": row["prompt_id"],
                    "domain": row.get("domain", ""),
                    "topic": row.get("topic", ""),
                    "prompt_type": row.get("prompt_type", ""),
                    "base_score": row[f"{metric}_base"],
                    "instruct_score": row[f"{metric}_instruct"],
                    "effect": row[effect_col],
                    "base_response": get_response_text(row["prompt_id"], "base"),
                    "instruct_response": get_response_text(row["prompt_id"], "instruct"),
                }
            )

    qualitative = pd.DataFrame(qualitative_rows)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        qualitative.to_csv(output_path, index=False)

    return qualitative

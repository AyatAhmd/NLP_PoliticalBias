"""Visualization utilities for the revised P9 political bias project.

All plots use matplotlib directly. Each function creates one independent figure
and avoids project-specific styling so the visuals remain simple and reproducible.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRIC_LABELS = {
    "leaning_score": "Leaning score",
    "neutrality_score": "Neutrality score",
    "bias_strength": "Bias strength",
    "hedging_score": "Hedging score",
    "response_length": "Response length",
    "output_validity": "Output validity",
}


def metric_label(metric: str) -> str:
    """Return a readable label for a metric column."""
    return METRIC_LABELS.get(metric, metric.replace("_", " ").title())


def _ensure_parent(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def plot_average_treatment_effects(summary: pd.DataFrame, output_path: str | Path) -> Path:
    """Plot average treatment effects with bootstrap confidence intervals."""
    if summary.empty:
        raise ValueError("summary is empty; cannot plot average treatment effects.")

    output_path = _ensure_parent(output_path)
    data = summary.copy()
    data["metric_label"] = data["metric"].map(metric_label)
    data = data.sort_values("average_treatment_effect")

    y_positions = np.arange(len(data))
    effects = data["average_treatment_effect"].to_numpy(dtype=float)
    lower_errors = effects - data["ci_low"].to_numpy(dtype=float)
    upper_errors = data["ci_high"].to_numpy(dtype=float) - effects

    fig, ax = plt.subplots(figsize=(9, max(4, len(data) * 0.55)))
    ax.barh(y_positions, effects)
    ax.errorbar(effects, y_positions, xerr=[lower_errors, upper_errors], fmt="none", capsize=4)
    ax.axvline(0, linewidth=1)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(data["metric_label"])
    ax.set_xlabel("Average treatment effect: instruct - base")
    ax.set_title("Average treatment effects of instruction tuning")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_group_treatment_effects(
    grouped_summary: pd.DataFrame,
    group_column: str,
    metric: str,
    output_path: str | Path,
) -> Path:
    """Plot treatment effects for one metric grouped by domain or prompt type."""
    if grouped_summary.empty:
        raise ValueError("grouped_summary is empty; cannot plot grouped treatment effects.")
    if group_column not in grouped_summary.columns:
        raise ValueError(f"{group_column!r} is not in grouped_summary.")

    metric_data = grouped_summary[grouped_summary["metric"] == metric].copy()
    if metric_data.empty:
        raise ValueError(f"Metric {metric!r} is not available in grouped_summary.")

    output_path = _ensure_parent(output_path)
    metric_data = metric_data.sort_values("average_treatment_effect")
    y_positions = np.arange(len(metric_data))
    effects = metric_data["average_treatment_effect"].to_numpy(dtype=float)
    lower_errors = effects - metric_data["ci_low"].to_numpy(dtype=float)
    upper_errors = metric_data["ci_high"].to_numpy(dtype=float) - effects

    fig, ax = plt.subplots(figsize=(9, max(4, len(metric_data) * 0.55)))
    ax.barh(y_positions, effects)
    ax.errorbar(effects, y_positions, xerr=[lower_errors, upper_errors], fmt="none", capsize=4)
    ax.axvline(0, linewidth=1)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(metric_data[group_column].astype(str))
    ax.set_xlabel("Average treatment effect: instruct - base")
    ax.set_title(f"Treatment effect on {metric_label(metric)} by {group_column.replace('_', ' ')}")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_model_score_distribution(
    scored_outputs: pd.DataFrame,
    metric: str,
    output_path: str | Path,
) -> Path:
    """Plot base vs instruct distributions for one metric."""
    if metric not in scored_outputs.columns:
        raise ValueError(f"Metric {metric!r} is not in scored_outputs.")

    output_path = _ensure_parent(output_path)
    base_values = scored_outputs.loc[scored_outputs["model_type"] == "base", metric].dropna().to_numpy()
    instruct_values = scored_outputs.loc[scored_outputs["model_type"] == "instruct", metric].dropna().to_numpy()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot([base_values, instruct_values])
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Base", "Instruct"])
    ax.set_ylabel(metric_label(metric))
    ax.set_title(f"Distribution of {metric_label(metric)} by model type")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_leaning_neutrality_scatter(scored_outputs: pd.DataFrame, output_path: str | Path) -> Path:
    """Plot leaning score against neutrality score for base and instruct outputs."""
    required = {"leaning_score", "neutrality_score", "model_type"}
    missing = required.difference(scored_outputs.columns)
    if missing:
        raise ValueError(f"Missing required columns for scatter plot: {sorted(missing)}")

    output_path = _ensure_parent(output_path)
    fig, ax = plt.subplots(figsize=(7, 5))

    for model_type, group in scored_outputs.groupby("model_type"):
        ax.scatter(
            group["leaning_score"],
            group["neutrality_score"],
            alpha=0.65,
            label=str(model_type).title(),
        )

    ax.axvline(0, linewidth=1)
    ax.axhline(0, linewidth=1)
    ax.set_xlabel("Leaning score: left similarity - right similarity")
    ax.set_ylabel("Neutrality score: center similarity - strongest side similarity")
    ax.set_title("Political leaning versus neutrality")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_all_figures(
    scored_outputs: pd.DataFrame,
    overall_summary: pd.DataFrame,
    grouped_summaries: dict[str, pd.DataFrame],
    output_dir: str | Path,
    metrics: Iterable[str],
) -> dict[str, Path]:
    """Create the standard figure set for the revised analysis."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}

    paths["average_treatment_effects"] = plot_average_treatment_effects(
        overall_summary,
        output_dir / "average_treatment_effects.png",
    )

    for metric in metrics:
        if metric in scored_outputs.columns:
            paths[f"distribution_{metric}"] = plot_model_score_distribution(
                scored_outputs,
                metric,
                output_dir / f"distribution_{metric}.png",
            )

    if {"leaning_score", "neutrality_score"}.issubset(scored_outputs.columns):
        paths["leaning_neutrality_scatter"] = plot_leaning_neutrality_scatter(
            scored_outputs,
            output_dir / "leaning_neutrality_scatter.png",
        )

    for group_column, group_summary in grouped_summaries.items():
        for metric in metrics:
            if metric in set(group_summary.get("metric", [])):
                paths[f"{group_column}_{metric}"] = plot_group_treatment_effects(
                    group_summary,
                    group_column=group_column,
                    metric=metric,
                    output_path=output_dir / f"{group_column}_effect_{metric}.png",
                )

    return paths

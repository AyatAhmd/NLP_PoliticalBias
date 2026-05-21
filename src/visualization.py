"""
Visualization utilities for treatment effects.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import METRICS


def plot_average_treatment_effects(summary_df: pd.DataFrame, output_path: Path | None = None) -> None:
    """Plot average treatment effects with bootstrap confidence intervals."""
    plot_df = summary_df.copy()

    plt.figure(figsize=(10, 5))
    x = np.arange(len(plot_df))
    y = plot_df["ate_instruct_minus_base"].to_numpy()
    yerr = np.vstack(
        [
            y - plot_df["bootstrap_ci_low"].to_numpy(),
            plot_df["bootstrap_ci_high"].to_numpy() - y,
        ]
    )

    plt.bar(x, y)
    plt.errorbar(x, y, yerr=yerr, fmt="none", capsize=5)
    plt.axhline(0, linewidth=1)
    plt.xticks(x, plot_df["metric"], rotation=30, ha="right")
    plt.ylabel("Average treatment effect (instruct - base)")
    plt.title("Average effect of instruction tuning by outcome metric")
    plt.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=200)

    plt.show()


def plot_group_effects(
    paired: pd.DataFrame,
    group_col: str,
    figures_dir: Path,
    metrics: list[str] | None = None,
    prefix: str | None = None,
) -> list[Path]:
    """Create one bar plot per metric for grouped treatment effects."""
    metrics = metrics or METRICS
    prefix = prefix or group_col
    figures_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []

    for metric in metrics:
        effect_col = f"effect_{metric}"
        grouped = paired.groupby(group_col)[effect_col].mean().sort_values()

        plt.figure(figsize=(9, 5))
        grouped.plot(kind="bar")
        plt.axhline(0, linewidth=1)
        plt.ylabel(f"Mean effect on {metric}")
        plt.title(f"Instruction-tuning effect by {group_col}: {metric}")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()

        fig_path = figures_dir / f"{prefix}_effect_{metric}.png"
        plt.savefig(fig_path, dpi=200)
        plt.show()

        saved_paths.append(fig_path)

    return saved_paths

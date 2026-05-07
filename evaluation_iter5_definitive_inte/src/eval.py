#!/usr/bin/env python3
"""Definitive Interpretability Analysis: Spectral CoI Arity Reduction vs. Random Oblique FIGS.

Six-part evaluation + per-fold accuracy reconstruction:
  A. Arity comparison table across all 60 conditions
  B. Accuracy-arity Pareto frontier analysis
  C. Wilcoxon signed-rank test on 20 paired arity conditions with bootstrap CIs
  D. Path length comparison
  E. Interpretability cost per accuracy point
  F. Practical interpretation for adult (d=6) and jannis (d=54)
  + Per-fold accuracy reconstruction from raw predictions for paired accuracy tests
"""

import gc
import json
import math
import os
import resource
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats
from sklearn.metrics import balanced_accuracy_score, r2_score

# ── Logging ──────────────────────────────────────────────────────────────────
WORKSPACE = Path("/ai-inventor/aii_pipeline/runs/jamnik-sgfigs-pid-v2/3_invention_loop/iter_5/gen_art/eval_id5_it5__opus")
LOG_DIR = WORKSPACE / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add(str(LOG_DIR / "run.log"), rotation="30 MB", level="DEBUG")


# ── Hardware Detection & Resource Limits ─────────────────────────────────────
def _detect_cpus() -> int:
    """Detect actual CPU allocation (containers/pods/bare metal)."""
    try:
        parts = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if parts[0] != "max":
            return math.ceil(int(parts[0]) / int(parts[1]))
    except (FileNotFoundError, ValueError):
        pass
    try:
        q = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text())
        p = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
        if q > 0:
            return math.ceil(q / p)
    except (FileNotFoundError, ValueError):
        pass
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        pass
    return os.cpu_count() or 1


def _container_ram_gb() -> float | None:
    """Read RAM limit from cgroup (containers/pods)."""
    for p in ["/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"]:
        try:
            v = Path(p).read_text().strip()
            if v != "max" and int(v) < 1_000_000_000_000:
                return int(v) / 1e9
        except (FileNotFoundError, ValueError):
            pass
    return None


NUM_CPUS = _detect_cpus()
TOTAL_RAM_GB = _container_ram_gb() or 42.0

# Set RAM limit: 20GB budget (out of ~42GB available), generous for ~210MB data
RAM_BUDGET = int(20 * 1e9)
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET * 3, RAM_BUDGET * 3))
resource.setrlimit(resource.RLIMIT_CPU, (3600, 3600))  # 1 hour CPU time

logger.info(f"Hardware: {NUM_CPUS} CPUs, {TOTAL_RAM_GB:.1f} GB RAM container limit")
logger.info(f"RAM budget: {RAM_BUDGET / 1e9:.0f} GB")


# ── Paths ────────────────────────────────────────────────────────────────────
EXP_DIR = Path(
    "/ai-inventor/aii_pipeline/runs/jamnik-sgfigs-pid-v2/"
    "3_invention_loop/iter_2/gen_art/exp_id2_it2__opus"
)
MINI_FILE = EXP_DIR / "mini_method_out.json"
PART_FILES = [EXP_DIR / "method_out" / f"full_method_out_{i}.json" for i in range(1, 5)]

METHODS = ["axis_aligned_figs", "random_oblique_figs", "signed_spectral_figs"]
METHOD_SHORT = {
    "axis_aligned_figs": "aa",
    "random_oblique_figs": "ro",
    "signed_spectral_figs": "ss",
}
DATASET_N_FEATURES = {
    "electricity": 7,
    "adult": 6,
    "california_housing": 8,
    "jannis": 54,
    "higgs_small": 28,
}


# ── Data Loading ─────────────────────────────────────────────────────────────
def load_aggregated_results() -> tuple[pd.DataFrame, dict]:
    """Load aggregated results from mini_method_out.json (primary source).

    Returns:
        df: DataFrame with all 60 rows (5 datasets x 3 methods x 4 max_splits)
            with a unified 'accuracy' column.
        best_ms: dict mapping "dataset__method" -> best max_splits int.
    """
    logger.info(f"Loading aggregated results from {MINI_FILE}")
    data = json.loads(MINI_FILE.read_text())
    agg = data["metadata"]["aggregated_results"]
    best_ms = data["metadata"]["best_max_splits"]
    logger.info(f"Loaded {len(agg)} aggregated result entries")

    df = pd.DataFrame(agg)

    # Unified accuracy column (balanced_accuracy for clf, r2 for reg)
    df["accuracy"] = df.apply(
        lambda r: r["balanced_accuracy_mean"]
        if r["task_type"] == "classification"
        else r["r2_mean"],
        axis=1,
    )
    df["accuracy_std"] = df.apply(
        lambda r: r["balanced_accuracy_std"]
        if r["task_type"] == "classification"
        else r["r2_std"],
        axis=1,
    )
    return df, best_ms


def load_per_example_predictions() -> dict[str, list[dict]]:
    """Load per-example predictions from full_method_out part files.

    Returns dict mapping dataset_name -> list of example dicts.
    """
    all_examples: dict[str, list[dict]] = defaultdict(list)
    for part_file in PART_FILES:
        if not part_file.exists():
            logger.warning(f"Part file not found: {part_file}")
            continue
        size_mb = part_file.stat().st_size / 1e6
        logger.info(f"Loading {part_file.name} ({size_mb:.1f} MB)")
        t0 = time.time()
        data = json.loads(part_file.read_text())
        for ds_section in data.get("datasets", []):
            ds_name = ds_section["dataset"]
            all_examples[ds_name].extend(ds_section["examples"])
        del data
        gc.collect()
        logger.info(f"  Loaded in {time.time() - t0:.1f}s")

    for ds, exs in all_examples.items():
        logger.info(f"  {ds}: {len(exs)} examples")
    return dict(all_examples)


# ── Analysis A: Arity Comparison Table ───────────────────────────────────────
def analysis_a_arity_table(df: pd.DataFrame) -> tuple[dict, list[dict]]:
    """Comprehensive arity comparison across all 60 conditions.

    Metrics:
      - arity_table_mean_delta_ss_ro
      - arity_table_pct_reduction_mean
      - arity_table_n_conditions_ss_lower
    """
    logger.info("=== Analysis A: Arity Comparison Table ===")

    pivot = df.pivot_table(
        index=["dataset", "max_splits"],
        columns="method",
        values="avg_split_arity_mean",
    ).reset_index()

    pivot["delta_ss_ro"] = (
        pivot["signed_spectral_figs"] - pivot["random_oblique_figs"]
    )
    # pct_reduction: positive means SS is more parsimonious
    pivot["pct_reduction"] = np.where(
        pivot["random_oblique_figs"] > 0,
        100.0 * (1.0 - pivot["signed_spectral_figs"] / pivot["random_oblique_figs"]),
        0.0,
    )

    mean_delta = float(pivot["delta_ss_ro"].mean())
    mean_pct = float(pivot["pct_reduction"].mean())
    n_ss_lower = int((pivot["delta_ss_ro"] < 0).sum())

    logger.info(f"  Mean delta (SS-RO): {mean_delta:.4f}")
    logger.info(f"  Mean pct reduction: {mean_pct:.2f}%")
    logger.info(f"  Conditions where SS < RO: {n_ss_lower}/20")

    metrics = {
        "arity_table_mean_delta_ss_ro": round(mean_delta, 6),
        "arity_table_pct_reduction_mean": round(mean_pct, 4),
        "arity_table_n_conditions_ss_lower": n_ss_lower,
    }

    examples = []
    for _, row in pivot.iterrows():
        examples.append(
            {
                "input": (
                    f"Arity comparison for {row['dataset']} "
                    f"at max_splits={int(row['max_splits'])}"
                ),
                "output": (
                    f"AA={row['axis_aligned_figs']:.4f}, "
                    f"RO={row['random_oblique_figs']:.4f}, "
                    f"SS={row['signed_spectral_figs']:.4f}, "
                    f"delta={row['delta_ss_ro']:.4f}"
                ),
                "metadata_dataset": row["dataset"],
                "metadata_max_splits": int(row["max_splits"]),
                "eval_arity_aa": round(float(row["axis_aligned_figs"]), 6),
                "eval_arity_ro": round(float(row["random_oblique_figs"]), 6),
                "eval_arity_ss": round(float(row["signed_spectral_figs"]), 6),
                "eval_delta_ss_ro": round(float(row["delta_ss_ro"]), 6),
                "eval_pct_reduction": round(float(row["pct_reduction"]), 4),
            }
        )

    return metrics, examples


# ── Analysis B: Pareto Frontier Analysis ─────────────────────────────────────
def analysis_b_pareto(df: pd.DataFrame) -> tuple[dict, list[dict]]:
    """Accuracy-arity Pareto frontier analysis.

    For each condition, determines which methods lie on the Pareto frontier
    of (lower arity, higher accuracy).
    """
    logger.info("=== Analysis B: Pareto Frontier Analysis ===")

    conditions = df.groupby(["dataset", "max_splits"])
    frontier_counts = {m: 0 for m in METHODS}
    ss_dominates_ro_count = 0
    ro_dominates_ss_count = 0
    n_conditions = 0
    examples = []

    for (ds, ms), group in conditions:
        n_conditions += 1
        points = {}
        for _, row in group.iterrows():
            points[row["method"]] = (
                row["avg_split_arity_mean"],
                row["accuracy"],
            )

        on_frontier = {}
        for m in METHODS:
            arity_m, acc_m = points[m]
            dominated = False
            for m2 in METHODS:
                if m2 == m:
                    continue
                arity_m2, acc_m2 = points[m2]
                # m2 dominates m if m2 has <= arity AND >= accuracy
                # with at least one strict inequality
                if (
                    arity_m2 <= arity_m
                    and acc_m2 >= acc_m
                    and (arity_m2 < arity_m or acc_m2 > acc_m)
                ):
                    dominated = True
                    break
            on_frontier[m] = 0 if dominated else 1

        for m in METHODS:
            frontier_counts[m] += on_frontier[m]

        # Check SS vs RO dominance
        ss_a, ss_acc = points["signed_spectral_figs"]
        ro_a, ro_acc = points["random_oblique_figs"]
        ss_dom_ro = int(
            ss_a <= ro_a and ss_acc >= ro_acc and (ss_a < ro_a or ss_acc > ro_acc)
        )
        ro_dom_ss = int(
            ro_a <= ss_a and ro_acc >= ss_acc and (ro_a < ss_a or ro_acc > ss_acc)
        )
        ss_dominates_ro_count += ss_dom_ro
        ro_dominates_ss_count += ro_dom_ss

        examples.append(
            {
                "input": f"Pareto analysis for {ds} at max_splits={int(ms)}",
                "output": (
                    f"Frontier: AA={on_frontier['axis_aligned_figs']}, "
                    f"RO={on_frontier['random_oblique_figs']}, "
                    f"SS={on_frontier['signed_spectral_figs']}"
                ),
                "metadata_dataset": ds,
                "metadata_max_splits": int(ms),
                "eval_on_frontier_aa": on_frontier["axis_aligned_figs"],
                "eval_on_frontier_ro": on_frontier["random_oblique_figs"],
                "eval_on_frontier_ss": on_frontier["signed_spectral_figs"],
                "eval_ss_dominates_ro": ss_dom_ro,
            }
        )

    metrics = {
        "pareto_rate_axis_aligned": round(
            frontier_counts["axis_aligned_figs"] / n_conditions, 4
        ),
        "pareto_rate_random_oblique": round(
            frontier_counts["random_oblique_figs"] / n_conditions, 4
        ),
        "pareto_rate_signed_spectral": round(
            frontier_counts["signed_spectral_figs"] / n_conditions, 4
        ),
        "pareto_dominates_ss_over_ro": ss_dominates_ro_count,
        "pareto_dominates_ro_over_ss": ro_dominates_ss_count,
    }

    logger.info(
        f"  Pareto rates: AA={metrics['pareto_rate_axis_aligned']:.2f}, "
        f"RO={metrics['pareto_rate_random_oblique']:.2f}, "
        f"SS={metrics['pareto_rate_signed_spectral']:.2f}"
    )
    logger.info(
        f"  SS dominates RO: {ss_dominates_ro_count}, "
        f"RO dominates SS: {ro_dominates_ss_count}"
    )

    return metrics, examples


# ── Analysis C: Arity Reduction Significance ─────────────────────────────────
def analysis_c_arity_significance(df: pd.DataFrame) -> tuple[dict, list[dict]]:
    """Wilcoxon signed-rank test on 20 paired arity conditions with bootstrap CIs.

    Tests H0: median(SS_arity - RO_arity) = 0 vs H1: < 0 (one-sided).
    """
    logger.info("=== Analysis C: Arity Reduction Significance ===")

    pivot = df.pivot_table(
        index=["dataset", "max_splits"],
        columns="method",
        values="avg_split_arity_mean",
    ).reset_index()

    ss_arity = pivot["signed_spectral_figs"].values
    ro_arity = pivot["random_oblique_figs"].values
    differences = ss_arity - ro_arity  # negative => SS more parsimonious

    # Wilcoxon signed-rank test (one-sided: SS < RO)
    try:
        w_stat, w_p = stats.wilcoxon(differences, alternative="less")
    except ValueError as e:
        logger.warning(f"Wilcoxon test issue: {e}")
        w_stat, w_p = 0.0, 1.0

    # Bootstrap 95% CI on mean difference
    rng = np.random.RandomState(42)
    n_boot = 10000
    boot_means = np.array(
        [
            np.mean(rng.choice(differences, size=len(differences), replace=True))
            for _ in range(n_boot)
        ]
    )
    ci_lower = float(np.percentile(boot_means, 2.5))
    ci_upper = float(np.percentile(boot_means, 97.5))

    # Cohen's d effect size
    mean_diff = float(np.mean(differences))
    std_diff = float(np.std(differences, ddof=1))
    cohens_d = mean_diff / std_diff if std_diff > 1e-12 else 0.0

    n_negative = int(np.sum(differences < 0))

    logger.info(f"  Wilcoxon stat={w_stat:.2f}, p={w_p:.8f}")
    logger.info(f"  Mean diff (SS-RO): {mean_diff:.4f}")
    logger.info(f"  Bootstrap 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
    logger.info(f"  Cohen's d: {cohens_d:.4f}")
    logger.info(f"  N conditions SS < RO: {n_negative}/20")

    metrics = {
        "wilcoxon_arity_ss_vs_ro_statistic": round(float(w_stat), 4),
        "wilcoxon_arity_ss_vs_ro_p": round(float(w_p), 8),
        "arity_diff_mean_ss_minus_ro": round(mean_diff, 6),
        "arity_diff_bootstrap_ci_lower": round(ci_lower, 6),
        "arity_diff_bootstrap_ci_upper": round(ci_upper, 6),
        "arity_diff_cohens_d": round(cohens_d, 6),
        "arity_diff_n_negative": n_negative,
    }

    # Build statistical test examples
    examples = [
        {
            "input": (
                "Wilcoxon signed-rank test: SS arity vs RO arity "
                "(20 paired conditions, one-sided H1: SS < RO)"
            ),
            "output": (
                f"W={w_stat:.4f}, p={w_p:.8f}, "
                f"mean_diff={mean_diff:.4f}, d={cohens_d:.4f}"
            ),
            "metadata_test": "wilcoxon_arity_ss_vs_ro",
            "metadata_n_pairs": 20,
            "metadata_alternative": "less",
            "eval_test_statistic": round(float(w_stat), 4),
            "eval_p_value": round(float(w_p), 8),
            "eval_effect_size": round(cohens_d, 6),
        },
        {
            "input": (
                "Bootstrap 95% CI on mean arity difference (SS - RO), "
                "10000 resamples of 20 paired conditions"
            ),
            "output": (
                f"CI=[{ci_lower:.6f}, {ci_upper:.6f}], "
                f"mean={mean_diff:.6f}"
            ),
            "metadata_test": "bootstrap_ci_arity_diff",
            "metadata_n_boot": 10000,
            "eval_test_statistic": round(mean_diff, 6),
            "eval_p_value": round(float(w_p), 8),
            "eval_effect_size": round(cohens_d, 6),
        },
    ]

    # Per-dataset sub-analysis
    for ds in sorted(pivot["dataset"].unique()):
        ds_mask = pivot["dataset"].values == ds
        ds_diffs = differences[ds_mask]
        ds_mean = float(np.mean(ds_diffs))
        ds_n_neg = int(np.sum(ds_diffs < 0))
        n_total = len(ds_diffs)
        examples.append(
            {
                "input": (
                    f"Arity difference sub-analysis for {ds}: "
                    f"{n_total} paired conditions (over max_splits)"
                ),
                "output": (
                    f"mean_diff={ds_mean:.4f}, "
                    f"{ds_n_neg}/{n_total} conditions SS < RO"
                ),
                "metadata_test": f"arity_subdataset_{ds}",
                "metadata_n_pairs": n_total,
                "eval_test_statistic": round(ds_mean, 6),
                "eval_p_value": 0.0,
                "eval_effect_size": round(ds_mean, 6),
            }
        )

    return metrics, examples


# ── Analysis D: Path Length Analysis ─────────────────────────────────────────
def analysis_d_path_length(df: pd.DataFrame) -> tuple[dict, list[dict]]:
    """Compare avg_path_length across methods.

    Shorter paths = more interpretable trees.
    """
    logger.info("=== Analysis D: Path Length Analysis ===")

    pivot = df.pivot_table(
        index=["dataset", "max_splits"],
        columns="method",
        values="avg_path_length_mean",
    ).reset_index()

    ss_path = pivot["signed_spectral_figs"].values
    ro_path = pivot["random_oblique_figs"].values
    aa_path = pivot["axis_aligned_figs"].values

    diff_ss_ro = ss_path - ro_path
    diff_ss_aa = ss_path - aa_path
    diff_ro_aa = ro_path - aa_path

    # Wilcoxon test: SS path < RO path?
    try:
        w_stat_path, p_ss_ro = stats.wilcoxon(diff_ss_ro, alternative="less")
    except ValueError:
        w_stat_path, p_ss_ro = 0.0, 1.0

    # Bootstrap CI on path length difference
    rng = np.random.RandomState(43)
    boot_path = np.array(
        [
            np.mean(rng.choice(diff_ss_ro, size=len(diff_ss_ro), replace=True))
            for _ in range(10000)
        ]
    )
    path_ci_lower = float(np.percentile(boot_path, 2.5))
    path_ci_upper = float(np.percentile(boot_path, 97.5))

    mean_ss_ro = float(np.mean(diff_ss_ro))
    std_ss_ro = float(np.std(diff_ss_ro, ddof=1))
    path_cohens_d = mean_ss_ro / std_ss_ro if std_ss_ro > 1e-12 else 0.0

    metrics = {
        "wilcoxon_path_ss_vs_ro_p": round(float(p_ss_ro), 8),
        "path_diff_mean_ss_minus_ro": round(float(np.mean(diff_ss_ro)), 6),
        "path_diff_mean_ss_minus_aa": round(float(np.mean(diff_ss_aa)), 6),
        "path_diff_mean_ro_minus_aa": round(float(np.mean(diff_ro_aa)), 6),
    }

    logger.info(f"  Mean path diff SS-RO: {metrics['path_diff_mean_ss_minus_ro']:.4f}")
    logger.info(f"  Mean path diff SS-AA: {metrics['path_diff_mean_ss_minus_aa']:.4f}")
    logger.info(f"  Mean path diff RO-AA: {metrics['path_diff_mean_ro_minus_aa']:.4f}")
    logger.info(f"  Wilcoxon p (SS vs RO path): {p_ss_ro:.8f}")
    logger.info(f"  Bootstrap CI: [{path_ci_lower:.4f}, {path_ci_upper:.4f}]")

    examples = [
        {
            "input": (
                "Wilcoxon signed-rank test: SS path length vs RO path length "
                "(20 paired conditions, one-sided H1: SS < RO)"
            ),
            "output": (
                f"W={w_stat_path:.4f}, p={p_ss_ro:.8f}, "
                f"mean_diff={mean_ss_ro:.4f}, d={path_cohens_d:.4f}"
            ),
            "metadata_test": "wilcoxon_path_ss_vs_ro",
            "metadata_n_pairs": 20,
            "eval_test_statistic": round(float(w_stat_path), 4),
            "eval_p_value": round(float(p_ss_ro), 8),
            "eval_effect_size": round(path_cohens_d, 6),
        },
        {
            "input": (
                "Path length comparison: both oblique methods vs axis-aligned "
                "(20 paired conditions)"
            ),
            "output": (
                f"SS-AA={np.mean(diff_ss_aa):.4f}, RO-AA={np.mean(diff_ro_aa):.4f}"
            ),
            "metadata_test": "path_oblique_vs_aa",
            "metadata_n_pairs": 20,
            "eval_test_statistic": round(float(np.mean(diff_ss_aa)), 6),
            "eval_p_value": round(float(p_ss_ro), 8),
            "eval_effect_size": round(float(np.mean(diff_ro_aa)), 6),
        },
    ]

    return metrics, examples


# ── Analysis E: Interpretability Cost ────────────────────────────────────────
def analysis_e_interp_cost(df: pd.DataFrame) -> tuple[dict, list[dict]]:
    """Quantify the 'interpretability price' of accuracy.

    Computes arity cost per accuracy point and arity-normalized accuracy.
    """
    logger.info("=== Analysis E: Interpretability Cost ===")

    pivot_arity = df.pivot_table(
        index=["dataset", "max_splits"],
        columns="method",
        values="avg_split_arity_mean",
    ).reset_index()

    pivot_acc = df.pivot_table(
        index=["dataset", "max_splits"],
        columns="method",
        values="accuracy",
    ).reset_index()

    aa_arity = pivot_arity["axis_aligned_figs"].values
    ro_arity = pivot_arity["random_oblique_figs"].values
    ss_arity = pivot_arity["signed_spectral_figs"].values

    aa_acc = pivot_acc["axis_aligned_figs"].values
    ro_acc = pivot_acc["random_oblique_figs"].values
    ss_acc = pivot_acc["signed_spectral_figs"].values

    # Arity increase over axis-aligned
    arity_inc_ro = ro_arity - aa_arity
    arity_inc_ss = ss_arity - aa_arity

    # Accuracy delta over axis-aligned
    acc_delta_ro = ro_acc - aa_acc
    acc_delta_ss = ss_acc - aa_acc

    # Arity cost per accuracy point (only where acc delta > 0)
    mask_ro = acc_delta_ro > 1e-8
    mask_ss = acc_delta_ss > 1e-8

    cost_ro = (
        arity_inc_ro[mask_ro] / acc_delta_ro[mask_ro] if mask_ro.any() else np.array([])
    )
    cost_ss = (
        arity_inc_ss[mask_ss] / acc_delta_ss[mask_ss] if mask_ss.any() else np.array([])
    )

    mean_cost_ro = float(np.mean(cost_ro)) if len(cost_ro) > 0 else 0.0
    mean_cost_ss = float(np.mean(cost_ss)) if len(cost_ss) > 0 else 0.0

    # Arity-normalized accuracy: accuracy / arity (higher = better)
    aa_norm = aa_acc / np.maximum(aa_arity, 1e-8)
    ro_norm = ro_acc / np.maximum(ro_arity, 1e-8)
    ss_norm = ss_acc / np.maximum(ss_arity, 1e-8)

    metrics = {
        "mean_arity_cost_per_acc_point_ss": round(mean_cost_ss, 6),
        "mean_arity_cost_per_acc_point_ro": round(mean_cost_ro, 6),
        "n_positive_acc_delta_ss": int(mask_ss.sum()),
        "n_positive_acc_delta_ro": int(mask_ro.sum()),
        "mean_arity_normalized_accuracy_aa": round(float(np.mean(aa_norm)), 6),
        "mean_arity_normalized_accuracy_ro": round(float(np.mean(ro_norm)), 6),
        "mean_arity_normalized_accuracy_ss": round(float(np.mean(ss_norm)), 6),
    }

    logger.info(
        f"  Mean arity cost/acc point: SS={mean_cost_ss:.4f}, RO={mean_cost_ro:.4f}"
    )
    logger.info(
        f"  N positive acc delta: SS={int(mask_ss.sum())}/20, "
        f"RO={int(mask_ro.sum())}/20"
    )
    logger.info(
        f"  Arity-normalized accuracy: AA={np.mean(aa_norm):.4f}, "
        f"RO={np.mean(ro_norm):.4f}, SS={np.mean(ss_norm):.4f}"
    )

    # No separate dataset section for Analysis E (metrics only)
    return metrics, []


# ── Analysis F: Practical Interpretation ─────────────────────────────────────
def analysis_f_practical(
    df: pd.DataFrame, best_ms: dict
) -> tuple[dict, list[dict]]:
    """Translate arity into concrete interpretability statements.

    Uses best_max_splits per (dataset, method) to select the most relevant config.
    """
    logger.info("=== Analysis F: Practical Interpretation ===")

    # Select the best-max-splits row for each (dataset, method)
    best_rows: dict[tuple[str, str], pd.Series] = {}
    for ds in df["dataset"].unique():
        for m in METHODS:
            key = f"{ds}__{m}"
            if key in best_ms:
                bms = best_ms[key]
                mask = (
                    (df["dataset"] == ds)
                    & (df["method"] == m)
                    & (df["max_splits"] == bms)
                )
                rows = df[mask]
                if len(rows) > 0:
                    best_rows[(ds, m)] = rows.iloc[0]

    # Compute cognitive complexity: arity * path_length
    cognitive: dict[str, dict[str, float]] = {}
    for ds in df["dataset"].unique():
        cognitive[ds] = {}
        for m in METHODS:
            if (ds, m) in best_rows:
                row = best_rows[(ds, m)]
                cognitive[ds][m] = (
                    row["avg_split_arity_mean"] * row["avg_path_length_mean"]
                )

    # Build metrics for adult and jannis
    metrics: dict[str, float] = {}
    for ds in ["adult", "jannis"]:
        if ds in cognitive:
            for m, short in METHOD_SHORT.items():
                mkey = f"{ds}_features_per_pred_{short}"
                metrics[mkey] = round(cognitive[ds].get(m, 0.0), 4)

    # Mean cognitive complexity reduction SS vs RO (all datasets)
    reductions = []
    for ds in cognitive:
        ro_cog = cognitive[ds].get("random_oblique_figs", 0.0)
        ss_cog = cognitive[ds].get("signed_spectral_figs", 0.0)
        if ro_cog > 1e-8:
            reductions.append(100.0 * (1.0 - ss_cog / ro_cog))
    metrics["mean_cognitive_complexity_reduction_ss_vs_ro"] = round(
        float(np.mean(reductions)) if reductions else 0.0, 4
    )

    logger.info(
        f"  Adult features/pred: "
        f"AA={metrics.get('adult_features_per_pred_aa', 'N/A')}, "
        f"RO={metrics.get('adult_features_per_pred_ro', 'N/A')}, "
        f"SS={metrics.get('adult_features_per_pred_ss', 'N/A')}"
    )
    logger.info(
        f"  Jannis features/pred: "
        f"AA={metrics.get('jannis_features_per_pred_aa', 'N/A')}, "
        f"RO={metrics.get('jannis_features_per_pred_ro', 'N/A')}, "
        f"SS={metrics.get('jannis_features_per_pred_ss', 'N/A')}"
    )
    logger.info(
        f"  Mean cognitive reduction SS vs RO: "
        f"{metrics['mean_cognitive_complexity_reduction_ss_vs_ro']:.2f}%"
    )

    # Build examples: one per dataset
    examples = []
    for ds in sorted(df["dataset"].unique()):
        if ds not in cognitive:
            continue
        n_feat = DATASET_N_FEATURES.get(ds, 0)
        ds_cog = cognitive[ds]

        # Get arities at best config for interpretability narrative
        arities = {}
        for m in METHODS:
            if (ds, m) in best_rows:
                arities[m] = best_rows[(ds, m)]["avg_split_arity_mean"]

        aa_cog = ds_cog.get("axis_aligned_figs", 0.0)
        ro_cog = ds_cog.get("random_oblique_figs", 0.0)
        ss_cog = ds_cog.get("signed_spectral_figs", 0.0)
        cog_red = (
            100.0 * (1.0 - ss_cog / ro_cog) if ro_cog > 1e-8 else 0.0
        )

        examples.append(
            {
                "input": f"Practical interpretation for {ds} (d={n_feat})",
                "output": (
                    f"Features inspected per prediction: "
                    f"AA={aa_cog:.1f}, RO={ro_cog:.1f}, SS={ss_cog:.1f}. "
                    f"Cognitive reduction SS vs RO: {cog_red:.1f}%"
                ),
                "metadata_dataset": ds,
                "metadata_n_features": n_feat,
                "eval_features_per_pred_aa": round(aa_cog, 4),
                "eval_features_per_pred_ro": round(ro_cog, 4),
                "eval_features_per_pred_ss": round(ss_cog, 4),
                "eval_cognitive_reduction_pct": round(cog_red, 4),
            }
        )

    return metrics, examples


# ── Per-Fold Accuracy Reconstruction ─────────────────────────────────────────
def analysis_perfold_accuracy(
    all_examples: dict[str, list[dict]],
) -> tuple[dict, list[dict]]:
    """Reconstruct per-fold accuracy from per-example predictions.

    For classification: balanced_accuracy_score per fold.
    For regression: r2_score per fold.
    Pools all (dataset, fold) pairs for Wilcoxon tests.
    """
    logger.info("=== Per-Fold Accuracy Reconstruction ===")

    if not all_examples:
        logger.warning("No per-example predictions loaded, skipping")
        return (
            {
                "perfold_wilcoxon_acc_ss_vs_ro_p": 1.0,
                "perfold_wilcoxon_acc_ss_vs_aa_p": 1.0,
                "perfold_mean_acc_diff_ss_minus_ro": 0.0,
            },
            [],
        )

    perfold_records = []

    for ds_name, examples in sorted(all_examples.items()):
        if not examples:
            continue
        task_type = examples[0].get("metadata_task_type", "classification")
        logger.info(
            f"  Processing {ds_name} ({len(examples)} examples, {task_type})"
        )

        # Group by fold
        fold_groups: dict[int, dict[str, list]] = defaultdict(
            lambda: {"y_true": [], "pred_aa": [], "pred_ro": [], "pred_ss": []}
        )

        for ex in examples:
            fold = ex.get("metadata_fold", 0)
            fg = fold_groups[fold]
            if task_type == "classification":
                fg["y_true"].append(int(float(ex["output"])))
                fg["pred_aa"].append(
                    int(float(ex.get("predict_axis_aligned_figs", "0")))
                )
                fg["pred_ro"].append(
                    int(float(ex.get("predict_random_oblique_figs", "0")))
                )
                fg["pred_ss"].append(
                    int(float(ex.get("predict_signed_spectral_figs", "0")))
                )
            else:
                fg["y_true"].append(float(ex["output"]))
                fg["pred_aa"].append(
                    float(ex.get("predict_axis_aligned_figs", "0"))
                )
                fg["pred_ro"].append(
                    float(ex.get("predict_random_oblique_figs", "0"))
                )
                fg["pred_ss"].append(
                    float(ex.get("predict_signed_spectral_figs", "0"))
                )

        for fold in sorted(fold_groups.keys()):
            fg = fold_groups[fold]
            y_true = np.array(fg["y_true"])
            if len(y_true) < 2:
                continue

            try:
                if task_type == "classification":
                    acc_aa = balanced_accuracy_score(y_true, np.array(fg["pred_aa"]))
                    acc_ro = balanced_accuracy_score(y_true, np.array(fg["pred_ro"]))
                    acc_ss = balanced_accuracy_score(y_true, np.array(fg["pred_ss"]))
                else:
                    acc_aa = r2_score(y_true, np.array(fg["pred_aa"]))
                    acc_ro = r2_score(y_true, np.array(fg["pred_ro"]))
                    acc_ss = r2_score(y_true, np.array(fg["pred_ss"]))
            except Exception:
                logger.exception(f"Error computing accuracy for {ds_name} fold {fold}")
                continue

            perfold_records.append(
                {
                    "dataset": ds_name,
                    "fold": fold,
                    "task_type": task_type,
                    "acc_aa": acc_aa,
                    "acc_ro": acc_ro,
                    "acc_ss": acc_ss,
                }
            )

    if not perfold_records:
        return (
            {
                "perfold_wilcoxon_acc_ss_vs_ro_p": 1.0,
                "perfold_wilcoxon_acc_ss_vs_aa_p": 1.0,
                "perfold_mean_acc_diff_ss_minus_ro": 0.0,
            },
            [],
        )

    pdf = pd.DataFrame(perfold_records)
    logger.info(f"  Reconstructed {len(pdf)} (dataset, fold) accuracy records")

    # Paired Wilcoxon tests pooled across all (dataset, fold) pairs
    diff_ss_ro = pdf["acc_ss"].values - pdf["acc_ro"].values
    diff_ss_aa = pdf["acc_ss"].values - pdf["acc_aa"].values

    try:
        _, p_ss_ro = stats.wilcoxon(diff_ss_ro, alternative="two-sided")
    except ValueError:
        p_ss_ro = 1.0
    try:
        _, p_ss_aa = stats.wilcoxon(diff_ss_aa, alternative="two-sided")
    except ValueError:
        p_ss_aa = 1.0

    mean_diff_ss_ro = float(np.mean(diff_ss_ro))

    logger.info(
        f"  Wilcoxon acc SS vs RO: p={p_ss_ro:.8f}, "
        f"mean_diff={mean_diff_ss_ro:.6f}"
    )
    logger.info(f"  Wilcoxon acc SS vs AA: p={p_ss_aa:.8f}")

    metrics = {
        "perfold_wilcoxon_acc_ss_vs_ro_p": round(float(p_ss_ro), 8),
        "perfold_wilcoxon_acc_ss_vs_aa_p": round(float(p_ss_aa), 8),
        "perfold_mean_acc_diff_ss_minus_ro": round(mean_diff_ss_ro, 8),
    }

    # Build examples: one per (dataset, fold)
    examples = []
    for _, row in pdf.iterrows():
        examples.append(
            {
                "input": (
                    f"Per-fold accuracy for {row['dataset']} "
                    f"fold {int(row['fold'])} ({row['task_type']})"
                ),
                "output": (
                    f"AA={row['acc_aa']:.6f}, "
                    f"RO={row['acc_ro']:.6f}, "
                    f"SS={row['acc_ss']:.6f}"
                ),
                "metadata_dataset": row["dataset"],
                "metadata_fold": int(row["fold"]),
                "eval_acc_aa": round(float(row["acc_aa"]), 8),
                "eval_acc_ro": round(float(row["acc_ro"]), 8),
                "eval_acc_ss": round(float(row["acc_ss"]), 8),
            }
        )

    return metrics, examples


# ── Main ─────────────────────────────────────────────────────────────────────
@logger.catch
def main():
    t_start = time.time()
    logger.info("=" * 70)
    logger.info("Definitive Interpretability Analysis: Spectral CoI vs Random Oblique")
    logger.info("=" * 70)

    # ── 1. Load aggregated results ───────────────────────
    df, best_ms = load_aggregated_results()

    # ── 2. Run Analyses A-F on aggregated data ───────────
    metrics_a, examples_a = analysis_a_arity_table(df)
    metrics_b, examples_b = analysis_b_pareto(df)
    metrics_c, stat_examples_c = analysis_c_arity_significance(df)
    metrics_d, stat_examples_d = analysis_d_path_length(df)
    metrics_e, _ = analysis_e_interp_cost(df)
    metrics_f, examples_f = analysis_f_practical(df, best_ms)

    # ── 3. Load per-example predictions & reconstruct accuracy ──
    logger.info("Loading per-example predictions for per-fold analysis...")
    all_examples = load_per_example_predictions()
    metrics_pf, examples_pf = analysis_perfold_accuracy(all_examples)
    del all_examples
    gc.collect()

    # ── 4. Merge all metrics ─────────────────────────────
    metrics_agg: dict[str, float | int] = {}
    for m in [metrics_a, metrics_b, metrics_c, metrics_d, metrics_e, metrics_f, metrics_pf]:
        metrics_agg.update(m)

    # Merge statistical test examples (Analysis C + D)
    stat_examples = stat_examples_c + stat_examples_d

    # ── 5. Build output following exp_eval_sol_out.json schema ──
    eval_out = {
        "metadata": {
            "evaluation": "definitive_interpretability_analysis",
            "description": (
                "Comprehensive 6-part interpretability evaluation of "
                "signed spectral FIGS vs. random oblique FIGS"
            ),
            "analyses": [
                "A_arity_table",
                "B_pareto",
                "C_arity_significance",
                "D_path_length",
                "E_interp_cost",
                "F_practical",
                "perfold_accuracy",
            ],
            "n_conditions": 20,
            "n_datasets": 5,
            "n_methods": 3,
            "n_max_splits_levels": 4,
        },
        "metrics_agg": metrics_agg,
        "datasets": [
            {"dataset": "arity_comparison", "examples": examples_a},
            {"dataset": "pareto_analysis", "examples": examples_b},
            {"dataset": "statistical_tests", "examples": stat_examples},
            {"dataset": "practical_interpretation", "examples": examples_f},
            {"dataset": "per_fold_accuracy", "examples": examples_pf},
        ],
    }

    # ── 6. Write output ──────────────────────────────────
    out_path = WORKSPACE / "eval_out.json"
    out_text = json.dumps(eval_out, indent=2)
    out_path.write_text(out_text)
    size_mb = out_path.stat().st_size / 1e6
    logger.info(f"Wrote eval_out.json ({size_mb:.2f} MB)")

    # ── 7. Summary ───────────────────────────────────────
    elapsed = time.time() - t_start
    logger.info("=" * 70)
    logger.info(f"COMPLETED in {elapsed:.1f}s")
    logger.info("=" * 70)
    logger.info("=== KEY METRICS SUMMARY ===")
    for k in sorted(metrics_agg.keys()):
        logger.info(f"  {k}: {metrics_agg[k]}")


if __name__ == "__main__":
    main()

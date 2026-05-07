#!/usr/bin/env python3
"""Interpretability Tradeoff Evaluation for 5 FIGS Variants.

Six analyses: (A) arity comparison table, (B) arity reduction significance tests,
(C) accuracy equivalence tests, (D) Pareto frontier counting, (E) path length analysis,
(F) cognitive complexity metric. All with paired non-parametric tests, bootstrap CIs,
and effect sizes. Output paper-ready JSON tables.
"""

import json
import math
import os
import resource
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add("logs/run.log", rotation="30 MB", level="DEBUG")

# ---------------------------------------------------------------------------
# Hardware-aware resource limits (cgroup v1 container)
# ---------------------------------------------------------------------------
def _container_ram_bytes() -> int:
    for p in ["/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"]:
        try:
            v = Path(p).read_text().strip()
            if v != "max" and int(v) < 1_000_000_000_000:
                return int(v)
        except (FileNotFoundError, ValueError):
            pass
    return 57 * 1024**3  # fallback

RAM_LIMIT = _container_ram_bytes()
RAM_BUDGET = int(RAM_LIMIT * 0.5)  # 50% of container — this script is lightweight
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET * 3, RAM_BUDGET * 3))
resource.setrlimit(resource.RLIMIT_CPU, (3600, 3600))
logger.info(f"RAM budget: {RAM_BUDGET / 1e9:.1f} GB, CPU limit: 3600s")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
WORKSPACE = Path(__file__).resolve().parent
DATA_PATH = WORKSPACE / "full_method_out.json"
OUTPUT_PATH = WORKSPACE / "eval_out.json"

METHODS = ["axis_aligned", "random_oblique", "unsigned_spectral", "signed_spectral", "hard_threshold"]
MAX_SPLITS_VALUES = [5, 10, 20]
N_BOOTSTRAP = 10_000
ROPE = 0.01  # region of practical equivalence for balanced accuracy
RNG = np.random.default_rng(42)
HIGH_DIM_DATASETS = ["jannis", "miniboone", "higgs_small"]

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _nanfloat(v: float) -> float | None:
    """Convert NaN/Inf to None for JSON-safe output."""
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return None
    return float(v)


def safe_wilcoxon(x: np.ndarray, y: np.ndarray) -> dict:
    """Wilcoxon signed-rank test with NaN filtering and fallback for constant differences."""
    # Drop NaN pairs
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    diff = x - y
    nonzero = diff[diff != 0]
    if len(nonzero) < 2:
        return {"W": None, "p": 1.0, "n_pairs": int(len(x)), "n_nonzero": int(len(nonzero))}
    try:
        res = stats.wilcoxon(x, y, alternative="two-sided")
        return {"W": float(res.statistic), "p": float(res.pvalue), "n_pairs": int(len(x)), "n_nonzero": int(len(nonzero))}
    except ValueError:
        return {"W": None, "p": 1.0, "n_pairs": int(len(x)), "n_nonzero": int(len(nonzero))}


def bootstrap_ci(diffs: np.ndarray, n_boot: int = N_BOOTSTRAP, alpha: float = 0.05) -> dict:
    """Bootstrap 95% CI on mean difference, dropping NaN."""
    diffs = diffs[~np.isnan(diffs)]
    if len(diffs) == 0:
        return {"mean": None, "ci_lo": None, "ci_hi": None, "n_boot": n_boot}
    means = np.array([RNG.choice(diffs, size=len(diffs), replace=True).mean() for _ in range(n_boot)])
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"mean": float(diffs.mean()), "ci_lo": float(lo), "ci_hi": float(hi), "n_boot": n_boot}


def cohens_d(x: np.ndarray, y: np.ndarray) -> dict:
    """Cohen's d with bootstrap 95% CI, dropping NaN pairs."""
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    if len(x) < 2:
        return {"d": None, "ci_lo": None, "ci_hi": None}
    diff = x - y
    pooled_std = np.sqrt((np.var(x, ddof=1) + np.var(y, ddof=1)) / 2)
    if pooled_std < 1e-12:
        return {"d": 0.0, "ci_lo": 0.0, "ci_hi": 0.0}
    d = diff.mean() / pooled_std
    # Bootstrap CI on d
    ds = []
    for _ in range(N_BOOTSTRAP):
        idx = RNG.integers(0, len(x), size=len(x))
        bx, by = x[idx], y[idx]
        ps = np.sqrt((np.var(bx, ddof=1) + np.var(by, ddof=1)) / 2)
        if ps > 1e-12:
            ds.append((bx - by).mean() / ps)
    ds = np.array(ds)
    if len(ds) == 0:
        return {"d": float(d), "ci_lo": float(d), "ci_hi": float(d)}
    lo, hi = np.percentile(ds, [2.5, 97.5])
    return {"d": float(d), "ci_lo": float(lo), "ci_hi": float(hi)}


def bayesian_sign_test_rope(x: np.ndarray, y: np.ndarray, rope: float = ROPE) -> dict:
    """Bayesian sign test with ROPE: fraction of differences in [-rope, +rope], positive, negative."""
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    diff = x - y
    n = len(diff)
    if n == 0:
        return {"p_equivalent": None, "p_method1_better": None, "p_method2_better": None, "rope": rope, "n": 0}
    n_equiv = int(np.sum(np.abs(diff) < rope))
    n_pos = int(np.sum(diff >= rope))
    n_neg = int(np.sum(diff <= -rope))
    return {
        "p_equivalent": round(n_equiv / n, 4),
        "p_method1_better": round(n_pos / n, 4),
        "p_method2_better": round(n_neg / n, 4),
        "rope": rope,
        "n": n,
    }


def pairwise_comparison(df: pd.DataFrame, method1: str, method2: str,
                        metric: str, datasets: list[str] | None = None) -> dict:
    """Run pooled and per-dataset Wilcoxon, bootstrap CI, Cohen's d on a metric."""
    sub = df if datasets is None else df[df["dataset"].isin(datasets)]
    m1 = sub[sub["method"] == method1].sort_values(["dataset", "max_splits", "fold"])[metric].values
    m2 = sub[sub["method"] == method2].sort_values(["dataset", "max_splits", "fold"])[metric].values
    if len(m1) != len(m2) or len(m1) == 0:
        logger.warning(f"Mismatched lengths for {method1} vs {method2} on {metric}: {len(m1)} vs {len(m2)}")
        return {}

    result = {
        "pooled_wilcoxon": safe_wilcoxon(m1, m2),
        "bootstrap_ci_mean_diff": bootstrap_ci(m1 - m2),
        "cohens_d": cohens_d(m1, m2),
    }

    # Per-dataset Wilcoxon
    per_ds = {}
    ds_list = sorted(sub["dataset"].unique())
    for ds in ds_list:
        ds_sub = sub[sub["dataset"] == ds]
        v1 = ds_sub[ds_sub["method"] == method1].sort_values(["max_splits", "fold"])[metric].values
        v2 = ds_sub[ds_sub["method"] == method2].sort_values(["max_splits", "fold"])[metric].values
        if len(v1) == len(v2) and len(v1) > 0:
            per_ds[ds] = safe_wilcoxon(v1, v2)
    result["per_dataset_wilcoxon"] = per_ds
    return result


def accuracy_equivalence(df: pd.DataFrame, method1: str, method2: str) -> dict:
    """Accuracy equivalence using unified 'performance' metric (balanced_accuracy or r2).
    Wilcoxon, Bayesian ROPE, bootstrap CI."""
    m1 = df[df["method"] == method1].sort_values(["dataset", "max_splits", "fold"])["performance"].values
    m2 = df[df["method"] == method2].sort_values(["dataset", "max_splits", "fold"])["performance"].values
    if len(m1) != len(m2) or len(m1) == 0:
        return {}
    return {
        "wilcoxon": safe_wilcoxon(m1, m2),
        "bayesian_rope": bayesian_sign_test_rope(m1, m2, rope=ROPE),
        "bootstrap_ci_mean_diff": bootstrap_ci(m1 - m2),
        "cohens_d": cohens_d(m1, m2),
        "mean_performance_method1": round(float(np.nanmean(m1)), 6),
        "mean_performance_method2": round(float(np.nanmean(m2)), 6),
        "note": "performance = balanced_accuracy for classification, r2 for regression",
    }


# ---------------------------------------------------------------------------
# Analysis blocks
# ---------------------------------------------------------------------------

def analysis_a_arity_table(df: pd.DataFrame) -> dict:
    """(A) Arity comparison table: mean/std per (dataset, method, max_splits)."""
    logger.info("Running Analysis A: Arity Comparison Table")
    table = {}
    for (ds, method, ms), grp in df.groupby(["dataset", "method", "max_splits"]):
        key = f"{ds}__{method}__ms{ms}"
        table[key] = {
            "dataset": ds,
            "method": method,
            "max_splits": int(ms),
            "arity_mean": round(float(grp["avg_split_arity"].mean()), 4),
            "arity_std": round(float(grp["avg_split_arity"].std()), 4),
            "n_folds": int(len(grp)),
        }
    logger.info(f"  Computed arity stats for {len(table)} cells")
    return table


def analysis_b_arity_significance(df: pd.DataFrame) -> dict:
    """(B) Arity reduction significance tests."""
    logger.info("Running Analysis B: Arity Reduction Significance")
    comparisons = {
        "unsigned_spectral_vs_random_oblique": ("unsigned_spectral", "random_oblique"),
        "unsigned_spectral_vs_hard_threshold": ("unsigned_spectral", "hard_threshold"),
        "signed_spectral_vs_random_oblique": ("signed_spectral", "random_oblique"),
    }
    results = {}
    for label, (m1, m2) in comparisons.items():
        logger.info(f"  Comparing {m1} vs {m2} on avg_split_arity")
        results[label] = pairwise_comparison(df, m1, m2, "avg_split_arity")
    return results


def analysis_c_accuracy_equivalence(df: pd.DataFrame) -> dict:
    """(C) Accuracy equivalence tests."""
    logger.info("Running Analysis C: Accuracy Equivalence")
    comparisons = {
        "unsigned_spectral_vs_random_oblique": ("unsigned_spectral", "random_oblique"),
        "unsigned_spectral_vs_hard_threshold": ("unsigned_spectral", "hard_threshold"),
        "unsigned_spectral_vs_axis_aligned": ("unsigned_spectral", "axis_aligned"),
    }
    results = {}
    for label, (m1, m2) in comparisons.items():
        logger.info(f"  Comparing {m1} vs {m2} on balanced_accuracy")
        results[label] = accuracy_equivalence(df, m1, m2)
    return results


def analysis_d_pareto(df: pd.DataFrame) -> dict:
    """(D) Pareto frontier analysis."""
    logger.info("Running Analysis D: Pareto Frontier")

    # Compute per-(dataset, method, max_splits) mean arity and performance
    summary = df.groupby(["dataset", "method", "max_splits"]).agg(
        arity_mean=("avg_split_arity", "mean"),
        accuracy_mean=("performance", "mean"),
    ).reset_index()

    datasets = sorted(df["dataset"].unique())
    frontier_counts = {m: 0 for m in METHODS}
    best_tradeoff_counts = {m: 0 for m in METHODS}
    n_settings = 0
    per_setting_details = {}

    for ds in datasets:
        for ms in MAX_SPLITS_VALUES:
            setting = f"{ds}__ms{ms}"
            sub = summary[(summary["dataset"] == ds) & (summary["max_splits"] == ms)]
            if len(sub) == 0:
                continue
            n_settings += 1

            # Find Pareto frontier: not dominated on (lower arity, higher accuracy)
            on_frontier = []
            for _, row in sub.iterrows():
                dominated = False
                for _, other in sub.iterrows():
                    if (other["arity_mean"] < row["arity_mean"] and
                            other["accuracy_mean"] > row["accuracy_mean"]):
                        dominated = True
                        break
                if not dominated:
                    on_frontier.append(row["method"])
                    frontier_counts[row["method"]] += 1

            # Best tradeoff: frontier method with lowest arity (excluding axis_aligned)
            frontier_non_aa = [m for m in on_frontier if m != "axis_aligned"]
            if frontier_non_aa:
                best = min(frontier_non_aa,
                           key=lambda m: sub[sub["method"] == m]["arity_mean"].values[0])
                best_tradeoff_counts[best] += 1

            per_setting_details[setting] = {
                "on_frontier": on_frontier,
                "best_tradeoff": best if frontier_non_aa else None,
            }

    # Compute percentages
    frontier_table = {}
    for m in METHODS:
        frontier_table[m] = {
            "count": frontier_counts[m],
            "percentage": round(100 * frontier_counts[m] / n_settings, 1) if n_settings else 0,
            "best_tradeoff_count": best_tradeoff_counts[m],
            "best_tradeoff_pct": round(100 * best_tradeoff_counts[m] / n_settings, 1) if n_settings else 0,
        }

    logger.info(f"  Pareto analysis over {n_settings} settings")
    for m in METHODS:
        logger.info(f"    {m}: frontier {frontier_counts[m]}/{n_settings} "
                     f"({frontier_table[m]['percentage']}%), "
                     f"best_tradeoff {best_tradeoff_counts[m]}/{n_settings}")

    return {
        "n_settings": n_settings,
        "frontier_table": frontier_table,
        "per_setting_details": per_setting_details,
    }


def analysis_e_path_length(df: pd.DataFrame) -> dict:
    """(E) Path length analysis."""
    logger.info("Running Analysis E: Path Length Analysis")

    # Descriptive stats
    path_table = {}
    for (ds, method, ms), grp in df.groupby(["dataset", "method", "max_splits"]):
        key = f"{ds}__{method}__ms{ms}"
        path_table[key] = {
            "dataset": ds,
            "method": method,
            "max_splits": int(ms),
            "path_length_mean": round(float(grp["avg_path_length"].mean()), 4),
            "path_length_std": round(float(grp["avg_path_length"].std()), 4),
        }

    # Significance tests
    comparisons = {
        "unsigned_spectral_vs_random_oblique": ("unsigned_spectral", "random_oblique"),
        "unsigned_spectral_vs_axis_aligned": ("unsigned_spectral", "axis_aligned"),
        "unsigned_spectral_vs_hard_threshold": ("unsigned_spectral", "hard_threshold"),
    }
    sig_tests = {}
    for label, (m1, m2) in comparisons.items():
        logger.info(f"  Path length: {m1} vs {m2}")
        sig_tests[label] = pairwise_comparison(df, m1, m2, "avg_path_length")

    return {"descriptive": path_table, "significance_tests": sig_tests}


def analysis_f_cognitive_complexity(df: pd.DataFrame) -> dict:
    """(F) Cognitive complexity = avg_path_length * avg_split_arity."""
    logger.info("Running Analysis F: Cognitive Complexity")

    df = df.copy()
    df["cognitive_complexity"] = df["avg_path_length"] * df["avg_split_arity"]

    # Descriptive stats for all
    cc_table = {}
    for (ds, method, ms), grp in df.groupby(["dataset", "method", "max_splits"]):
        key = f"{ds}__{method}__ms{ms}"
        cc_table[key] = {
            "dataset": ds,
            "method": method,
            "max_splits": int(ms),
            "cc_mean": round(float(grp["cognitive_complexity"].mean()), 4),
            "cc_std": round(float(grp["cognitive_complexity"].std()), 4),
        }

    # Focus on high-dimensional datasets
    high_dim = df[df["dataset"].isin(HIGH_DIM_DATASETS)]

    # Wilcoxon on high-dim pooled
    us_vals = high_dim[high_dim["method"] == "unsigned_spectral"].sort_values(
        ["dataset", "max_splits", "fold"])["cognitive_complexity"].values
    ro_vals = high_dim[high_dim["method"] == "random_oblique"].sort_values(
        ["dataset", "max_splits", "fold"])["cognitive_complexity"].values

    high_dim_tests = {}
    if len(us_vals) == len(ro_vals) and len(us_vals) > 0:
        high_dim_tests["pooled_wilcoxon"] = safe_wilcoxon(us_vals, ro_vals)
        high_dim_tests["bootstrap_ci"] = bootstrap_ci(us_vals - ro_vals)
        high_dim_tests["cohens_d"] = cohens_d(us_vals, ro_vals)

        # Percentage reduction
        ro_mean = ro_vals.mean()
        us_mean = us_vals.mean()
        if ro_mean > 0:
            pct_reduction = (ro_mean - us_mean) / ro_mean * 100
        else:
            pct_reduction = 0.0
        high_dim_tests["pct_reduction"] = round(float(pct_reduction), 2)
        high_dim_tests["us_mean_cc"] = round(float(us_mean), 4)
        high_dim_tests["ro_mean_cc"] = round(float(ro_mean), 4)

    # Per high-dim dataset tests
    per_ds_tests = {}
    for ds in HIGH_DIM_DATASETS:
        ds_sub = df[df["dataset"] == ds]
        v1 = ds_sub[ds_sub["method"] == "unsigned_spectral"].sort_values(
            ["max_splits", "fold"])["cognitive_complexity"].values
        v2 = ds_sub[ds_sub["method"] == "random_oblique"].sort_values(
            ["max_splits", "fold"])["cognitive_complexity"].values
        if len(v1) == len(v2) and len(v1) > 0:
            ro_m = v2.mean()
            us_m = v1.mean()
            per_ds_tests[ds] = {
                "wilcoxon": safe_wilcoxon(v1, v2),
                "us_mean_cc": round(float(us_m), 4),
                "ro_mean_cc": round(float(ro_m), 4),
                "pct_reduction": round(float((ro_m - us_m) / ro_m * 100), 2) if ro_m > 0 else 0.0,
            }

    # Practical interpretation
    interpretation = (
        "Cognitive complexity = path_length x arity measures total features inspected per prediction. "
        "For a 4-split path: arity 1.8 => ~7.2 feature values inspected; arity 3.0 => ~12.0 feature values. "
        "Lower CC means the model is easier for domain experts to understand."
    )

    return {
        "descriptive": cc_table,
        "high_dim_analysis": {
            "datasets": HIGH_DIM_DATASETS,
            "pooled_tests": high_dim_tests,
            "per_dataset": per_ds_tests,
        },
        "interpretation": interpretation,
    }


# ---------------------------------------------------------------------------
# Build output schema
# ---------------------------------------------------------------------------

def build_output(raw_data: dict, df: pd.DataFrame, analyses: dict) -> dict:
    """Build output conforming to exp_eval_sol_out.json schema."""

    # --- metrics_agg: key summary statistics ---
    metrics_agg = {}

    # From analysis B: pooled arity Wilcoxon p-values
    for comp_key, comp in analyses["B_arity_significance"].items():
        safe_key = comp_key.replace(".", "_")
        pw = comp.get("pooled_wilcoxon", {})
        metrics_agg[f"arity_{safe_key}_wilcoxon_p"] = pw.get("p", 1.0)
        cd = comp.get("cohens_d", {})
        metrics_agg[f"arity_{safe_key}_cohens_d"] = cd.get("d", 0.0)

    # From analysis C: accuracy equivalence p-values and ROPE
    for comp_key, comp in analyses["C_accuracy_equivalence"].items():
        safe_key = comp_key.replace(".", "_")
        wil = comp.get("wilcoxon", {})
        metrics_agg[f"accuracy_{safe_key}_wilcoxon_p"] = wil.get("p", 1.0)
        rope = comp.get("bayesian_rope", {})
        metrics_agg[f"accuracy_{safe_key}_rope_p_equiv"] = rope.get("p_equivalent", 0.0)

    # From analysis D: Pareto frontier percentages
    ft = analyses["D_pareto"]["frontier_table"]
    for m in METHODS:
        metrics_agg[f"pareto_frontier_pct_{m}"] = ft[m]["percentage"]
        metrics_agg[f"pareto_best_tradeoff_pct_{m}"] = ft[m]["best_tradeoff_pct"]

    # From analysis F: cognitive complexity reduction
    hdim = analyses["F_cognitive_complexity"]["high_dim_analysis"]["pooled_tests"]
    metrics_agg["cc_high_dim_pct_reduction"] = hdim.get("pct_reduction", 0.0)
    metrics_agg["cc_high_dim_us_mean"] = hdim.get("us_mean_cc", 0.0)
    metrics_agg["cc_high_dim_ro_mean"] = hdim.get("ro_mean_cc", 0.0)

    # Round all metrics; replace None with 0.0 for schema compliance (metrics_agg requires numbers)
    cleaned = {}
    for k, v in metrics_agg.items():
        if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
            cleaned[k] = 0.0
        else:
            cleaned[k] = round(float(v), 6)
    metrics_agg = cleaned

    # --- datasets: one per real dataset, examples = per-fold results ---
    dataset_list = sorted(df["dataset"].unique())
    datasets_out = []

    for ds in dataset_list:
        ds_df = df[df["dataset"] == ds].sort_values(["method", "max_splits", "fold"])
        examples = []
        for _, row in ds_df.iterrows():
            method = row["method"]
            ms = int(row["max_splits"])
            fold = int(row["fold"])

            # Cognitive complexity
            cc = row["avg_path_length"] * row["avg_split_arity"]

            # Arity table lookup
            arity_key = f"{ds}__{method}__ms{ms}"
            arity_info = analyses["A_arity_table"].get(arity_key, {})

            # Use performance (balanced_accuracy or r2)
            perf = row["performance"]
            perf_val = round(float(perf), 6) if not pd.isna(perf) else 0.0

            input_str = json.dumps({
                "dataset": ds,
                "method": method,
                "max_splits": ms,
                "fold": fold,
                "n_features": int(row["n_features"]),
            })

            output_str = json.dumps({
                "performance": perf_val,
                "avg_split_arity": round(float(row["avg_split_arity"]), 4),
                "avg_path_length": round(float(row["avg_path_length"]), 4),
                "cognitive_complexity": round(float(cc), 4),
            })

            example = {
                "input": input_str,
                "output": output_str,
                "metadata_method": method,
                "metadata_max_splits": ms,
                "metadata_fold": fold,
                "metadata_n_features": int(row["n_features"]),
                "metadata_task_type": row.get("task_type", "unknown"),
                "eval_performance": perf_val,
                "eval_avg_split_arity": round(float(row["avg_split_arity"]), 4),
                "eval_avg_path_length": round(float(row["avg_path_length"]), 4),
                "eval_cognitive_complexity": round(float(cc), 4),
                "eval_fit_time_s": round(float(row["fit_time_s"]), 4),
            }
            examples.append(example)

        datasets_out.append({"dataset": ds, "examples": examples})

    # Build final metadata with all analysis results
    metadata = {
        "evaluation_name": "interpretability_tradeoff_evaluation",
        "description": (
            "Six-block evaluation of FIGS interpretability tradeoffs: "
            "(A) arity comparison, (B) arity significance, (C) accuracy equivalence, "
            "(D) Pareto frontier, (E) path length, (F) cognitive complexity."
        ),
        "methods": METHODS,
        "max_splits_values": MAX_SPLITS_VALUES,
        "n_datasets": len(dataset_list),
        "n_folds": 5,
        "n_bootstrap": N_BOOTSTRAP,
        "rope": ROPE,
        "high_dim_datasets": HIGH_DIM_DATASETS,
        "analyses": analyses,
    }

    return {
        "metadata": metadata,
        "metrics_agg": metrics_agg,
        "datasets": datasets_out,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@logger.catch
def main():
    logger.info(f"Loading data from {DATA_PATH}")
    raw_data = json.loads(DATA_PATH.read_text())

    results_per_fold = raw_data["metadata"]["results_per_fold"]
    logger.info(f"Loaded {len(results_per_fold)} per-fold results")

    df = pd.DataFrame(results_per_fold)
    logger.info(f"DataFrame shape: {df.shape}")
    logger.info(f"Datasets: {sorted(df['dataset'].unique())}")
    logger.info(f"Methods: {sorted(df['method'].unique())}")
    logger.info(f"Max splits: {sorted(df['max_splits'].unique())}")

    # Validate expected shape
    n_datasets = df["dataset"].nunique()
    n_methods = df["method"].nunique()
    n_splits = df["max_splits"].nunique()
    n_folds = df.groupby(["dataset", "method", "max_splits"]).size().max()
    logger.info(f"Grid: {n_datasets} datasets x {n_methods} methods x {n_splits} max_splits x {n_folds} folds = {len(df)} rows")

    # Create unified performance metric: balanced_accuracy for classification, r2 for regression
    df["performance"] = df["balanced_accuracy"]
    regression_mask = df["balanced_accuracy"].isna() & df["r2"].notna()
    df.loc[regression_mask, "performance"] = df.loc[regression_mask, "r2"]
    logger.info(f"Performance column: {df['performance'].notna().sum()} non-null "
                f"({regression_mask.sum()} from r2, {(~regression_mask & df['performance'].notna()).sum()} from balanced_accuracy)")

    # Run all 6 analyses
    analyses = {}
    analyses["A_arity_table"] = analysis_a_arity_table(df)
    analyses["B_arity_significance"] = analysis_b_arity_significance(df)
    analyses["C_accuracy_equivalence"] = analysis_c_accuracy_equivalence(df)
    analyses["D_pareto"] = analysis_d_pareto(df)
    analyses["E_path_length"] = analysis_e_path_length(df)
    analyses["F_cognitive_complexity"] = analysis_f_cognitive_complexity(df)

    # Build output
    output = build_output(raw_data, df, analyses)

    # Sanitize: replace any residual NaN/Inf with None for JSON compliance
    def _sanitize(obj):
        if isinstance(obj, dict):
            return {k: _sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_sanitize(v) for v in obj]
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            v = float(obj)
            return None if (math.isnan(v) or math.isinf(v)) else v
        if isinstance(obj, np.ndarray):
            return _sanitize(obj.tolist())
        return obj

    output = _sanitize(output)

    # Write output
    OUTPUT_PATH.write_text(json.dumps(output, indent=2))
    logger.info(f"Wrote eval output to {OUTPUT_PATH}")
    logger.info(f"Output size: {OUTPUT_PATH.stat().st_size / 1024:.1f} KB")
    logger.info(f"metrics_agg keys: {list(output['metrics_agg'].keys())}")
    logger.info(f"Number of datasets in output: {len(output['datasets'])}")
    total_examples = sum(len(d['examples']) for d in output['datasets'])
    logger.info(f"Total examples: {total_examples}")

    # Log key results
    logger.info("=== KEY RESULTS ===")
    ma = output["metrics_agg"]
    logger.info(f"Arity: unsigned_spectral vs random_oblique Wilcoxon p={ma.get('arity_unsigned_spectral_vs_random_oblique_wilcoxon_p', 'N/A')}")
    logger.info(f"Arity: unsigned_spectral vs random_oblique Cohen's d={ma.get('arity_unsigned_spectral_vs_random_oblique_cohens_d', 'N/A')}")
    logger.info(f"Accuracy equiv: unsigned_spectral vs random_oblique p={ma.get('accuracy_unsigned_spectral_vs_random_oblique_wilcoxon_p', 'N/A')}")
    logger.info(f"Accuracy ROPE P(equiv)={ma.get('accuracy_unsigned_spectral_vs_random_oblique_rope_p_equiv', 'N/A')}")
    logger.info(f"Pareto frontier %: unsigned_spectral={ma.get('pareto_frontier_pct_unsigned_spectral', 'N/A')}")
    logger.info(f"CC high-dim reduction: {ma.get('cc_high_dim_pct_reduction', 'N/A')}%")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Definitive Main Results Table: 8 Methods x 8 Datasets with Statistical Tests.

Loads per-fold results from two dependency experiments (5 FIGS variants + 3 baselines),
selects best hyperparameters for FIGS, merges into a unified 8x8 matrix, and computes
Friedman/Nemenyi, Bayesian signed-rank, Cohen's d, winner analysis, and ranking --
all output as JSON arrays ready for LaTeX.
"""

import json
import sys
import os
import math
import gc
import resource
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from scipy.stats import friedmanchisquare, rankdata
import scikit_posthocs as sp
from loguru import logger
import psutil

# ─── Setup ───────────────────────────────────────────────────────────────
WORK_DIR = Path(__file__).parent
LOG_DIR = WORK_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add(str(LOG_DIR / "run.log"), rotation="30 MB", level="DEBUG")


# ─── Hardware Detection ──────────────────────────────────────────────────
def _detect_cpus() -> int:
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
    for p in ["/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"]:
        try:
            v = Path(p).read_text().strip()
            if v != "max" and int(v) < 1_000_000_000_000:
                return int(v) / 1e9
        except (FileNotFoundError, ValueError):
            pass
    return None


NUM_CPUS = _detect_cpus()
TOTAL_RAM_GB = _container_ram_gb() or 57.0

# Memory limits - 8GB budget (data < 25MB total)
_avail = psutil.virtual_memory().available
RAM_BUDGET = int(8 * 1024**3)
assert RAM_BUDGET < _avail, f"Budget {RAM_BUDGET / 1e9:.1f}GB > available {_avail / 1e9:.1f}GB"
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET * 3, RAM_BUDGET * 3))
logger.info(f"Hardware: {NUM_CPUS} CPUs, {TOTAL_RAM_GB:.1f} GB RAM, budget={RAM_BUDGET / 1e9:.1f} GB")

# ─── Constants ───────────────────────────────────────────────────────────
FIGS_METHODS = [
    "axis_aligned", "random_oblique", "unsigned_spectral",
    "signed_spectral", "hard_threshold",
]
BASELINE_METHODS = ["ebm", "random_forest", "linear"]
ALL_METHODS = FIGS_METHODS + BASELINE_METHODS
N_METHODS = len(ALL_METHODS)  # 8

DATASETS = [
    "electricity", "adult", "california_housing", "jannis",
    "higgs_small", "eye_movements", "credit", "miniboone",
]
CLASSIFICATION_DATASETS = [d for d in DATASETS if d != "california_housing"]
N_DATASETS = len(DATASETS)  # 8

MAX_SPLITS_VALUES = [5, 10, 20]
ROPE = 0.01
ALPHA = 0.05

DEP1_PATH = Path(
    "/ai-inventor/aii_pipeline/runs/jamnik-sgfigs-pid-v2"
    "/3_invention_loop/iter_5/gen_art/exp_id1_it5__opus"
)
DEP2_PATH = Path(
    "/ai-inventor/aii_pipeline/runs/jamnik-sgfigs-pid-v2"
    "/3_invention_loop/iter_4/gen_art/exp_id2_it4__opus"
)


# ─── Helpers ─────────────────────────────────────────────────────────────
def get_primary_metric_name(dataset: str) -> str:
    """Return the primary metric field name for a dataset."""
    return "r2" if dataset == "california_housing" else "balanced_accuracy"


def is_classification(dataset: str) -> bool:
    return dataset != "california_housing"


# ─── Data Loading ────────────────────────────────────────────────────────
def load_figs_data() -> dict:
    """Load FIGS experiment (dep1): 5 methods x 8 datasets x 3 max_splits x 5 folds."""
    path = DEP1_PATH / "full_method_out.json"
    logger.info(f"Loading FIGS data from {path} ({path.stat().st_size / 1024:.1f} KB)")
    data = json.loads(path.read_text())
    n_folds = len(data["metadata"]["results_per_fold"])
    n_summary = len(data["metadata"]["results_summary"])
    logger.info(f"FIGS: {n_folds} per-fold results, {n_summary} summary entries")
    return data


def load_baselines_data() -> dict:
    """Load baselines experiment (dep2): EBM, RF, Linear x 8 datasets x 5 folds."""
    path = DEP2_PATH / "full_method_out.json"
    logger.info(f"Loading baselines data from {path} ({path.stat().st_size / 1024:.1f} KB)")
    data = json.loads(path.read_text())
    n_ds = len(data["metadata"]["per_dataset_results"])
    logger.info(f"Baselines: {n_ds} datasets")
    return data


def extract_figs_per_fold(figs_data: dict) -> pd.DataFrame:
    """Extract FIGS per-fold results into a tidy DataFrame."""
    rows = []
    for r in figs_data["metadata"]["results_per_fold"]:
        dataset = r["dataset"]
        task_type = r.get("task_type", "classification")

        # Primary metric: r2 for regression, balanced_accuracy for classification
        if dataset == "california_housing":
            primary_val = r.get("r2")
            if primary_val is None:
                primary_val = r.get("balanced_accuracy")
        else:
            primary_val = r.get("balanced_accuracy")

        rows.append({
            "dataset": dataset,
            "method": r["method"],
            "max_splits": r["max_splits"],
            "fold": r["fold"],
            "task_type": task_type,
            "primary_metric": primary_val,
            "balanced_accuracy": r.get("balanced_accuracy"),
            "auc": r.get("auc"),
            "r2": r.get("r2"),
            "fit_time_s": r.get("fit_time_s"),
        })

    df = pd.DataFrame(rows)
    logger.info(
        f"FIGS per-fold: {len(df)} rows, "
        f"methods={sorted(df['method'].unique().tolist())}, "
        f"datasets={sorted(df['dataset'].unique().tolist())}"
    )
    return df


def extract_baselines_per_fold(baselines_data: dict) -> pd.DataFrame:
    """Extract baselines per-fold results into a tidy DataFrame."""
    rows = []
    per_ds = baselines_data["metadata"]["per_dataset_results"]
    for dataset, methods_dict in per_ds.items():
        task_type = "regression" if dataset == "california_housing" else "classification"

        for method, mdata in methods_dict.items():
            for fr in mdata["fold_results"]:
                if fr.get("status") != "success":
                    logger.warning(f"Skipping failed fold: {dataset}/{method}/fold{fr.get('fold')}")
                    continue

                if dataset == "california_housing":
                    primary_val = fr.get("r2")
                else:
                    primary_val = fr.get("balanced_accuracy")

                rows.append({
                    "dataset": dataset,
                    "method": method,
                    "max_splits": None,
                    "fold": fr["fold"],
                    "task_type": task_type,
                    "primary_metric": primary_val,
                    "balanced_accuracy": fr.get("balanced_accuracy"),
                    "auc": fr.get("auc"),
                    "r2": fr.get("r2"),
                    "fit_time_s": fr.get("fit_time"),
                })

    df = pd.DataFrame(rows)
    logger.info(
        f"Baselines per-fold: {len(df)} rows, "
        f"methods={sorted(df['method'].unique().tolist())}, "
        f"datasets={sorted(df['dataset'].unique().tolist())}"
    )
    return df


# ─── Hyperparameter Selection ────────────────────────────────────────────
def select_best_max_splits(figs_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Select best max_splits per (dataset, method) by mean primary metric.

    Returns:
        best_df: DataFrame filtered to best max_splits per (dataset, method, fold)
        best_splits: dict mapping (dataset, method) -> best max_splits value
    """
    agg = (
        figs_df.groupby(["dataset", "method", "max_splits"])["primary_metric"]
        .mean()
        .reset_index()
    )

    best_splits = {}
    idx_best = agg.groupby(["dataset", "method"])["primary_metric"].idxmax()
    for idx in idx_best:
        row = agg.loc[idx]
        key = (row["dataset"], row["method"])
        best_splits[key] = int(row["max_splits"])

    logger.info("Best max_splits selected:")
    for (ds, method), ms in sorted(best_splits.items()):
        logger.debug(f"  {ds}/{method}: max_splits={ms}")

    # Filter to keep only best max_splits per (dataset, method)
    keep_mask = pd.Series(False, index=figs_df.index)
    for (ds, method), ms in best_splits.items():
        mask = (
            (figs_df["dataset"] == ds)
            & (figs_df["method"] == method)
            & (figs_df["max_splits"] == ms)
        )
        keep_mask |= mask

    best_df = figs_df[keep_mask].copy()
    logger.info(f"After selection: {len(best_df)} rows (from {len(figs_df)})")
    return best_df, best_splits


# ─── Unified Results Matrix ──────────────────────────────────────────────
def build_unified_fold_results(
    figs_best_df: pd.DataFrame, baselines_df: pd.DataFrame
) -> pd.DataFrame:
    """Merge FIGS (best splits) and baselines into one tidy DataFrame."""
    cols = [
        "dataset", "method", "fold", "task_type",
        "primary_metric", "balanced_accuracy", "auc", "r2", "fit_time_s",
    ]

    figs_sub = figs_best_df[cols].copy()
    baselines_sub = baselines_df[cols].copy()
    unified = pd.concat([figs_sub, baselines_sub], ignore_index=True)

    # Verify completeness
    for ds in DATASETS:
        for method in ALL_METHODS:
            count = len(unified[(unified["dataset"] == ds) & (unified["method"] == method)])
            if count != 5:
                logger.warning(f"Expected 5 folds for {ds}/{method}, got {count}")

    logger.info(
        f"Unified: {len(unified)} rows, "
        f"{unified['dataset'].nunique()} datasets, "
        f"{unified['method'].nunique()} methods"
    )
    return unified


def build_mean_matrix(unified_df: pd.DataFrame) -> pd.DataFrame:
    """Build matrix of mean primary metric: rows=datasets, columns=methods."""
    pivot = (
        unified_df.groupby(["dataset", "method"])["primary_metric"]
        .mean()
        .unstack("method")
    )
    pivot = pivot.reindex(index=DATASETS, columns=ALL_METHODS)
    logger.info(f"Mean matrix shape: {pivot.shape}")
    logger.info(f"\n{pivot.to_string()}")
    return pivot


# ─── Metric A: Main Results Table ────────────────────────────────────────
def compute_main_results_table(
    unified_df: pd.DataFrame, best_splits: dict
) -> list[dict]:
    """8x8 table: mean +/- std for primary metric and AUC, plus best max_splits."""
    table = []
    for ds in DATASETS:
        for method in ALL_METHODS:
            mask = (unified_df["dataset"] == ds) & (unified_df["method"] == method)
            subset = unified_df[mask]

            if len(subset) == 0:
                logger.warning(f"No data for {ds}/{method}")
                continue

            primary_vals = subset["primary_metric"].dropna().values
            auc_vals = subset["auc"].dropna().values

            entry = {
                "dataset": ds,
                "method": method,
                "task_type": "regression" if ds == "california_housing" else "classification",
                "primary_metric_name": get_primary_metric_name(ds),
                "primary_mean": round(float(np.mean(primary_vals)), 6),
                "primary_std": round(
                    float(np.std(primary_vals, ddof=1)) if len(primary_vals) > 1 else 0.0, 6
                ),
                "n_folds": int(len(primary_vals)),
            }

            if is_classification(ds) and len(auc_vals) > 0:
                entry["auc_mean"] = round(float(np.mean(auc_vals)), 6)
                entry["auc_std"] = round(
                    float(np.std(auc_vals, ddof=1)) if len(auc_vals) > 1 else 0.0, 6
                )
            else:
                entry["auc_mean"] = None
                entry["auc_std"] = None

            if method in FIGS_METHODS:
                entry["best_max_splits"] = best_splits.get((ds, method))
            else:
                entry["best_max_splits"] = None

            table.append(entry)

    logger.info(f"Main results table: {len(table)} entries")
    return table


# ─── Metric B: Friedman + Nemenyi ────────────────────────────────────────
def compute_nemenyi_cd(n_methods: int, n_datasets: int, alpha: float = 0.05) -> float:
    """Compute Nemenyi critical difference.

    CD = (q_alpha / sqrt(2)) * sqrt(k*(k+1)/(6*N))
    where q_alpha is from the Studentized Range distribution.
    """
    try:
        from scipy.stats import studentized_range
        q_alpha = studentized_range.ppf(1 - alpha, n_methods, 1e6)
        cd = (q_alpha / np.sqrt(2)) * np.sqrt(
            n_methods * (n_methods + 1) / (6 * n_datasets)
        )
    except Exception:
        logger.warning("scipy studentized_range failed, using table lookup")
        # Fallback: manual critical values q_alpha/(sqrt(2)) for common k values
        # From Demsar 2006 Table 5, alpha=0.05
        nemenyi_q = {
            3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850,
            7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164,
        }
        q_val = nemenyi_q.get(n_methods, 3.031)
        cd = q_val * np.sqrt(n_methods * (n_methods + 1) / (6 * n_datasets))
    return cd


def compute_friedman_nemenyi(
    mean_matrix: pd.DataFrame, label: str = "all"
) -> dict:
    """Friedman test + Nemenyi post-hoc on per-dataset means."""
    logger.info(f"Computing Friedman/Nemenyi ({label}): shape={mean_matrix.shape}")

    n_ds = mean_matrix.shape[0]
    n_meth = mean_matrix.shape[1]

    # Average ranks (higher metric = rank 1 = best)
    ranks = mean_matrix.rank(axis=1, ascending=False)
    avg_ranks = ranks.mean(axis=0)

    logger.info(f"Average ranks ({label}):")
    for method in sorted(avg_ranks.index, key=lambda m: avg_ranks[m]):
        logger.info(f"  {method}: {avg_ranks[method]:.3f}")

    # Friedman test
    cols = [mean_matrix[m].values for m in mean_matrix.columns]
    chi2, p_value = friedmanchisquare(*cols)
    logger.info(f"Friedman ({label}): chi2={chi2:.4f}, p={p_value:.6f}")

    # Nemenyi post-hoc pairwise p-values
    nemenyi_pvals_dict = {}
    try:
        nemenyi_result = sp.posthoc_nemenyi_friedman(mean_matrix.values)
        nemenyi_result.index = mean_matrix.columns
        nemenyi_result.columns = mean_matrix.columns
        for i, m1 in enumerate(mean_matrix.columns):
            for j, m2 in enumerate(mean_matrix.columns):
                if i < j:
                    nemenyi_pvals_dict[f"{m1}_vs_{m2}"] = round(
                        float(nemenyi_result.loc[m1, m2]), 6
                    )
    except Exception:
        logger.exception("Nemenyi post-hoc failed")

    # Nemenyi critical difference
    cd = compute_nemenyi_cd(n_meth, n_ds, ALPHA)
    logger.info(f"Nemenyi CD ({label}): {cd:.4f}")

    # Significance matrix: True if rank difference exceeds CD
    sig_matrix = {}
    for m1 in mean_matrix.columns:
        for m2 in mean_matrix.columns:
            if m1 < m2:
                rank_diff = abs(avg_ranks[m1] - avg_ranks[m2])
                sig_matrix[f"{m1}_vs_{m2}"] = bool(rank_diff > cd)

    result = {
        "friedman_chi2": round(float(chi2), 6),
        "friedman_pvalue": float(p_value),
        "significant": bool(p_value < ALPHA),
        "n_datasets": n_ds,
        "n_methods": n_meth,
        "alpha": ALPHA,
        "avg_ranks": {m: round(float(avg_ranks[m]), 4) for m in mean_matrix.columns},
        "nemenyi_cd": round(float(cd), 4),
        "nemenyi_pairwise_pvalues": nemenyi_pvals_dict,
        "significance_matrix": sig_matrix,
        "cd_diagram_data": {
            "avg_ranks": {m: round(float(avg_ranks[m]), 4) for m in mean_matrix.columns},
            "cd": round(float(cd), 4),
            "n_datasets": n_ds,
            "n_methods": n_meth,
        },
    }
    return result


# ─── Metric C: Bayesian Signed-Rank ─────────────────────────────────────
def compute_bayesian_tests(mean_matrix: pd.DataFrame) -> dict:
    """Bayesian signed-rank tests for 4 key pairwise comparisons."""
    comparisons = [
        ("unsigned_spectral", "ebm"),
        ("unsigned_spectral", "random_forest"),
        ("unsigned_spectral", "random_oblique"),
        ("unsigned_spectral", "axis_aligned"),
    ]

    results = {}

    try:
        from baycomp import SignedRankTest
        baycomp_available = True
    except ImportError:
        logger.warning("baycomp not importable, trying alternative API")
        baycomp_available = False

    for left, right in comparisons:
        key = f"{left}_vs_{right}"
        try:
            x = mean_matrix[left].values
            y = mean_matrix[right].values

            if baycomp_available:
                probs = SignedRankTest.probs(x, y, rope=ROPE)
                p_left, p_rope, p_right = float(probs[0]), float(probs[1]), float(probs[2])
            else:
                import baycomp
                probs = baycomp.two_on_multiple(x, y, rope=ROPE)
                p_left, p_rope, p_right = float(probs[0]), float(probs[1]), float(probs[2])

            results[key] = {
                "p_left": round(p_left, 6),
                "p_rope": round(p_rope, 6),
                "p_right": round(p_right, 6),
                "rope": ROPE,
                "interpretation": (
                    f"P({left} > {right}) = {p_left:.4f}, "
                    f"P(equivalent) = {p_rope:.4f}, "
                    f"P({right} > {left}) = {p_right:.4f}"
                ),
            }
            logger.info(
                f"Bayesian {left} vs {right}: "
                f"P(left)={p_left:.4f}, P(rope)={p_rope:.4f}, P(right)={p_right:.4f}"
            )
        except Exception:
            logger.exception(f"Bayesian test failed for {key}")
            results[key] = {
                "p_left": 0.0, "p_rope": 0.0, "p_right": 0.0,
                "rope": ROPE, "error": "computation failed",
            }

    return results


# ─── Metric D: Cohen's d ────────────────────────────────────────────────
def compute_cohens_d(unified_df: pd.DataFrame) -> dict:
    """Cohen's d for 5 FIGS x 3 baselines = 15 pairs.

    Uses rank-normalized per-fold scores: for each (dataset, fold), ranks all 8
    methods, then pools ranks across datasets and computes Cohen's d.
    Positive d = FIGS method has *better* (lower) average rank.
    """
    logger.info("Computing Cohen's d effect sizes (rank-normalized)")

    # Build per-fold per-method primary scores
    # shape: dict[(dataset, method)] -> list of per-fold values sorted by fold
    fold_scores = {}
    for ds in DATASETS:
        for method in ALL_METHODS:
            mask = (unified_df["dataset"] == ds) & (unified_df["method"] == method)
            vals = unified_df[mask].sort_values("fold")["primary_metric"].values
            fold_scores[(ds, method)] = vals

    # Rank within each (dataset, fold) combination
    # Higher primary metric = rank 1 (best)
    method_ranks = defaultdict(list)  # method -> list of ranks
    for ds in DATASETS:
        n_folds = 5
        for fold_idx in range(n_folds):
            fold_vals = {}
            for method in ALL_METHODS:
                arr = fold_scores.get((ds, method), np.array([]))
                if fold_idx < len(arr):
                    fold_vals[method] = arr[fold_idx]

            if not fold_vals:
                continue

            # Rank: higher value = rank 1
            methods_sorted = sorted(fold_vals.keys(), key=lambda m: fold_vals[m], reverse=True)
            for rank, method in enumerate(methods_sorted, 1):
                method_ranks[method].append(rank)

    # Compute Cohen's d for each FIGS-vs-baseline pair
    results = {}
    for figs_method in FIGS_METHODS:
        for baseline in BASELINE_METHODS:
            pair_key = f"{figs_method}_vs_{baseline}"

            figs_arr = np.array(method_ranks.get(figs_method, []), dtype=float)
            base_arr = np.array(method_ranks.get(baseline, []), dtype=float)

            if len(figs_arr) < 2 or len(base_arr) < 2:
                results[pair_key] = None
                continue

            n1, n2 = len(figs_arr), len(base_arr)
            mean1, mean2 = np.mean(figs_arr), np.mean(base_arr)
            var1 = np.var(figs_arr, ddof=1)
            var2 = np.var(base_arr, ddof=1)
            pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

            if pooled_std < 1e-12:
                d = 0.0
            else:
                # Positive d = FIGS better (lower mean rank)
                d = (mean2 - mean1) / pooled_std

            abs_d = abs(d)
            if abs_d < 0.2:
                magnitude = "negligible"
            elif abs_d < 0.5:
                magnitude = "small"
            elif abs_d < 0.8:
                magnitude = "medium"
            else:
                magnitude = "large"

            results[pair_key] = {
                "cohens_d": round(float(d), 4),
                "magnitude": magnitude,
                "figs_mean_rank": round(float(mean1), 4),
                "baseline_mean_rank": round(float(mean2), 4),
                "pooled_std": round(float(pooled_std), 4),
                "n_observations": int(n1),
            }
            logger.debug(f"  {pair_key}: d={d:.4f} ({magnitude})")

    logger.info(f"Cohen's d: computed {sum(1 for v in results.values() if v)} / {len(results)} pairs")
    return results


# ─── Metric E: Winner Analysis ──────────────────────────────────────────
def compute_winner_analysis(
    mean_matrix: pd.DataFrame, clustering_info: dict
) -> dict:
    """Per-dataset winner + pattern analysis (n_features vs winner)."""
    logger.info("Computing winner analysis")

    winners = {}
    for ds in mean_matrix.index:
        row = mean_matrix.loc[ds]
        winner = row.idxmax()
        winners[ds] = {
            "winner": winner,
            "score": round(float(row[winner]), 6),
            "all_scores": {
                m: round(float(row[m]), 6)
                for m in mean_matrix.columns
                if pd.notna(row[m])
            },
        }
        logger.info(f"  {ds}: winner={winner} ({row[winner]:.4f})")

    # Win counts
    win_counts = defaultdict(int)
    for ds, info in winners.items():
        win_counts[info["winner"]] += 1
    # Ensure all methods have entries (even 0)
    for m in ALL_METHODS:
        if m not in win_counts:
            win_counts[m] = 0

    # Pattern analysis: n_features vs winner
    pattern = []
    for ds, info in winners.items():
        ci = clustering_info.get(ds, {})
        n_features = ci.get("n_valid_features")
        pattern.append({
            "dataset": ds,
            "winner": info["winner"],
            "n_features": n_features,
            "is_spectral_winner": info["winner"] in ("unsigned_spectral", "signed_spectral"),
        })

    pattern.sort(key=lambda x: x.get("n_features") or 0)

    # High-dimensionality analysis
    n_spectral_total = sum(1 for p in pattern if p["is_spectral_winner"])
    high_dim = [p for p in pattern if (p.get("n_features") or 0) >= 20]
    n_spectral_high_dim = sum(1 for p in high_dim if p["is_spectral_winner"])

    result = {
        "per_dataset_winners": winners,
        "win_counts": dict(win_counts),
        "feature_pattern": pattern,
        "n_spectral_wins_total": n_spectral_total,
        "n_spectral_wins_high_dim": n_spectral_high_dim,
        "n_high_dim_datasets": len(high_dim),
    }
    return result


# ─── Metric F: Method Ranking ───────────────────────────────────────────
def compute_rankings(
    mean_matrix: pd.DataFrame, classif_matrix: pd.DataFrame
) -> dict:
    """Average rank across all 8 datasets and classification-only 7."""
    logger.info("Computing method rankings")

    ranks_all = mean_matrix.rank(axis=1, ascending=False)
    avg_rank_all = ranks_all.mean(axis=0)

    ranks_classif = classif_matrix.rank(axis=1, ascending=False)
    avg_rank_classif = ranks_classif.mean(axis=0)

    result = {
        "all_8_datasets": {
            "avg_ranks": {m: round(float(avg_rank_all[m]), 4) for m in mean_matrix.columns},
            "ranking": sorted(mean_matrix.columns, key=lambda m: avg_rank_all[m]),
        },
        "classification_7_datasets": {
            "avg_ranks": {
                m: round(float(avg_rank_classif[m]), 4) for m in classif_matrix.columns
            },
            "ranking": sorted(classif_matrix.columns, key=lambda m: avg_rank_classif[m]),
        },
    }

    logger.info(f"Ranking (all 8): {result['all_8_datasets']['ranking']}")
    logger.info(f"Ranking (classif 7): {result['classification_7_datasets']['ranking']}")
    return result


# ─── Additional Metrics ──────────────────────────────────────────────────
def compute_improvement_analysis(mean_matrix: pd.DataFrame) -> dict:
    """Per-dataset improvement of unsigned_spectral over each baseline."""
    improvements = {}
    for ds in mean_matrix.index:
        us_score = mean_matrix.loc[ds, "unsigned_spectral"]
        for baseline in BASELINE_METHODS:
            bl_score = mean_matrix.loc[ds, baseline]
            diff = us_score - bl_score
            if bl_score != 0:
                pct = (diff / abs(bl_score)) * 100
            else:
                pct = 0.0
            improvements[f"{ds}_unsigned_spectral_vs_{baseline}"] = {
                "absolute_diff": round(float(diff), 6),
                "percent_improvement": round(float(pct), 4),
            }
    return improvements


def compute_top3_analysis(mean_matrix: pd.DataFrame) -> dict:
    """For each method, count how many datasets it appears in top-3."""
    top3_counts = defaultdict(int)
    for ds in mean_matrix.index:
        row = mean_matrix.loc[ds]
        top3 = row.nlargest(3).index.tolist()
        for m in top3:
            top3_counts[m] += 1
    return {m: top3_counts.get(m, 0) for m in ALL_METHODS}


# ─── Merge Examples for Output ───────────────────────────────────────────
def merge_examples(figs_data: dict, baselines_data: dict) -> list[dict]:
    """Merge per-example predictions from both experiments into output datasets."""
    logger.info("Merging per-example predictions for output")

    # Build lookup from dep1 (FIGS): (dataset, input_str) -> predict fields
    figs_lookup = {}
    if "datasets" in figs_data:
        for ds_entry in figs_data["datasets"]:
            ds_name = ds_entry["dataset"]
            for ex in ds_entry.get("examples", []):
                key = (ds_name, ex.get("input", ""))
                figs_lookup[key] = {
                    k: v for k, v in ex.items()
                    if k.startswith("predict_") or k.startswith("metadata_")
                }
    logger.info(f"FIGS lookup: {len(figs_lookup)} examples")

    # Use dep2 (baselines) as base — 2000 examples per dataset
    output_datasets = []
    if "datasets" in baselines_data:
        for ds_entry in baselines_data["datasets"]:
            ds_name = ds_entry["dataset"]
            merged_examples = []

            for ex in ds_entry.get("examples", []):
                merged = {
                    "input": ex.get("input", ""),
                    "output": str(ex.get("output", "")),
                }

                # Add metadata fields
                for k, v in ex.items():
                    if k.startswith("metadata_"):
                        merged[k] = v

                # Add baseline predictions
                for k, v in ex.items():
                    if k.startswith("predict_"):
                        merged[k] = str(v)

                # Merge FIGS predictions if available
                figs_key = (ds_name, ex.get("input", ""))
                if figs_key in figs_lookup:
                    for k, v in figs_lookup[figs_key].items():
                        if k.startswith("predict_") and k not in merged:
                            merged[k] = str(v)
                        elif k.startswith("metadata_") and k not in merged:
                            merged[k] = v

                # Add per-example correctness eval metrics
                ground_truth = str(ex.get("output", ""))
                for k, v in list(merged.items()):
                    if k.startswith("predict_"):
                        method_name = k.replace("predict_", "")
                        eval_key = f"eval_correct_{method_name}"
                        try:
                            # For classification: exact match
                            # For regression: use a tolerance
                            if ds_name == "california_housing":
                                gt_val = float(ground_truth)
                                pred_val = float(str(v))
                                # Within 10% relative error
                                rel_err = abs(pred_val - gt_val) / max(abs(gt_val), 1e-8)
                                merged[eval_key] = 1 if rel_err < 0.1 else 0
                            else:
                                merged[eval_key] = 1 if str(v) == ground_truth else 0
                        except (ValueError, TypeError):
                            merged[eval_key] = 0

                merged_examples.append(merged)

            output_datasets.append({
                "dataset": ds_name,
                "examples": merged_examples,
            })

    total_examples = sum(len(d["examples"]) for d in output_datasets)
    logger.info(f"Merged: {len(output_datasets)} datasets, {total_examples} total examples")
    return output_datasets


# ─── Build Output JSON ───────────────────────────────────────────────────
def build_output(
    main_table: list[dict],
    friedman_all: dict,
    friedman_classif: dict,
    bayesian_tests: dict,
    cohens_d: dict,
    winner_analysis: dict,
    rankings: dict,
    best_splits: dict,
    merged_datasets: list[dict],
    clustering_info: dict,
    improvement_analysis: dict,
    top3_counts: dict,
) -> dict:
    """Build output JSON following exp_eval_sol_out schema."""
    logger.info("Building output JSON")

    # ── metrics_agg: flat numeric values only ──
    metrics_agg = {}

    # Friedman stats (all 8)
    metrics_agg["friedman_chi2_all_8"] = friedman_all["friedman_chi2"]
    metrics_agg["friedman_pvalue_all_8"] = friedman_all["friedman_pvalue"]
    metrics_agg["nemenyi_cd_all_8"] = friedman_all["nemenyi_cd"]

    # Friedman stats (classification 7)
    metrics_agg["friedman_chi2_classif_7"] = friedman_classif["friedman_chi2"]
    metrics_agg["friedman_pvalue_classif_7"] = friedman_classif["friedman_pvalue"]
    metrics_agg["nemenyi_cd_classif_7"] = friedman_classif["nemenyi_cd"]

    # Average ranks (all 8)
    for method, rank in friedman_all["avg_ranks"].items():
        metrics_agg[f"rank_all_{method}"] = rank

    # Average ranks (classification 7)
    for method, rank in friedman_classif["avg_ranks"].items():
        metrics_agg[f"rank_classif_{method}"] = rank

    # Bayesian test probabilities
    for comp_name, comp_data in bayesian_tests.items():
        safe = comp_name.replace(" ", "_")
        if isinstance(comp_data.get("p_left"), (int, float)):
            metrics_agg[f"bayes_{safe}_p_left"] = comp_data["p_left"]
            metrics_agg[f"bayes_{safe}_p_rope"] = comp_data["p_rope"]
            metrics_agg[f"bayes_{safe}_p_right"] = comp_data["p_right"]

    # Cohen's d values
    for pair_name, pair_data in cohens_d.items():
        if pair_data is not None and isinstance(pair_data, dict):
            metrics_agg[f"cohens_d_{pair_name}"] = pair_data["cohens_d"]

    # Per-dataset per-method primary metric means
    for entry in main_table:
        ds = entry["dataset"]
        method = entry["method"]
        metrics_agg[f"primary_{ds}_{method}"] = entry["primary_mean"]
        metrics_agg[f"primary_std_{ds}_{method}"] = entry["primary_std"]
        if entry.get("auc_mean") is not None:
            metrics_agg[f"auc_{ds}_{method}"] = entry["auc_mean"]

    # Win counts
    for method, count in winner_analysis.get("win_counts", {}).items():
        metrics_agg[f"n_wins_{method}"] = count

    # Top-3 counts
    for method, count in top3_counts.items():
        metrics_agg[f"n_top3_{method}"] = count

    # ── Ensure all values are numeric ──
    clean_agg = {}
    for k, v in metrics_agg.items():
        if isinstance(v, (int, float)) and not (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
            clean_agg[k] = v
        elif v is None:
            clean_agg[k] = 0.0
        else:
            try:
                fv = float(v)
                if np.isnan(fv) or np.isinf(fv):
                    clean_agg[k] = 0.0
                else:
                    clean_agg[k] = fv
            except (ValueError, TypeError):
                logger.warning(f"Dropping non-numeric metrics_agg key: {k}={v}")

    # ── metadata: rich structured data ──
    metadata = {
        "evaluation_name": "definitive_main_results_8x8",
        "description": (
            "8 methods (5 FIGS + 3 baselines) x 8 datasets with "
            "Friedman/Nemenyi, Bayesian signed-rank, Cohen's d, "
            "winner analysis, and ranking"
        ),
        "methods": ALL_METHODS,
        "figs_methods": FIGS_METHODS,
        "baseline_methods": BASELINE_METHODS,
        "datasets": DATASETS,
        "classification_datasets": CLASSIFICATION_DATASETS,
        "alpha": ALPHA,
        "rope": ROPE,
        "main_results_table": main_table,
        "best_max_splits": {
            f"{ds}_{method}": ms for (ds, method), ms in best_splits.items()
        },
        "friedman_all": friedman_all,
        "friedman_classif": friedman_classif,
        "bayesian_tests": bayesian_tests,
        "cohens_d": cohens_d,
        "winner_analysis": winner_analysis,
        "rankings": rankings,
        "improvement_analysis": improvement_analysis,
        "top3_counts": top3_counts,
        "clustering_info": clustering_info,
    }

    output = {
        "metadata": metadata,
        "metrics_agg": clean_agg,
        "datasets": merged_datasets,
    }

    logger.info(f"Output: {len(clean_agg)} aggregate metrics, {len(merged_datasets)} datasets")
    return output


# ─── Main ────────────────────────────────────────────────────────────────
@logger.catch
def main():
    logger.info("=" * 60)
    logger.info("Starting evaluation: 8 Methods x 8 Datasets")
    logger.info("=" * 60)

    # 1. Load data
    logger.info("Step 1: Loading data")
    figs_data = load_figs_data()
    baselines_data = load_baselines_data()

    # 2. Extract per-fold results
    logger.info("Step 2: Extracting per-fold results")
    figs_df = extract_figs_per_fold(figs_data)
    baselines_df = extract_baselines_per_fold(baselines_data)

    # 3. Select best max_splits for FIGS
    logger.info("Step 3: Selecting best hyperparameters")
    figs_best_df, best_splits = select_best_max_splits(figs_df)

    # 4. Build unified matrix
    logger.info("Step 4: Building unified results matrix")
    unified_df = build_unified_fold_results(figs_best_df, baselines_df)
    mean_matrix = build_mean_matrix(unified_df)

    # Check for NaN
    if mean_matrix.isnull().any().any():
        nan_cols = mean_matrix.columns[mean_matrix.isnull().any()].tolist()
        logger.warning(f"NaN values found in columns: {nan_cols}")
        mean_matrix = mean_matrix.fillna(mean_matrix.mean())

    classif_matrix = mean_matrix.loc[CLASSIFICATION_DATASETS]

    # 5. Compute all metrics
    logger.info("Step 5: Computing metrics")

    # A: Main results table
    logger.info("  A: Main results table")
    main_table = compute_main_results_table(unified_df, best_splits)

    # B: Friedman + Nemenyi (all 8)
    logger.info("  B: Friedman + Nemenyi (all 8 datasets)")
    friedman_all = compute_friedman_nemenyi(mean_matrix, label="all_8")

    # C: Bayesian signed-rank tests
    logger.info("  C: Bayesian signed-rank tests")
    bayesian_tests = compute_bayesian_tests(mean_matrix)

    # D: Cohen's d
    logger.info("  D: Cohen's d effect sizes")
    cohens_d = compute_cohens_d(unified_df)

    # E: Winner analysis
    logger.info("  E: Winner analysis")
    clustering_info = figs_data.get("metadata", {}).get("clustering_info", {})
    winner_analysis = compute_winner_analysis(mean_matrix, clustering_info)

    # F: Method rankings
    logger.info("  F: Method rankings")
    rankings = compute_rankings(mean_matrix, classif_matrix)

    # G: Classification-only Friedman
    logger.info("  G: Classification-only Friedman + Nemenyi (7 datasets)")
    friedman_classif = compute_friedman_nemenyi(classif_matrix, label="classif_7")

    # Additional metrics
    logger.info("  Extra: Improvement analysis and top-3 counts")
    improvement_analysis = compute_improvement_analysis(mean_matrix)
    top3_counts = compute_top3_analysis(mean_matrix)

    # 6. Merge examples
    logger.info("Step 6: Merging examples for output")
    merged_datasets = merge_examples(figs_data, baselines_data)

    # 7. Build output
    logger.info("Step 7: Building output JSON")
    output = build_output(
        main_table=main_table,
        friedman_all=friedman_all,
        friedman_classif=friedman_classif,
        bayesian_tests=bayesian_tests,
        cohens_d=cohens_d,
        winner_analysis=winner_analysis,
        rankings=rankings,
        best_splits=best_splits,
        merged_datasets=merged_datasets,
        clustering_info=clustering_info,
        improvement_analysis=improvement_analysis,
        top3_counts=top3_counts,
    )

    # 8. Write output
    logger.info("Step 8: Writing output")
    output_path = WORK_DIR / "eval_out.json"
    output_path.write_text(json.dumps(output, indent=2, default=str))
    file_size_bytes = output_path.stat().st_size
    logger.info(f"Output written: {output_path} ({file_size_bytes / 1024:.1f} KB)")

    # Free memory
    del figs_data, baselines_data, figs_df, baselines_df
    gc.collect()

    logger.info("=" * 60)
    logger.info("Evaluation complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Signed vs Unsigned Spectral Ablation: Statistical Analysis with Root Cause Synthesis.

Comprehensive evaluation comparing signed (SPONGE) vs unsigned spectral clustering
for CoI-guided oblique FIGS splits. Synthesises evidence from three experiments:
  1. Real-data 8-dataset benchmark (exp_id1_it5__opus)
  2. Synthetic 6-variant benchmark with ground-truth module recovery (exp_id3_it3__opus)
  3. CoI estimator bias diagnostics (exp_id3_it4__opus)

Produces paper-ready statistical tables, effect sizes, and a causal narrative linking
estimator bias -> sign collapse -> L_pos degeneration -> SPONGE failure.
"""

from __future__ import annotations

import gc
import json
import math
import os
import resource
import sys
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
# Hardware-aware resource limits (cgroup-safe)
# ---------------------------------------------------------------------------

def _container_ram_bytes() -> int | None:
    for p in ["/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"]:
        try:
            v = Path(p).read_text().strip()
            if v != "max" and int(v) < 1_000_000_000_000:
                return int(v)
        except (FileNotFoundError, ValueError):
            pass
    return None


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


NUM_CPUS = _detect_cpus()
_container_ram = _container_ram_bytes()
TOTAL_RAM_BYTES = _container_ram if _container_ram else 50 * 1024**3
RAM_BUDGET = int(TOTAL_RAM_BYTES * 0.4)  # 40% — evaluation is lightweight
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET * 3, RAM_BUDGET * 3))
resource.setrlimit(resource.RLIMIT_CPU, (3600, 3600))
logger.info(f"Hardware: {NUM_CPUS} CPUs, {TOTAL_RAM_BYTES/1e9:.1f} GB RAM, budget {RAM_BUDGET/1e9:.1f} GB")

# ---------------------------------------------------------------------------
# Dependency paths
# ---------------------------------------------------------------------------
WORKSPACE = Path(__file__).resolve().parent
DEP1_DIR = Path("/ai-inventor/aii_pipeline/runs/jamnik-sgfigs-pid-v2/3_invention_loop/iter_5/gen_art/exp_id1_it5__opus")
DEP2_DIR = Path("/ai-inventor/aii_pipeline/runs/jamnik-sgfigs-pid-v2/3_invention_loop/iter_3/gen_art/exp_id3_it3__opus")
DEP3_DIR = Path("/ai-inventor/aii_pipeline/runs/jamnik-sgfigs-pid-v2/3_invention_loop/iter_4/gen_art/exp_id3_it4__opus")

# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def load_json(path: Path) -> dict:
    """Load JSON from path with logging."""
    logger.info(f"Loading {path.name} ({path.stat().st_size / 1e6:.1f} MB)")
    data = json.loads(path.read_text())
    logger.info(f"  Loaded successfully")
    return data


def hedges_g(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Hedges' g effect size for paired or independent samples.

    Uses pooled std with Bessel correction and small-sample bias correction.
    """
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    n1, n2 = len(x), len(y)
    if n1 < 2 or n2 < 2:
        return 0.0
    s1, s2 = np.std(x, ddof=1), np.std(y, ddof=1)
    sp = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    if sp < 1e-15:
        return 0.0
    d = (np.mean(x) - np.mean(y)) / sp
    # Hedges' correction factor for small samples
    df = n1 + n2 - 2
    correction = 1 - 3 / (4 * df - 1)
    return float(d * correction)


def hedges_g_paired(diffs: np.ndarray) -> float:
    """Compute Hedges' g for paired differences (one-sample effect size)."""
    diffs = diffs[~np.isnan(diffs)]
    n = len(diffs)
    if n < 2:
        return 0.0
    sd = np.std(diffs, ddof=1)
    if sd < 1e-15:
        return 0.0
    d = np.mean(diffs) / sd
    correction = 1 - 3 / (4 * n - 5) if n > 3 else 1.0
    return float(d * correction)


def classify_effect_size(g: float) -> str:
    """Classify |g|: negligible < 0.2, small < 0.5, medium < 0.8, large >= 0.8."""
    if np.isnan(g):
        return "undefined"
    ag = abs(g)
    if ag < 0.2:
        return "negligible"
    elif ag < 0.5:
        return "small"
    elif ag < 0.8:
        return "medium"
    else:
        return "large"


def safe_wilcoxon(x: np.ndarray) -> tuple[float, float]:
    """Wilcoxon signed-rank test, handling edge cases."""
    x = x[~np.isnan(x)]
    x = x[np.abs(x) > 1e-15]  # remove exact zeros
    if len(x) < 6:
        return (float("nan"), float("nan"))
    try:
        stat, p = stats.wilcoxon(x, alternative="two-sided")
        return (float(stat), float(p))
    except Exception:
        return (float("nan"), float("nan"))


def holm_bonferroni(pvalues: list[float]) -> list[float]:
    """Holm-Bonferroni correction for multiple comparisons."""
    n = len(pvalues)
    indexed = sorted(enumerate(pvalues), key=lambda x: x[1])
    adjusted = [0.0] * n
    cummax = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        adj_p = min(p * (n - rank), 1.0)
        cummax = max(cummax, adj_p)
        adjusted[orig_idx] = cummax
    return adjusted


# ===================================================================
# SECTION A: Real-Data Ablation (from exp_id1_it5__opus)
# ===================================================================

def section_a_real_data(dep1: dict) -> dict:
    """Compute all Section A metrics from real-data benchmark."""
    logger.info("=== SECTION A: Real-Data Ablation ===")
    meta = dep1["metadata"]
    results_per_fold = meta["results_per_fold"]
    clustering_info = meta["clustering_info"]

    # Build a DataFrame from per-fold results
    df = pd.DataFrame(results_per_fold)
    datasets = sorted(df["dataset"].unique())
    max_splits_vals = sorted(df["max_splits"].unique())
    logger.info(f"  Datasets: {datasets}")
    logger.info(f"  Max splits: {max_splits_vals}")
    logger.info(f"  Methods: {sorted(df['method'].unique())}")
    logger.info(f"  Total rows: {len(df)}")

    # Filter to unsigned_spectral and signed_spectral only
    unsigned_df = df[df["method"] == "unsigned_spectral"].copy()
    signed_df = df[df["method"] == "signed_spectral"].copy()

    # ---- A1: Per-fold balanced accuracy delta ----
    # Merge on (dataset, max_splits, fold)
    merge_keys = ["dataset", "max_splits", "fold"]
    paired = unsigned_df[merge_keys + ["balanced_accuracy", "auc", "avg_split_arity"]].merge(
        signed_df[merge_keys + ["balanced_accuracy", "auc", "avg_split_arity"]],
        on=merge_keys,
        suffixes=("_unsigned", "_signed")
    )
    paired["delta_bacc"] = paired["balanced_accuracy_unsigned"] - paired["balanced_accuracy_signed"]
    paired["delta_auc"] = paired["auc_unsigned"] - paired["auc_signed"]

    # Drop rows where both methods have NaN (e.g. regression datasets like california_housing)
    n_before = len(paired)
    paired_valid = paired.dropna(subset=["delta_bacc"])
    n_dropped = n_before - len(paired_valid)
    if n_dropped > 0:
        dropped_ds = paired[paired["delta_bacc"].isna()]["dataset"].unique()
        logger.warning(f"  Dropped {n_dropped} NaN rows from datasets: {list(dropped_ds)}")

    n_paired = len(paired_valid)
    logger.info(f"  A1: {n_paired} paired observations (dropped {n_dropped} NaN)")

    deltas = paired_valid["delta_bacc"].values
    delta_auc = paired_valid["delta_auc"].dropna().values

    # ---- A2: Aggregate Wilcoxon signed-rank test ----
    w_stat, w_p = safe_wilcoxon(deltas)
    logger.info(f"  A2: Wilcoxon W={w_stat:.1f}, p={w_p:.6f}")

    # Per-dataset Wilcoxon
    per_dataset_wilcoxon = {}
    raw_pvals = []
    for ds in datasets:
        ds_deltas = paired_valid[paired_valid["dataset"] == ds]["delta_bacc"].values
        ws, wp = safe_wilcoxon(ds_deltas)
        per_dataset_wilcoxon[ds] = {"W": ws, "p_raw": wp, "n": len(ds_deltas)}
        raw_pvals.append(wp)

    # Holm-Bonferroni correction
    adj_pvals = holm_bonferroni(raw_pvals)
    for i, ds in enumerate(datasets):
        per_dataset_wilcoxon[ds]["p_adjusted"] = adj_pvals[i]

    # ---- A3: Hedges' g effect size ----
    agg_g = hedges_g_paired(deltas)
    agg_g_class = classify_effect_size(agg_g)
    logger.info(f"  A3: Aggregate Hedges' g = {agg_g:.4f} ({agg_g_class})")

    per_dataset_g = {}
    for ds in datasets:
        ds_deltas = paired_valid[paired_valid["dataset"] == ds]["delta_bacc"].values
        g = hedges_g_paired(ds_deltas)
        per_dataset_g[ds] = {"hedges_g": g, "classification": classify_effect_size(g)}

    # ---- A4: Win/loss/tie table ----
    TOLERANCE = 0.001
    wins, losses, ties = 0, 0, 0
    win_loss_detail = []
    for ds in datasets:
        for ms in max_splits_vals:
            sub = paired_valid[(paired_valid["dataset"] == ds) & (paired_valid["max_splits"] == ms)]
            w = int((sub["delta_bacc"] > TOLERANCE).sum())
            l = int((sub["delta_bacc"] < -TOLERANCE).sum())
            t = int(len(sub)) - w - l
            wins += w
            losses += l
            ties += t
            win_loss_detail.append({
                "dataset": ds, "max_splits": int(ms),
                "unsigned_wins": w, "signed_wins": l, "ties": t
            })

    total_configs = len(win_loss_detail)
    logger.info(f"  A4: Wins={wins}, Losses={losses}, Ties={ties} across {total_configs} configs")

    # ---- A5: Per-dataset comparison table ----
    per_dataset_table = []
    for ds in datasets:
        for ms in max_splits_vals:
            u_sub = unsigned_df[(unsigned_df["dataset"] == ds) & (unsigned_df["max_splits"] == ms)].dropna(subset=["balanced_accuracy"])
            s_sub = signed_df[(signed_df["dataset"] == ds) & (signed_df["max_splits"] == ms)].dropna(subset=["balanced_accuracy"])
            p_sub = paired_valid[(paired_valid["dataset"] == ds) & (paired_valid["max_splits"] == ms)]

            if len(u_sub) == 0 or len(s_sub) == 0:
                continue

            u_mean = float(u_sub["balanced_accuracy"].mean())
            u_std = float(u_sub["balanced_accuracy"].std())
            s_mean = float(s_sub["balanced_accuracy"].mean())
            s_std = float(s_sub["balanced_accuracy"].std())
            delta_mean = u_mean - s_mean

            ds_deltas = p_sub["delta_bacc"].values
            _, p_val = safe_wilcoxon(ds_deltas) if len(ds_deltas) >= 6 else (float("nan"), float("nan"))
            g = hedges_g_paired(ds_deltas)

            u_arity = float(u_sub["avg_split_arity"].mean()) if "avg_split_arity" in u_sub.columns else float("nan")
            s_arity = float(s_sub["avg_split_arity"].mean()) if "avg_split_arity" in s_sub.columns else float("nan")

            per_dataset_table.append({
                "dataset": ds, "max_splits": int(ms),
                "unsigned_bacc_mean": round(u_mean, 6),
                "unsigned_bacc_std": round(u_std, 6),
                "signed_bacc_mean": round(s_mean, 6),
                "signed_bacc_std": round(s_std, 6),
                "delta_mean": round(delta_mean, 6),
                "p_value": round(p_val, 6) if not np.isnan(p_val) else None,
                "hedges_g": round(g, 4),
                "unsigned_avg_arity": round(u_arity, 4),
                "signed_avg_arity": round(s_arity, 4),
            })

    # ---- A6: AUC delta analysis ----
    w_auc_stat, w_auc_p = safe_wilcoxon(delta_auc)
    auc_g = hedges_g_paired(delta_auc)
    logger.info(f"  A6: AUC Wilcoxon p={w_auc_p:.6f}, Hedges' g={auc_g:.4f}")

    # ---- A7: CoI sign distribution context ----
    coi_sign_data = []
    for ds in datasets:
        info = clustering_info.get(ds, {})
        n_pos = info.get("n_positive_coi_pairs", 0)
        n_neg = info.get("n_negative_coi_pairs", 0)
        total = n_pos + n_neg
        frac_neg = n_neg / total if total > 0 else 0.0

        # Get dataset-level delta
        ds_deltas = paired_valid[paired_valid["dataset"] == ds]["delta_bacc"].values
        mean_delta = float(np.nanmean(ds_deltas)) if len(ds_deltas) > 0 else 0.0

        coi_sign_data.append({
            "dataset": ds,
            "n_positive_coi_pairs": n_pos,
            "n_negative_coi_pairs": n_neg,
            "frac_negative": round(frac_neg, 4),
            "mean_delta_bacc": round(mean_delta, 6),
            "frustration_index": info.get("signed_spectral", {}).get("frustration_index", None),
        })

    # Correlation: frac_negative vs mean_delta (only datasets with valid paired data)
    valid_coi = [c for c in coi_sign_data
                 if c["mean_delta_bacc"] is not None
                 and not np.isnan(c["mean_delta_bacc"])
                 and c["frac_negative"] is not None]
    frac_negs = np.array([c["frac_negative"] for c in valid_coi])
    mean_deltas = np.array([c["mean_delta_bacc"] for c in valid_coi])
    if len(frac_negs) > 2 and np.std(frac_negs) > 1e-10 and np.std(mean_deltas) > 1e-10:
        corr, corr_p = stats.spearmanr(frac_negs, mean_deltas)
    else:
        corr, corr_p = float("nan"), float("nan")
    logger.info(f"  A7: Spearman corr(frac_negative, delta_bacc) = {corr:.4f}, p={corr_p:.4f}")

    return {
        "n_paired_observations": n_paired,
        "aggregate_wilcoxon": {"W": w_stat, "p": w_p},
        "aggregate_hedges_g": {"g": round(agg_g, 4), "classification": agg_g_class},
        "per_dataset_wilcoxon": per_dataset_wilcoxon,
        "per_dataset_hedges_g": per_dataset_g,
        "win_loss_tie": {
            "total_unsigned_wins": wins,
            "total_signed_wins": losses,
            "total_ties": ties,
            "total_configs": total_configs,
            "detail": win_loss_detail,
        },
        "per_dataset_comparison_table": per_dataset_table,
        "auc_delta": {
            "wilcoxon_W": w_auc_stat,
            "wilcoxon_p": w_auc_p,
            "hedges_g": round(auc_g, 4),
            "classification": classify_effect_size(auc_g),
            "mean_delta_auc": round(float(np.mean(delta_auc)), 6),
        },
        "coi_sign_distribution": coi_sign_data,
        "coi_sign_correlation": {
            "spearman_rho": round(float(corr), 4) if not np.isnan(corr) else None,
            "spearman_p": round(float(corr_p), 4) if not np.isnan(corr_p) else None,
        },
        "mean_delta_bacc_unsigned_minus_signed": round(float(np.nanmean(deltas)), 6),
        "median_delta_bacc": round(float(np.nanmedian(deltas)), 6),
    }


# ===================================================================
# SECTION B: Synthetic-Data Ablation (from exp_id3_it3__opus)
# ===================================================================

def section_b_synthetic(dep2: dict) -> dict:
    """Compute all Section B metrics from synthetic benchmark."""
    logger.info("=== SECTION B: Synthetic-Data Ablation ===")
    meta = dep2["metadata"]
    per_variant = meta["per_variant_results"]
    variants = sorted(per_variant.keys())
    logger.info(f"  Variants: {variants}")

    # ---- B1: Per-variant accuracy comparison ----
    variant_comparison = []
    all_signed_bacc = []
    all_unsigned_bacc = []
    paired_deltas_best = []  # at best_max_splits only
    paired_deltas_all = []   # all (variant x max_splits x fold)

    for variant in variants:
        vdata = per_variant[variant]
        methods = vdata["methods"]

        signed = methods.get("signed_spectral", {})
        unsigned = methods.get("unsigned_spectral", {})

        s_mean = signed.get("mean_balanced_accuracy", float("nan"))
        u_mean = unsigned.get("mean_balanced_accuracy", float("nan"))

        # Collect best_folds for paired comparison
        s_best_folds = signed.get("best_folds", [])
        u_best_folds = unsigned.get("best_folds", [])

        # Pair by fold index
        s_by_fold = {f["fold"]: f for f in s_best_folds}
        u_by_fold = {f["fold"]: f for f in u_best_folds}

        per_fold_deltas = []
        per_fold_ari_signed = []
        per_fold_ari_unsigned = []
        per_fold_jaccard_signed = []
        per_fold_jaccard_unsigned = []

        for fold_id in sorted(set(s_by_fold.keys()) & set(u_by_fold.keys())):
            sf = s_by_fold[fold_id]
            uf = u_by_fold[fold_id]
            d = uf["balanced_accuracy"] - sf["balanced_accuracy"]
            per_fold_deltas.append(d)
            paired_deltas_best.append(d)

            all_signed_bacc.append(sf["balanced_accuracy"])
            all_unsigned_bacc.append(uf["balanced_accuracy"])

            # Module recovery
            if sf.get("module_recovery_ari") is not None:
                per_fold_ari_signed.append(sf["module_recovery_ari"])
            if uf.get("module_recovery_ari") is not None:
                per_fold_ari_unsigned.append(uf["module_recovery_ari"])
            if sf.get("module_recovery_jaccard") is not None:
                per_fold_jaccard_signed.append(sf["module_recovery_jaccard"])
            if uf.get("module_recovery_jaccard") is not None:
                per_fold_jaccard_unsigned.append(uf["module_recovery_jaccard"])

        # Also collect all folds across all max_splits
        s_all_folds = signed.get("folds", [])
        u_all_folds = unsigned.get("folds", [])
        s_all_by_key = {(f["fold"], f["max_splits"]): f for f in s_all_folds}
        u_all_by_key = {(f["fold"], f["max_splits"]): f for f in u_all_folds}
        # Also add best_folds
        for f in s_best_folds:
            s_all_by_key[(f["fold"], f["max_splits"])] = f
        for f in u_best_folds:
            u_all_by_key[(f["fold"], f["max_splits"])] = f

        common_keys = sorted(set(s_all_by_key.keys()) & set(u_all_by_key.keys()))
        for key in common_keys:
            d = u_all_by_key[key]["balanced_accuracy"] - s_all_by_key[key]["balanced_accuracy"]
            paired_deltas_all.append(d)

        variant_comparison.append({
            "variant": variant,
            "signed_mean_bacc": round(s_mean, 4) if not np.isnan(s_mean) else None,
            "unsigned_mean_bacc": round(u_mean, 4) if not np.isnan(u_mean) else None,
            "delta_mean": round(u_mean - s_mean, 4) if not (np.isnan(s_mean) or np.isnan(u_mean)) else None,
            "n_paired_folds_best": len(per_fold_deltas),
            "mean_fold_delta_best": round(float(np.mean(per_fold_deltas)), 4) if per_fold_deltas else None,
        })

    logger.info(f"  B1: {len(variant_comparison)} variant comparisons")

    # ---- B2: Module recovery ARI comparison ----
    ari_comparison = []
    for variant in variants:
        vdata = per_variant[variant]
        methods = vdata["methods"]

        for method_name in ["signed_spectral", "unsigned_spectral"]:
            method = methods.get(method_name, {})
            best_folds = method.get("best_folds", [])
            aris = [f["module_recovery_ari"] for f in best_folds if f.get("module_recovery_ari") is not None]
            if aris:
                ari_comparison.append({
                    "variant": variant,
                    "method": method_name,
                    "mean_ari": round(float(np.mean(aris)), 4),
                    "std_ari": round(float(np.std(aris)), 4),
                    "n_folds": len(aris),
                })

    # ---- B3: Module recovery Jaccard comparison ----
    jaccard_comparison = []
    for variant in variants:
        vdata = per_variant[variant]
        methods = vdata["methods"]

        for method_name in ["signed_spectral", "unsigned_spectral"]:
            method = methods.get(method_name, {})
            best_folds = method.get("best_folds", [])
            jaccards = [f["module_recovery_jaccard"] for f in best_folds if f.get("module_recovery_jaccard") is not None]
            if jaccards:
                jaccard_comparison.append({
                    "variant": variant,
                    "method": method_name,
                    "mean_jaccard": round(float(np.mean(jaccards)), 4),
                    "std_jaccard": round(float(np.std(jaccards)), 4),
                    "n_folds": len(jaccards),
                })

    # ---- B4: Synthetic Wilcoxon + Hedges' g ----
    paired_deltas_best_arr = np.array(paired_deltas_best)
    paired_deltas_all_arr = np.array(paired_deltas_all)

    w_best_stat, w_best_p = safe_wilcoxon(paired_deltas_best_arr)
    g_best = hedges_g_paired(paired_deltas_best_arr)

    w_all_stat, w_all_p = safe_wilcoxon(paired_deltas_all_arr)
    g_all = hedges_g_paired(paired_deltas_all_arr)

    logger.info(f"  B4 (best): Wilcoxon p={w_best_p:.6f}, g={g_best:.4f} ({len(paired_deltas_best_arr)} obs)")
    logger.info(f"  B4 (all):  Wilcoxon p={w_all_p:.6f}, g={g_all:.4f} ({len(paired_deltas_all_arr)} obs)")

    return {
        "variant_comparison": variant_comparison,
        "module_recovery_ari": ari_comparison,
        "module_recovery_jaccard": jaccard_comparison,
        "paired_test_best_max_splits": {
            "n_observations": len(paired_deltas_best_arr),
            "wilcoxon_W": w_best_stat,
            "wilcoxon_p": w_best_p,
            "hedges_g": round(g_best, 4),
            "classification": classify_effect_size(g_best),
            "mean_delta": round(float(np.mean(paired_deltas_best_arr)), 6),
            "median_delta": round(float(np.median(paired_deltas_best_arr)), 6),
        },
        "paired_test_all_configs": {
            "n_observations": len(paired_deltas_all_arr),
            "wilcoxon_W": w_all_stat,
            "wilcoxon_p": w_all_p,
            "hedges_g": round(g_all, 4),
            "classification": classify_effect_size(g_all),
            "mean_delta": round(float(np.mean(paired_deltas_all_arr)), 6),
        },
    }


# ===================================================================
# SECTION C: Root Cause Analysis (from exp_id3_it4__opus)
# ===================================================================

def section_c_root_cause(dep3: dict) -> dict:
    """Compute all Section C metrics from CoI estimator bias diagnosis."""
    logger.info("=== SECTION C: Root Cause Analysis ===")
    meta = dep3["metadata"]
    part1 = meta["part1_estimator_bias"]
    part2 = meta["part2_sponge_diagnosis"]
    conclusions = meta["conclusions"]

    # ---- C1: CoI sign distribution by estimator ----
    frac_neg_by_method = conclusions.get("frac_negative_by_method", {})
    logger.info(f"  C1: {len(frac_neg_by_method)} estimator x dataset combos")

    # Restructure into table
    coi_sign_table = []
    for key, val in frac_neg_by_method.items():
        parts = key.split("/")
        if len(parts) == 2:
            coi_sign_table.append({
                "dataset": parts[0],
                "estimator": parts[1],
                "frac_negative": round(val, 4),
            })

    # Summarise by estimator
    estimator_summary = {}
    for entry in coi_sign_table:
        est = entry["estimator"]
        if est not in estimator_summary:
            estimator_summary[est] = []
        estimator_summary[est].append(entry["frac_negative"])

    estimator_agg = {}
    for est, vals in estimator_summary.items():
        estimator_agg[est] = {
            "mean_frac_negative": round(float(np.mean(vals)), 4),
            "min_frac_negative": round(float(np.min(vals)), 4),
            "max_frac_negative": round(float(np.max(vals)), 4),
            "n_datasets": len(vals),
        }

    # Also extract from part1 for all datasets x methods with full details
    all_estimators = set()
    all_datasets_part1 = sorted(part1.keys())
    for ds_data in part1.values():
        for method_name in ds_data:
            if method_name not in ("analytical_ground_truth", "bias_analysis"):
                all_estimators.add(method_name)
    all_estimators = sorted(all_estimators)

    full_sign_table = []
    for ds in all_datasets_part1:
        for est in all_estimators:
            if est in part1[ds] and est not in ("analytical_ground_truth", "bias_analysis"):
                sd = part1[ds][est].get("sign_distribution", {})
                full_sign_table.append({
                    "dataset": ds,
                    "estimator": est,
                    "frac_positive": sd.get("frac_positive", None),
                    "frac_negative": sd.get("frac_negative", None),
                    "frac_near_zero": sd.get("frac_near_zero", None),
                    "n_pairs": part1[ds][est].get("n_pairs", None),
                })

    # ---- C2: Individual MI negativity evidence ----
    raw_ksg_neg = conclusions.get("raw_ksg_negative_individual_mi", {})
    mi_negativity = {}
    for ds, info in raw_ksg_neg.items():
        mi_negativity[ds] = {
            "n_negative_individual_mi": info.get("n_negative_individual_mi", 0),
            "negative_mi_values": info.get("negative_mi_values", []),
        }

    # ---- C3: L_pos eigenspectrum analysis ----
    eigenspectrum_analysis = {}
    for ds in sorted(part2.keys()):
        es = part2[ds].get("eigenspectrum", {})

        l_pos = es.get("L_pos", {})
        l_neg = es.get("L_neg", {})
        l_abs = es.get("L_abs", {})

        l_pos_evals = l_pos.get("eigenvalues", [])
        l_neg_evals = l_neg.get("eigenvalues", [])
        l_abs_evals = l_abs.get("eigenvalues", [])

        # Effective rank: eigenvalues > 0.01 * max_eigenvalue
        max_eval_pos = l_pos.get("max_eval", 0)
        threshold = 0.01 * max_eval_pos if max_eval_pos > 0 else 0.01
        # We only have partial eigenvalues in preview; use rank from metadata

        eigenspectrum_analysis[ds] = {
            "L_pos_rank": l_pos.get("rank"),
            "L_pos_min_eval": l_pos.get("min_eval"),
            "L_pos_max_eval": l_pos.get("max_eval"),
            "L_neg_rank": l_neg.get("rank"),
            "L_neg_min_eval": l_neg.get("min_eval"),
            "L_neg_max_eval": l_neg.get("max_eval"),
            "L_abs_rank": l_abs.get("rank"),
            "L_abs_min_eval": l_abs.get("min_eval"),
            "L_abs_max_eval": l_abs.get("max_eval"),
            "positive_edge_fraction": es.get("positive_edge_fraction"),
            "negative_edge_fraction": es.get("negative_edge_fraction"),
        }

    # ---- C4: SPONGE condition number analysis ----
    condition_analysis = {}
    for ds in sorted(part2.keys()):
        conds = part2[ds].get("condition_numbers", [])
        condition_analysis[ds] = conds

    # ---- C5: Edge injection failure documentation ----
    edge_injection = {}
    for ds in sorted(part2.keys()):
        inj = part2[ds].get("edge_injection", {})
        strategies = inj.get("strategies", {})
        edge_injection[ds] = {
            "strategies": {
                name: {
                    "ari": s.get("ari"),
                    "n_positive_edges": s.get("n_positive_edges"),
                    "frac_positive": s.get("frac_positive"),
                }
                for name, s in strategies.items()
            },
            "all_failed": all(s.get("ari", 0) < 0 for s in strategies.values()) if strategies else None,
        }

    # ---- C6: Clustering comparison summary ----
    clustering_summary = []
    for ds in sorted(part2.keys()):
        cc = part2[ds].get("clustering_comparison", {})
        clustering_summary.append({
            "dataset": cc.get("dataset", ds),
            "k_true": cc.get("k_true"),
            "unsigned_spectral_ari": cc.get("unsigned_spectral_ari"),
            "sponge_sym_weighted_ari": cc.get("sponge_sym_weighted_ari"),
            "sponge_sym_unweighted_ari": cc.get("sponge_sym_unweighted_ari"),
        })

    # Also get diagnostic details from conclusions
    sponge_details = conclusions.get("sponge_diagnostic_details", [])

    return {
        "coi_sign_by_estimator_conclusions": coi_sign_table,
        "estimator_aggregate": estimator_agg,
        "full_sign_distribution_table": full_sign_table,
        "individual_mi_negativity": mi_negativity,
        "eigenspectrum_analysis": eigenspectrum_analysis,
        "condition_number_analysis": condition_analysis,
        "edge_injection_failure": edge_injection,
        "clustering_comparison": clustering_summary,
        "sponge_diagnostic_details": sponge_details,
        "sponge_failure_mechanism": conclusions.get("sponge_failure_mechanism", ""),
    }


# ===================================================================
# SECTION D: Unified Synthesis Metrics
# ===================================================================

def section_d_synthesis(
    section_a: dict,
    section_b: dict,
    section_c: dict,
    dep1: dict,
    dep2: dict,
    dep3: dict,
) -> dict:
    """Compute unified synthesis metrics across all three experiments."""
    logger.info("=== SECTION D: Unified Synthesis ===")

    # ---- D1: Aggregate Hedges' g (real + synthetic combined) ----
    # Collect all paired deltas from Section A
    meta1 = dep1["metadata"]
    df1 = pd.DataFrame(meta1["results_per_fold"])
    unsigned1 = df1[df1["method"] == "unsigned_spectral"]
    signed1 = df1[df1["method"] == "signed_spectral"]
    merge_keys = ["dataset", "max_splits", "fold"]
    paired1 = unsigned1[merge_keys + ["balanced_accuracy"]].merge(
        signed1[merge_keys + ["balanced_accuracy"]],
        on=merge_keys, suffixes=("_u", "_s")
    )
    real_deltas_raw = (paired1["balanced_accuracy_u"] - paired1["balanced_accuracy_s"]).values
    real_deltas = real_deltas_raw[~np.isnan(real_deltas_raw)]

    # Collect from Section B (all configs)
    meta2 = dep2["metadata"]
    per_variant = meta2["per_variant_results"]
    synth_deltas = []
    for variant in sorted(per_variant.keys()):
        methods = per_variant[variant]["methods"]
        signed = methods.get("signed_spectral", {})
        unsigned = methods.get("unsigned_spectral", {})

        s_all = {(f["fold"], f["max_splits"]): f for f in signed.get("folds", [])}
        u_all = {(f["fold"], f["max_splits"]): f for f in unsigned.get("folds", [])}
        for f in signed.get("best_folds", []):
            s_all[(f["fold"], f["max_splits"])] = f
        for f in unsigned.get("best_folds", []):
            u_all[(f["fold"], f["max_splits"])] = f

        for key in sorted(set(s_all.keys()) & set(u_all.keys())):
            synth_deltas.append(u_all[key]["balanced_accuracy"] - s_all[key]["balanced_accuracy"])

    synth_deltas = np.array(synth_deltas)
    combined_deltas = np.concatenate([real_deltas, synth_deltas])

    combined_g = hedges_g_paired(combined_deltas)
    logger.info(f"  D1: Combined Hedges' g = {combined_g:.4f} ({len(combined_deltas)} total obs)")

    # ---- D2: Causal chain quantification ----
    meta3 = dep3["metadata"]
    part2 = meta3["part2_sponge_diagnosis"]
    conclusions = meta3["conclusions"]
    frac_neg_by_method = conclusions.get("frac_negative_by_method", {})

    causal_chain = []
    # For synthetic datasets where we have both part1 and part2 data
    synth_datasets_with_diag = sorted(part2.keys())
    for ds in synth_datasets_with_diag:
        diag = part2[ds]

        # Estimator bias: use binned_10 frac_negative (the method used in the actual pipeline)
        frac_neg_key = f"{ds}/binned_10"
        frac_neg = frac_neg_by_method.get(frac_neg_key)
        if frac_neg is None:
            frac_neg_key = f"{ds}/binned_20"
            frac_neg = frac_neg_by_method.get(frac_neg_key)

        # L_pos effective rank
        es = diag.get("eigenspectrum", {})
        l_pos_rank = es.get("L_pos", {}).get("rank")

        # SPONGE condition number (tau=0.01)
        cond_nums = diag.get("condition_numbers", [])
        sponge_cond_01 = None
        for cn in cond_nums:
            if cn.get("tau") == 0.01:
                sponge_cond_01 = cn.get("sponge_cond")

        # Clustering ARI
        cc = diag.get("clustering_comparison", {})
        unsigned_ari = cc.get("unsigned_spectral_ari")
        sponge_ari = cc.get("sponge_sym_weighted_ari")

        # Downstream accuracy delta (from dep2 if available)
        variant_data = per_variant.get(ds, {}).get("methods", {})
        s_bacc = variant_data.get("signed_spectral", {}).get("mean_balanced_accuracy")
        u_bacc = variant_data.get("unsigned_spectral", {}).get("mean_balanced_accuracy")
        delta_bacc = None
        if s_bacc is not None and u_bacc is not None:
            delta_bacc = round(u_bacc - s_bacc, 4)

        causal_chain.append({
            "dataset": ds,
            "estimator_frac_negative": frac_neg,
            "L_pos_effective_rank": l_pos_rank,
            "sponge_condition_number_tau001": sponge_cond_01,
            "unsigned_spectral_ari": unsigned_ari,
            "sponge_ari": sponge_ari,
            "downstream_delta_bacc": delta_bacc,
        })

    # ---- D3: Frustration index analysis ----
    # Real data frustration indices
    clustering_info = dep1["metadata"].get("clustering_info", {})
    real_frustrations = []
    real_deltas_by_ds = {}

    df_real = pd.DataFrame(dep1["metadata"]["results_per_fold"])
    u_real = df_real[df_real["method"] == "unsigned_spectral"]
    s_real = df_real[df_real["method"] == "signed_spectral"]
    paired_real = u_real[["dataset", "max_splits", "fold", "balanced_accuracy"]].merge(
        s_real[["dataset", "max_splits", "fold", "balanced_accuracy"]],
        on=["dataset", "max_splits", "fold"], suffixes=("_u", "_s")
    )
    paired_real["delta"] = paired_real["balanced_accuracy_u"] - paired_real["balanced_accuracy_s"]
    paired_real = paired_real.dropna(subset=["delta"])

    for ds in sorted(clustering_info.keys()):
        fi = clustering_info[ds].get("signed_spectral", {}).get("frustration_index")
        ds_sub = paired_real[paired_real["dataset"] == ds]
        if len(ds_sub) == 0:
            continue
        ds_delta = ds_sub["delta"].mean()
        if fi is not None:
            real_frustrations.append({
                "dataset": ds,
                "frustration_index": fi,
                "mean_delta_bacc": round(float(ds_delta), 6),
                "source": "real",
            })

    # Synthetic frustration indices
    synth_frustrations = []
    for variant in sorted(per_variant.keys()):
        methods = per_variant[variant]["methods"]
        # Get frustration_index from any fold
        for method_name in ["signed_spectral", "unsigned_spectral"]:
            best_folds = methods.get(method_name, {}).get("best_folds", [])
            if best_folds and best_folds[0].get("frustration_index") is not None:
                fi = best_folds[0]["frustration_index"]
                s_bacc = methods.get("signed_spectral", {}).get("mean_balanced_accuracy", 0)
                u_bacc = methods.get("unsigned_spectral", {}).get("mean_balanced_accuracy", 0)
                synth_frustrations.append({
                    "dataset": variant,
                    "frustration_index": fi,
                    "mean_delta_bacc": round(u_bacc - s_bacc, 6) if s_bacc and u_bacc else None,
                    "source": "synthetic",
                })
                break  # Only need one per variant

    all_frustrations = real_frustrations + synth_frustrations
    valid_fi = [
        f for f in all_frustrations
        if f["frustration_index"] is not None
        and f["mean_delta_bacc"] is not None
        and not np.isnan(f["frustration_index"])
        and not np.isnan(f["mean_delta_bacc"])
    ]
    fi_vals = np.array([f["frustration_index"] for f in valid_fi])
    delta_vals = np.array([f["mean_delta_bacc"] for f in valid_fi])

    if len(fi_vals) > 2 and np.std(fi_vals) > 1e-10 and np.std(delta_vals) > 1e-10:
        fi_corr, fi_corr_p = stats.spearmanr(fi_vals, delta_vals)
    else:
        fi_corr, fi_corr_p = float("nan"), float("nan")

    logger.info(f"  D3: Frustration index correlation with delta_bacc: rho={fi_corr:.4f}, p={fi_corr_p:.4f}")

    return {
        "combined_hedges_g": {
            "g": round(combined_g, 4),
            "classification": classify_effect_size(combined_g),
            "n_real_obs": len(real_deltas),
            "n_synth_obs": len(synth_deltas),
            "n_total_obs": len(combined_deltas),
            "mean_delta_combined": round(float(np.mean(combined_deltas)), 6),
        },
        "causal_chain": causal_chain,
        "frustration_index_analysis": {
            "all_frustrations": all_frustrations,
            "spearman_rho": round(float(fi_corr), 4) if not np.isnan(fi_corr) else None,
            "spearman_p": round(float(fi_corr_p), 4) if not np.isnan(fi_corr_p) else None,
            "n_datasets": len(all_frustrations),
        },
    }


# ===================================================================
# Build output conforming to exp_eval_sol_out schema
# ===================================================================

def build_output(
    section_a: dict,
    section_b: dict,
    section_c: dict,
    section_d: dict,
    dep1: dict,
    dep2: dict,
    dep3: dict,
) -> dict:
    """Build final output conforming to exp_eval_sol_out.json schema."""
    logger.info("=== Building final output ===")

    # ---- metrics_agg: aggregate numerical metrics ----
    metrics_agg = {}

    # A metrics
    a_wil = section_a["aggregate_wilcoxon"]
    metrics_agg["real_wilcoxon_p"] = round(a_wil["p"], 8) if not np.isnan(a_wil["p"]) else 1.0
    metrics_agg["real_hedges_g"] = section_a["aggregate_hedges_g"]["g"]
    metrics_agg["real_mean_delta_bacc"] = section_a["mean_delta_bacc_unsigned_minus_signed"]
    metrics_agg["real_median_delta_bacc"] = section_a["median_delta_bacc"]
    metrics_agg["real_unsigned_wins"] = section_a["win_loss_tie"]["total_unsigned_wins"]
    metrics_agg["real_signed_wins"] = section_a["win_loss_tie"]["total_signed_wins"]
    metrics_agg["real_ties"] = section_a["win_loss_tie"]["total_ties"]
    metrics_agg["real_n_paired"] = section_a["n_paired_observations"]

    # AUC
    metrics_agg["real_auc_wilcoxon_p"] = round(section_a["auc_delta"]["wilcoxon_p"], 8) if not np.isnan(section_a["auc_delta"]["wilcoxon_p"]) else 1.0
    metrics_agg["real_auc_hedges_g"] = section_a["auc_delta"]["hedges_g"]
    metrics_agg["real_auc_mean_delta"] = section_a["auc_delta"]["mean_delta_auc"]

    # CoI sign correlation
    coi_corr = section_a["coi_sign_correlation"]
    metrics_agg["real_coi_sign_spearman_rho"] = coi_corr["spearman_rho"] if coi_corr["spearman_rho"] is not None else 0.0

    # B metrics
    b_best = section_b["paired_test_best_max_splits"]
    metrics_agg["synth_wilcoxon_p_best"] = round(b_best["wilcoxon_p"], 8) if not np.isnan(b_best["wilcoxon_p"]) else 1.0
    metrics_agg["synth_hedges_g_best"] = b_best["hedges_g"]
    metrics_agg["synth_mean_delta_best"] = b_best["mean_delta"]
    metrics_agg["synth_n_paired_best"] = b_best["n_observations"]

    b_all = section_b["paired_test_all_configs"]
    metrics_agg["synth_wilcoxon_p_all"] = round(b_all["wilcoxon_p"], 8) if not np.isnan(b_all["wilcoxon_p"]) else 1.0
    metrics_agg["synth_hedges_g_all"] = b_all["hedges_g"]

    # D metrics
    metrics_agg["combined_hedges_g"] = section_d["combined_hedges_g"]["g"]
    metrics_agg["combined_n_obs"] = section_d["combined_hedges_g"]["n_total_obs"]
    metrics_agg["combined_mean_delta"] = section_d["combined_hedges_g"]["mean_delta_combined"]

    fi_analysis = section_d["frustration_index_analysis"]
    metrics_agg["frustration_spearman_rho"] = fi_analysis["spearman_rho"] if fi_analysis["spearman_rho"] is not None else 0.0

    # Ensure all values are numbers (schema requirement)
    for k, v in metrics_agg.items():
        if v is None or (isinstance(v, float) and np.isnan(v)):
            metrics_agg[k] = 0.0
        metrics_agg[k] = float(metrics_agg[k]) if not isinstance(metrics_agg[k], int) else metrics_agg[k]

    # ---- datasets: array of dataset objects with examples ----
    datasets = []

    # Dataset 1: Real-data per-dataset comparison table
    real_examples = []
    for entry in section_a["per_dataset_comparison_table"]:
        input_str = json.dumps({
            "experiment": "real_data_ablation",
            "dataset": entry["dataset"],
            "max_splits": entry["max_splits"],
            "metric": "balanced_accuracy",
        })
        output_str = json.dumps({
            "unsigned_mean": entry["unsigned_bacc_mean"],
            "signed_mean": entry["signed_bacc_mean"],
            "delta": entry["delta_mean"],
            "hedges_g": entry["hedges_g"],
        })
        real_examples.append({
            "input": input_str,
            "output": output_str,
            "metadata_dataset": entry["dataset"],
            "metadata_max_splits": entry["max_splits"],
            "predict_unsigned_spectral": str(round(entry["unsigned_bacc_mean"], 6)),
            "predict_signed_spectral": str(round(entry["signed_bacc_mean"], 6)),
            "eval_delta_bacc": round(entry["delta_mean"], 6),
            "eval_hedges_g": round(entry["hedges_g"], 4),
        })

    datasets.append({
        "dataset": "real_data_ablation",
        "examples": real_examples,
    })

    # Dataset 2: Synthetic variant comparison
    synth_examples = []
    for entry in section_b["variant_comparison"]:
        input_str = json.dumps({
            "experiment": "synthetic_ablation",
            "variant": entry["variant"],
            "metric": "balanced_accuracy",
        })
        output_str = json.dumps({
            "unsigned_mean": entry["unsigned_mean_bacc"],
            "signed_mean": entry["signed_mean_bacc"],
            "delta": entry["delta_mean"],
        })
        delta = entry["delta_mean"] if entry["delta_mean"] is not None else 0.0
        u_val = str(round(entry["unsigned_mean_bacc"], 4)) if entry["unsigned_mean_bacc"] is not None else "N/A"
        s_val = str(round(entry["signed_mean_bacc"], 4)) if entry["signed_mean_bacc"] is not None else "N/A"
        synth_examples.append({
            "input": input_str,
            "output": output_str,
            "metadata_variant": entry["variant"],
            "predict_unsigned_spectral": u_val,
            "predict_signed_spectral": s_val,
            "eval_delta_bacc": round(delta, 6),
        })

    datasets.append({
        "dataset": "synthetic_ablation",
        "examples": synth_examples,
    })

    # Dataset 3: Module recovery ARI
    ari_examples = []
    for entry in section_b["module_recovery_ari"]:
        input_str = json.dumps({
            "experiment": "module_recovery_ari",
            "variant": entry["variant"],
            "method": entry["method"],
        })
        output_str = json.dumps({
            "mean_ari": entry["mean_ari"],
            "std_ari": entry["std_ari"],
        })
        ari_examples.append({
            "input": input_str,
            "output": output_str,
            "metadata_variant": entry["variant"],
            "metadata_method": entry["method"],
            "predict_ari": str(round(entry["mean_ari"], 4)),
            "eval_mean_ari": round(entry["mean_ari"], 4),
        })

    if ari_examples:
        datasets.append({
            "dataset": "module_recovery",
            "examples": ari_examples,
        })

    # Dataset 4: CoI sign distribution by estimator
    coi_examples = []
    for entry in section_c["full_sign_distribution_table"]:
        input_str = json.dumps({
            "experiment": "coi_estimator_bias",
            "dataset": entry["dataset"],
            "estimator": entry["estimator"],
        })
        frac_neg = entry["frac_negative"] if entry["frac_negative"] is not None else 0.0
        output_str = json.dumps({
            "frac_positive": entry["frac_positive"],
            "frac_negative": entry["frac_negative"],
            "frac_near_zero": entry["frac_near_zero"],
        })
        coi_examples.append({
            "input": input_str,
            "output": output_str,
            "metadata_dataset": entry["dataset"],
            "metadata_estimator": entry["estimator"],
            "predict_frac_negative": str(round(frac_neg, 4)),
            "eval_frac_negative": round(frac_neg, 4),
        })

    if coi_examples:
        datasets.append({
            "dataset": "coi_estimator_bias",
            "examples": coi_examples,
        })

    # Dataset 5: SPONGE clustering comparison
    clust_examples = []
    for entry in section_c["clustering_comparison"]:
        input_str = json.dumps({
            "experiment": "clustering_comparison",
            "dataset": entry["dataset"],
            "k_true": entry["k_true"],
        })
        unsigned_ari = entry["unsigned_spectral_ari"] if entry["unsigned_spectral_ari"] is not None else 0.0
        sponge_ari = entry["sponge_sym_weighted_ari"] if entry["sponge_sym_weighted_ari"] is not None else 0.0
        output_str = json.dumps({
            "unsigned_spectral_ari": entry["unsigned_spectral_ari"],
            "sponge_sym_weighted_ari": entry["sponge_sym_weighted_ari"],
        })
        clust_examples.append({
            "input": input_str,
            "output": output_str,
            "metadata_dataset": entry["dataset"],
            "predict_unsigned_spectral": str(round(unsigned_ari, 4)),
            "predict_sponge_sym": str(round(sponge_ari, 4)),
            "eval_unsigned_ari": round(unsigned_ari, 4),
            "eval_sponge_ari": round(sponge_ari, 4),
            "eval_ari_delta": round(unsigned_ari - sponge_ari, 4),
        })

    if clust_examples:
        datasets.append({
            "dataset": "sponge_clustering",
            "examples": clust_examples,
        })

    # Dataset 6: Causal chain
    chain_examples = []
    for entry in section_d["causal_chain"]:
        input_str = json.dumps({
            "experiment": "causal_chain",
            "dataset": entry["dataset"],
        })
        output_str = json.dumps(entry)
        frac_neg = entry["estimator_frac_negative"] if entry["estimator_frac_negative"] is not None else 0.0
        u_ari = round(entry["unsigned_spectral_ari"], 4) if entry["unsigned_spectral_ari"] is not None else 0.0
        s_ari = round(entry["sponge_ari"], 4) if entry["sponge_ari"] is not None else 0.0
        chain_examples.append({
            "input": input_str,
            "output": output_str,
            "metadata_dataset": entry["dataset"],
            "predict_unsigned_ari": str(u_ari),
            "predict_sponge_ari": str(s_ari),
            "eval_estimator_frac_negative": round(frac_neg, 4),
            "eval_sponge_ari": s_ari,
            "eval_unsigned_ari": u_ari,
        })

    if chain_examples:
        datasets.append({
            "dataset": "causal_chain",
            "examples": chain_examples,
        })

    # ---- metadata: full analysis details ----
    metadata = {
        "evaluation_name": "signed_vs_unsigned_spectral_ablation",
        "description": (
            "Comprehensive ablation evaluation comparing signed (SPONGE) vs unsigned spectral "
            "clustering for CoI-guided oblique FIGS splits. Synthesises evidence from real-data "
            "benchmark (8 datasets), synthetic benchmark (6 variants), and CoI estimator bias "
            "diagnostics to produce a causal narrative of SPONGE failure."
        ),
        "section_a_real_data": section_a,
        "section_b_synthetic": section_b,
        "section_c_root_cause": section_c,
        "section_d_synthesis": section_d,
    }

    output = {
        "metadata": metadata,
        "metrics_agg": metrics_agg,
        "datasets": datasets,
    }

    return output


# ===================================================================
# Main
# ===================================================================

@logger.catch
def main():
    logger.info("Starting evaluation: Signed vs Unsigned Spectral Ablation")

    # Load all three dependencies
    dep1 = load_json(DEP1_DIR / "full_method_out.json")
    dep2 = load_json(DEP2_DIR / "full_method_out.json")
    dep3 = load_json(DEP3_DIR / "full_method_out.json")

    # Run all sections
    section_a = section_a_real_data(dep1)
    gc.collect()

    section_b = section_b_synthetic(dep2)
    gc.collect()

    section_c = section_c_root_cause(dep3)
    gc.collect()

    section_d = section_d_synthesis(section_a, section_b, section_c, dep1, dep2, dep3)
    gc.collect()

    # Build output
    output = build_output(section_a, section_b, section_c, section_d, dep1, dep2, dep3)

    # Save
    out_path = WORKSPACE / "method_out.json"
    out_path.write_text(json.dumps(output, indent=2, default=str))
    logger.info(f"Saved output to {out_path} ({out_path.stat().st_size / 1e6:.2f} MB)")

    # Print key metrics summary
    ma = output["metrics_agg"]
    logger.info("=" * 60)
    logger.info("KEY RESULTS SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Real-data: Wilcoxon p={ma['real_wilcoxon_p']:.6f}, Hedges' g={ma['real_hedges_g']:.4f}")
    logger.info(f"  Real-data: mean delta_bacc={ma['real_mean_delta_bacc']:.6f}, unsigned wins={ma['real_unsigned_wins']}")
    logger.info(f"  Real-data AUC: Wilcoxon p={ma['real_auc_wilcoxon_p']:.6f}, Hedges' g={ma['real_auc_hedges_g']:.4f}")
    logger.info(f"  Synthetic: Wilcoxon p={ma['synth_wilcoxon_p_best']:.6f}, Hedges' g={ma['synth_hedges_g_best']:.4f}")
    logger.info(f"  Combined: Hedges' g={ma['combined_hedges_g']:.4f}, N={ma['combined_n_obs']}")
    logger.info(f"  Frustration index correlation: rho={ma['frustration_spearman_rho']:.4f}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

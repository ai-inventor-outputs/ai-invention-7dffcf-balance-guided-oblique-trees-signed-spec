#!/usr/bin/env python3
"""Scalability & Computational Cost Analysis for Balance-Guided Oblique Trees.

Evaluates computational cost by analyzing timing data from two experiments:
- exp_id1 (5 FIGS variants): axis_aligned, random_oblique, unsigned_spectral, signed_spectral, hard_threshold
- exp_id2 (3 baselines): ebm, random_forest, linear

Produces six analyses:
(A) Per-dataset timing breakdown table
(B) Scaling with n (sample count)
(C) Scaling with d (feature count)
(D) Overhead ratio vs accuracy benefit
(E) Cross-method timing comparison
(F) Summary statistics
"""

import json
import math
import os
import resource
import sys
from pathlib import Path

import numpy as np
from loguru import logger

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
WORKSPACE = Path(__file__).resolve().parent
LOG_DIR = WORKSPACE / "logs"
LOG_DIR.mkdir(exist_ok=True)

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add(str(LOG_DIR / "run.log"), rotation="30 MB", level="DEBUG")

# ---------------------------------------------------------------------------
# Hardware detection & memory limits
# ---------------------------------------------------------------------------
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
TOTAL_RAM_GB = _container_ram_gb() or 32.0

# Conservative RAM budget: this is a small-data evaluation, 4GB is plenty
RAM_BUDGET = int(4 * 1024**3)
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET * 3, RAM_BUDGET * 3))
resource.setrlimit(resource.RLIMIT_CPU, (3600, 3600))

logger.info(f"Hardware: {NUM_CPUS} CPUs, {TOTAL_RAM_GB:.1f} GB RAM, budget {RAM_BUDGET / 1e9:.1f} GB")

# ---------------------------------------------------------------------------
# Paths to dependency data
# ---------------------------------------------------------------------------
EXP1_DIR = Path("/ai-inventor/aii_pipeline/runs/jamnik-sgfigs-pid-v2/3_invention_loop/iter_5/gen_art/exp_id1_it5__opus")
EXP2_DIR = Path("/ai-inventor/aii_pipeline/runs/jamnik-sgfigs-pid-v2/3_invention_loop/iter_4/gen_art/exp_id2_it4__opus")

FIGS_METHODS = ["axis_aligned", "random_oblique", "unsigned_spectral", "signed_spectral", "hard_threshold"]
SPECTRAL_METHODS = ["unsigned_spectral", "signed_spectral", "hard_threshold"]
BASELINE_METHODS = ["ebm", "random_forest", "linear"]
ALL_METHODS = FIGS_METHODS + BASELINE_METHODS
DATASETS = ["adult", "electricity", "eye_movements", "credit", "california_housing", "higgs_small", "jannis", "miniboone"]


def load_experiment_data() -> tuple[dict, dict]:
    """Load full JSON data from both experiments."""
    logger.info(f"Loading FIGS experiment from {EXP1_DIR / 'full_method_out.json'}")
    exp1 = json.loads((EXP1_DIR / "full_method_out.json").read_text())
    logger.info(f"Loading baselines experiment from {EXP2_DIR / 'full_method_out.json'}")
    exp2 = json.loads((EXP2_DIR / "full_method_out.json").read_text())
    return exp1, exp2


def get_dataset_info(exp1: dict) -> dict[str, dict]:
    """Extract n_samples and n_features per dataset from results_summary."""
    info = {}
    for entry in exp1["metadata"]["results_summary"]:
        ds = entry["dataset"]
        if ds not in info:
            info[ds] = {
                "n_samples": entry["n_samples"],
                "n_features": entry["n_features"],
                "task_type": entry["task_type"],
            }
    return info


# =========================================================================
# (A) Timing Breakdown Table
# =========================================================================
def compute_timing_breakdown(exp1: dict, exp2: dict) -> list[dict]:
    """Per-dataset timing breakdown for all 8 methods at max_splits=10."""
    logger.info("Computing (A) Timing Breakdown Table")
    ds_info = get_dataset_info(exp1)
    clustering = exp1["metadata"]["clustering_info"]
    results_summary = exp1["metadata"]["results_summary"]
    results_per_fold = exp1["metadata"]["results_per_fold"]
    per_dataset_results = exp2["metadata"]["per_dataset_results"]

    table = []

    for ds in DATASETS:
        n_samples = ds_info[ds]["n_samples"]
        n_features = ds_info[ds]["n_features"]
        coi_time = clustering[ds]["coi_time_s"]

        # FIGS methods (use max_splits=10)
        for method in FIGS_METHODS:
            # Get summary at max_splits=10
            summary_entry = None
            for entry in results_summary:
                if entry["dataset"] == ds and entry["method"] == method and entry["max_splits"] == 10:
                    summary_entry = entry
                    break

            if summary_entry is None:
                logger.warning(f"No summary for {ds}/{method}/max_splits=10")
                continue

            mean_time = summary_entry["fit_time_s_mean"]
            std_time = summary_entry["fit_time_s_std"]

            # For spectral methods, decompose into CoI + tree fitting
            if method in SPECTRAL_METHODS:
                tree_fit_time = mean_time
                coi_overhead_pct = coi_time / (coi_time + mean_time) * 100 if (coi_time + mean_time) > 0 else 0.0
                row_coi = coi_time
            else:
                tree_fit_time = mean_time
                row_coi = 0.0
                coi_overhead_pct = 0.0

            table.append({
                "dataset": ds,
                "n_samples": n_samples,
                "n_features": n_features,
                "method": method,
                "mean_time_s": round(mean_time, 6),
                "std_time_s": round(std_time, 6),
                "coi_time_s": round(row_coi, 4),
                "tree_fit_time_s": round(tree_fit_time, 6),
                "coi_overhead_pct": round(coi_overhead_pct, 2),
            })

        # Baseline methods
        for method in BASELINE_METHODS:
            if ds not in per_dataset_results:
                logger.warning(f"Dataset {ds} not in baselines")
                continue
            if method not in per_dataset_results[ds]:
                logger.warning(f"Method {method} not in baselines for {ds}")
                continue

            agg = per_dataset_results[ds][method]["aggregate"]
            mean_time = agg["fit_time_mean"]
            std_time = agg["fit_time_std"]

            table.append({
                "dataset": ds,
                "n_samples": n_samples,
                "n_features": n_features,
                "method": method,
                "mean_time_s": round(mean_time, 6),
                "std_time_s": round(std_time, 6),
                "coi_time_s": 0.0,
                "tree_fit_time_s": round(mean_time, 6),
                "coi_overhead_pct": 0.0,
            })

    logger.info(f"Timing breakdown: {len(table)} rows")
    return table


# =========================================================================
# (B) Scaling with n (sample count)
# =========================================================================
def fit_power_law(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Fit power law y = a * x^b via log-log regression. Returns (a, b, r2)."""
    mask = (x > 0) & (y > 0)
    if mask.sum() < 2:
        return 0.0, 0.0, 0.0
    lx = np.log(x[mask])
    ly = np.log(y[mask])
    n = len(lx)
    sx = lx.sum()
    sy = ly.sum()
    sxx = (lx * lx).sum()
    sxy = (lx * ly).sum()
    denom = n * sxx - sx * sx
    if abs(denom) < 1e-15:
        return 0.0, 0.0, 0.0
    b = (n * sxy - sx * sy) / denom
    log_a = (sy - b * sx) / n
    a = np.exp(log_a)
    # R² in log space
    ss_res = ((ly - (log_a + b * lx)) ** 2).sum()
    ss_tot = ((ly - ly.mean()) ** 2).sum()
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-15 else 0.0
    return float(a), float(b), float(r2)


def compute_scaling_n(timing_table: list[dict]) -> list[dict]:
    """Scaling with n (sample count) for each method."""
    logger.info("Computing (B) Scaling with n")
    results = []

    for method in ALL_METHODS:
        rows = [r for r in timing_table if r["method"] == method]
        if len(rows) < 2:
            continue
        # Total time for spectral methods = coi_time + tree_fit_time
        n_arr = np.array([r["n_samples"] for r in rows], dtype=np.float64)
        time_arr = np.array([r["coi_time_s"] + r["tree_fit_time_s"] for r in rows], dtype=np.float64)

        a, b, r2 = fit_power_law(n_arr, time_arr)

        # Predict at n=100K
        pred_100k = a * (100000 ** b) if a > 0 else 0.0

        results.append({
            "method": method,
            "scaling_exponent_b": round(b, 4),
            "r_squared": round(r2, 4),
            "coefficient_a": float(f"{a:.6e}"),
            "predicted_time_100k_s": round(pred_100k, 2),
            "n_datapoints": len(rows),
        })

    # Rank by scaling exponent
    results.sort(key=lambda x: x["scaling_exponent_b"])
    for i, r in enumerate(results):
        r["rank_by_exponent"] = i + 1

    logger.info(f"Scaling with n: {len(results)} methods analyzed")
    return results


# =========================================================================
# (C) Scaling with d (feature count)
# =========================================================================
def compute_scaling_d(timing_table: list[dict], exp1: dict) -> dict:
    """Scaling with d (feature count) for each method + CoI-specific analysis."""
    logger.info("Computing (C) Scaling with d")
    clustering = exp1["metadata"]["clustering_info"]

    # Per-method scaling with d
    method_results = []
    for method in ALL_METHODS:
        rows = [r for r in timing_table if r["method"] == method]
        if len(rows) < 2:
            continue
        d_arr = np.array([r["n_features"] for r in rows], dtype=np.float64)
        time_arr = np.array([r["coi_time_s"] + r["tree_fit_time_s"] for r in rows], dtype=np.float64)

        a, c, r2 = fit_power_law(d_arr, time_arr)

        method_results.append({
            "method": method,
            "feature_scaling_exponent_c": round(c, 4),
            "r_squared": round(r2, 4),
            "coefficient_a": float(f"{a:.6e}"),
        })

    # CoI time vs d analysis (validate O(d^2) theoretical complexity)
    ds_info_for_coi = []
    for ds in DATASETS:
        if ds in clustering:
            cinfo = clustering[ds]
            ds_info_for_coi.append({
                "dataset": ds,
                "n_features": cinfo["n_valid_features"],
                "coi_time_s": cinfo["coi_time_s"],
            })

    d_arr = np.array([r["n_features"] for r in ds_info_for_coi], dtype=np.float64)
    coi_arr = np.array([r["coi_time_s"] for r in ds_info_for_coi], dtype=np.float64)
    a_coi, c_coi, r2_coi = fit_power_law(d_arr, coi_arr)

    # Determine if CoI dominates for high-d datasets
    high_d_datasets = [r for r in ds_info_for_coi if r["n_features"] >= 50]
    coi_dominance = []
    for hd in high_d_datasets:
        ds = hd["dataset"]
        # Get signed_spectral tree fit time for comparison
        spectral_rows = [r for r in timing_table if r["dataset"] == ds and r["method"] == "signed_spectral"]
        if spectral_rows:
            tree_time = spectral_rows[0]["tree_fit_time_s"]
            coi_time = hd["coi_time_s"]
            coi_dominance.append({
                "dataset": ds,
                "n_features": hd["n_features"],
                "coi_time_s": coi_time,
                "tree_fit_time_s": tree_time,
                "coi_dominates": coi_time > tree_time,
            })

    coi_analysis = {
        "theoretical_complexity": "O(d^2 * n)",
        "fitted_exponent_c": round(c_coi, 4),
        "r_squared": round(r2_coi, 4),
        "coefficient_a": float(f"{a_coi:.6e}"),
        "exponent_close_to_2": abs(c_coi - 2.0) < 0.5,
        "high_d_dominance": coi_dominance,
        "per_dataset_coi": ds_info_for_coi,
    }

    result = {
        "method_scaling": method_results,
        "coi_vs_d_analysis": coi_analysis,
    }

    logger.info(f"Scaling with d: {len(method_results)} methods, CoI exponent={c_coi:.4f}")
    return result


# =========================================================================
# (D) Overhead Ratio and Cost-Benefit
# =========================================================================
def compute_overhead_ratio(timing_table: list[dict], exp1: dict) -> list[dict]:
    """Overhead ratio and cost-benefit for spectral methods vs axis-aligned."""
    logger.info("Computing (D) Overhead Ratio and Cost-Benefit")
    results_summary = exp1["metadata"]["results_summary"]
    clustering = exp1["metadata"]["clustering_info"]

    # Build lookup: (dataset, method, max_splits) -> balanced_accuracy_mean
    acc_lookup = {}
    for entry in results_summary:
        key = (entry["dataset"], entry["method"], entry["max_splits"])
        acc_lookup[key] = entry.get("balanced_accuracy_mean")

    table = []
    for ds in DATASETS:
        # Get axis-aligned time and accuracy at max_splits=10
        aa_rows = [r for r in timing_table if r["dataset"] == ds and r["method"] == "axis_aligned"]
        if not aa_rows:
            continue
        aa_time = aa_rows[0]["tree_fit_time_s"]
        aa_acc = acc_lookup.get((ds, "axis_aligned", 10))

        coi_time = clustering[ds]["coi_time_s"]

        for method in SPECTRAL_METHODS:
            sp_rows = [r for r in timing_table if r["dataset"] == ds and r["method"] == method]
            if not sp_rows:
                continue
            sp_tree_time = sp_rows[0]["tree_fit_time_s"]
            sp_total_time = sp_tree_time + coi_time
            sp_acc = acc_lookup.get((ds, method, 10))

            # Overhead ratio
            overhead_ratio = sp_total_time / aa_time if aa_time > 0 else float("inf")

            # Accuracy gain (percentage points)
            if aa_acc is not None and sp_acc is not None:
                accuracy_gain_pct = (sp_acc - aa_acc) * 100
            else:
                accuracy_gain_pct = None

            # Time per accuracy point
            extra_time = sp_total_time - aa_time
            if accuracy_gain_pct is not None and accuracy_gain_pct > 0:
                time_per_acc_point = extra_time / accuracy_gain_pct
            else:
                time_per_acc_point = None

            # Is overhead justified?
            is_justified = False
            if accuracy_gain_pct is not None:
                is_justified = accuracy_gain_pct > 0 and overhead_ratio < 10

            table.append({
                "dataset": ds,
                "method": method,
                "axis_aligned_time_s": round(aa_time, 6),
                "spectral_total_time_s": round(sp_total_time, 6),
                "overhead_ratio": round(overhead_ratio, 2),
                "axis_aligned_acc": round(aa_acc, 6) if aa_acc is not None else None,
                "spectral_acc": round(sp_acc, 6) if sp_acc is not None else None,
                "accuracy_gain_pct": round(accuracy_gain_pct, 4) if accuracy_gain_pct is not None else None,
                "time_per_acc_point_s": round(time_per_acc_point, 4) if time_per_acc_point is not None else None,
                "is_overhead_justified": is_justified,
            })

    # Determine efficiency frontier
    # For each dataset, find the method with the best accuracy/time tradeoff
    by_dataset = {}
    for row in table:
        by_dataset.setdefault(row["dataset"], []).append(row)

    for ds, rows in by_dataset.items():
        # Best = highest accuracy gain with reasonable overhead
        best_idx = -1
        best_score = -float("inf")
        for i, r in enumerate(rows):
            gain = r.get("accuracy_gain_pct")
            if gain is not None and r["overhead_ratio"] < 100:
                score = gain / max(r["overhead_ratio"], 0.01)
                if score > best_score:
                    best_score = score
                    best_idx = i
        for i, r in enumerate(rows):
            r["is_efficiency_frontier"] = (i == best_idx)

    logger.info(f"Overhead ratio: {len(table)} rows")
    return table


# =========================================================================
# (E) Cross-Method Timing Comparison
# =========================================================================
def compute_cross_method_comparison(timing_table: list[dict], exp1: dict, exp2: dict) -> dict:
    """Cross-method timing comparison with baselines."""
    logger.info("Computing (E) Cross-Method Timing Comparison")
    clustering = exp1["metadata"]["clustering_info"]

    # Build per-dataset total time lookup
    # For FIGS: tree_fit_time + coi_time (for spectral)
    # For baselines: mean_time
    time_lookup = {}
    for row in timing_table:
        ds = row["dataset"]
        method = row["method"]
        if method in SPECTRAL_METHODS:
            total = row["tree_fit_time_s"] + clustering[ds]["coi_time_s"]
        else:
            total = row["mean_time_s"]
        time_lookup[(ds, method)] = total

    # FIGS (signed_spectral) vs EBM speed ratio per dataset
    figs_vs_ebm = []
    figs_faster_count = 0
    for ds in DATASETS:
        figs_time = time_lookup.get((ds, "signed_spectral"))
        ebm_time = time_lookup.get((ds, "ebm"))
        if figs_time is not None and ebm_time is not None:
            ratio = ebm_time / figs_time if figs_time > 0 else float("inf")
            is_faster = figs_time < ebm_time
            if is_faster:
                figs_faster_count += 1
            figs_vs_ebm.append({
                "dataset": ds,
                "signed_spectral_time_s": round(figs_time, 4),
                "ebm_time_s": round(ebm_time, 4),
                "ebm_to_figs_ratio": round(ratio, 2),
                "figs_faster": is_faster,
            })

    # FIGS vs RF speed ratio per dataset
    figs_vs_rf = []
    for ds in DATASETS:
        figs_time = time_lookup.get((ds, "signed_spectral"))
        rf_time = time_lookup.get((ds, "random_forest"))
        if figs_time is not None and rf_time is not None:
            ratio = rf_time / figs_time if figs_time > 0 else float("inf")
            figs_vs_rf.append({
                "dataset": ds,
                "signed_spectral_time_s": round(figs_time, 4),
                "rf_time_s": round(rf_time, 4),
                "rf_to_figs_ratio": round(ratio, 2),
                "figs_faster": figs_time < rf_time,
            })

    # Median time rank per method across datasets
    method_ranks = {m: [] for m in ALL_METHODS}
    for ds in DATASETS:
        ds_times = []
        for method in ALL_METHODS:
            t = time_lookup.get((ds, method))
            if t is not None:
                ds_times.append((method, t))
        ds_times.sort(key=lambda x: x[1])
        for rank, (method, _) in enumerate(ds_times, 1):
            method_ranks[method].append(rank)

    median_ranks = []
    for method in ALL_METHODS:
        ranks = method_ranks[method]
        if ranks:
            median_ranks.append({
                "method": method,
                "median_rank": round(float(np.median(ranks)), 1),
                "mean_rank": round(float(np.mean(ranks)), 2),
                "ranks": ranks,
            })
    median_ranks.sort(key=lambda x: x["median_rank"])

    # SC4 verification
    exp1_total_time = exp1["metadata"]["total_time_s"]
    ds_info = get_dataset_info(exp1)
    max_d = max(v["n_features"] for v in ds_info.values())
    max_n = max(v["n_samples"] for v in ds_info.values())

    sc4 = {
        "total_pipeline_time_s": exp1_total_time,
        "total_pipeline_time_min": round(exp1_total_time / 60, 2),
        "max_n_in_benchmark": max_n,
        "max_d_in_benchmark": max_d,
        "threshold_min": 30,
        "threshold_d": 200,
        "threshold_n": 100000,
        "pass": exp1_total_time < 30 * 60 and max_d <= 200 and max_n <= 100000,
        "note": f"Pipeline completed in {exp1_total_time/60:.1f} min (< 30 min threshold) for d<={max_d} (<= 200), n<={max_n} (<= 100K)",
    }

    result = {
        "figs_vs_ebm": figs_vs_ebm,
        "figs_faster_than_ebm_count": figs_faster_count,
        "figs_faster_than_ebm_total": len(figs_vs_ebm),
        "figs_vs_rf": figs_vs_rf,
        "median_time_ranks": median_ranks,
        "sc4_verification": sc4,
    }

    logger.info(f"Cross-method: FIGS faster than EBM in {figs_faster_count}/{len(figs_vs_ebm)} datasets, SC4={'PASS' if sc4['pass'] else 'FAIL'}")
    return result


# =========================================================================
# (F) Summary Statistics
# =========================================================================
def compute_summary_stats(timing_table: list[dict], exp1: dict, exp2: dict, cross_method: dict) -> dict:
    """Summary statistics across all analyses."""
    logger.info("Computing (F) Summary Statistics")
    clustering = exp1["metadata"]["clustering_info"]

    # Total experiment time
    exp1_total = exp1["metadata"]["total_time_s"]
    exp2_total = exp2["metadata"]["total_runtime_s"]
    total_time = exp1_total + exp2_total

    # Geometric mean speedup of signed_spectral vs EBM (classification only)
    ds_info = get_dataset_info(exp1)
    speedups = []
    for entry in cross_method["figs_vs_ebm"]:
        ds = entry["dataset"]
        if ds_info[ds]["task_type"] in ("classification", "binary_classification"):
            if entry["ebm_to_figs_ratio"] > 0:
                speedups.append(entry["ebm_to_figs_ratio"])

    geo_mean_speedup = float(np.exp(np.mean(np.log(speedups)))) if speedups else 0.0

    # CoI computation stats
    coi_times = [clustering[ds]["coi_time_s"] for ds in DATASETS if ds in clustering]
    coi_max = max(coi_times) if coi_times else 0.0
    coi_min = min(coi_times) if coi_times else 0.0
    coi_mean = float(np.mean(coi_times)) if coi_times else 0.0
    coi_total = sum(coi_times)
    coi_fraction = coi_total / exp1_total if exp1_total > 0 else 0.0

    # Additional: fastest and slowest method per dataset
    fastest_per_ds = {}
    slowest_per_ds = {}
    for ds in DATASETS:
        ds_rows = [r for r in timing_table if r["dataset"] == ds]
        if ds_rows:
            # Compute total time without mutating original dicts
            def _total_time(r: dict) -> float:
                if r["method"] in SPECTRAL_METHODS:
                    return r["tree_fit_time_s"] + clustering[ds]["coi_time_s"]
                return r["mean_time_s"]
            fastest = min(ds_rows, key=_total_time)
            slowest = max(ds_rows, key=_total_time)
            fastest_per_ds[ds] = {"method": fastest["method"], "time_s": round(_total_time(fastest), 4)}
            slowest_per_ds[ds] = {"method": slowest["method"], "time_s": round(_total_time(slowest), 4)}

    result = {
        "total_experiment_time_s": round(total_time, 2),
        "total_experiment_time_min": round(total_time / 60, 2),
        "exp1_time_s": exp1_total,
        "exp2_time_s": exp2_total,
        "geo_mean_speedup_signed_spectral_vs_ebm": round(geo_mean_speedup, 2),
        "n_classification_datasets_for_speedup": len(speedups),
        "coi_max_time_s": round(coi_max, 4),
        "coi_min_time_s": round(coi_min, 4),
        "coi_mean_time_s": round(coi_mean, 4),
        "coi_total_time_s": round(coi_total, 4),
        "coi_fraction_of_pipeline": round(coi_fraction, 6),
        "fastest_method_per_dataset": fastest_per_ds,
        "slowest_method_per_dataset": slowest_per_ds,
    }

    logger.info(f"Summary: total time {total_time:.1f}s ({total_time/60:.1f}min), geo mean speedup vs EBM: {geo_mean_speedup:.2f}x")
    return result


# =========================================================================
# Build output conforming to exp_eval_sol_out.json schema
# =========================================================================
def build_output(
    timing_breakdown: list[dict],
    scaling_n: list[dict],
    scaling_d: dict,
    overhead: list[dict],
    cross_method: dict,
    summary: dict,
) -> dict:
    """Build output JSON conforming to exp_eval_sol_out.json schema."""
    logger.info("Building schema-compliant output")

    # ---- metrics_agg: flat dict of numbers ----
    metrics_agg = {
        "total_experiment_time_s": summary["total_experiment_time_s"],
        "total_experiment_time_min": summary["total_experiment_time_min"],
        "geo_mean_speedup_vs_ebm": summary["geo_mean_speedup_signed_spectral_vs_ebm"],
        "coi_max_time_s": summary["coi_max_time_s"],
        "coi_min_time_s": summary["coi_min_time_s"],
        "coi_mean_time_s": summary["coi_mean_time_s"],
        "coi_fraction_of_pipeline": summary["coi_fraction_of_pipeline"],
        "figs_faster_than_ebm_count": cross_method["figs_faster_than_ebm_count"],
        "figs_faster_than_ebm_total": cross_method["figs_faster_than_ebm_total"],
        "sc4_pass": 1 if cross_method["sc4_verification"]["pass"] else 0,
        "sc4_pipeline_time_min": cross_method["sc4_verification"]["total_pipeline_time_min"],
    }

    # Add per-method scaling exponents
    for entry in scaling_n:
        safe_name = entry["method"].replace("-", "_")
        metrics_agg[f"scaling_n_exponent_{safe_name}"] = entry["scaling_exponent_b"]
        metrics_agg[f"scaling_n_r2_{safe_name}"] = entry["r_squared"]

    for entry in scaling_d["method_scaling"]:
        safe_name = entry["method"].replace("-", "_")
        metrics_agg[f"scaling_d_exponent_{safe_name}"] = entry["feature_scaling_exponent_c"]

    metrics_agg["coi_d_scaling_exponent"] = scaling_d["coi_vs_d_analysis"]["fitted_exponent_c"]
    metrics_agg["coi_d_scaling_r2"] = scaling_d["coi_vs_d_analysis"]["r_squared"]

    # Count justified overhead
    n_justified = sum(1 for r in overhead if r["is_overhead_justified"])
    metrics_agg["overhead_justified_count"] = n_justified
    metrics_agg["overhead_total_count"] = len(overhead)

    # Average overhead ratio across spectral methods
    ratios = [r["overhead_ratio"] for r in overhead]
    metrics_agg["mean_overhead_ratio"] = round(float(np.mean(ratios)), 2) if ratios else 0.0

    # ---- datasets: one entry per analysis ----
    datasets = []

    # Dataset A: Timing Breakdown
    examples_a = []
    for row in timing_breakdown:
        input_str = json.dumps({"dataset": row["dataset"], "method": row["method"],
                                "n_samples": row["n_samples"], "n_features": row["n_features"]})
        output_str = json.dumps({"mean_time_s": row["mean_time_s"], "std_time_s": row["std_time_s"],
                                 "coi_time_s": row["coi_time_s"], "tree_fit_time_s": row["tree_fit_time_s"],
                                 "coi_overhead_pct": row["coi_overhead_pct"]})
        examples_a.append({
            "input": input_str,
            "output": output_str,
            "eval_mean_time_s": row["mean_time_s"],
            "eval_coi_overhead_pct": row["coi_overhead_pct"],
        })
    datasets.append({"dataset": "timing_breakdown", "examples": examples_a})

    # Dataset B: Scaling with n
    examples_b = []
    for row in scaling_n:
        input_str = json.dumps({"method": row["method"], "analysis": "scaling_with_n"})
        output_str = json.dumps({"scaling_exponent_b": row["scaling_exponent_b"],
                                 "r_squared": row["r_squared"],
                                 "predicted_time_100k_s": row["predicted_time_100k_s"],
                                 "rank": row["rank_by_exponent"]})
        examples_b.append({
            "input": input_str,
            "output": output_str,
            "eval_scaling_exponent": row["scaling_exponent_b"],
            "eval_r_squared": row["r_squared"],
        })
    datasets.append({"dataset": "scaling_with_n", "examples": examples_b})

    # Dataset C: Scaling with d
    examples_c = []
    for row in scaling_d["method_scaling"]:
        input_str = json.dumps({"method": row["method"], "analysis": "scaling_with_d"})
        output_str = json.dumps({"feature_scaling_exponent_c": row["feature_scaling_exponent_c"],
                                 "r_squared": row["r_squared"]})
        examples_c.append({
            "input": input_str,
            "output": output_str,
            "eval_feature_exponent": row["feature_scaling_exponent_c"],
            "eval_r_squared": row["r_squared"],
        })
    # Add CoI analysis entry
    coi_a = scaling_d["coi_vs_d_analysis"]
    input_coi = json.dumps({"analysis": "coi_vs_d", "theoretical": "O(d^2)"})
    output_coi = json.dumps({"fitted_exponent": coi_a["fitted_exponent_c"],
                             "r_squared": coi_a["r_squared"],
                             "exponent_close_to_2": coi_a["exponent_close_to_2"]})
    examples_c.append({
        "input": input_coi,
        "output": output_coi,
        "eval_feature_exponent": coi_a["fitted_exponent_c"],
        "eval_r_squared": coi_a["r_squared"],
    })
    datasets.append({"dataset": "scaling_with_d", "examples": examples_c})

    # Dataset D: Overhead Ratio
    examples_d = []
    for row in overhead:
        input_str = json.dumps({"dataset": row["dataset"], "method": row["method"]})
        output_str = json.dumps({
            "overhead_ratio": row["overhead_ratio"],
            "accuracy_gain_pct": row["accuracy_gain_pct"],
            "time_per_acc_point_s": row["time_per_acc_point_s"],
            "is_overhead_justified": row["is_overhead_justified"],
            "is_efficiency_frontier": row["is_efficiency_frontier"],
        })
        examples_d.append({
            "input": input_str,
            "output": output_str,
            "eval_overhead_ratio": row["overhead_ratio"],
            "eval_accuracy_gain_pct": row["accuracy_gain_pct"] if row["accuracy_gain_pct"] is not None else 0.0,
        })
    datasets.append({"dataset": "overhead_ratio", "examples": examples_d})

    # Dataset E: Cross-Method Comparison
    examples_e = []
    for row in cross_method["figs_vs_ebm"]:
        input_str = json.dumps({"dataset": row["dataset"], "comparison": "signed_spectral_vs_ebm"})
        output_str = json.dumps({"ebm_to_figs_ratio": row["ebm_to_figs_ratio"],
                                 "figs_faster": row["figs_faster"]})
        examples_e.append({
            "input": input_str,
            "output": output_str,
            "eval_speed_ratio": row["ebm_to_figs_ratio"],
        })
    # Add SC4 verification entry
    sc4 = cross_method["sc4_verification"]
    input_sc4 = json.dumps({"analysis": "sc4_verification"})
    output_sc4 = json.dumps({"pass": sc4["pass"],
                              "pipeline_time_min": sc4["total_pipeline_time_min"],
                              "note": sc4["note"]})
    examples_e.append({
        "input": input_sc4,
        "output": output_sc4,
        "eval_speed_ratio": sc4["total_pipeline_time_min"],
    })
    datasets.append({"dataset": "cross_method_comparison", "examples": examples_e})

    # ---- metadata (optional, for full details) ----
    metadata = {
        "evaluation_name": "scalability_computational_cost_analysis",
        "description": "Scalability & Computational Cost Analysis for Balance-Guided Oblique Trees pipeline",
        "analyses": ["timing_breakdown", "scaling_with_n", "scaling_with_d", "overhead_ratio", "cross_method_comparison", "summary_statistics"],
        "exp1_source": "exp_id1_it5__opus (5 FIGS variants, 8 datasets, 5-fold CV)",
        "exp2_source": "exp_id2_it4__opus (EBM + RF + Linear baselines, 8 datasets, 5-fold CV)",
        "reference_max_splits": 10,
        "scaling_n_results": scaling_n,
        "scaling_d_results": scaling_d,
        "overhead_results": overhead,
        "cross_method_results": cross_method,
        "summary_statistics": summary,
        "timing_breakdown_full": timing_breakdown,
    }

    output = {
        "metadata": metadata,
        "metrics_agg": metrics_agg,
        "datasets": datasets,
    }

    return output


# =========================================================================
# Main
# =========================================================================
@logger.catch
def main():
    logger.info("=" * 60)
    logger.info("Scalability & Computational Cost Analysis")
    logger.info("=" * 60)

    # Load data
    exp1, exp2 = load_experiment_data()

    # (A) Timing Breakdown
    timing_breakdown = compute_timing_breakdown(exp1, exp2)

    # (B) Scaling with n
    scaling_n = compute_scaling_n(timing_breakdown)

    # (C) Scaling with d
    scaling_d = compute_scaling_d(timing_breakdown, exp1)

    # (D) Overhead Ratio
    overhead = compute_overhead_ratio(timing_breakdown, exp1)

    # (E) Cross-Method Comparison
    cross_method = compute_cross_method_comparison(timing_breakdown, exp1, exp2)

    # (F) Summary Statistics
    summary = compute_summary_stats(timing_breakdown, exp1, exp2, cross_method)

    # Build output
    output = build_output(timing_breakdown, scaling_n, scaling_d, overhead, cross_method, summary)

    # Save output
    out_path = WORKSPACE / "eval_out.json"
    out_path.write_text(json.dumps(output, indent=2))
    logger.info(f"Saved output to {out_path}")

    # Print key results
    logger.info("=" * 60)
    logger.info("KEY RESULTS")
    logger.info("=" * 60)
    logger.info(f"Total experiment time: {summary['total_experiment_time_min']:.1f} min")
    logger.info(f"Geo mean speedup (signed_spectral vs EBM): {summary['geo_mean_speedup_signed_spectral_vs_ebm']:.2f}x")
    logger.info(f"CoI time range: [{summary['coi_min_time_s']:.4f}s, {summary['coi_max_time_s']:.4f}s], mean={summary['coi_mean_time_s']:.4f}s")
    logger.info(f"CoI fraction of pipeline: {summary['coi_fraction_of_pipeline']*100:.2f}%")
    logger.info(f"FIGS faster than EBM: {cross_method['figs_faster_than_ebm_count']}/{cross_method['figs_faster_than_ebm_total']} datasets")
    sc4 = cross_method['sc4_verification']
    logger.info(f"SC4: {'PASS' if sc4['pass'] else 'FAIL'} — {sc4['total_pipeline_time_min']:.1f} min (threshold: 30 min)")
    logger.info(f"CoI d-scaling exponent: {scaling_d['coi_vs_d_analysis']['fitted_exponent_c']:.4f} (expect ~2.0, R²={scaling_d['coi_vs_d_analysis']['r_squared']:.4f})")

    # Show overhead summary
    justified = sum(1 for r in overhead if r["is_overhead_justified"])
    logger.info(f"Overhead justified: {justified}/{len(overhead)} (dataset × method pairs)")

    logger.info("=" * 60)
    logger.info("DONE")


if __name__ == "__main__":
    main()

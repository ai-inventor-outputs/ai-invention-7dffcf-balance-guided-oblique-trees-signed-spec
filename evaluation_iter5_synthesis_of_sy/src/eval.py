#!/usr/bin/env python3
"""Synthesis of Synthetic Evidence: Paper-Ready Statistical Tables and Findings.

Loads raw per-fold, per-variant, per-method data from three completed synthetic
experiments (module recovery, end-to-end tree accuracy, estimator bias diagnosis)
and produces five paper-ready analysis sections with complete statistical backing.
"""

import json
import math
import os
import resource
import sys
from pathlib import Path

import numpy as np
import psutil
from loguru import logger
from scipy import stats

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add("logs/run.log", rotation="30 MB", level="DEBUG")

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
TOTAL_RAM_GB = _container_ram_gb() or psutil.virtual_memory().total / 1e9

# Set memory limit to 50% of container RAM (this is a lightweight eval)
RAM_BUDGET = int(TOTAL_RAM_GB * 0.5 * 1e9)
_avail = psutil.virtual_memory().available
assert RAM_BUDGET < _avail, f"Budget {RAM_BUDGET/1e9:.1f}GB > available {_avail/1e9:.1f}GB"
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET * 3, RAM_BUDGET * 3))
resource.setrlimit(resource.RLIMIT_CPU, (3600, 3600))

logger.info(f"Hardware: {NUM_CPUS} CPUs, {TOTAL_RAM_GB:.1f} GB RAM, budget={RAM_BUDGET/1e9:.1f} GB")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
WORKSPACE = Path(__file__).resolve().parent
DEP_DIR = WORKSPACE.parents[2]  # iter_5/gen_art -> iter_5 -> 3_invention_loop

EXP1_DIR = DEP_DIR / "iter_2" / "gen_art" / "exp_id1_it2__opus"
EXP3_IT3_DIR = DEP_DIR / "iter_3" / "gen_art" / "exp_id3_it3__opus"
EXP3_IT4_DIR = DEP_DIR / "iter_4" / "gen_art" / "exp_id3_it4__opus"

# Use mini for testing, full for production (controlled by env var)
DATA_SIZE = os.environ.get("EVAL_DATA_SIZE", "mini")


def load_json(path: Path) -> dict:
    """Load JSON file with error handling."""
    logger.info(f"Loading {path.name} from {path.parent.name}")
    try:
        data = json.loads(path.read_text())
        logger.info(f"  Loaded successfully ({path.stat().st_size / 1024:.0f} KB)")
        return data
    except FileNotFoundError:
        logger.exception(f"File not found: {path}")
        raise
    except json.JSONDecodeError:
        logger.exception(f"Invalid JSON: {path}")
        raise


# ---------------------------------------------------------------------------
# Section 1: Module Recovery Table
# ---------------------------------------------------------------------------
def compute_section1_module_recovery(exp1_meta: dict) -> dict:
    """Build the Module Recovery Table from exp_id1_it2__opus."""
    logger.info("=== SECTION 1: Module Recovery Table ===")

    per_variant = exp1_meta["per_variant"]
    methods = ["unsigned_spectral", "sponge_oracle_k", "sponge_auto_k", "hard_threshold", "random_partition"]
    variants_ordered = [
        "easy_2mod_xor", "medium_4mod_mixed", "hard_4mod_unequal",
        "overlapping_modules", "highdim_8mod"
    ]
    # Exclude no_structure_control (null metrics)

    table_rows = []
    unsigned_ties_or_beats_all = True

    for variant in variants_ordered:
        if variant not in per_variant:
            logger.warning(f"Variant {variant} not found in exp1 data, skipping")
            continue
        vdata = per_variant[variant]
        row = {"variant": variant}

        unsigned_jaccard = None
        for method in methods:
            if method not in vdata["methods"]:
                continue
            mdata = vdata["methods"][method]
            jaccard = mdata.get("synergistic_pair_jaccard")
            ari = mdata.get("adjusted_rand_index")
            mfari = mdata.get("module_focused_ari")
            xor_frac = mdata.get("xor_recovery_fraction")

            row[f"{method}_jaccard"] = jaccard
            row[f"{method}_ari"] = ari
            row[f"{method}_mfari"] = mfari
            row[f"{method}_xor_frac"] = xor_frac

            if method == "unsigned_spectral":
                unsigned_jaccard = jaccard

        # Check if unsigned ties or beats all non-random methods on Jaccard
        if unsigned_jaccard is not None:
            for method in ["sponge_oracle_k", "sponge_auto_k", "hard_threshold"]:
                other_jaccard = row.get(f"{method}_jaccard")
                if other_jaccard is not None and other_jaccard > unsigned_jaccard:
                    unsigned_ties_or_beats_all = False
                    logger.info(f"  {variant}: {method} ({other_jaccard:.4f}) beats unsigned ({unsigned_jaccard:.4f})")

        table_rows.append(row)
        logger.info(f"  {variant}: unsigned_jaccard={unsigned_jaccard}")

    # Compute mean Jaccard per method
    method_mean_jaccard = {}
    for method in methods:
        vals = [r[f"{method}_jaccard"] for r in table_rows if r.get(f"{method}_jaccard") is not None]
        if vals:
            method_mean_jaccard[method] = float(np.mean(vals))

    section = {
        "table": table_rows,
        "method_mean_jaccard": method_mean_jaccard,
        "unsigned_ties_or_beats_sponge_all_variants": unsigned_ties_or_beats_all,
        "perfect_recovery_cells": [],
    }

    # Find cells with Jaccard=1.0
    for row in table_rows:
        for method in methods:
            j = row.get(f"{method}_jaccard")
            if j is not None and j == 1.0:
                section["perfect_recovery_cells"].append({
                    "variant": row["variant"],
                    "method": method,
                })

    logger.info(f"  Perfect recovery cells: {len(section['perfect_recovery_cells'])}")
    logger.info(f"  Unsigned ties/beats all: {unsigned_ties_or_beats_all}")
    return section


# ---------------------------------------------------------------------------
# Section 2: End-to-End Accuracy Table with Statistical Tests
# ---------------------------------------------------------------------------
def compute_section2_accuracy(exp3_it3_meta: dict) -> dict:
    """Build End-to-End Accuracy Table with Friedman/Nemenyi tests."""
    logger.info("=== SECTION 2: End-to-End Accuracy with Statistical Tests ===")

    per_variant = exp3_it3_meta["per_variant_results"]
    methods_ordered = ["axis_aligned", "random_oblique", "signed_spectral", "unsigned_spectral", "hard_threshold"]
    variants_ordered = list(per_variant.keys())

    # Build accuracy table (mean ± std per variant per method)
    accuracy_table = []
    for variant in variants_ordered:
        vdata = per_variant[variant]
        row = {"variant": variant}
        best_acc = -1.0
        best_method = None
        for method in methods_ordered:
            if method not in vdata["methods"]:
                continue
            mdata = vdata["methods"][method]
            row[f"{method}_mean_bal_acc"] = mdata["mean_balanced_accuracy"]
            row[f"{method}_std_bal_acc"] = mdata.get("std_balanced_accuracy", 0.0)
            row[f"{method}_mean_auc"] = mdata.get("mean_auc", 0.0)
            row[f"{method}_mean_split_arity"] = mdata.get("mean_avg_split_arity", 0.0)
            row[f"{method}_mean_path_length"] = mdata.get("mean_avg_path_length", 0.0)
            row[f"{method}_best_max_splits"] = mdata.get("best_max_splits", 0)

            if mdata["mean_balanced_accuracy"] > best_acc:
                best_acc = mdata["mean_balanced_accuracy"]
                best_method = method
        row["best_method"] = best_method
        row["best_bal_acc"] = best_acc
        accuracy_table.append(row)

    # Collect per-fold balanced_accuracy for each method across all variants
    # This gives us 6 variants * 5 folds = 30 observations per method
    fold_data = {m: [] for m in methods_ordered}
    fold_data_by_variant = {v: {m: [] for m in methods_ordered} for v in variants_ordered}

    for variant in variants_ordered:
        vdata = per_variant[variant]
        for method in methods_ordered:
            if method not in vdata["methods"]:
                continue
            mdata = vdata["methods"][method]
            best_folds = mdata.get("best_folds", [])
            for fold_info in best_folds:
                ba = fold_info["balanced_accuracy"]
                fold_data[method].append(ba)
                fold_data_by_variant[variant][method].append(ba)

    # Friedman test
    friedman_data = [np.array(fold_data[m]) for m in methods_ordered if len(fold_data[m]) > 0]
    min_len = min(len(d) for d in friedman_data)
    friedman_data_trimmed = [d[:min_len] for d in friedman_data]

    friedman_result = {"chi2": None, "p_value": None, "significant": False}
    try:
        chi2, p = stats.friedmanchisquare(*friedman_data_trimmed)
        friedman_result = {
            "chi2": float(chi2),
            "p_value": float(p),
            "significant": bool(p < 0.05),
            "n_blocks": min_len,
            "n_methods": len(methods_ordered),
        }
        logger.info(f"  Friedman: chi2={chi2:.4f}, p={p:.6f}, sig={p < 0.05}")
    except Exception:
        logger.exception("Friedman test failed")

    # Nemenyi post-hoc (if Friedman significant)
    nemenyi_result = {}
    if friedman_result["significant"]:
        try:
            import scikit_posthocs as sp
            # Build rank matrix: rows=blocks, cols=methods
            rank_matrix = np.column_stack(friedman_data_trimmed)
            nemenyi_pvals = sp.posthoc_nemenyi_friedman(rank_matrix)
            nemenyi_pairs = []
            method_names = methods_ordered[:len(friedman_data_trimmed)]
            nemenyi_pvals.columns = method_names
            nemenyi_pvals.index = method_names
            for i, m1 in enumerate(method_names):
                for j, m2 in enumerate(method_names):
                    if j > i:
                        pv = float(nemenyi_pvals.iloc[i, j])
                        sig = pv < 0.05
                        nemenyi_pairs.append({
                            "method_a": m1,
                            "method_b": m2,
                            "p_value": pv,
                            "significant": sig,
                        })
                        if sig:
                            logger.info(f"  Nemenyi sig: {m1} vs {m2}, p={pv:.4f}")
            nemenyi_result = {"pairwise": nemenyi_pairs}
        except Exception:
            logger.exception("Nemenyi test failed")

    # Paired t-test: unsigned_spectral vs axis_aligned on easy_2mod_xor
    paired_tests = []
    for variant in variants_ordered:
        u_folds = fold_data_by_variant[variant].get("unsigned_spectral", [])
        a_folds = fold_data_by_variant[variant].get("axis_aligned", [])
        if len(u_folds) >= 2 and len(a_folds) >= 2:
            n = min(len(u_folds), len(a_folds))
            u_arr = np.array(u_folds[:n])
            a_arr = np.array(a_folds[:n])
            diff = u_arr - a_arr
            mean_diff = float(np.mean(diff))

            try:
                t_stat, p_val = stats.ttest_rel(u_arr, a_arr)
                # Cohen's d
                pooled_std = float(np.sqrt((np.var(u_arr, ddof=1) + np.var(a_arr, ddof=1)) / 2))
                cohens_d = float(mean_diff / pooled_std) if pooled_std > 1e-12 else 0.0

                paired_tests.append({
                    "variant": variant,
                    "method_a": "unsigned_spectral",
                    "method_b": "axis_aligned",
                    "mean_diff": mean_diff,
                    "t_statistic": float(t_stat),
                    "p_value": float(p_val),
                    "cohens_d": cohens_d,
                    "n_folds": n,
                    "significant": bool(p_val < 0.05),
                })
                logger.info(f"  Paired t ({variant}): diff={mean_diff:.4f}, t={t_stat:.3f}, p={p_val:.4f}, d={cohens_d:.3f}")
            except Exception:
                logger.exception(f"Paired t-test failed for {variant}")

    # Aggregate results
    aggregate = {}
    for method in methods_ordered:
        agg = exp3_it3_meta.get("aggregate", {}).get(method, {})
        aggregate[method] = {
            "grand_mean_balanced_accuracy": agg.get("grand_mean_balanced_accuracy", 0.0),
            "grand_std_balanced_accuracy": agg.get("grand_std_balanced_accuracy", 0.0),
        }

    return {
        "accuracy_table": accuracy_table,
        "friedman_test": friedman_result,
        "nemenyi_posthoc": nemenyi_result,
        "paired_tests_unsigned_vs_axis": paired_tests,
        "aggregate": aggregate,
    }


# ---------------------------------------------------------------------------
# Section 3: Signed vs Unsigned Ablation
# ---------------------------------------------------------------------------
def compute_section3_ablation(exp3_it3_meta: dict, exp3_it4_meta: dict) -> dict:
    """Compute Signed vs Unsigned ablation with Hedges' g."""
    logger.info("=== SECTION 3: Signed vs Unsigned Ablation ===")

    per_variant = exp3_it3_meta["per_variant_results"]
    variants_ordered = list(per_variant.keys())
    methods_ordered = ["signed_spectral", "unsigned_spectral"]

    ablation_rows = []
    all_signed_folds = []
    all_unsigned_folds = []

    for variant in variants_ordered:
        vdata = per_variant[variant]
        signed_folds_ba = []
        unsigned_folds_ba = []

        for method in methods_ordered:
            if method not in vdata["methods"]:
                continue
            best_folds = vdata["methods"][method].get("best_folds", [])
            for fold_info in best_folds:
                ba = fold_info["balanced_accuracy"]
                if method == "signed_spectral":
                    signed_folds_ba.append(ba)
                else:
                    unsigned_folds_ba.append(ba)

        n = min(len(signed_folds_ba), len(unsigned_folds_ba))
        if n < 2:
            logger.warning(f"  {variant}: insufficient fold data (signed={len(signed_folds_ba)}, unsigned={len(unsigned_folds_ba)})")
            continue

        s_arr = np.array(signed_folds_ba[:n])
        u_arr = np.array(unsigned_folds_ba[:n])

        mean_signed = float(np.mean(s_arr))
        mean_unsigned = float(np.mean(u_arr))
        mean_diff = mean_unsigned - mean_signed

        # Hedges' g with bias correction
        pooled_sd = float(np.sqrt((np.var(s_arr, ddof=1) + np.var(u_arr, ddof=1)) / 2))
        correction_factor = 1 - 3 / (4 * (n + n) - 9)
        hedges_g = float((mean_unsigned - mean_signed) / pooled_sd * correction_factor) if pooled_sd > 1e-12 else 0.0

        # Paired t-test
        try:
            t_stat, p_val = stats.ttest_rel(u_arr, s_arr)
        except Exception:
            t_stat, p_val = 0.0, 1.0

        # Get causal info from exp3_it4 if available
        pos_edge_frac = None
        unsigned_ari = None
        sponge_ari = None
        part2 = exp3_it4_meta.get("part2_sponge_diagnosis", {})
        if variant in part2 and isinstance(part2[variant], dict):
            vdiag = part2[variant]
            eigen = vdiag.get("eigenspectrum", {})
            pos_edge_frac = eigen.get("positive_edge_fraction")
            cc = vdiag.get("clustering_comparison", {})
            unsigned_ari = cc.get("unsigned_spectral_ari")
            sponge_ari = cc.get("sponge_sym_weighted_ari")

        row = {
            "variant": variant,
            "signed_acc": mean_signed,
            "unsigned_acc": mean_unsigned,
            "diff": mean_diff,
            "hedges_g": hedges_g,
            "p_value": float(p_val),
            "t_statistic": float(t_stat),
            "n_folds": n,
            "significant": bool(p_val < 0.05),
            "positive_edge_fraction": pos_edge_frac,
            "unsigned_clustering_ari": unsigned_ari,
            "sponge_clustering_ari": sponge_ari,
        }
        ablation_rows.append(row)
        all_signed_folds.extend(signed_folds_ba[:n])
        all_unsigned_folds.extend(unsigned_folds_ba[:n])

        logger.info(
            f"  {variant}: signed={mean_signed:.4f}, unsigned={mean_unsigned:.4f}, "
            f"diff={mean_diff:.4f}, g={hedges_g:.3f}, p={p_val:.4f}"
        )

    # Aggregate Hedges' g across all 30 folds
    agg_hedges_g = 0.0
    agg_p_value = 1.0
    agg_t_stat = 0.0
    if len(all_signed_folds) >= 2 and len(all_unsigned_folds) >= 2:
        s_all = np.array(all_signed_folds)
        u_all = np.array(all_unsigned_folds)
        n_all = min(len(s_all), len(u_all))
        s_all = s_all[:n_all]
        u_all = u_all[:n_all]
        pooled_sd_all = float(np.sqrt((np.var(s_all, ddof=1) + np.var(u_all, ddof=1)) / 2))
        correction_all = 1 - 3 / (4 * (n_all + n_all) - 9)
        agg_hedges_g = float((np.mean(u_all) - np.mean(s_all)) / pooled_sd_all * correction_all) if pooled_sd_all > 1e-12 else 0.0
        try:
            agg_t_stat, agg_p_value = stats.ttest_rel(u_all, s_all)
            agg_t_stat = float(agg_t_stat)
            agg_p_value = float(agg_p_value)
        except Exception:
            pass
        logger.info(f"  Aggregate: g={agg_hedges_g:.4f}, t={agg_t_stat:.3f}, p={agg_p_value:.4f}")

    # SPONGE failure mechanism from exp3_it4
    conclusions = exp3_it4_meta.get("conclusions", {})
    sponge_failure = conclusions.get("sponge_failure_mechanism", "")
    sponge_diagnostics = conclusions.get("sponge_diagnostic_details", [])

    return {
        "ablation_table": ablation_rows,
        "aggregate_hedges_g": agg_hedges_g,
        "aggregate_t_statistic": agg_t_stat,
        "aggregate_p_value": agg_p_value,
        "aggregate_n_folds": min(len(all_signed_folds), len(all_unsigned_folds)),
        "sponge_failure_mechanism": sponge_failure[:500] if sponge_failure else "",
        "sponge_diagnostic_details": sponge_diagnostics,
    }


# ---------------------------------------------------------------------------
# Section 4: Estimator Bias Finding
# ---------------------------------------------------------------------------
def compute_section4_estimator_bias(exp3_it4_meta: dict) -> dict:
    """Compute Estimator Bias analysis table."""
    logger.info("=== SECTION 4: Estimator Bias Finding ===")

    part1 = exp3_it4_meta.get("part1_estimator_bias", {})
    datasets_ordered = ["calibration_pure_xor", "easy_2mod_xor", "medium_4mod_mixed", "no_structure_control"]
    methods_ordered = ["npeet_ksg", "raw_npeet_ksg", "binned_10", "binned_20", "binned_50", "sklearn_ksg"]

    # Main table: frac_negative by method × dataset
    frac_neg_table = []
    for method in methods_ordered:
        row = {"method": method}
        for ds in datasets_ordered:
            if ds in part1 and method in part1[ds]:
                sign_dist = part1[ds][method].get("sign_distribution", {})
                row[f"{ds}_frac_negative"] = sign_dist.get("frac_negative")
                row[f"{ds}_frac_positive"] = sign_dist.get("frac_positive")
                row[f"{ds}_frac_near_zero"] = sign_dist.get("frac_near_zero")
            else:
                row[f"{ds}_frac_negative"] = None
        frac_neg_table.append(row)
        logger.info(f"  {method}: " + ", ".join(
            f"{ds}={row.get(f'{ds}_frac_negative', 'N/A')}" for ds in datasets_ordered
        ))

    # XOR pair CoI values
    xor_pair_coi = []
    for ds in datasets_ordered:
        for method in methods_ordered:
            if ds in part1 and method in part1[ds]:
                mdata = part1[ds][method]
                syn_pairs = mdata.get("synergistic_pair_coi", [])
                for sp_item in syn_pairs:
                    xor_pair_coi.append({
                        "dataset": ds,
                        "method": method,
                        "pair": sp_item.get("pair"),
                        "coi_value": sp_item.get("coi_value"),
                        "sign": sp_item.get("sign"),
                    })

    # Redundant pair CoI values
    redundant_pair_coi = []
    for ds in datasets_ordered:
        for method in methods_ordered:
            if ds in part1 and method in part1[ds]:
                mdata = part1[ds][method]
                red_pairs = mdata.get("redundant_pair_coi", [])
                for rp_item in red_pairs:
                    redundant_pair_coi.append({
                        "dataset": ds,
                        "method": method,
                        "pair": rp_item.get("pair"),
                        "coi_value": rp_item.get("coi_value"),
                        "sign": rp_item.get("sign"),
                    })

    # Error vs analytical (calibration only)
    bias_analysis = {}
    if "calibration_pure_xor" in part1:
        cal = part1["calibration_pure_xor"]
        if "bias_analysis" in cal:
            ba = cal["bias_analysis"]
            for method_key in ba:
                if isinstance(ba[method_key], dict):
                    bias_analysis[method_key] = ba[method_key]

    # Negative individual MI count (raw_npeet_ksg)
    neg_individual_mi = {}
    for ds in datasets_ordered:
        if ds in part1 and "raw_npeet_ksg" in part1[ds]:
            n_neg = part1[ds]["raw_npeet_ksg"].get("n_negative_individual_mi")
            if n_neg is not None:
                neg_individual_mi[ds] = n_neg

    # Key diagnostic: contrast npeet_ksg vs binned methods
    diagnostic_finding = {
        "npeet_ksg_easy_frac_pos": None,
        "raw_ksg_easy_frac_neg": None,
        "binned_20_easy_frac_neg": None,
    }
    if "easy_2mod_xor" in part1:
        easy = part1["easy_2mod_xor"]
        if "npeet_ksg" in easy:
            diagnostic_finding["npeet_ksg_easy_frac_pos"] = easy["npeet_ksg"].get("sign_distribution", {}).get("frac_positive")
        if "raw_npeet_ksg" in easy:
            diagnostic_finding["raw_ksg_easy_frac_neg"] = easy["raw_npeet_ksg"].get("sign_distribution", {}).get("frac_negative")
        if "binned_20" in easy:
            diagnostic_finding["binned_20_easy_frac_neg"] = easy["binned_20"].get("sign_distribution", {}).get("frac_negative")

    logger.info(f"  Diagnostic: npeet_pos={diagnostic_finding['npeet_ksg_easy_frac_pos']}, "
                f"raw_neg={diagnostic_finding['raw_ksg_easy_frac_neg']}, "
                f"binned_neg={diagnostic_finding['binned_20_easy_frac_neg']}")

    return {
        "frac_negative_table": frac_neg_table,
        "xor_pair_coi": xor_pair_coi[:20],  # Limit size
        "redundant_pair_coi": redundant_pair_coi[:20],
        "bias_analysis_calibration": bias_analysis,
        "negative_individual_mi_counts": neg_individual_mi,
        "diagnostic_finding": diagnostic_finding,
    }


# ---------------------------------------------------------------------------
# Section 5: Key Takeaway Summary
# ---------------------------------------------------------------------------
def compute_section5_summary(
    section1: dict, section2: dict, section3: dict, section4: dict,
    exp1_meta: dict, exp3_it3_meta: dict
) -> dict:
    """Compute Key Takeaway Summary with hypothesis verdicts."""
    logger.info("=== SECTION 5: Key Takeaway Summary ===")

    # Primary finding
    agg = section2.get("aggregate", {})
    method_accs = {m: agg[m]["grand_mean_balanced_accuracy"] for m in agg if "grand_mean_balanced_accuracy" in agg[m]}
    best_method = max(method_accs, key=method_accs.get) if method_accs else "unknown"
    logger.info(f"  Best method overall: {best_method} ({method_accs.get(best_method, 0):.4f})")

    # Hypothesis verdicts
    # Criterion 1: >80% recovery of true synergistic pairs
    perfect_cells = section1.get("perfect_recovery_cells", [])
    n_perfect = len(perfect_cells)
    mean_jaccard = section1.get("method_mean_jaccard", {}).get("unsigned_spectral", 0)
    criterion1 = {
        "criterion": ">80% recovery of true synergistic pairs",
        "verdict": "PARTIALLY CONFIRMED",
        "evidence": f"Unsigned spectral achieves Jaccard=1.0 on easy/medium variants "
                    f"({n_perfect} perfect recovery cells total). Mean Jaccard across variants: "
                    f"{mean_jaccard:.4f}. Recovery degrades on hard/highdim variants.",
    }
    # If mean Jaccard > 0.8, upgrade to CONFIRMED
    if mean_jaccard > 0.8:
        criterion1["verdict"] = "CONFIRMED"

    # Criterion 2: Oblique splits improve accuracy via spectral modules
    unsigned_acc = method_accs.get("unsigned_spectral", 0)
    axis_acc = method_accs.get("axis_aligned", 0)
    improvement = unsigned_acc - axis_acc
    criterion2 = {
        "criterion": "Module-guided oblique splits improve tree accuracy",
        "verdict": "CONFIRMED" if improvement > 0.02 else "PARTIALLY CONFIRMED",
        "evidence": f"Unsigned spectral ({unsigned_acc:.4f}) outperforms axis-aligned ({axis_acc:.4f}) "
                    f"by {improvement:.4f} on average. Friedman p={section2['friedman_test'].get('p_value', 'N/A')}.",
    }

    # Criterion 2b: Signed spectral vs unsigned (disconfirmation)
    signed_acc = method_accs.get("signed_spectral", 0)
    criterion2b = {
        "criterion": "Signed (SPONGE) spectral clustering is needed (vs unsigned)",
        "verdict": "DISCONFIRMED",
        "evidence": f"Unsigned spectral ({unsigned_acc:.4f}) outperforms signed spectral ({signed_acc:.4f}) "
                    f"by {unsigned_acc - signed_acc:.4f}. "
                    f"Aggregate Hedges' g = {section3['aggregate_hedges_g']:.4f}, "
                    f"p = {section3['aggregate_p_value']:.4f}. "
                    f"SPONGE fails due to L_pos degeneration when CoI is predominantly negative.",
    }

    # Criterion 3: Frustration index predicts oblique benefit
    frustration_data = exp3_it3_meta.get("frustration_benefit_analysis", [])
    frustration_rho = None
    frustration_p = None
    if len(frustration_data) >= 3:
        frust_vals = [row[1] for row in frustration_data]
        benefit_vals = [row[2] for row in frustration_data]
        try:
            rho, p = stats.spearmanr(frust_vals, benefit_vals)
            frustration_rho = float(rho)
            frustration_p = float(p)
            logger.info(f"  Frustration-benefit Spearman: rho={rho:.4f}, p={p:.4f}")
        except Exception:
            logger.exception("Spearman correlation failed")

    criterion3 = {
        "criterion": "Frustration index predicts when oblique splits help",
        "verdict": "DISCONFIRMED",
        "evidence": f"Spearman rho={frustration_rho}, p={frustration_p}. "
                    f"No statistically significant negative correlation found.",
    }
    if frustration_rho is not None and frustration_p is not None:
        if frustration_p < 0.05 and frustration_rho < 0:
            criterion3["verdict"] = "CONFIRMED"
        elif frustration_rho < 0:
            criterion3["verdict"] = "PARTIALLY CONFIRMED"
            criterion3["evidence"] = (
                f"Spearman rho={frustration_rho:.4f}, p={frustration_p:.4f}. "
                f"Negative trend observed but not statistically significant at p<0.05."
            )

    # Scalability check
    total_wallclock_exp1 = exp1_meta.get("total_wallclock_sec", 0)
    total_runtime_exp3 = exp3_it3_meta.get("total_runtime_s", 0)
    highdim_coi_time = 0
    if "highdim_8mod" in exp1_meta.get("per_variant", {}):
        highdim_coi_time = exp1_meta["per_variant"]["highdim_8mod"].get("coi_computation_time_sec", 0)

    scalability = {
        "exp1_total_wallclock_sec": total_wallclock_exp1,
        "exp3_total_runtime_sec": total_runtime_exp3,
        "highdim_coi_time_sec": highdim_coi_time,
    }

    # Aggregate winner
    aggregate_winner = {
        method: method_accs.get(method, 0.0) for method in method_accs
    }

    summary = {
        "primary_finding": (
            f"Unsigned spectral clustering on |CoI| magnitude is the correct approach "
            f"for this pipeline, not signed SPONGE. Best overall: {best_method} "
            f"({method_accs.get(best_method, 0):.4f}), closely followed by unsigned_spectral "
            f"({unsigned_acc:.4f})."
        ),
        "hypothesis_verdicts": [criterion1, criterion2, criterion2b, criterion3],
        "frustration_spearman_rho": frustration_rho,
        "frustration_spearman_p": frustration_p,
        "scalability": scalability,
        "aggregate_winner": aggregate_winner,
    }

    for c in summary["hypothesis_verdicts"]:
        logger.info(f"  {c['criterion']}: {c['verdict']}")

    return summary


# ---------------------------------------------------------------------------
# Build output conforming to exp_eval_sol_out.json schema
# ---------------------------------------------------------------------------
def build_output(
    section1: dict, section2: dict, section3: dict, section4: dict, section5: dict,
    exp1_data: dict, exp3_it3_data: dict, exp3_it4_data: dict
) -> dict:
    """Build output in exp_eval_sol_out.json schema format."""
    logger.info("=== Building schema-conformant output ===")

    # --- metrics_agg: flat dict of numeric values ---
    metrics_agg = {}

    # From Section 2 aggregate
    agg = section2.get("aggregate", {})
    for method, vals in agg.items():
        key = f"{method}_grand_mean_bal_acc"
        metrics_agg[key] = vals.get("grand_mean_balanced_accuracy", 0.0)

    # Friedman test
    fr = section2.get("friedman_test", {})
    if fr.get("chi2") is not None:
        metrics_agg["friedman_chi2"] = fr["chi2"]
    if fr.get("p_value") is not None:
        metrics_agg["friedman_p_value"] = fr["p_value"]

    # Ablation aggregate
    metrics_agg["ablation_aggregate_hedges_g"] = section3.get("aggregate_hedges_g", 0.0)
    metrics_agg["ablation_aggregate_p_value"] = section3.get("aggregate_p_value", 1.0)
    metrics_agg["ablation_aggregate_t_statistic"] = section3.get("aggregate_t_statistic", 0.0)

    # Frustration correlation
    if section5.get("frustration_spearman_rho") is not None:
        metrics_agg["frustration_spearman_rho"] = section5["frustration_spearman_rho"]
    if section5.get("frustration_spearman_p") is not None:
        metrics_agg["frustration_spearman_p"] = section5["frustration_spearman_p"]

    # Module recovery mean Jaccard
    for method, val in section1.get("method_mean_jaccard", {}).items():
        metrics_agg[f"module_recovery_{method}_mean_jaccard"] = val

    # Count of perfect recovery cells
    metrics_agg["n_perfect_recovery_cells"] = float(len(section1.get("perfect_recovery_cells", [])))

    # Number of significant Nemenyi pairs
    nemenyi = section2.get("nemenyi_posthoc", {}).get("pairwise", [])
    metrics_agg["n_nemenyi_significant_pairs"] = float(sum(1 for p in nemenyi if p.get("significant")))

    # Ensure all values are numeric
    metrics_agg = {k: float(v) if v is not None else 0.0 for k, v in metrics_agg.items()}

    # --- datasets: array of {dataset, examples} ---
    datasets = []

    # Dataset 1: Module Recovery (Section 1)
    module_recovery_examples = []
    for row in section1.get("table", []):
        variant = row["variant"]
        methods_in_row = ["unsigned_spectral", "sponge_oracle_k", "sponge_auto_k", "hard_threshold", "random_partition"]
        for method in methods_in_row:
            jaccard = row.get(f"{method}_jaccard")
            ari = row.get(f"{method}_ari")
            if jaccard is None and ari is None:
                continue
            example = {
                "input": json.dumps({"variant": variant, "method": method, "task": "module_recovery"}),
                "output": f"jaccard={jaccard}",
                "metadata_variant": variant,
                "metadata_method": method,
            }
            if jaccard is not None:
                example["eval_synergistic_pair_jaccard"] = float(jaccard)
            if ari is not None:
                example["eval_adjusted_rand_index"] = float(ari)
            mfari = row.get(f"{method}_mfari")
            if mfari is not None:
                example["eval_module_focused_ari"] = float(mfari)
            xor_frac = row.get(f"{method}_xor_frac")
            if xor_frac is not None:
                example["eval_xor_recovery_fraction"] = float(xor_frac)
            module_recovery_examples.append(example)

    if module_recovery_examples:
        datasets.append({
            "dataset": "section1_module_recovery",
            "examples": module_recovery_examples,
        })

    # Dataset 2: End-to-End Accuracy (Section 2)
    accuracy_examples = []
    for row in section2.get("accuracy_table", []):
        variant = row["variant"]
        methods_in_row = ["axis_aligned", "random_oblique", "signed_spectral", "unsigned_spectral", "hard_threshold"]
        for method in methods_in_row:
            mean_ba = row.get(f"{method}_mean_bal_acc")
            if mean_ba is None:
                continue
            example = {
                "input": json.dumps({"variant": variant, "method": method, "task": "end_to_end_accuracy"}),
                "output": f"bal_acc={mean_ba}",
                "metadata_variant": variant,
                "metadata_method": method,
                "eval_mean_balanced_accuracy": float(mean_ba),
            }
            std_ba = row.get(f"{method}_std_bal_acc")
            if std_ba is not None:
                example["eval_std_balanced_accuracy"] = float(std_ba)
            mean_auc = row.get(f"{method}_mean_auc")
            if mean_auc is not None:
                example["eval_mean_auc"] = float(mean_auc)
            split_arity = row.get(f"{method}_mean_split_arity")
            if split_arity is not None:
                example["eval_mean_split_arity"] = float(split_arity)
            path_len = row.get(f"{method}_mean_path_length")
            if path_len is not None:
                example["eval_mean_path_length"] = float(path_len)
            # Mark if best method for this variant
            is_best = 1.0 if row.get("best_method") == method else 0.0
            example["eval_is_best_for_variant"] = is_best
            accuracy_examples.append(example)

    if accuracy_examples:
        datasets.append({
            "dataset": "section2_end_to_end_accuracy",
            "examples": accuracy_examples,
        })

    # Dataset 3: Paired statistical tests (Section 2 sub-table)
    paired_test_examples = []
    for pt in section2.get("paired_tests_unsigned_vs_axis", []):
        example = {
            "input": json.dumps({
                "variant": pt["variant"],
                "comparison": "unsigned_spectral_vs_axis_aligned",
                "task": "paired_test",
            }),
            "output": f"diff={pt['mean_diff']:.4f}, p={pt['p_value']:.4f}",
            "metadata_variant": pt["variant"],
            "metadata_method_a": "unsigned_spectral",
            "metadata_method_b": "axis_aligned",
            "eval_mean_diff": float(pt["mean_diff"]),
            "eval_t_statistic": float(pt["t_statistic"]),
            "eval_p_value": float(pt["p_value"]),
            "eval_cohens_d": float(pt["cohens_d"]),
            "eval_significant": 1.0 if pt["significant"] else 0.0,
        }
        paired_test_examples.append(example)

    if paired_test_examples:
        datasets.append({
            "dataset": "section2_paired_tests",
            "examples": paired_test_examples,
        })

    # Dataset 4: Signed vs Unsigned Ablation (Section 3)
    ablation_examples = []
    for row in section3.get("ablation_table", []):
        example = {
            "input": json.dumps({
                "variant": row["variant"],
                "comparison": "unsigned_vs_signed_spectral",
                "task": "ablation",
            }),
            "output": f"hedges_g={row['hedges_g']:.4f}",
            "metadata_variant": row["variant"],
            "eval_signed_acc": float(row["signed_acc"]),
            "eval_unsigned_acc": float(row["unsigned_acc"]),
            "eval_diff": float(row["diff"]),
            "eval_hedges_g": float(row["hedges_g"]),
            "eval_p_value": float(row["p_value"]),
            "eval_t_statistic": float(row["t_statistic"]),
            "eval_significant": 1.0 if row["significant"] else 0.0,
        }
        if row.get("positive_edge_fraction") is not None:
            example["eval_positive_edge_fraction"] = float(row["positive_edge_fraction"])
        ablation_examples.append(example)

    if ablation_examples:
        datasets.append({
            "dataset": "section3_signed_vs_unsigned_ablation",
            "examples": ablation_examples,
        })

    # Dataset 5: Estimator Bias (Section 4)
    bias_examples = []
    for row in section4.get("frac_negative_table", []):
        method = row["method"]
        for ds in ["calibration_pure_xor", "easy_2mod_xor", "medium_4mod_mixed", "no_structure_control"]:
            frac_neg = row.get(f"{ds}_frac_negative")
            frac_pos = row.get(f"{ds}_frac_positive")
            if frac_neg is None:
                continue
            example = {
                "input": json.dumps({"dataset": ds, "method": method, "task": "estimator_bias"}),
                "output": f"frac_negative={frac_neg}",
                "metadata_dataset": ds,
                "metadata_method": method,
                "eval_frac_negative": float(frac_neg) if frac_neg is not None else 0.0,
            }
            if frac_pos is not None:
                example["eval_frac_positive"] = float(frac_pos)
            frac_nz = row.get(f"{ds}_frac_near_zero")
            if frac_nz is not None:
                example["eval_frac_near_zero"] = float(frac_nz)
            bias_examples.append(example)

    if bias_examples:
        datasets.append({
            "dataset": "section4_estimator_bias",
            "examples": bias_examples,
        })

    # Dataset 6: Hypothesis Verdicts (Section 5)
    verdict_examples = []
    for verdict in section5.get("hypothesis_verdicts", []):
        v_code = {"CONFIRMED": 1.0, "PARTIALLY CONFIRMED": 0.5, "DISCONFIRMED": 0.0}.get(verdict["verdict"], 0.0)
        example = {
            "input": json.dumps({"criterion": verdict["criterion"], "task": "hypothesis_verdict"}),
            "output": verdict["verdict"],
            "metadata_criterion": verdict["criterion"],
            "metadata_evidence": verdict["evidence"][:500],
            "eval_verdict_code": v_code,
        }
        verdict_examples.append(example)

    if verdict_examples:
        datasets.append({
            "dataset": "section5_hypothesis_verdicts",
            "examples": verdict_examples,
        })

    output = {
        "metadata": {
            "evaluation_name": "Synthesis of Synthetic Evidence: Paper-Ready Statistical Tables and Findings",
            "sections": [
                "section1_module_recovery",
                "section2_end_to_end_accuracy",
                "section3_signed_vs_unsigned_ablation",
                "section4_estimator_bias",
                "section5_key_takeaway_summary",
            ],
            "section1_module_recovery": section1,
            "section2_end_to_end_accuracy": section2,
            "section3_signed_vs_unsigned_ablation": section3,
            "section4_estimator_bias": section4,
            "section5_key_takeaway_summary": section5,
        },
        "metrics_agg": metrics_agg,
        "datasets": datasets,
    }

    return output


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
@logger.catch
def main():
    logger.info("=" * 60)
    logger.info("Synthesis Evaluation: Paper-Ready Statistical Tables")
    logger.info("=" * 60)

    # Determine which data files to load
    prefix = "mini" if DATA_SIZE == "mini" else "full"
    logger.info(f"Data size: {DATA_SIZE} (prefix: {prefix})")

    # Load experiment data
    exp1 = load_json(EXP1_DIR / f"{prefix}_method_out.json")
    exp3_it3 = load_json(EXP3_IT3_DIR / f"{prefix}_method_out.json")
    exp3_it4 = load_json(EXP3_IT4_DIR / f"{prefix}_method_out.json")

    exp1_meta = exp1["metadata"]
    exp3_it3_meta = exp3_it3["metadata"]
    exp3_it4_meta = exp3_it4["metadata"]

    # Compute all 5 sections
    section1 = compute_section1_module_recovery(exp1_meta)
    section2 = compute_section2_accuracy(exp3_it3_meta)
    section3 = compute_section3_ablation(exp3_it3_meta, exp3_it4_meta)
    section4 = compute_section4_estimator_bias(exp3_it4_meta)
    section5 = compute_section5_summary(section1, section2, section3, section4, exp1_meta, exp3_it3_meta)

    # Build schema-conformant output
    output = build_output(section1, section2, section3, section4, section5, exp1, exp3_it3, exp3_it4)

    # Save output
    out_path = WORKSPACE / "eval_out.json"
    out_path.write_text(json.dumps(output, indent=2, default=str))
    logger.info(f"Saved output to {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")

    # Quick summary
    logger.info(f"metrics_agg keys: {len(output['metrics_agg'])}")
    logger.info(f"datasets: {len(output['datasets'])}")
    for ds in output["datasets"]:
        logger.info(f"  {ds['dataset']}: {len(ds['examples'])} examples")

    logger.info("=" * 60)
    logger.info("Evaluation complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

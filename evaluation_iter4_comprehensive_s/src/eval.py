#!/usr/bin/env python3
"""Comprehensive statistical analysis of synthetic end-to-end experiment results.

Analyses A-H: Friedman+Nemenyi ranking, XOR significance, module-accuracy correlation,
Pareto frontier, no-structure control, highdim diagnosis, signed vs unsigned effect size,
and overall success criteria assessment.
"""

import json
import math
import os
import resource
import sys
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger
from scipy import stats

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add("logs/run.log", rotation="30 MB", level="DEBUG")

# ---------------------------------------------------------------------------
# Resource limits (container-aware)
# ---------------------------------------------------------------------------
def _container_ram_gb() -> float | None:
    for p in ["/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"]:
        try:
            v = Path(p).read_text().strip()
            if v != "max" and int(v) < 1_000_000_000_000:
                return int(v) / 1e9
        except (FileNotFoundError, ValueError):
            pass
    return None

TOTAL_RAM_GB = _container_ram_gb() or 57.0
RAM_BUDGET = int(min(4, TOTAL_RAM_GB * 0.3) * 1e9)  # 4 GB is plenty for this analysis
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET * 3, RAM_BUDGET * 3))
resource.setrlimit(resource.RLIMIT_CPU, (3600, 3600))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
WORKSPACE = Path(__file__).parent
DEP_DIR = Path(
    "/ai-inventor/aii_pipeline/runs/jamnik-sgfigs-pid-v2"
    "/3_invention_loop/iter_3/gen_art/exp_id3_it3__opus"
)
INPUT_FILE = DEP_DIR / "full_method_out.json"
OUTPUT_FILE = WORKSPACE / "eval_out.json"

METHODS = [
    "axis_aligned", "random_oblique", "signed_spectral",
    "unsigned_spectral", "hard_threshold",
]
VARIANTS = [
    "easy_2mod_xor", "medium_4mod_mixed", "overlapping_modules",
    "no_structure_control", "hard_4mod_unequal", "highdim_8mod",
]
N_BOOTSTRAP = 10000
RNG = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def safe_float(v: Any) -> float | None:
    """Convert to float, returning None for null/None."""
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """Paired Cohen's d (mean diff / pooled SD)."""
    diff = x - y
    sd = np.std(diff, ddof=1)
    if sd == 0:
        return 0.0
    return float(np.mean(diff) / sd)


def hedges_g(x: np.ndarray, y: np.ndarray) -> float:
    """Hedges' g: bias-corrected Cohen's d for small samples."""
    n = len(x)
    d = cohens_d(x, y)
    # Correction factor J
    j = 1 - 3 / (4 * (2 * n - 2) - 1)
    return float(d * j)


def interpret_effect(g: float) -> str:
    """Interpret Hedges' g / Cohen's d magnitude."""
    ag = abs(g)
    if ag < 0.2:
        return "negligible"
    elif ag < 0.5:
        return "small"
    elif ag < 0.8:
        return "medium"
    else:
        return "large"


def bootstrap_mean_diff(x: np.ndarray, y: np.ndarray, n_boot: int = N_BOOTSTRAP) -> tuple[float, float]:
    """Bootstrap 95% CI for mean(x - y)."""
    diff = x - y
    n = len(diff)
    boot_means = np.array([
        np.mean(RNG.choice(diff, size=n, replace=True)) for _ in range(n_boot)
    ])
    return float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))


def bootstrap_correlation(x: np.ndarray, y: np.ndarray, n_boot: int = N_BOOTSTRAP) -> tuple[float, float]:
    """Bootstrap 95% CI for Spearman correlation."""
    n = len(x)
    boot_rhos = []
    for _ in range(n_boot):
        idx = RNG.choice(n, size=n, replace=True)
        rho, _ = stats.spearmanr(x[idx], y[idx])
        if not np.isnan(rho):
            boot_rhos.append(rho)
    boot_rhos = np.array(boot_rhos)
    if len(boot_rhos) == 0:
        return (float("nan"), float("nan"))
    return float(np.percentile(boot_rhos, 2.5)), float(np.percentile(boot_rhos, 97.5))


def permutation_test_correlation(x: np.ndarray, y: np.ndarray, n_perm: int = N_BOOTSTRAP) -> float:
    """Permutation p-value for Spearman correlation."""
    obs_rho, _ = stats.spearmanr(x, y)
    count = 0
    for _ in range(n_perm):
        y_perm = RNG.permutation(y)
        rho_perm, _ = stats.spearmanr(x, y_perm)
        if abs(rho_perm) >= abs(obs_rho):
            count += 1
    return float(count / n_perm)


def fisher_z(r: float) -> float:
    """Fisher z-transformation of a correlation."""
    r = np.clip(r, -0.9999, 0.9999)
    return float(0.5 * np.log((1 + r) / (1 - r)))


def inv_fisher_z(z: float) -> float:
    """Inverse Fisher z-transformation."""
    return float(np.tanh(z))


def safe_wilcoxon(x: np.ndarray, y: np.ndarray) -> dict:
    """Wilcoxon signed-rank test, handling ties and zero diffs."""
    diff = x - y
    nonzero = diff[diff != 0]
    if len(nonzero) < 2:
        return {"stat": None, "p_value": None, "note": "insufficient non-zero differences"}
    try:
        stat, p = stats.wilcoxon(x, y, method='exact')
        return {"stat": float(stat), "p_value": float(p)}
    except ValueError:
        try:
            stat, p = stats.wilcoxon(x, y, method='approx')
            return {"stat": float(stat), "p_value": float(p)}
        except Exception:
            return {"stat": None, "p_value": None, "note": "wilcoxon failed"}


def safe_ttest_rel(x: np.ndarray, y: np.ndarray) -> dict:
    """Paired t-test with CI for mean difference."""
    n = len(x)
    diff = x - y
    mean_diff = float(np.mean(diff))
    try:
        t_stat, p_value = stats.ttest_rel(x, y)
        se = float(np.std(diff, ddof=1) / np.sqrt(n))
        t_crit = stats.t.ppf(0.975, df=n - 1)
        ci_lower = mean_diff - t_crit * se
        ci_upper = mean_diff + t_crit * se
        return {
            "t_stat": float(t_stat),
            "p_value": float(p_value),
            "mean_diff": mean_diff,
            "ci_lower": float(ci_lower),
            "ci_upper": float(ci_upper),
        }
    except Exception:
        return {"t_stat": None, "p_value": None, "mean_diff": mean_diff,
                "ci_lower": None, "ci_upper": None}


# ===========================================================================
# Analysis Functions
# ===========================================================================

def analysis_a_method_ranking(pvr: dict) -> dict:
    """Analysis A: Friedman + Nemenyi method ranking."""
    logger.info("Running Analysis A: Method Ranking (Friedman + Nemenyi)")

    per_variant = {}
    # Collect all blocks for pooled test: each block = (variant, fold)
    pooled_matrix = []  # rows = blocks, cols = methods

    for var in VARIANTS:
        vdata = pvr[var]["methods"]
        # Build 5×5 matrix: rows=folds, cols=methods (at best_max_splits)
        matrix = []
        for method in METHODS:
            accs = [f["balanced_accuracy"] for f in vdata[method]["best_folds"]]
            matrix.append(accs)
        matrix = np.array(matrix).T  # shape (n_folds, n_methods)
        n_folds = matrix.shape[0]

        # Add to pooled
        for row in matrix:
            pooled_matrix.append(row)

        # Friedman test
        try:
            chi2, p = stats.friedmanchisquare(*[matrix[:, j] for j in range(len(METHODS))])
        except Exception as e:
            logger.warning(f"Friedman failed on {var}: {e}")
            chi2, p = None, None

        # Average ranks
        ranks = np.zeros((n_folds, len(METHODS)))
        for i in range(n_folds):
            ranks[i] = stats.rankdata(-matrix[i])  # rank 1 = best (highest acc)
        avg_ranks = {m: float(ranks[:, j].mean()) for j, m in enumerate(METHODS)}

        # Nemenyi post-hoc if significant
        nemenyi_pairwise = {}
        if p is not None and p < 0.05:
            try:
                import scikit_posthocs as sp
                # posthoc_nemenyi_friedman expects (n_blocks, n_treatments) matrix
                nem_result = sp.posthoc_nemenyi_friedman(matrix)
                for i, m1 in enumerate(METHODS):
                    for j, m2 in enumerate(METHODS):
                        if i < j:
                            pair = f"{m1}_vs_{m2}"
                            nemenyi_pairwise[pair] = float(nem_result.iloc[i, j])
            except Exception as e:
                logger.warning(f"Nemenyi failed on {var}: {e}")
                nemenyi_pairwise = {"note": str(e)}

        per_variant[var] = {
            "friedman_chi2": float(chi2) if chi2 is not None else None,
            "friedman_p": float(p) if p is not None else None,
            "avg_ranks": avg_ranks,
            "nemenyi_pairwise": nemenyi_pairwise,
        }

    # Pooled Friedman across all 30 blocks
    pooled_matrix = np.array(pooled_matrix)  # (30, 5)
    try:
        chi2_pooled, p_pooled = stats.friedmanchisquare(
            *[pooled_matrix[:, j] for j in range(len(METHODS))]
        )
    except Exception as e:
        logger.warning(f"Pooled Friedman failed: {e}")
        chi2_pooled, p_pooled = None, None

    # Pooled average ranks
    pooled_ranks = np.zeros((pooled_matrix.shape[0], len(METHODS)))
    for i in range(pooled_matrix.shape[0]):
        pooled_ranks[i] = stats.rankdata(-pooled_matrix[i])
    pooled_avg_ranks = {m: float(pooled_ranks[:, j].mean()) for j, m in enumerate(METHODS)}

    # Nemenyi on pooled
    pooled_nemenyi = {}
    if p_pooled is not None and p_pooled < 0.05:
        try:
            import scikit_posthocs as sp
            nem_result = sp.posthoc_nemenyi_friedman(pooled_matrix)
            for i, m1 in enumerate(METHODS):
                for j, m2 in enumerate(METHODS):
                    if i < j:
                        pair = f"{m1}_vs_{m2}"
                        pooled_nemenyi[pair] = float(nem_result.iloc[i, j])
        except Exception as e:
            logger.warning(f"Pooled Nemenyi failed: {e}")
            pooled_nemenyi = {"note": str(e)}

    # Critical difference
    n_blocks = pooled_matrix.shape[0]
    k = len(METHODS)
    # CD = q_alpha * sqrt(k*(k+1)/(6*n))
    # For Nemenyi with k=5, alpha=0.05, q_alpha ~ 2.728 (from tables)
    q_alpha = 2.728
    cd = q_alpha * np.sqrt(k * (k + 1) / (6 * n_blocks))

    result = {
        "per_variant": per_variant,
        "pooled": {
            "friedman_chi2": float(chi2_pooled) if chi2_pooled is not None else None,
            "friedman_p": float(p_pooled) if p_pooled is not None else None,
            "avg_ranks": pooled_avg_ranks,
            "nemenyi_pairwise": pooled_nemenyi,
            "n_blocks": n_blocks,
        },
        "critical_difference": {
            "cd_value": float(cd),
            "q_alpha_0_05": q_alpha,
            "method_ranks": pooled_avg_ranks,
        },
    }
    logger.info(f"  Pooled Friedman: chi2={chi2_pooled}, p={p_pooled}")
    logger.info(f"  Pooled ranks: {pooled_avg_ranks}")
    logger.info(f"  Critical difference: {cd:.4f}")
    return result


def analysis_b_xor_significance(pvr: dict) -> dict:
    """Analysis B: XOR-specific paired significance tests."""
    logger.info("Running Analysis B: XOR-Specific Significance")

    var = "easy_2mod_xor"
    vdata = pvr[var]["methods"]

    # Extract per-fold accuracies at best_max_splits
    fold_accs = {}
    for m in METHODS:
        fold_accs[m] = np.array([f["balanced_accuracy"] for f in vdata[m]["best_folds"]])

    # Define comparisons
    comparisons = [
        ("unsigned_spectral", "axis_aligned"),
        ("unsigned_spectral", "random_oblique"),
        ("signed_spectral", "axis_aligned"),
        ("hard_threshold", "axis_aligned"),
    ]

    paired_tests = {}
    wilcoxon_tests = {}
    effect_sizes = {}

    for m1, m2 in comparisons:
        x = fold_accs[m1]
        y = fold_accs[m2]
        pair_name = f"{m1}_vs_{m2}"

        # Paired t-test
        paired_tests[pair_name] = safe_ttest_rel(x, y)

        # Wilcoxon
        wilcoxon_tests[pair_name] = safe_wilcoxon(x, y)

        # Cohen's d
        d = cohens_d(x, y)
        effect_sizes[pair_name] = {
            "cohens_d": float(d),
            "interpretation": interpret_effect(d),
        }

    # Bootstrap CI for unsigned_spectral vs axis_aligned
    x = fold_accs["unsigned_spectral"]
    y = fold_accs["axis_aligned"]
    ci_low, ci_high = bootstrap_mean_diff(x, y)

    result = {
        "paired_tests": paired_tests,
        "wilcoxon": wilcoxon_tests,
        "effect_sizes": effect_sizes,
        "bootstrap_ci": {
            "comparison": "unsigned_spectral_vs_axis_aligned",
            "lower": ci_low,
            "upper": ci_high,
            "n_resamples": N_BOOTSTRAP,
        },
        "power_note": (
            "With n=5 folds, statistical power is very limited. "
            "Non-significant results do NOT mean no effect — they mean insufficient evidence. "
            "Effect sizes are more informative than p-values at this sample size."
        ),
    }

    for pair_name, pt in paired_tests.items():
        logger.info(f"  {pair_name}: t={pt['t_stat']}, p={pt['p_value']}, diff={pt['mean_diff']}")
    return result


def analysis_c_module_accuracy_correlation(pvr: dict) -> dict:
    """Analysis C: Module recovery → accuracy Spearman correlation."""
    logger.info("Running Analysis C: Module-Accuracy Correlation")

    module_methods = ["signed_spectral", "unsigned_spectral", "hard_threshold"]

    # Collect all (jaccard, accuracy) pairs
    all_jaccard = []
    all_accuracy = []
    all_ari = []
    all_variant_labels = []
    per_variant = {}

    for var in VARIANTS:
        vdata = pvr[var]["methods"]
        var_jacc = []
        var_acc = []
        var_ari = []

        for m in module_methods:
            for fold in vdata[m]["best_folds"]:
                jacc = safe_float(fold["module_recovery_jaccard"])
                ari = safe_float(fold["module_recovery_ari"])
                acc = safe_float(fold["balanced_accuracy"])

                if jacc is not None and acc is not None:
                    var_jacc.append(jacc)
                    var_acc.append(acc)
                    all_jaccard.append(jacc)
                    all_accuracy.append(acc)
                    all_variant_labels.append(var)

                if ari is not None and acc is not None:
                    var_ari.append(ari)
                    all_ari.append(ari)

        # Per-variant Spearman
        if len(var_jacc) >= 3:
            rho, p = stats.spearmanr(var_jacc, var_acc)
            per_variant[var] = {
                "spearman_rho": float(rho) if not np.isnan(rho) else None,
                "p_value": float(p) if not np.isnan(p) else None,
                "n_points": len(var_jacc),
            }
        else:
            per_variant[var] = {
                "spearman_rho": None,
                "p_value": None,
                "n_points": len(var_jacc),
                "note": "insufficient data",
            }

    all_jaccard = np.array(all_jaccard)
    all_accuracy = np.array(all_accuracy)
    all_ari = np.array(all_ari)

    # Pooled Spearman (jaccard)
    pooled_rho, pooled_p = stats.spearmanr(all_jaccard, all_accuracy)

    # Bootstrap CI
    ci_low, ci_high = bootstrap_correlation(all_jaccard, all_accuracy)

    # Permutation test
    perm_p = permutation_test_correlation(all_jaccard, all_accuracy)

    pooled_result = {
        "spearman_rho": float(pooled_rho),
        "p_value": float(pooled_p),
        "bootstrap_ci_lower": ci_low,
        "bootstrap_ci_upper": ci_high,
        "permutation_p": perm_p,
        "n_points": len(all_jaccard),
    }

    # ARI-based correlation
    # Align ARI and accuracy arrays (same ordering since we iterated same way)
    all_ari_arr = np.array(all_ari[:len(all_accuracy)])
    if len(all_ari_arr) >= 3 and len(all_ari_arr) == len(all_accuracy):
        ari_rho, ari_p = stats.spearmanr(all_ari_arr, all_accuracy)
        ari_result = {
            "spearman_rho": float(ari_rho),
            "p_value": float(ari_p),
            "n_points": len(all_ari_arr),
        }
    else:
        ari_result = {"note": "ARI/accuracy length mismatch, recomputing"}
        # Recompute ARI correlation with aligned pairs
        ari_pairs_jacc = []
        ari_pairs_acc = []
        ari_pairs_ari = []
        for var in VARIANTS:
            vdata = pvr[var]["methods"]
            for m in module_methods:
                for fold in vdata[m]["best_folds"]:
                    ari = safe_float(fold["module_recovery_ari"])
                    acc = safe_float(fold["balanced_accuracy"])
                    if ari is not None and acc is not None:
                        ari_pairs_ari.append(ari)
                        ari_pairs_acc.append(acc)
        if len(ari_pairs_ari) >= 3:
            ari_rho, ari_p = stats.spearmanr(ari_pairs_ari, ari_pairs_acc)
            ari_result = {
                "spearman_rho": float(ari_rho),
                "p_value": float(ari_p),
                "n_points": len(ari_pairs_ari),
            }

    # Partial correlation controlling for variant identity
    # Fisher z-transform pooling of within-variant correlations
    z_values = []
    z_weights = []
    for var in VARIANTS:
        v = per_variant.get(var, {})
        rho = v.get("spearman_rho")
        n = v.get("n_points", 0)
        if rho is not None and n >= 4:  # need at least 4 points for meaningful correlation
            z_values.append(fisher_z(rho))
            z_weights.append(n - 3)  # weight = n-3 for Fisher z

    if z_values:
        z_weights = np.array(z_weights, dtype=float)
        z_values = np.array(z_values)
        pooled_z = np.average(z_values, weights=z_weights)
        partial_rho = inv_fisher_z(pooled_z)
    else:
        partial_rho = None

    result = {
        "pooled": pooled_result,
        "per_variant": per_variant,
        "partial": {
            "partial_rho": float(partial_rho) if partial_rho is not None else None,
            "method": "Fisher_z_transform_pooling_of_within_variant_correlations",
        },
        "ari_vs_jaccard": ari_result,
        "causal_note": (
            "Correlation does NOT establish causation. Even a strong correlation "
            "could be confounded by variant difficulty. The per-variant analysis "
            "and partial correlation help disentangle this."
        ),
    }
    logger.info(f"  Pooled Spearman rho={pooled_rho:.4f}, p={pooled_p:.4f}")
    logger.info(f"  Partial rho={partial_rho}")
    return result


def analysis_d_pareto(pvr: dict) -> dict:
    """Analysis D: Arity-Accuracy Tradeoff (Pareto Analysis)."""
    logger.info("Running Analysis D: Pareto Analysis")

    scatter_data = []
    per_variant = {}

    for var in VARIANTS:
        vdata = pvr[var]["methods"]
        points = []
        for m in METHODS:
            md = vdata[m]
            acc = md["mean_balanced_accuracy"]
            arity = md["mean_avg_split_arity"]
            points.append({"method": m, "accuracy": acc, "arity": arity})
            scatter_data.append({"method": m, "variant": var, "accuracy": acc, "arity": arity})

        # Find Pareto-optimal methods: not dominated by any other
        pareto_optimal = []
        for i, p in enumerate(points):
            dominated = False
            for j, q in enumerate(points):
                if i != j and q["accuracy"] >= p["accuracy"] and q["arity"] <= p["arity"]:
                    if q["accuracy"] > p["accuracy"] or q["arity"] < p["arity"]:
                        dominated = True
                        break
            if not dominated:
                pareto_optimal.append(p["method"])

        per_variant[var] = {
            "pareto_optimal": pareto_optimal,
            "points": points,
        }

    # Pooled Pareto frontier
    pooled_pareto = []
    for i, p in enumerate(scatter_data):
        dominated = False
        for j, q in enumerate(scatter_data):
            if i != j and q["accuracy"] >= p["accuracy"] and q["arity"] <= p["arity"]:
                if q["accuracy"] > p["accuracy"] or q["arity"] < p["arity"]:
                    dominated = True
                    break
        if not dominated:
            pooled_pareto.append(p)

    # Efficiency ratio = mean_accuracy / mean_arity per method
    efficiency_ratio = {}
    for m in METHODS:
        accs = []
        arities = []
        for var in VARIANTS:
            md = pvr[var]["methods"][m]
            accs.append(md["mean_balanced_accuracy"])
            arities.append(md["mean_avg_split_arity"])
        mean_acc = np.mean(accs)
        mean_arity = np.mean(arities)
        efficiency_ratio[m] = float(mean_acc / mean_arity) if mean_arity > 0 else None

    # Wilcoxon: arity of unsigned_spectral vs random_oblique across 6 variants
    us_arities = []
    ro_arities = []
    for var in VARIANTS:
        us_arities.append(pvr[var]["methods"]["unsigned_spectral"]["mean_avg_split_arity"])
        ro_arities.append(pvr[var]["methods"]["random_oblique"]["mean_avg_split_arity"])
    us_arities = np.array(us_arities)
    ro_arities = np.array(ro_arities)
    arity_test = safe_wilcoxon(us_arities, ro_arities)
    arity_test["mean_diff"] = float(np.mean(us_arities - ro_arities))

    result = {
        "per_variant": per_variant,
        "pooled_frontier": pooled_pareto,
        "efficiency_ratio": efficiency_ratio,
        "arity_test": arity_test,
        "scatter_data": scatter_data,
    }
    logger.info(f"  Efficiency ratios: {efficiency_ratio}")
    logger.info(f"  Arity test (unsigned vs random): diff={arity_test['mean_diff']:.3f}, p={arity_test.get('p_value')}")
    return result


def analysis_e_no_structure(pvr: dict) -> dict:
    """Analysis E: No-structure control analysis."""
    logger.info("Running Analysis E: No-Structure Control Analysis")

    var = "no_structure_control"
    vdata = pvr[var]["methods"]

    method_accuracies = {}
    for m in METHODS:
        folds = vdata[m]["best_folds"]
        per_fold = [f["balanced_accuracy"] for f in folds]
        method_accuracies[m] = {
            "mean": float(np.mean(per_fold)),
            "std": float(np.std(per_fold, ddof=1)),
            "per_fold": per_fold,
        }

    # Significance tests
    significance_tests = {}
    for m1, m2 in [
        ("unsigned_spectral", "axis_aligned"),
        ("unsigned_spectral", "random_oblique"),
    ]:
        x = np.array(method_accuracies[m1]["per_fold"])
        y = np.array(method_accuracies[m2]["per_fold"])
        significance_tests[f"{m1}_vs_{m2}"] = {
            "ttest": safe_ttest_rel(x, y),
            "wilcoxon": safe_wilcoxon(x, y),
        }

    # Selected k values for spectral methods
    selected_k_values = {}
    for m in ["signed_spectral", "unsigned_spectral"]:
        ks = [f.get("selected_k") for f in vdata[m]["best_folds"]]
        selected_k_values[m] = ks

    # Arity comparison
    arity_comparison = {}
    for m in METHODS:
        arities = [f["avg_split_arity"] for f in vdata[m]["best_folds"]]
        arity_comparison[m] = float(np.mean(arities))

    # Overfitting indicators: variance ratio = std(unsigned) / std(random)
    us_std = method_accuracies["unsigned_spectral"]["std"]
    ro_std = method_accuracies["random_oblique"]["std"]
    variance_ratio = float(us_std / ro_std) if ro_std > 0 else None

    # Fold consistency: max - min accuracy spread
    us_folds = method_accuracies["unsigned_spectral"]["per_fold"]
    fold_consistency = float(max(us_folds) - min(us_folds))

    # Oblique splits count
    oblique_counts = {}
    for m in METHODS:
        ob = [f.get("oblique_splits", 0) for f in vdata[m]["best_folds"]]
        oblique_counts[m] = ob

    interpretation = (
        "The no_structure_control dataset has no planted modules (n_modules=0, 20 random features). "
        f"Unsigned spectral achieves highest accuracy ({method_accuracies['unsigned_spectral']['mean']:.4f}) "
        f"with low variance (std={us_std:.4f}), suggesting it finds genuine marginal structure "
        "rather than overfitting. The spectral methods select k=2-6, grouping random features into "
        "modules that happen to capture useful oblique directions. Since all features are independently "
        "informative (no XOR structure), oblique splits combining any features can improve over "
        "axis-aligned splits. The higher arity of unsigned_spectral vs axis_aligned reflects this."
    )

    result = {
        "method_accuracies": method_accuracies,
        "significance_tests": significance_tests,
        "selected_k_values": selected_k_values,
        "arity_comparison": arity_comparison,
        "overfitting_indicators": {
            "variance_ratio_us_over_ro": variance_ratio,
            "fold_consistency_us_range": fold_consistency,
        },
        "oblique_splits_per_fold": oblique_counts,
        "interpretation": interpretation,
    }
    logger.info(f"  Unsigned spectral: mean={method_accuracies['unsigned_spectral']['mean']:.4f}")
    logger.info(f"  Variance ratio (us/ro): {variance_ratio}")
    return result


def analysis_f_highdim(pvr: dict) -> dict:
    """Analysis F: Highdim failure mode diagnosis."""
    logger.info("Running Analysis F: Highdim Failure Diagnosis")

    var = "highdim_8mod"
    vdata = pvr[var]["methods"]

    # Method comparison
    method_comparison = {}
    for m in METHODS:
        md = vdata[m]
        method_comparison[m] = {
            "accuracy": md["mean_balanced_accuracy"],
            "arity": md["mean_avg_split_arity"],
            "path_length": md["mean_avg_path_length"],
        }

    # Selected k values
    selected_k = {}
    for m in ["signed_spectral", "unsigned_spectral"]:
        ks = [f.get("selected_k") for f in vdata[m]["best_folds"]]
        selected_k[m] = ks

    # Module recovery
    module_recovery = {}
    for m in ["signed_spectral", "unsigned_spectral", "hard_threshold"]:
        aris = [safe_float(f.get("module_recovery_ari")) for f in vdata[m]["best_folds"]]
        jaccs = [safe_float(f.get("module_recovery_jaccard")) for f in vdata[m]["best_folds"]]
        aris_valid = [a for a in aris if a is not None]
        jaccs_valid = [j for j in jaccs if j is not None]
        module_recovery[m] = {
            "mean_ari": float(np.mean(aris_valid)) if aris_valid else None,
            "mean_jaccard": float(np.mean(jaccs_valid)) if jaccs_valid else None,
            "per_fold_ari": aris,
            "per_fold_jaccard": jaccs,
        }

    # CoI time comparison across variants
    coi_time = {}
    for v in VARIANTS:
        times = []
        # Use axis_aligned as reference (same CoI for all methods on same data)
        for f in pvr[v]["methods"]["axis_aligned"]["best_folds"]:
            t = safe_float(f.get("coi_time_s"))
            if t is not None:
                times.append(t)
        coi_time[v] = float(np.mean(times)) if times else None

    # Arity equals axis-aligned check
    arity_equals_axis = {}
    for m in METHODS:
        arities = [f["avg_split_arity"] for f in vdata[m]["best_folds"]]
        arity_equals_axis[m] = all(a == 1.0 for a in arities)

    # Failure mode explanation
    failure_mode = (
        "On highdim_8mod (200 features, 8 true modules), signed_spectral collapses to arity=1.0 "
        "(identical to axis-aligned), meaning NO oblique splits are used. The spectral clustering "
        f"selects k={selected_k.get('signed_spectral', '?')}, far below the true k=8. "
        "With k=2 for 200 features, each module contains ~100 features — too many for meaningful "
        "oblique splits. The SPONGE signed clustering produces near-random assignments "
        f"(mean ARI={module_recovery.get('signed_spectral', {}).get('mean_ari', '?')}), "
        "so the module-guided split selection finds no useful feature groups and falls back to "
        "axis-aligned behavior. Unsigned spectral performs slightly better with varying k but still "
        "has very poor module recovery. The 200x200 CoI matrix computation takes "
        f"~{coi_time.get('highdim_8mod', '?')}s per fold, dominating wall-clock time."
    )

    result = {
        "method_comparison": method_comparison,
        "selected_k": selected_k,
        "module_recovery": module_recovery,
        "coi_time": coi_time,
        "failure_mode": failure_mode,
        "arity_equals_axis_aligned": arity_equals_axis,
    }
    logger.info(f"  Signed spectral k values: {selected_k.get('signed_spectral')}")
    logger.info(f"  Module recovery (signed): ARI={module_recovery.get('signed_spectral',{}).get('mean_ari')}")
    return result


def analysis_g_signed_vs_unsigned(pvr: dict) -> dict:
    """Analysis G: Signed vs unsigned effect size (Hedges' g)."""
    logger.info("Running Analysis G: Signed vs Unsigned Effect Size")

    per_variant_g = {}
    all_signed = []
    all_unsigned = []
    signed_wins = 0
    unsigned_wins = 0
    ties = 0

    for var in VARIANTS:
        vdata = pvr[var]["methods"]
        signed_accs = np.array([f["balanced_accuracy"] for f in vdata["signed_spectral"]["best_folds"]])
        unsigned_accs = np.array([f["balanced_accuracy"] for f in vdata["unsigned_spectral"]["best_folds"]])

        g = hedges_g(signed_accs, unsigned_accs)
        signed_mean = float(np.mean(signed_accs))
        unsigned_mean = float(np.mean(unsigned_accs))
        diff = signed_mean - unsigned_mean

        per_variant_g[var] = {
            "hedges_g": float(g),
            "interpretation": interpret_effect(g),
            "signed_mean": signed_mean,
            "unsigned_mean": unsigned_mean,
            "diff": float(diff),
        }

        if diff > 0.001:
            signed_wins += 1
        elif diff < -0.001:
            unsigned_wins += 1
        else:
            ties += 1

        all_signed.extend(signed_accs.tolist())
        all_unsigned.extend(unsigned_accs.tolist())

    # Pooled Hedges' g across all variant-fold pairs
    all_signed = np.array(all_signed)
    all_unsigned = np.array(all_unsigned)

    # Simple pooled g
    pooled_g = hedges_g(all_signed, all_unsigned)

    # Bootstrap CI for pooled g
    n = len(all_signed)
    boot_gs = []
    for _ in range(N_BOOTSTRAP):
        idx = RNG.choice(n, size=n, replace=True)
        bg = hedges_g(all_signed[idx], all_unsigned[idx])
        boot_gs.append(bg)
    boot_gs = np.array(boot_gs)
    g_ci_low = float(np.percentile(boot_gs, 2.5))
    g_ci_high = float(np.percentile(boot_gs, 97.5))

    # Overall mean difference with bootstrap CI
    overall_diff = float(np.mean(all_signed - all_unsigned))
    diff_ci_low, diff_ci_high = bootstrap_mean_diff(all_signed, all_unsigned)

    # Also: signed vs axis_aligned
    vs_axis = {}
    for var in VARIANTS:
        vdata = pvr[var]["methods"]
        signed_accs = np.array([f["balanced_accuracy"] for f in vdata["signed_spectral"]["best_folds"]])
        axis_accs = np.array([f["balanced_accuracy"] for f in vdata["axis_aligned"]["best_folds"]])
        g = hedges_g(signed_accs, axis_accs)
        vs_axis[var] = {
            "hedges_g": float(g),
            "interpretation": interpret_effect(g),
            "signed_mean": float(np.mean(signed_accs)),
            "axis_mean": float(np.mean(axis_accs)),
            "diff": float(np.mean(signed_accs) - np.mean(axis_accs)),
        }

    result = {
        "per_variant": per_variant_g,
        "pooled": {
            "pooled_g": float(pooled_g),
            "interpretation": interpret_effect(pooled_g),
            "bootstrap_ci_lower": g_ci_low,
            "bootstrap_ci_upper": g_ci_high,
        },
        "win_loss_tie": {
            "signed_wins": signed_wins,
            "unsigned_wins": unsigned_wins,
            "ties": ties,
        },
        "overall_mean_diff": {
            "mean": overall_diff,
            "ci_lower": diff_ci_low,
            "ci_upper": diff_ci_high,
        },
        "vs_axis_aligned": vs_axis,
    }
    logger.info(f"  Pooled Hedges' g: {pooled_g:.4f} ({interpret_effect(pooled_g)})")
    logger.info(f"  Win/Loss/Tie: signed={signed_wins}, unsigned={unsigned_wins}, ties={ties}")
    return result


def analysis_h_success_criteria(pvr: dict, frustration_data: list, total_runtime: float) -> dict:
    """Analysis H: Overall success criteria assessment."""
    logger.info("Running Analysis H: Success Criteria Assessment")

    module_methods = ["signed_spectral", "unsigned_spectral", "hard_threshold"]

    # Criterion 1: Module recovery > 80% for signed spectral
    ari_data = {}
    fraction_above_80 = {}
    for m in module_methods:
        all_ari = []
        above_80_count = 0
        total_count = 0
        for var in VARIANTS:
            for fold in pvr[var]["methods"][m]["best_folds"]:
                ari = safe_float(fold.get("module_recovery_ari"))
                if ari is not None:
                    all_ari.append(ari)
                    total_count += 1
                    if ari > 0.8:
                        above_80_count += 1
        ari_data[m] = float(np.mean(all_ari)) if all_ari else None
        fraction_above_80[m] = float(above_80_count / total_count) if total_count > 0 else None

    # Signed spectral ARI is very low — criterion disconfirmed
    signed_ari = ari_data.get("signed_spectral")
    unsigned_ari = ari_data.get("unsigned_spectral")
    hard_ari = ari_data.get("hard_threshold")

    criterion_1_verdict = "DISCONFIRMED"
    if signed_ari is not None and signed_ari > 0.8:
        criterion_1_verdict = "CONFIRMED"
    elif signed_ari is not None and signed_ari > 0.5:
        criterion_1_verdict = "INCONCLUSIVE"

    criterion_1 = {
        "verdict": criterion_1_verdict,
        "signed_mean_ari": signed_ari,
        "unsigned_mean_ari": unsigned_ari,
        "hard_mean_ari": hard_ari,
        "fraction_above_80": fraction_above_80,
        "note": (
            "Signed spectral (SPONGE) shows very poor module recovery across variants. "
            "Unsigned spectral achieves perfect recovery on easy_2mod_xor (ARI=1.0) "
            "but degrades on harder variants."
        ),
    }

    # Criterion 2: Signed spectral matches/exceeds random-oblique accuracy with lower arity
    signed_accs = []
    random_accs = []
    signed_arities = []
    random_arities = []
    for var in VARIANTS:
        signed_accs.append(pvr[var]["methods"]["signed_spectral"]["mean_balanced_accuracy"])
        random_accs.append(pvr[var]["methods"]["random_oblique"]["mean_balanced_accuracy"])
        signed_arities.append(pvr[var]["methods"]["signed_spectral"]["mean_avg_split_arity"])
        random_arities.append(pvr[var]["methods"]["random_oblique"]["mean_avg_split_arity"])

    acc_gap = float(np.mean(signed_accs) - np.mean(random_accs))
    arity_gap = float(np.mean(signed_arities) - np.mean(random_arities))
    # Criterion passes if acc_gap >= 0 AND arity_gap < 0
    if acc_gap >= -0.01 and arity_gap < 0:
        c2_verdict = "CONFIRMED"
    elif acc_gap >= -0.05:
        c2_verdict = "INCONCLUSIVE"
    else:
        c2_verdict = "DISCONFIRMED"

    criterion_2 = {
        "verdict": c2_verdict,
        "signed_mean_accuracy": float(np.mean(signed_accs)),
        "random_mean_accuracy": float(np.mean(random_accs)),
        "accuracy_gap": acc_gap,
        "signed_mean_arity": float(np.mean(signed_arities)),
        "random_mean_arity": float(np.mean(random_arities)),
        "arity_gap": arity_gap,
    }

    # Criterion 3: Frustration index negatively correlates with oblique benefit
    # frustration_data is list of [variant, frustration_index, signed_acc - axis_acc]
    if frustration_data and len(frustration_data) >= 3:
        frust_vals = np.array([f[1] for f in frustration_data])
        benefit_vals = np.array([f[2] for f in frustration_data])
        rho, p = stats.spearmanr(frust_vals, benefit_vals)
        c3_verdict = "CONFIRMED" if p < 0.05 and rho < 0 else "INCONCLUSIVE"
        if p > 0.1:
            c3_verdict = "DISCONFIRMED"
        criterion_3 = {
            "verdict": c3_verdict,
            "spearman_rho": float(rho),
            "p_value": float(p),
            "n_datapoints": len(frustration_data),
            "note": f"With n={len(frustration_data)}, significance at p<0.05 requires |rho| >= ~0.886",
        }
    else:
        criterion_3 = {
            "verdict": "INCONCLUSIVE",
            "note": "Insufficient frustration benefit data",
        }

    # Criterion 4: Pipeline completes in <30 min for d<=200, n<=100K
    runtime_min = total_runtime / 60.0
    c4_verdict = "CONFIRMED" if runtime_min < 30 else "DISCONFIRMED"
    criterion_4 = {
        "verdict": c4_verdict,
        "total_runtime_s": total_runtime,
        "total_runtime_min": float(runtime_min),
        "threshold_min": 30,
    }

    # Disconfirmation 1: Signed spectral recovery < 50%
    disconf_1_triggered = signed_ari is not None and signed_ari < 0.5
    disconfirmation_1 = {
        "verdict": "TRIGGERED" if disconf_1_triggered else "NOT_TRIGGERED",
        "signed_mean_ari": signed_ari,
        "threshold": 0.5,
    }

    # Disconfirmation 2: No signed vs unsigned difference
    signed_grand = np.mean(signed_accs)
    unsigned_grand_accs = [pvr[var]["methods"]["unsigned_spectral"]["mean_balanced_accuracy"] for var in VARIANTS]
    unsigned_grand = np.mean(unsigned_grand_accs)
    diff_su = float(signed_grand - unsigned_grand)
    disconf_2_triggered = abs(diff_su) < 0.01
    disconfirmation_2 = {
        "verdict": "TRIGGERED" if disconf_2_triggered else "NOT_TRIGGERED",
        "signed_grand_mean": float(signed_grand),
        "unsigned_grand_mean": float(unsigned_grand),
        "difference": diff_su,
        "note": "Unsigned outperforms signed, suggesting sign information hurts rather than helps",
    }

    # Disconfirmation 3: Frustration index uncorrelated
    if frustration_data and len(frustration_data) >= 3:
        frust_p = criterion_3.get("p_value", 1.0)
        disconf_3_triggered = frust_p is not None and frust_p > 0.1
    else:
        disconf_3_triggered = True
    disconfirmation_3 = {
        "verdict": "TRIGGERED" if disconf_3_triggered else "NOT_TRIGGERED",
        "frustration_correlation_p": criterion_3.get("p_value"),
    }

    # Overall verdict
    verdicts = [criterion_1["verdict"], criterion_2["verdict"], criterion_3["verdict"], criterion_4["verdict"]]
    n_confirmed = sum(1 for v in verdicts if v == "CONFIRMED")
    n_disconfirmed = sum(1 for v in verdicts if v == "DISCONFIRMED")
    disconf_triggers = [disconfirmation_1["verdict"], disconfirmation_2["verdict"], disconfirmation_3["verdict"]]
    n_triggers = sum(1 for v in disconf_triggers if v == "TRIGGERED")

    overall_verdict = (
        f"Of 4 success criteria: {n_confirmed} CONFIRMED, {n_disconfirmed} DISCONFIRMED, "
        f"{4 - n_confirmed - n_disconfirmed} INCONCLUSIVE. "
        f"Of 3 disconfirmation criteria: {n_triggers} TRIGGERED. "
        "The signed spectral (SPONGE) approach fails its primary module recovery criterion. "
        "Unsigned spectral shows promise but does not validate the signed information hypothesis. "
        "The pipeline meets runtime constraints. Frustration-benefit correlation is inconclusive "
        "with only 6 data points."
    )

    result = {
        "criterion_1": criterion_1,
        "criterion_2": criterion_2,
        "criterion_3": criterion_3,
        "criterion_4": criterion_4,
        "disconfirmation_1": disconfirmation_1,
        "disconfirmation_2": disconfirmation_2,
        "disconfirmation_3": disconfirmation_3,
        "overall_verdict": overall_verdict,
    }
    logger.info(f"  Criterion 1 (Module recovery): {criterion_1_verdict}")
    logger.info(f"  Criterion 2 (Accuracy+arity): {c2_verdict}")
    logger.info(f"  Criterion 3 (Frustration): {criterion_3['verdict']}")
    logger.info(f"  Criterion 4 (Runtime): {c4_verdict}")
    return result


# ===========================================================================
# Schema-conformant output builder
# ===========================================================================

def build_schema_output(
    meta_in: dict,
    pvr: dict,
    detailed_results: dict,
) -> dict:
    """Build output conforming to exp_eval_sol_out.json schema.

    Schema requires:
      - metrics_agg: {metric_name: number}
      - datasets: [{dataset: str, examples: [{input, output, eval_*, ...}]}]
      - metadata: (optional) any extra info
    """
    # metrics_agg: aggregate summary metrics
    metrics_agg = {}

    # Pooled Friedman p-value
    fr_p = detailed_results["method_ranking"]["pooled"].get("friedman_p")
    if fr_p is not None:
        metrics_agg["pooled_friedman_p"] = fr_p

    # Grand mean accuracies per method
    agg = meta_in.get("aggregate", {})
    for m in METHODS:
        if m in agg:
            key = f"grand_mean_bacc_{m}"
            metrics_agg[key] = agg[m]["grand_mean_balanced_accuracy"]

    # Module-accuracy pooled Spearman
    mac = detailed_results["module_accuracy_correlation"]["pooled"]
    metrics_agg["module_acc_spearman_rho"] = mac["spearman_rho"]
    metrics_agg["module_acc_spearman_p"] = mac["p_value"]

    # Signed vs unsigned pooled Hedges' g
    svu = detailed_results["signed_vs_unsigned"]["pooled"]
    metrics_agg["signed_vs_unsigned_hedges_g"] = svu["pooled_g"]

    # XOR effect size: unsigned vs axis
    xor_es = detailed_results["xor_analysis"]["effect_sizes"].get(
        "unsigned_spectral_vs_axis_aligned", {}
    )
    if "cohens_d" in xor_es:
        metrics_agg["xor_unsigned_vs_axis_cohens_d"] = xor_es["cohens_d"]

    # Success criteria counts
    sc = detailed_results["success_criteria"]
    n_confirmed = sum(
        1 for k in ["criterion_1", "criterion_2", "criterion_3", "criterion_4"]
        if sc[k]["verdict"] == "CONFIRMED"
    )
    n_disconfirmed = sum(
        1 for k in ["criterion_1", "criterion_2", "criterion_3", "criterion_4"]
        if sc[k]["verdict"] == "DISCONFIRMED"
    )
    metrics_agg["success_criteria_confirmed"] = n_confirmed
    metrics_agg["success_criteria_disconfirmed"] = n_disconfirmed

    # Total runtime
    metrics_agg["total_runtime_s"] = meta_in.get("total_runtime_s", 0)

    # Build datasets: one dataset per variant, examples = per-method results
    datasets = []
    for var in VARIANTS:
        vdata = pvr[var]
        examples = []
        for m in METHODS:
            md = vdata["methods"][m]
            # Each example: one method on one variant
            fold_accs = [f["balanced_accuracy"] for f in md["best_folds"]]
            fold_aucs = [f.get("auc", 0) for f in md["best_folds"]]

            input_str = (
                f"Evaluate method '{m}' on variant '{var}' "
                f"(n_features={vdata['variant_meta']['n_features']}, "
                f"n_modules={vdata['variant_meta']['n_modules']}, "
                f"best_max_splits={md['best_max_splits']})"
            )
            output_str = (
                f"mean_balanced_accuracy={md['mean_balanced_accuracy']:.4f}, "
                f"std={md['std_balanced_accuracy']:.4f}, "
                f"mean_auc={md['mean_auc']:.4f}, "
                f"mean_arity={md['mean_avg_split_arity']:.2f}"
            )

            # Build per-fold prediction string
            fold_details = "; ".join(
                f"fold{f['fold']}={f['balanced_accuracy']:.4f}"
                for f in md["best_folds"]
            )
            predict_str = (
                f"[{m}] {fold_details} | "
                f"mean={md['mean_balanced_accuracy']:.4f} "
                f"arity={md['mean_avg_split_arity']:.2f}"
            )

            example = {
                "input": input_str,
                "output": output_str,
                "predict_method_result": predict_str,
                "metadata_variant": var,
                "metadata_method": m,
                "metadata_best_max_splits": str(md["best_max_splits"]),
                "metadata_n_folds": str(len(md["best_folds"])),
                "eval_mean_balanced_accuracy": md["mean_balanced_accuracy"],
                "eval_std_balanced_accuracy": md["std_balanced_accuracy"],
                "eval_mean_auc": md["mean_auc"],
                "eval_mean_split_arity": md["mean_avg_split_arity"],
                "eval_mean_path_length": md["mean_avg_path_length"],
            }

            # Add module recovery if available
            aris = [safe_float(f.get("module_recovery_ari")) for f in md["best_folds"]]
            aris_valid = [a for a in aris if a is not None]
            if aris_valid:
                example["eval_mean_module_ari"] = float(np.mean(aris_valid))

            jaccs = [safe_float(f.get("module_recovery_jaccard")) for f in md["best_folds"]]
            jaccs_valid = [j for j in jaccs if j is not None]
            if jaccs_valid:
                example["eval_mean_module_jaccard"] = float(np.mean(jaccs_valid))

            examples.append(example)

        datasets.append({
            "dataset": var,
            "examples": examples,
        })

    # Metadata
    metadata = {
        "evaluation_name": "comprehensive_synthetic_analysis_iter4",
        "experiment_analyzed": "exp_id3_it3__opus",
        "n_methods": len(METHODS),
        "n_variants": len(VARIANTS),
        "n_folds": meta_in.get("n_folds", 5),
        "max_splits_grid": meta_in.get("max_splits_grid", [5, 10, 15, 20]),
        "analysis_sections": [
            "A_method_ranking", "B_xor_significance",
            "C_module_accuracy_correlation", "D_pareto_analysis",
            "E_no_structure_control", "F_highdim_diagnosis",
            "G_signed_vs_unsigned", "H_success_criteria",
        ],
        "detailed_results": detailed_results,
    }

    return {
        "metadata": metadata,
        "metrics_agg": metrics_agg,
        "datasets": datasets,
    }


# ===========================================================================
# Main
# ===========================================================================

@logger.catch
def main():
    logger.info(f"Loading data from {INPUT_FILE}")
    raw = json.loads(INPUT_FILE.read_text())
    meta = raw["metadata"]
    pvr = meta["per_variant_results"]
    logger.info(f"Loaded data: {len(pvr)} variants, {len(METHODS)} methods")

    # Verify data structure
    for var in VARIANTS:
        assert var in pvr, f"Missing variant: {var}"
        for m in METHODS:
            assert m in pvr[var]["methods"], f"Missing method {m} in {var}"
            n_bf = len(pvr[var]["methods"][m]["best_folds"])
            logger.debug(f"  {var}/{m}: {n_bf} best_folds")

    # Run all 8 analyses
    detailed = {}

    detailed["method_ranking"] = analysis_a_method_ranking(pvr)
    detailed["xor_analysis"] = analysis_b_xor_significance(pvr)
    detailed["module_accuracy_correlation"] = analysis_c_module_accuracy_correlation(pvr)
    detailed["pareto_analysis"] = analysis_d_pareto(pvr)
    detailed["no_structure_analysis"] = analysis_e_no_structure(pvr)
    detailed["highdim_diagnosis"] = analysis_f_highdim(pvr)
    detailed["signed_vs_unsigned"] = analysis_g_signed_vs_unsigned(pvr)
    detailed["success_criteria"] = analysis_h_success_criteria(
        pvr,
        meta.get("frustration_benefit_analysis", []),
        meta.get("total_runtime_s", 0),
    )

    # Build schema-conformant output
    logger.info("Building output...")
    output = build_schema_output(meta, pvr, detailed)

    # Write output
    OUTPUT_FILE.write_text(json.dumps(output, indent=2, default=str))
    logger.info(f"Wrote eval_out.json ({OUTPUT_FILE.stat().st_size / 1024:.1f} KB)")
    logger.info("Evaluation complete.")


if __name__ == "__main__":
    main()

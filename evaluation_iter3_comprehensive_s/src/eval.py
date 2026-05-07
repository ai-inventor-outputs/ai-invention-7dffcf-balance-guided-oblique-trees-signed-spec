#!/usr/bin/env python3
"""Comprehensive Statistical Analysis of Iteration 2 Results.

Performs rigorous statistical analysis of all iteration 2 experiment outputs:
- Real benchmarks (exp_id2): Friedman/Nemenyi, Effect Sizes, Interpretability
- Synthetic recovery (exp_id1): Method ranking, SPONGE vs baselines
- Estimator validation (exp_id4): Subsampling stability, noise floor
- Cross-cutting: Frustration-accuracy correlation, Bayesian tests, Success criteria
"""

import gc
import json
import math
import os
import resource
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
from loguru import logger

# ─── Logging setup ───
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
WORKSPACE = Path(__file__).parent
(WORKSPACE / "logs").mkdir(exist_ok=True)
logger.add(WORKSPACE / "logs" / "run.log", rotation="30 MB", level="DEBUG")


# ─── Hardware detection ───
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


def _container_ram_gb() -> Optional[float]:
    for p in ["/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"]:
        try:
            v = Path(p).read_text().strip()
            if v != "max" and int(v) < 1_000_000_000_000:
                return int(v) / 1e9
        except (FileNotFoundError, ValueError):
            pass
    return None


NUM_CPUS = _detect_cpus()
TOTAL_RAM_GB = _container_ram_gb() or 29.0

# Set memory limit (50% of container - metadata-only analysis, not memory-heavy)
RAM_BUDGET = int(TOTAL_RAM_GB * 0.5 * 1e9)
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET * 3, RAM_BUDGET * 3))
resource.setrlimit(resource.RLIMIT_CPU, (3600, 3600))

logger.info(f"Hardware: {NUM_CPUS} CPUs, {TOTAL_RAM_GB:.1f} GB RAM, budget={RAM_BUDGET/1e9:.1f} GB")

# ─── Constants ───
DEP_BASE = Path("/ai-inventor/aii_pipeline/runs/jamnik-sgfigs-pid-v2/3_invention_loop/iter_2/gen_art")
EXP_ID2_PATH = DEP_BASE / "exp_id2_it2__opus" / "full_method_out.json"
EXP_ID1_PATH = DEP_BASE / "exp_id1_it2__opus" / "full_method_out.json"
EXP_ID4_PATH = DEP_BASE / "exp_id4_it2__opus" / "full_method_out.json"

DATASETS = ["electricity", "adult", "california_housing", "jannis", "higgs_small"]
METHODS = ["axis_aligned_figs", "random_oblique_figs", "signed_spectral_figs"]
METHOD_SHORT = {"axis_aligned_figs": "aa", "random_oblique_figs": "ro", "signed_spectral_figs": "ss"}
MAX_SPLITS_GRID = [5, 10, 15, 20]
SYNTHETIC_VARIANTS_STRUCTURED = [
    "easy_2mod_xor", "medium_4mod_mixed", "hard_4mod_unequal",
    "overlapping_modules", "highdim_8mod",
]
SYNTHETIC_METHODS = [
    "sponge_auto_k", "sponge_oracle_k", "hard_threshold",
    "unsigned_spectral", "random_partition",
]


# ─── Data loading ───
def load_json(path: Path) -> dict:
    logger.info(f"Loading {path.name} ({path.stat().st_size / 1e6:.1f} MB)")
    data = json.loads(path.read_text())
    logger.info(f"Loaded {path.name}")
    return data


def get_perf(record: dict) -> tuple:
    """Get (mean, std) performance metric from an aggregated result record."""
    if record["task_type"] == "classification":
        return record["balanced_accuracy_mean"], record["balanced_accuracy_std"]
    else:
        return record["r2_mean"], record["r2_std"]


def build_perf_lookup(agg_results: list) -> dict:
    """Build dict: (dataset, method, max_splits) -> {mean, std, task_type, arity, path_length, fit_time}."""
    lookup = {}
    for r in agg_results:
        mean_val, std_val = get_perf(r)
        key = (r["dataset"], r["method"], r["max_splits"])
        lookup[key] = {
            "mean": mean_val,
            "std": std_val,
            "task_type": r["task_type"],
            "arity": r["avg_split_arity_mean"],
            "path_length": r["avg_path_length_mean"],
            "fit_time": r["fit_time_sec_mean"],
        }
    return lookup


def safe_float(v: Any) -> float:
    """Convert value to float, replacing None/NaN/Inf with sentinel -999.0."""
    if v is None:
        return -999.0
    try:
        f = float(v)
        if np.isnan(f) or np.isinf(f):
            return -999.0
        return f
    except (TypeError, ValueError):
        return -999.0


# ═══════════════════════════════════════════════════════════════
# A. FRIEDMAN / NEMENYI TESTS
# ═══════════════════════════════════════════════════════════════
def compute_friedman_nemenyi(lookup: dict, best_max_splits: dict) -> dict:
    """Compute Friedman + Nemenyi tests at each max_splits level and best-max_splits."""
    results = {}

    # Try to import scikit-posthocs for Nemenyi
    try:
        import scikit_posthocs as sp
        has_posthocs = True
        logger.info("scikit_posthocs available for Nemenyi post-hoc")
    except ImportError:
        logger.warning("scikit_posthocs not available, Nemenyi post-hoc will be skipped")
        has_posthocs = False

    def run_friedman(matrix_5x3: np.ndarray, label: str) -> dict:
        """Run Friedman test on a 5x3 matrix (5 datasets, 3 methods)."""
        try:
            chi2, p_value = stats.friedmanchisquare(
                matrix_5x3[:, 0], matrix_5x3[:, 1], matrix_5x3[:, 2]
            )
        except Exception as e:
            logger.warning(f"Friedman test failed for {label}: {e}")
            return {"friedman_chi2": -999.0, "friedman_p": -999.0, "error": str(e)}

        # Compute ranks within each dataset (row); higher performance = rank 1
        ranks = np.zeros_like(matrix_5x3)
        for i in range(matrix_5x3.shape[0]):
            ranks[i] = stats.rankdata(-matrix_5x3[i])  # negative => descending
        avg_ranks = ranks.mean(axis=0)

        result = {
            "friedman_chi2": float(chi2),
            "friedman_p": float(p_value),
            "avg_rank_aa": float(avg_ranks[0]),
            "avg_rank_ro": float(avg_ranks[1]),
            "avg_rank_ss": float(avg_ranks[2]),
        }

        # Nemenyi post-hoc if significant
        if has_posthocs and p_value < 0.05:
            try:
                nemenyi = sp.posthoc_nemenyi_friedman(matrix_5x3)
                result["nemenyi_aa_vs_ro"] = float(nemenyi.iloc[0, 1])
                result["nemenyi_aa_vs_ss"] = float(nemenyi.iloc[0, 2])
                result["nemenyi_ro_vs_ss"] = float(nemenyi.iloc[1, 2])
            except Exception as e:
                logger.warning(f"Nemenyi failed for {label}: {e}")

        return result

    # Per max_splits level
    for ms in MAX_SPLITS_GRID:
        matrix = np.zeros((5, 3))
        for i, ds in enumerate(DATASETS):
            for j, method in enumerate(METHODS):
                key = (ds, method, ms)
                if key in lookup:
                    matrix[i, j] = lookup[key]["mean"]
                else:
                    logger.warning(f"Missing: {key}")

        results[f"max_splits_{ms}"] = run_friedman(matrix, f"ms={ms}")
        fr = results[f"max_splits_{ms}"]
        logger.info(
            f"Friedman ms={ms}: chi2={fr.get('friedman_chi2', -1):.3f}, "
            f"p={fr.get('friedman_p', -1):.4f}, "
            f"ranks=[aa={fr.get('avg_rank_aa', -1):.2f}, "
            f"ro={fr.get('avg_rank_ro', -1):.2f}, "
            f"ss={fr.get('avg_rank_ss', -1):.2f}]"
        )

    # Best max_splits variant: for each (dataset, method), use performance at its best max_splits
    matrix_best = np.zeros((5, 3))
    for i, ds in enumerate(DATASETS):
        for j, method in enumerate(METHODS):
            best_ms_key = f"{ds}__{method}"
            best_ms = best_max_splits.get(best_ms_key, 20)
            key = (ds, method, best_ms)
            if key in lookup:
                matrix_best[i, j] = lookup[key]["mean"]

    results["best_max_splits"] = run_friedman(matrix_best, "best_ms")
    fr = results["best_max_splits"]
    logger.info(
        f"Friedman best_ms: chi2={fr.get('friedman_chi2', -1):.3f}, "
        f"p={fr.get('friedman_p', -1):.4f}, "
        f"ranks=[aa={fr.get('avg_rank_aa', -1):.2f}, "
        f"ro={fr.get('avg_rank_ro', -1):.2f}, "
        f"ss={fr.get('avg_rank_ss', -1):.2f}]"
    )

    return results


# ═══════════════════════════════════════════════════════════════
# B. COHEN'S D EFFECT SIZES
# ═══════════════════════════════════════════════════════════════
def compute_cohens_d_analysis(lookup: dict) -> dict:
    """Cohen's d for signed_spectral vs axis_aligned and vs random_oblique."""
    per_dataset_ms = []

    wins_vs_aa = {"positive": 0, "negligible": 0, "negative": 0}
    wins_vs_ro = {"positive": 0, "negligible": 0, "negative": 0}

    def interpret_d(d_val: float) -> str:
        ad = abs(d_val)
        if ad < 0.2:
            return "negligible"
        elif ad < 0.8:
            return "medium"
        else:
            return "large"

    for ds in DATASETS:
        for ms in MAX_SPLITS_GRID:
            ss = lookup.get((ds, "signed_spectral_figs", ms))
            aa = lookup.get((ds, "axis_aligned_figs", ms))
            ro = lookup.get((ds, "random_oblique_figs", ms))

            if not (ss and aa and ro):
                continue

            # Cohen's d = (M1 - M2) / sqrt((SD1^2 + SD2^2) / 2)
            pooled_aa = math.sqrt((ss["std"] ** 2 + aa["std"] ** 2) / 2)
            d_ss_vs_aa = (ss["mean"] - aa["mean"]) / max(pooled_aa, 1e-10)

            pooled_ro = math.sqrt((ss["std"] ** 2 + ro["std"] ** 2) / 2)
            d_ss_vs_ro = (ss["mean"] - ro["mean"]) / max(pooled_ro, 1e-10)

            # Win/loss/tie classification
            if d_ss_vs_aa > 0.2:
                wins_vs_aa["positive"] += 1
            elif d_ss_vs_aa < -0.2:
                wins_vs_aa["negative"] += 1
            else:
                wins_vs_aa["negligible"] += 1

            if d_ss_vs_ro > 0.2:
                wins_vs_ro["positive"] += 1
            elif d_ss_vs_ro < -0.2:
                wins_vs_ro["negative"] += 1
            else:
                wins_vs_ro["negligible"] += 1

            per_dataset_ms.append({
                "dataset": ds,
                "max_splits": ms,
                "d_ss_vs_aa": float(d_ss_vs_aa),
                "d_ss_vs_ro": float(d_ss_vs_ro),
                "interp_ss_vs_aa": interpret_d(d_ss_vs_aa),
                "interp_ss_vs_ro": interpret_d(d_ss_vs_ro),
                "ss_mean": ss["mean"],
                "aa_mean": aa["mean"],
                "ro_mean": ro["mean"],
                "ss_std": ss["std"],
                "aa_std": aa["std"],
                "ro_std": ro["std"],
            })

    result = {
        "per_dataset_ms": per_dataset_ms,
        "win_loss_tie": {
            "ss_vs_aa": wins_vs_aa,
            "ss_vs_ro": wins_vs_ro,
        },
    }

    logger.info(
        f"Cohen's d: ss_vs_aa wins={wins_vs_aa['positive']}, "
        f"ties={wins_vs_aa['negligible']}, losses={wins_vs_aa['negative']}"
    )
    logger.info(
        f"Cohen's d: ss_vs_ro wins={wins_vs_ro['positive']}, "
        f"ties={wins_vs_ro['negligible']}, losses={wins_vs_ro['negative']}"
    )
    return result


# ═══════════════════════════════════════════════════════════════
# C. BAYESIAN SIGNED-RANK TEST
# ═══════════════════════════════════════════════════════════════
def compute_bayesian_test(lookup: dict, best_max_splits: dict) -> dict:
    """Bayesian signed-rank test with ROPE for method pairs. Bootstrap fallback."""
    ROPE = 0.01
    N_BOOTSTRAP = 10000

    # Get best-ms performance for each (dataset, method)
    best_perf = {}
    for ds in DATASETS:
        for method in METHODS:
            best_ms_key = f"{ds}__{method}"
            best_ms = best_max_splits.get(best_ms_key, 20)
            key = (ds, method, best_ms)
            if key in lookup:
                best_perf[(ds, method)] = lookup[key]["mean"]

    # Try baycomp first
    has_baycomp = False
    try:
        from baycomp import SignedRankTest
        has_baycomp = True
        logger.info("Using baycomp for Bayesian signed-rank test")
    except ImportError:
        logger.info("baycomp not available, using bootstrap approximation")

    pairs = [
        ("signed_spectral_figs", "axis_aligned_figs", "ss_vs_aa"),
        ("signed_spectral_figs", "random_oblique_figs", "ss_vs_ro"),
        ("random_oblique_figs", "axis_aligned_figs", "ro_vs_aa"),
    ]

    results = {}
    rng = np.random.RandomState(42)

    for m1, m2, label in pairs:
        scores_m1 = np.array([best_perf.get((ds, m1), np.nan) for ds in DATASETS])
        scores_m2 = np.array([best_perf.get((ds, m2), np.nan) for ds in DATASETS])

        valid = ~(np.isnan(scores_m1) | np.isnan(scores_m2))
        scores_m1 = scores_m1[valid]
        scores_m2 = scores_m2[valid]

        if len(scores_m1) < 3:
            results[label] = {"p_left": -999.0, "p_rope": -999.0, "p_right": -999.0, "error": "insufficient data"}
            continue

        used_baycomp = False
        if has_baycomp:
            try:
                test_result = SignedRankTest(scores_m1, scores_m2, rope=ROPE)
                p_left, p_rope, p_right = test_result.probs()
                results[label] = {
                    "p_left": float(p_left),
                    "p_rope": float(p_rope),
                    "p_right": float(p_right),
                    "method": "baycomp_signed_rank",
                }
                used_baycomp = True
            except Exception as e:
                logger.warning(f"baycomp failed for {label}: {e}, falling back to bootstrap")

        if not used_baycomp:
            # Bootstrap approximation
            diffs = scores_m1 - scores_m2
            n = len(diffs)
            boot_means = np.array([
                rng.choice(diffs, size=n, replace=True).mean()
                for _ in range(N_BOOTSTRAP)
            ])
            p_left = float(np.mean(boot_means < -ROPE))
            p_rope = float(np.mean((boot_means >= -ROPE) & (boot_means <= ROPE)))
            p_right = float(np.mean(boot_means > ROPE))
            results[label] = {
                "p_left": p_left,
                "p_rope": p_rope,
                "p_right": p_right,
                "method": "bootstrap_10000",
            }

        # Interpret
        r = results[label]
        if r["p_right"] > 0.95:
            r["interpretation"] = f"{METHOD_SHORT.get(m1, m1)} significantly better than {METHOD_SHORT.get(m2, m2)}"
        elif r["p_left"] > 0.95:
            r["interpretation"] = f"{METHOD_SHORT.get(m2, m2)} significantly better than {METHOD_SHORT.get(m1, m1)}"
        elif r["p_rope"] > 0.5:
            r["interpretation"] = "practically equivalent"
        else:
            r["interpretation"] = "inconclusive"

        logger.info(
            f"Bayesian {label}: P(left)={r['p_left']:.3f}, "
            f"P(rope)={r['p_rope']:.3f}, P(right)={r['p_right']:.3f} "
            f"-> {r['interpretation']}"
        )

    return results


# ═══════════════════════════════════════════════════════════════
# D. FRUSTRATION-ACCURACY CORRELATION
# ═══════════════════════════════════════════════════════════════
def compute_frustration_correlation(frustration_analysis: dict) -> dict:
    """Spearman correlation between frustration_index and ss_minus_aa gap."""
    ds_names = list(frustration_analysis.keys())
    frustrations = np.array([frustration_analysis[ds]["frustration_index"] for ds in ds_names])
    gaps = np.array([frustration_analysis[ds]["ss_minus_aa"] for ds in ds_names])
    n = len(ds_names)

    logger.info(f"Frustration-accuracy pairs (n={n}):")
    for i, ds in enumerate(ds_names):
        logger.info(f"  {ds}: frustration={frustrations[i]:.6f}, gap={gaps[i]:.6f}")

    # Spearman correlation
    rho, p_value = stats.spearmanr(frustrations, gaps)

    # Bootstrap 95% CI
    rng = np.random.RandomState(42)
    n_boot = 10000
    boot_rhos = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        if len(set(idx)) < 2:
            continue
        try:
            r, _ = stats.spearmanr(frustrations[idx], gaps[idx])
            if not np.isnan(r):
                boot_rhos.append(r)
        except Exception:
            pass

    boot_rhos = np.array(boot_rhos)
    ci_lower = float(np.percentile(boot_rhos, 2.5)) if len(boot_rhos) > 0 else float("nan")
    ci_upper = float(np.percentile(boot_rhos, 97.5)) if len(boot_rhos) > 0 else float("nan")

    # Power analysis: required n for 80% power at observed |rho|
    # Formula: n = (z_alpha + z_beta)^2 / (0.5 * ln((1+|rho|)/(1-|rho|)))^2 + 3
    z_alpha = 1.96  # two-sided alpha=0.05
    z_beta = 0.842  # power=0.80
    if 0.01 < abs(rho) < 0.999:
        fisher_z = 0.5 * math.log((1 + abs(rho)) / (1 - abs(rho)))
        required_n = math.ceil((z_alpha + z_beta) ** 2 / fisher_z ** 2 + 3)
    else:
        required_n = -1  # undefined

    result = {
        "all_datasets": {
            "n": n,
            "spearman_rho": float(rho),
            "p_value": float(p_value),
            "bootstrap_ci_lower": safe_float(ci_lower),
            "bootstrap_ci_upper": safe_float(ci_upper),
            "required_n_for_power_80": required_n,
            "datasets": ds_names,
            "frustrations": frustrations.tolist(),
            "gaps": gaps.tolist(),
        },
    }

    # Excluding california_housing (only regression dataset)
    excl_idx = [i for i, ds in enumerate(ds_names) if ds != "california_housing"]
    if len(excl_idx) >= 3:
        frust_excl = frustrations[excl_idx]
        gaps_excl = gaps[excl_idx]
        rho_excl, p_excl = stats.spearmanr(frust_excl, gaps_excl)

        boot_rhos_excl = []
        for _ in range(n_boot):
            idx = rng.choice(len(excl_idx), size=len(excl_idx), replace=True)
            if len(set(idx)) < 2:
                continue
            try:
                r, _ = stats.spearmanr(frust_excl[idx], gaps_excl[idx])
                if not np.isnan(r):
                    boot_rhos_excl.append(r)
            except Exception:
                pass
        boot_rhos_excl = np.array(boot_rhos_excl)

        result["excluding_california"] = {
            "n": len(excl_idx),
            "spearman_rho": float(rho_excl),
            "p_value": float(p_excl),
            "bootstrap_ci_lower": safe_float(
                float(np.percentile(boot_rhos_excl, 2.5)) if len(boot_rhos_excl) > 0 else float("nan")
            ),
            "bootstrap_ci_upper": safe_float(
                float(np.percentile(boot_rhos_excl, 97.5)) if len(boot_rhos_excl) > 0 else float("nan")
            ),
        }

    logger.info(f"Frustration-accuracy: rho={rho:.3f}, p={p_value:.4f}, CI=[{ci_lower:.3f}, {ci_upper:.3f}]")
    logger.info(f"  Required n for 80% power: {required_n}")

    return result


# ═══════════════════════════════════════════════════════════════
# E. SUBSAMPLING IMPACT QUANTIFICATION
# ═══════════════════════════════════════════════════════════════
def compute_subsampling_impact(stability_data: dict) -> dict:
    """Fit noise-floor model and estimate recommended subsample size."""
    results_by_sub = stability_data["results_by_subsample"]

    ns = np.array([r["n_sub"] for r in results_by_sub], dtype=float)
    sign_flips = np.array([r["sign_flip_rate"] for r in results_by_sub])
    spearmans = np.array([r["spearman_r"] for r in results_by_sub])
    aris = np.array([r["sponge_ari"] for r in results_by_sub])

    model_type = "none"
    model_params = {}
    recommended_n_from_model = None

    # Fit exponential decay to sign_flip_rate: y = a * exp(-b * n) + c
    try:
        def exp_decay(x, a, b, c):
            return a * np.exp(-b * x) + c

        popt, _ = curve_fit(
            exp_decay, ns, sign_flips,
            p0=[0.5, 1e-4, 0.0],
            bounds=([0, 0, -0.1], [1.0, 1.0, 0.5]),
            maxfev=10000,
        )
        model_params = {"a": float(popt[0]), "b": float(popt[1]), "c": float(popt[2])}
        model_type = "exponential_decay"

        # Find n where sign_flip = 0.1
        # 0.1 = a * exp(-b * n) + c  =>  n = -ln((0.1 - c) / a) / b
        target = 0.1
        ratio = (target - popt[2]) / popt[0]
        if ratio > 0 and popt[1] > 0:
            recommended_n_from_model = int(-math.log(ratio) / popt[1])
        logger.info(f"Exponential decay fit: a={popt[0]:.4f}, b={popt[1]:.6f}, c={popt[2]:.4f}")
    except Exception as e:
        logger.warning(f"Exponential fit failed: {e}, trying power law")
        # Fallback: power law
        try:
            mask = sign_flips > 0
            def power_law(x, a, b):
                return a * np.power(x, -b)
            popt2, _ = curve_fit(power_law, ns[mask], sign_flips[mask], p0=[100, 0.5], maxfev=10000)
            model_params = {"a": float(popt2[0]), "b": float(popt2[1])}
            model_type = "power_law"
            recommended_n_from_model = int((popt2[0] / 0.1) ** (1 / popt2[1]))
        except Exception as e2:
            logger.warning(f"Power law fit also failed: {e2}")

    # Find actual inflection from data
    recommended_n_data = None
    for r in results_by_sub:
        if r["spearman_r"] > 0.7 and r["sign_flip_rate"] < 0.1:
            recommended_n_data = r["n_sub"]
            break
    if recommended_n_data is None:
        recommended_n_data = stability_data.get("minimum_stable_n", 20000)

    # Impact at n=10K (the value used in exp_id2)
    n10k = next((r for r in results_by_sub if r["n_sub"] == 10000), None)
    n15k = next((r for r in results_by_sub if r["n_sub"] == 15000), None)
    n20k = next((r for r in results_by_sub if r["n_sub"] == 20000), None)

    result = {
        "model_type": model_type,
        "model_params": model_params,
        "data_points": {
            "subsample_sizes": ns.tolist(),
            "sign_flip_rates": sign_flips.tolist(),
            "spearman_correlations": spearmans.tolist(),
            "sponge_aris": aris.tolist(),
        },
        "at_experiment_n_10k": {
            "sign_flip_rate": n10k["sign_flip_rate"] if n10k else None,
            "spearman_r": n10k["spearman_r"] if n10k else None,
            "sponge_ari": n10k["sponge_ari"] if n10k else None,
        },
        "ari_improvement": {
            "n10k_to_n15k": round(n15k["sponge_ari"] - n10k["sponge_ari"], 4) if n10k and n15k else None,
            "n15k_to_n20k": round(n20k["sponge_ari"] - n15k["sponge_ari"], 4) if n15k and n20k else None,
        },
        "recommended_minimum_n": recommended_n_data,
        "recommended_n_from_model": recommended_n_from_model,
    }

    logger.info(
        f"Subsampling: recommended min n={recommended_n_data}, model={model_type}, "
        f"model_n={recommended_n_from_model}"
    )
    return result


# ═══════════════════════════════════════════════════════════════
# F. INTERPRETABILITY ANALYSIS (Split Arity & Path Length)
# ═══════════════════════════════════════════════════════════════
def compute_interpretability(lookup: dict) -> dict:
    """Compare split arity and path length across methods, Wilcoxon tests, Pareto analysis."""
    arity_ss = []
    arity_ro = []
    path_ss = []
    path_ro = []
    per_entry = []

    for ds in DATASETS:
        for ms in MAX_SPLITS_GRID:
            aa = lookup.get((ds, "axis_aligned_figs", ms))
            ro = lookup.get((ds, "random_oblique_figs", ms))
            ss = lookup.get((ds, "signed_spectral_figs", ms))

            if not (aa and ro and ss):
                continue

            entry = {
                "dataset": ds,
                "max_splits": ms,
                "aa_arity": aa["arity"],
                "ro_arity": ro["arity"],
                "ss_arity": ss["arity"],
                "aa_path": aa["path_length"],
                "ro_path": ro["path_length"],
                "ss_path": ss["path_length"],
                "ss_ro_arity_ratio": round(ss["arity"] / ro["arity"], 4) if ro["arity"] > 0 else None,
            }
            per_entry.append(entry)

            arity_ss.append(ss["arity"])
            arity_ro.append(ro["arity"])
            path_ss.append(ss["path_length"])
            path_ro.append(ro["path_length"])

    arity_ss_arr = np.array(arity_ss)
    arity_ro_arr = np.array(arity_ro)
    path_ss_arr = np.array(path_ss)
    path_ro_arr = np.array(path_ro)

    arity_diffs = arity_ss_arr - arity_ro_arr
    path_diffs = path_ss_arr - path_ro_arr

    # Wilcoxon signed-rank test on arity (ss vs ro)
    nonzero_arity = arity_diffs[arity_diffs != 0]
    try:
        w_arity, p_arity = stats.wilcoxon(nonzero_arity)
        w_arity, p_arity = float(w_arity), float(p_arity)
    except Exception:
        w_arity, p_arity = None, None

    # Wilcoxon signed-rank test on path length (ss vs ro)
    nonzero_path = path_diffs[path_diffs != 0]
    try:
        w_path, p_path = stats.wilcoxon(nonzero_path)
        w_path, p_path = float(w_path), float(p_path)
    except Exception:
        w_path, p_path = None, None

    # Pareto efficiency at max_splits=20
    pareto_analysis = []
    for ds in DATASETS:
        aa = lookup.get((ds, "axis_aligned_figs", 20))
        ro = lookup.get((ds, "random_oblique_figs", 20))
        ss = lookup.get((ds, "signed_spectral_figs", 20))

        if not (aa and ro and ss):
            continue

        methods_data = {
            "axis_aligned": {"perf": aa["mean"], "arity": aa["arity"], "path": aa["path_length"]},
            "random_oblique": {"perf": ro["mean"], "arity": ro["arity"], "path": ro["path_length"]},
            "signed_spectral": {"perf": ss["mean"], "arity": ss["arity"], "path": ss["path_length"]},
        }

        # Check Pareto dominance: better perf AND lower arity
        pareto_dominant = []
        for m1_name, m1 in methods_data.items():
            dominates_all = True
            for m2_name, m2 in methods_data.items():
                if m1_name == m2_name:
                    continue
                if not (m1["perf"] >= m2["perf"] and m1["arity"] <= m2["arity"]):
                    dominates_all = False
                    break
            if dominates_all:
                pareto_dominant.append(m1_name)

        pareto_analysis.append({
            "dataset": ds,
            "pareto_dominant": pareto_dominant if pareto_dominant else ["none"],
            "aa_perf": aa["mean"], "ro_perf": ro["mean"], "ss_perf": ss["mean"],
            "aa_arity": aa["arity"], "ro_arity": ro["arity"], "ss_arity": ss["arity"],
            "aa_path": aa["path_length"], "ro_path": ro["path_length"], "ss_path": ss["path_length"],
        })

    result = {
        "wilcoxon_arity": {
            "statistic": w_arity,
            "p_value": p_arity,
            "n_pairs": len(arity_ss),
            "n_nonzero": len(nonzero_arity),
            "mean_diff_ss_minus_ro": float(np.mean(arity_diffs)),
            "interpretation": "ss has lower arity" if np.mean(arity_diffs) < 0 else "ss has higher or equal arity",
        },
        "wilcoxon_path": {
            "statistic": w_path,
            "p_value": p_path,
            "n_pairs": len(path_ss),
            "n_nonzero": len(nonzero_path),
            "mean_diff_ss_minus_ro": float(np.mean(path_diffs)),
        },
        "pareto_at_ms20": pareto_analysis,
        "per_entry": per_entry,
    }

    logger.info(
        f"Interpretability: arity Wilcoxon p={p_arity}, "
        f"mean arity diff (ss-ro)={np.mean(arity_diffs):.3f}"
    )
    return result


# ═══════════════════════════════════════════════════════════════
# G. CALIFORNIA HOUSING REGRESSION ANOMALY DIAGNOSIS
# ═══════════════════════════════════════════════════════════════
def compute_california_diagnosis(lookup: dict) -> dict:
    """Diagnose california_housing regression anomaly."""
    entries = []

    for ms in MAX_SPLITS_GRID:
        aa = lookup.get(("california_housing", "axis_aligned_figs", ms))
        ro = lookup.get(("california_housing", "random_oblique_figs", ms))
        ss = lookup.get(("california_housing", "signed_spectral_figs", ms))

        if not (aa and ro and ss):
            continue

        entries.append({
            "max_splits": ms,
            "aa_r2": aa["mean"], "aa_r2_std": aa["std"],
            "ro_r2": ro["mean"], "ro_r2_std": ro["std"],
            "ss_r2": ss["mean"], "ss_r2_std": ss["std"],
            "gap_ss_minus_aa": ss["mean"] - aa["mean"],
            "gap_ss_minus_ro": ss["mean"] - ro["mean"],
            "aa_arity": aa["arity"], "ro_arity": ro["arity"], "ss_arity": ss["arity"],
            "aa_path": aa["path_length"], "ro_path": ro["path_length"], "ss_path": ss["path_length"],
        })

    # Cross-dataset arity comparison at ms=20 (contextualize high arity)
    arity_by_ds = {}
    for ds in DATASETS:
        ss = lookup.get((ds, "signed_spectral_figs", 20))
        ro = lookup.get((ds, "random_oblique_figs", 20))
        if ss and ro:
            arity_by_ds[ds] = {
                "ss_arity": ss["arity"],
                "ro_arity": ro["arity"],
                "ss_std": ss["std"],
            }

    # Variance comparison: ss std vs aa std
    variance_ratio = {}
    for ms in MAX_SPLITS_GRID:
        ss = lookup.get(("california_housing", "signed_spectral_figs", ms))
        aa = lookup.get(("california_housing", "axis_aligned_figs", ms))
        if ss and aa and aa["std"] > 0:
            variance_ratio[f"ms_{ms}"] = {
                "ss_std": ss["std"],
                "aa_std": aa["std"],
                "ratio": ss["std"] / aa["std"],
            }

    # Classification vs regression gap comparison at ms=20
    clf_gaps = []
    reg_gaps = []
    for ds in DATASETS:
        ss = lookup.get((ds, "signed_spectral_figs", 20))
        aa = lookup.get((ds, "axis_aligned_figs", 20))
        if ss and aa:
            gap = ss["mean"] - aa["mean"]
            if ss["task_type"] == "classification":
                clf_gaps.append(gap)
            else:
                reg_gaps.append(gap)

    result = {
        "per_max_splits": entries,
        "arity_cross_dataset": arity_by_ds,
        "variance_ratio": variance_ratio,
        "classification_vs_regression": {
            "mean_clf_gap": float(np.mean(clf_gaps)) if clf_gaps else None,
            "mean_reg_gap": float(np.mean(reg_gaps)) if reg_gaps else None,
            "clf_gaps": clf_gaps,
            "reg_gaps": reg_gaps,
            "regression_underperforms": bool(np.mean(reg_gaps) < np.mean(clf_gaps)) if clf_gaps and reg_gaps else None,
        },
    }

    if entries:
        logger.info(
            f"California diagnosis: gap at ms=20: ss-aa={entries[-1]['gap_ss_minus_aa']:.4f}, "
            f"ss_arity={entries[-1]['ss_arity']:.1f}, ss_std={entries[-1]['ss_r2_std']:.4f}"
        )
    return result


# ═══════════════════════════════════════════════════════════════
# H. SYNTHETIC RECOVERY STATISTICAL SUMMARY
# ═══════════════════════════════════════════════════════════════
def compute_synthetic_summary(per_variant: dict) -> dict:
    """Statistical summary of synthetic recovery experiments."""
    structured_variants = [v for v in SYNTHETIC_VARIANTS_STRUCTURED if v in per_variant]

    metrics = ["adjusted_rand_index", "module_focused_ari", "synergistic_pair_jaccard", "xor_recovery_fraction"]
    method_ranks = {m: {metric: [] for metric in metrics} for m in SYNTHETIC_METHODS}

    for variant_name in structured_variants:
        vdata = per_variant[variant_name]
        for metric in metrics:
            vals = {}
            for method in SYNTHETIC_METHODS:
                v = vdata["methods"][method].get(metric)
                if v is not None:
                    vals[method] = v

            if not vals:
                continue

            # Rank (highest is best, rank 1 = best)
            sorted_methods = sorted(vals.keys(), key=lambda m: vals[m], reverse=True)
            for rank, m in enumerate(sorted_methods, 1):
                method_ranks[m][metric].append(rank)

    avg_ranks = {}
    for method in SYNTHETIC_METHODS:
        for metric in metrics:
            ranks = method_ranks[method][metric]
            if ranks:
                avg_ranks[f"{method}__{metric}"] = float(np.mean(ranks))

    # SPONGE vs baselines: pairwise win rates
    comparisons = [
        ("sponge_oracle_k", "hard_threshold", "sponge_oracle_vs_hard_threshold"),
        ("sponge_oracle_k", "unsigned_spectral", "sponge_oracle_vs_unsigned_spectral"),
        ("sponge_auto_k", "hard_threshold", "sponge_auto_vs_hard_threshold"),
        ("sponge_auto_k", "unsigned_spectral", "sponge_auto_vs_unsigned_spectral"),
    ]

    win_rates = {}
    for m1, m2, comp_label in comparisons:
        metric_wins = {metric: 0 for metric in metrics}
        n_comp = {metric: 0 for metric in metrics}

        for variant_name in structured_variants:
            vdata = per_variant[variant_name]
            for metric in metrics:
                v1 = vdata["methods"].get(m1, {}).get(metric)
                v2 = vdata["methods"].get(m2, {}).get(metric)
                if v1 is not None and v2 is not None:
                    n_comp[metric] += 1
                    if v1 > v2:
                        metric_wins[metric] += 1

        win_rates[comp_label] = {
            metric: float(metric_wins[metric] / n_comp[metric]) if n_comp[metric] > 0 else None
            for metric in metrics
        }

    # Frustration diagnostic check
    no_structure = per_variant.get("no_structure_control", {})
    easy = per_variant.get("easy_2mod_xor", {})
    frustration_diagnostic = {
        "no_structure_frustration": no_structure.get("frustration_index"),
        "easy_structured_frustration": easy.get("frustration_index"),
        "no_structure_higher": (
            no_structure.get("frustration_index", 0) > easy.get("frustration_index", 1)
        ),
    }

    # Scalability: CoI computation time vs number of features
    times_features = []
    for variant_name, vdata in per_variant.items():
        d = vdata["n_features"]
        t = vdata["coi_computation_time_sec"]
        n = vdata["n_samples"]
        times_features.append({"variant": variant_name, "d": d, "n": n, "time_sec": t})

    # Fit quadratic model: t = a * d^2 * n + b
    ds_arr = np.array([r["d"] for r in times_features], dtype=float)
    ns_arr = np.array([r["n"] for r in times_features], dtype=float)
    ts_arr = np.array([r["time_sec"] for r in times_features], dtype=float)

    try:
        X = (ds_arr ** 2 * ns_arr).reshape(-1, 1)
        A = np.column_stack([X, np.ones(len(X))])
        coeffs, _, _, _ = np.linalg.lstsq(A, ts_arr, rcond=None)
        residuals = ts_arr - A @ coeffs
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((ts_arr - ts_arr.mean()) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        scalability_model = {
            "model": "t = a * d^2 * n + b",
            "a_coeff": float(coeffs[0]),
            "b_intercept": float(coeffs[1]),
            "r_squared": float(r_squared),
        }
    except Exception as e:
        scalability_model = {"error": str(e)}

    result = {
        "method_avg_ranks": avg_ranks,
        "sponge_win_rates": win_rates,
        "frustration_diagnostic": frustration_diagnostic,
        "scalability": {
            "data": times_features,
            "model": scalability_model,
        },
    }

    logger.info(f"Synthetic summary: {len(structured_variants)} structured variants analyzed")
    return result


# ═══════════════════════════════════════════════════════════════
# I. SUCCESS CRITERIA EVALUATION
# ═══════════════════════════════════════════════════════════════
def compute_success_criteria(
    per_variant: dict,
    lookup: dict,
    best_max_splits: dict,
    frustration_result: dict,
    total_wallclock: float,
) -> dict:
    """Check each success criterion from the hypothesis."""

    # Criterion 1: On synthetic data, signed spectral recovers >80% of true synergistic pairs
    easy = per_variant.get("easy_2mod_xor", {})
    medium = per_variant.get("medium_4mod_mixed", {})

    max_jac_easy = max(
        easy.get("methods", {}).get("sponge_oracle_k", {}).get("synergistic_pair_jaccard", 0) or 0,
        easy.get("methods", {}).get("sponge_auto_k", {}).get("synergistic_pair_jaccard", 0) or 0,
    )
    max_jac_medium = max(
        medium.get("methods", {}).get("sponge_oracle_k", {}).get("synergistic_pair_jaccard", 0) or 0,
        medium.get("methods", {}).get("sponge_auto_k", {}).get("synergistic_pair_jaccard", 0) or 0,
    )

    criterion1 = {
        "description": "Signed spectral recovers >80% of true synergistic pairs on synthetic data",
        "max_jaccard_easy": max_jac_easy,
        "max_jaccard_medium": max_jac_medium,
        "passed": bool(max_jac_easy > 0.8 and max_jac_medium > 0.8),
    }

    # Criterion 2: On real benchmarks, signed spectral matches or exceeds random-oblique with lower arity
    matches_count = 0
    total_count = 0
    for ds in DATASETS:
        ss_key = f"{ds}__signed_spectral_figs"
        ro_key = f"{ds}__random_oblique_figs"
        ss_ms = best_max_splits.get(ss_key, 20)
        ro_ms = best_max_splits.get(ro_key, 20)

        ss = lookup.get((ds, "signed_spectral_figs", ss_ms))
        ro = lookup.get((ds, "random_oblique_figs", ro_ms))

        if ss and ro:
            total_count += 1
            # Match or exceed performance (within 0.5%) AND lower or equal arity
            if ss["mean"] >= ro["mean"] - 0.005 and ss["arity"] <= ro["arity"]:
                matches_count += 1

    criterion2 = {
        "description": "Signed spectral matches/exceeds random-oblique accuracy with lower arity",
        "matching_datasets": matches_count,
        "total_datasets": total_count,
        "passed": bool(matches_count > total_count / 2),
    }

    # Criterion 3: Frustration-accuracy correlation significant (p < 0.05)
    frust_all = frustration_result.get("all_datasets", {})
    criterion3 = {
        "description": "Frustration-accuracy correlation significant (p < 0.05)",
        "spearman_rho": frust_all.get("spearman_rho"),
        "p_value": frust_all.get("p_value"),
        "passed": bool(frust_all.get("p_value", 1.0) < 0.05),
    }

    # Criterion 4: Pipeline completes within 30 min for d<=200, n<=100K
    criterion4 = {
        "description": "Pipeline completes within 30 min for d<=200, n<=100K",
        "total_wallclock_sec": total_wallclock,
        "passed": bool(total_wallclock < 1800),
    }

    overall = sum([
        criterion1["passed"],
        criterion2["passed"],
        criterion3["passed"],
        criterion4["passed"],
    ])

    result = {
        "criterion1_synthetic_recovery": criterion1,
        "criterion2_real_benchmark": criterion2,
        "criterion3_frustration_correlation": criterion3,
        "criterion4_timing": criterion4,
        "overall_assessment": overall,
    }

    for i, c in enumerate([criterion1, criterion2, criterion3, criterion4], 1):
        status = "PASS" if c["passed"] else "FAIL"
        logger.info(f"Criterion {i}: {status} - {c['description']}")

    return result


# ═══════════════════════════════════════════════════════════════
# OUTPUT FORMATTING
# ═══════════════════════════════════════════════════════════════
def build_eval_output(
    friedman_results: dict,
    cohens_d_results: dict,
    bayesian_results: dict,
    frustration_results: dict,
    subsampling_results: dict,
    interpretability_results: dict,
    california_results: dict,
    synthetic_results: dict,
    success_results: dict,
    per_variant: dict,
) -> dict:
    """Build eval_out.json in schema-compliant format."""

    # ─── metrics_agg (all values must be numbers) ───
    metrics_agg = {}

    # A. Friedman p-values
    for ms in MAX_SPLITS_GRID:
        key = f"max_splits_{ms}"
        if key in friedman_results and "friedman_p" in friedman_results[key]:
            metrics_agg[f"friedman_p_ms{ms}"] = safe_float(friedman_results[key]["friedman_p"])
            metrics_agg[f"friedman_chi2_ms{ms}"] = safe_float(friedman_results[key]["friedman_chi2"])
    if "best_max_splits" in friedman_results and "friedman_p" in friedman_results["best_max_splits"]:
        metrics_agg["friedman_p_best_ms"] = safe_float(friedman_results["best_max_splits"]["friedman_p"])
        metrics_agg["friedman_chi2_best_ms"] = safe_float(friedman_results["best_max_splits"]["friedman_chi2"])

    # B. Cohen's d summary
    d_vals_aa = [e["d_ss_vs_aa"] for e in cohens_d_results["per_dataset_ms"]]
    d_vals_ro = [e["d_ss_vs_ro"] for e in cohens_d_results["per_dataset_ms"]]
    metrics_agg["mean_cohens_d_ss_vs_aa"] = safe_float(np.mean(d_vals_aa))
    metrics_agg["mean_cohens_d_ss_vs_ro"] = safe_float(np.mean(d_vals_ro))
    metrics_agg["median_cohens_d_ss_vs_aa"] = safe_float(np.median(d_vals_aa))
    metrics_agg["median_cohens_d_ss_vs_ro"] = safe_float(np.median(d_vals_ro))
    metrics_agg["wins_ss_vs_aa"] = float(cohens_d_results["win_loss_tie"]["ss_vs_aa"]["positive"])
    metrics_agg["losses_ss_vs_aa"] = float(cohens_d_results["win_loss_tie"]["ss_vs_aa"]["negative"])
    metrics_agg["ties_ss_vs_aa"] = float(cohens_d_results["win_loss_tie"]["ss_vs_aa"]["negligible"])
    metrics_agg["wins_ss_vs_ro"] = float(cohens_d_results["win_loss_tie"]["ss_vs_ro"]["positive"])
    metrics_agg["losses_ss_vs_ro"] = float(cohens_d_results["win_loss_tie"]["ss_vs_ro"]["negative"])
    metrics_agg["ties_ss_vs_ro"] = float(cohens_d_results["win_loss_tie"]["ss_vs_ro"]["negligible"])

    # C. Bayesian test
    for label in ["ss_vs_aa", "ss_vs_ro", "ro_vs_aa"]:
        if label in bayesian_results and "p_right" in bayesian_results[label]:
            metrics_agg[f"bayesian_p_right_{label}"] = safe_float(bayesian_results[label]["p_right"])
            metrics_agg[f"bayesian_p_rope_{label}"] = safe_float(bayesian_results[label]["p_rope"])
            metrics_agg[f"bayesian_p_left_{label}"] = safe_float(bayesian_results[label]["p_left"])

    # D. Frustration correlation
    frust_all = frustration_results.get("all_datasets", {})
    metrics_agg["frustration_spearman_rho"] = safe_float(frust_all.get("spearman_rho", 0.0))
    metrics_agg["frustration_p_value"] = safe_float(frust_all.get("p_value", 1.0))
    metrics_agg["frustration_required_n_power80"] = safe_float(frust_all.get("required_n_for_power_80", -1))
    frust_excl = frustration_results.get("excluding_california", {})
    if frust_excl:
        metrics_agg["frustration_rho_excl_calif"] = safe_float(frust_excl.get("spearman_rho"))
        metrics_agg["frustration_p_excl_calif"] = safe_float(frust_excl.get("p_value"))

    # E. Subsampling
    metrics_agg["subsampling_recommended_n"] = safe_float(subsampling_results.get("recommended_minimum_n", 20000))
    at_10k = subsampling_results.get("at_experiment_n_10k", {})
    metrics_agg["sign_flip_rate_at_10k"] = safe_float(at_10k.get("sign_flip_rate"))
    metrics_agg["spearman_r_at_10k"] = safe_float(at_10k.get("spearman_r"))
    metrics_agg["sponge_ari_at_10k"] = safe_float(at_10k.get("sponge_ari"))

    # F. Interpretability
    interp_arity = interpretability_results.get("wilcoxon_arity", {})
    metrics_agg["wilcoxon_arity_p"] = safe_float(interp_arity.get("p_value"))
    metrics_agg["mean_arity_diff_ss_ro"] = safe_float(interp_arity.get("mean_diff_ss_minus_ro"))
    interp_path = interpretability_results.get("wilcoxon_path", {})
    metrics_agg["wilcoxon_path_p"] = safe_float(interp_path.get("p_value"))
    metrics_agg["mean_path_diff_ss_ro"] = safe_float(interp_path.get("mean_diff_ss_minus_ro"))

    # G. California housing
    if california_results.get("per_max_splits"):
        last = california_results["per_max_splits"][-1]
        metrics_agg["calif_gap_ss_aa_ms20"] = safe_float(last["gap_ss_minus_aa"])
        metrics_agg["calif_ss_std_ms20"] = safe_float(last["ss_r2_std"])
        metrics_agg["calif_ss_arity_ms20"] = safe_float(last["ss_arity"])
    clf_vs_reg = california_results.get("classification_vs_regression", {})
    metrics_agg["mean_clf_gap_ss_aa"] = safe_float(clf_vs_reg.get("mean_clf_gap"))
    metrics_agg["mean_reg_gap_ss_aa"] = safe_float(clf_vs_reg.get("mean_reg_gap"))

    # I. Success criteria
    metrics_agg["success_criteria_passed"] = float(success_results.get("overall_assessment", 0))
    metrics_agg["criterion1_passed"] = 1.0 if success_results["criterion1_synthetic_recovery"]["passed"] else 0.0
    metrics_agg["criterion2_passed"] = 1.0 if success_results["criterion2_real_benchmark"]["passed"] else 0.0
    metrics_agg["criterion3_passed"] = 1.0 if success_results["criterion3_frustration_correlation"]["passed"] else 0.0
    metrics_agg["criterion4_passed"] = 1.0 if success_results["criterion4_timing"]["passed"] else 0.0

    # ─── datasets (schema: array of {dataset: str, examples: [{input, output, eval_*, metadata_*, predict_*}]}) ───
    datasets = []

    # Dataset 1: Friedman/Nemenyi tests
    friedman_examples = []
    for ms_label in [f"max_splits_{ms}" for ms in MAX_SPLITS_GRID] + ["best_max_splits"]:
        fr = friedman_results.get(ms_label, {})
        if "friedman_chi2" not in fr:
            continue
        ms_val = ms_label.replace("max_splits_", "") if ms_label.startswith("max_splits_") else "best"
        ex = {
            "input": json.dumps({"analysis": "friedman_nemenyi", "max_splits": ms_val}),
            "output": json.dumps({"significant": fr["friedman_p"] < 0.05}),
            "eval_friedman_chi2": safe_float(fr["friedman_chi2"]),
            "eval_friedman_p": safe_float(fr["friedman_p"]),
            "eval_avg_rank_aa": safe_float(fr.get("avg_rank_aa")),
            "eval_avg_rank_ro": safe_float(fr.get("avg_rank_ro")),
            "eval_avg_rank_ss": safe_float(fr.get("avg_rank_ss")),
            "metadata_max_splits": ms_val,
        }
        if "nemenyi_aa_vs_ro" in fr:
            ex["eval_nemenyi_aa_vs_ro"] = safe_float(fr["nemenyi_aa_vs_ro"])
            ex["eval_nemenyi_aa_vs_ss"] = safe_float(fr["nemenyi_aa_vs_ss"])
            ex["eval_nemenyi_ro_vs_ss"] = safe_float(fr["nemenyi_ro_vs_ss"])
        friedman_examples.append(ex)
    if friedman_examples:
        datasets.append({"dataset": "friedman_nemenyi_tests", "examples": friedman_examples})

    # Dataset 2: Cohen's d effect sizes
    effect_examples = []
    for entry in cohens_d_results["per_dataset_ms"]:
        effect_examples.append({
            "input": json.dumps({"dataset": entry["dataset"], "max_splits": entry["max_splits"]}),
            "output": json.dumps({
                "d_ss_vs_aa": round(entry["d_ss_vs_aa"], 4),
                "d_ss_vs_ro": round(entry["d_ss_vs_ro"], 4),
            }),
            "eval_d_ss_vs_aa": safe_float(entry["d_ss_vs_aa"]),
            "eval_d_ss_vs_ro": safe_float(entry["d_ss_vs_ro"]),
            "eval_ss_mean": safe_float(entry["ss_mean"]),
            "eval_aa_mean": safe_float(entry["aa_mean"]),
            "eval_ro_mean": safe_float(entry["ro_mean"]),
            "metadata_dataset": entry["dataset"],
            "metadata_max_splits": entry["max_splits"],
            "metadata_interp_vs_aa": entry["interp_ss_vs_aa"],
            "metadata_interp_vs_ro": entry["interp_ss_vs_ro"],
        })
    if effect_examples:
        datasets.append({"dataset": "cohens_d_effect_sizes", "examples": effect_examples})

    # Dataset 3: Bayesian signed-rank tests
    bayesian_examples = []
    for label, res in bayesian_results.items():
        if "p_right" not in res:
            continue
        bayesian_examples.append({
            "input": json.dumps({"comparison": label}),
            "output": json.dumps({"interpretation": res.get("interpretation", "N/A")}),
            "eval_p_left": safe_float(res["p_left"]),
            "eval_p_rope": safe_float(res["p_rope"]),
            "eval_p_right": safe_float(res["p_right"]),
            "metadata_comparison": label,
            "metadata_method": res.get("method", "unknown"),
        })
    if bayesian_examples:
        datasets.append({"dataset": "bayesian_signed_rank", "examples": bayesian_examples})

    # Dataset 4: Frustration-accuracy correlation
    frust_ds = frustration_results.get("all_datasets", {})
    frust_examples = []
    ds_names = frust_ds.get("datasets", [])
    frusts = frust_ds.get("frustrations", [])
    gaps = frust_ds.get("gaps", [])
    for i in range(len(ds_names)):
        frust_examples.append({
            "input": json.dumps({"dataset": ds_names[i]}),
            "output": json.dumps({"frustration": round(frusts[i], 6), "gap": round(gaps[i], 6)}),
            "eval_frustration_index": safe_float(frusts[i]),
            "eval_ss_minus_aa_gap": safe_float(gaps[i]),
            "metadata_dataset": ds_names[i],
        })
    if frust_examples:
        datasets.append({"dataset": "frustration_correlation", "examples": frust_examples})

    # Dataset 5: Subsampling stability
    sub_data = subsampling_results.get("data_points", {})
    sub_examples = []
    sub_ns = sub_data.get("subsample_sizes", [])
    sub_flips = sub_data.get("sign_flip_rates", [])
    sub_spears = sub_data.get("spearman_correlations", [])
    sub_aris = sub_data.get("sponge_aris", [])
    for i in range(len(sub_ns)):
        sub_examples.append({
            "input": json.dumps({"subsample_n": int(sub_ns[i])}),
            "output": json.dumps({"sign_flip_rate": sub_flips[i], "spearman": sub_spears[i]}),
            "eval_sign_flip_rate": safe_float(sub_flips[i]),
            "eval_spearman_r": safe_float(sub_spears[i]),
            "eval_sponge_ari": safe_float(sub_aris[i]),
            "metadata_subsample_n": int(sub_ns[i]),
        })
    if sub_examples:
        datasets.append({"dataset": "subsampling_stability", "examples": sub_examples})

    # Dataset 6: Interpretability analysis
    interp_examples = []
    for entry in interpretability_results.get("per_entry", []):
        ratio_val = entry.get("ss_ro_arity_ratio")
        interp_examples.append({
            "input": json.dumps({"dataset": entry["dataset"], "max_splits": entry["max_splits"]}),
            "output": json.dumps({"ss_ro_arity_ratio": ratio_val}),
            "eval_aa_arity": safe_float(entry["aa_arity"]),
            "eval_ro_arity": safe_float(entry["ro_arity"]),
            "eval_ss_arity": safe_float(entry["ss_arity"]),
            "eval_aa_path": safe_float(entry["aa_path"]),
            "eval_ro_path": safe_float(entry["ro_path"]),
            "eval_ss_path": safe_float(entry["ss_path"]),
            "metadata_dataset": entry["dataset"],
            "metadata_max_splits": entry["max_splits"],
        })
    if interp_examples:
        datasets.append({"dataset": "interpretability_analysis", "examples": interp_examples})

    # Dataset 7: California housing diagnosis
    calif_examples = []
    for entry in california_results.get("per_max_splits", []):
        calif_examples.append({
            "input": json.dumps({"dataset": "california_housing", "max_splits": entry["max_splits"]}),
            "output": json.dumps({"gap_ss_aa": round(entry["gap_ss_minus_aa"], 4)}),
            "eval_aa_r2": safe_float(entry["aa_r2"]),
            "eval_ro_r2": safe_float(entry["ro_r2"]),
            "eval_ss_r2": safe_float(entry["ss_r2"]),
            "eval_gap_ss_minus_aa": safe_float(entry["gap_ss_minus_aa"]),
            "eval_gap_ss_minus_ro": safe_float(entry["gap_ss_minus_ro"]),
            "eval_ss_r2_std": safe_float(entry["ss_r2_std"]),
            "eval_aa_r2_std": safe_float(entry["aa_r2_std"]),
            "eval_ss_arity": safe_float(entry["ss_arity"]),
            "eval_ro_arity": safe_float(entry["ro_arity"]),
            "metadata_max_splits": entry["max_splits"],
        })
    if calif_examples:
        datasets.append({"dataset": "california_housing_diagnosis", "examples": calif_examples})

    # Dataset 8: Synthetic recovery
    synth_examples = []
    for variant_name in SYNTHETIC_VARIANTS_STRUCTURED:
        if variant_name not in per_variant:
            continue
        vdata = per_variant[variant_name]
        for method in SYNTHETIC_METHODS:
            mdata = vdata["methods"].get(method, {})
            ari = mdata.get("adjusted_rand_index")
            if ari is None:
                continue
            synth_examples.append({
                "input": json.dumps({"variant": variant_name, "method": method}),
                "output": json.dumps({"ari": ari}),
                "eval_ari": safe_float(ari),
                "eval_mfari": safe_float(mdata.get("module_focused_ari")),
                "eval_jaccard": safe_float(mdata.get("synergistic_pair_jaccard")),
                "eval_xor_recovery": safe_float(mdata.get("xor_recovery_fraction")),
                "metadata_variant": variant_name,
                "metadata_method": method,
                "metadata_n_features": vdata["n_features"],
            })
    if synth_examples:
        datasets.append({"dataset": "synthetic_recovery", "examples": synth_examples})

    # Dataset 9: Success criteria
    success_examples = []
    for crit_key, crit_val in success_results.items():
        if crit_key == "overall_assessment":
            continue
        success_examples.append({
            "input": json.dumps({"criterion": crit_key}),
            "output": json.dumps({"passed": crit_val["passed"], "description": crit_val["description"]}),
            "eval_passed": 1.0 if crit_val["passed"] else 0.0,
            "metadata_criterion": crit_key,
        })
    if success_examples:
        datasets.append({"dataset": "success_criteria", "examples": success_examples})

    # ─── metadata ───
    metadata = {
        "evaluation_name": "comprehensive_statistical_analysis_iter2",
        "description": "Rigorous statistical analysis of all iteration 2 experiment outputs",
        "experiments_analyzed": [
            "exp_id2_real_benchmarks",
            "exp_id1_synthetic_recovery",
            "exp_id4_estimator_validation",
        ],
        "analysis_sections": [
            "A_friedman_nemenyi", "B_cohens_d", "C_bayesian_signed_rank",
            "D_frustration_correlation", "E_subsampling_impact", "F_interpretability",
            "G_california_diagnosis", "H_synthetic_recovery", "I_success_criteria",
        ],
        "detailed_results": {
            "friedman_nemenyi": friedman_results,
            "cohens_d": {"win_loss_tie": cohens_d_results["win_loss_tie"]},
            "bayesian_tests": bayesian_results,
            "frustration_correlation": frustration_results,
            "subsampling_impact": subsampling_results,
            "interpretability": {
                "wilcoxon_arity": interpretability_results["wilcoxon_arity"],
                "wilcoxon_path": interpretability_results["wilcoxon_path"],
                "pareto_at_ms20": interpretability_results["pareto_at_ms20"],
            },
            "california_diagnosis": california_results,
            "synthetic_summary": synthetic_results,
            "success_criteria": success_results,
        },
    }

    return {"metadata": metadata, "metrics_agg": metrics_agg, "datasets": datasets}


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
@logger.catch
def main():
    logger.info("=" * 60)
    logger.info("Starting comprehensive statistical evaluation")
    logger.info("=" * 60)

    # Load data
    logger.info("Loading experiment data...")
    try:
        exp2 = load_json(EXP_ID2_PATH)
    except FileNotFoundError:
        logger.exception(f"exp_id2 not found at {EXP_ID2_PATH}")
        raise
    try:
        exp1 = load_json(EXP_ID1_PATH)
    except FileNotFoundError:
        logger.exception(f"exp_id1 not found at {EXP_ID1_PATH}")
        raise
    try:
        exp4 = load_json(EXP_ID4_PATH)
    except FileNotFoundError:
        logger.exception(f"exp_id4 not found at {EXP_ID4_PATH}")
        raise

    meta2 = exp2["metadata"]
    meta1 = exp1["metadata"]
    meta4 = exp4["metadata"]

    # Build lookup table from aggregated results
    agg_results = meta2["aggregated_results"]
    lookup = build_perf_lookup(agg_results)
    best_max_splits = meta2["best_max_splits"]
    frustration_analysis = meta2["frustration_analysis"]
    per_variant = meta1["per_variant"]
    stability_data = meta4["subsampling_stability"]

    # Free raw per-example data to save memory
    if "datasets" in exp2:
        del exp2["datasets"]
    if "datasets" in exp1:
        del exp1["datasets"]
    if "datasets" in exp4:
        del exp4["datasets"]
    gc.collect()

    logger.info(
        f"Loaded: {len(agg_results)} aggregated results, "
        f"{len(per_variant)} synthetic variants, "
        f"{len(stability_data['results_by_subsample'])} subsampling points"
    )

    # ─── Compute all analyses ───
    logger.info("=" * 40)
    logger.info("A. Friedman/Nemenyi Tests")
    friedman_results = compute_friedman_nemenyi(lookup, best_max_splits)

    logger.info("=" * 40)
    logger.info("B. Cohen's d Effect Sizes")
    cohens_d_results = compute_cohens_d_analysis(lookup)

    logger.info("=" * 40)
    logger.info("C. Bayesian Signed-Rank Test")
    bayesian_results = compute_bayesian_test(lookup, best_max_splits)

    logger.info("=" * 40)
    logger.info("D. Frustration-Accuracy Correlation")
    frustration_results = compute_frustration_correlation(frustration_analysis)

    logger.info("=" * 40)
    logger.info("E. Subsampling Impact")
    subsampling_results = compute_subsampling_impact(stability_data)

    logger.info("=" * 40)
    logger.info("F. Interpretability Analysis")
    interpretability_results = compute_interpretability(lookup)

    logger.info("=" * 40)
    logger.info("G. California Housing Diagnosis")
    california_results = compute_california_diagnosis(lookup)

    logger.info("=" * 40)
    logger.info("H. Synthetic Recovery Summary")
    synthetic_results = compute_synthetic_summary(per_variant)

    logger.info("=" * 40)
    logger.info("I. Success Criteria Evaluation")
    total_wallclock = meta1.get("total_wallclock_sec", 130.0)
    success_results = compute_success_criteria(
        per_variant, lookup, best_max_splits,
        frustration_results, total_wallclock,
    )

    # ─── Build output ───
    logger.info("=" * 40)
    logger.info("Building eval_out.json")
    output = build_eval_output(
        friedman_results, cohens_d_results, bayesian_results,
        frustration_results, subsampling_results, interpretability_results,
        california_results, synthetic_results, success_results,
        per_variant,
    )

    # Save
    out_path = WORKSPACE / "eval_out.json"
    out_path.write_text(json.dumps(output, indent=2, default=str))
    size_mb = out_path.stat().st_size / 1e6
    logger.info(f"Saved eval_out.json ({size_mb:.2f} MB)")

    # Summary
    logger.info("=" * 60)
    logger.info("EVALUATION SUMMARY")
    logger.info(f"  metrics_agg keys: {len(output['metrics_agg'])}")
    logger.info(f"  datasets: {len(output['datasets'])}")
    for ds in output["datasets"]:
        logger.info(f"    {ds['dataset']}: {len(ds['examples'])} examples")
    logger.info(f"  Success criteria passed: {success_results['overall_assessment']}/4")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

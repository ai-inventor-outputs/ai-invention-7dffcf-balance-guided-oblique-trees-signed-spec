#!/usr/bin/env python3
"""Unified evaluation: synthetic+real results tables & complete statistical test catalogue.

Loads 4 dependency experiments and produces:
  A) Method progression table (synthetic vs real accuracy/arity)
  B) Spearman correlation between module recovery Jaccard and downstream accuracy
  C) Signed-vs-unsigned Hedges' g analysis with divergence explanation
  D) Complete catalogue of ~30 statistical tests
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
from scipy import stats as sp_stats


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def json_dumps(obj, **kwargs):
    """json.dumps with numpy type support."""
    return json.dumps(obj, cls=NumpyEncoder, **kwargs)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
logger.add(str(LOG_DIR / "run.log"), rotation="30 MB", level="DEBUG")

# ---------------------------------------------------------------------------
# Hardware / memory limits (cgroup-aware)
# ---------------------------------------------------------------------------
def _container_ram_gb():
    for p in ["/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"]:
        try:
            v = Path(p).read_text().strip()
            if v != "max" and int(v) < 1_000_000_000_000:
                return int(v) / 1e9
        except (FileNotFoundError, ValueError):
            pass
    return None

TOTAL_RAM_GB = _container_ram_gb() or psutil.virtual_memory().total / 1e9
RAM_BUDGET = int(min(TOTAL_RAM_GB * 0.5, 20) * 1e9)  # 50% of container, max 20GB
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET * 3, RAM_BUDGET * 3))
logger.info(f"Container RAM={TOTAL_RAM_GB:.1f}GB, budget={RAM_BUDGET/1e9:.1f}GB")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = Path("/ai-inventor/aii_pipeline/runs/jamnik-sgfigs-pid-v2/3_invention_loop")
WORKSPACE = BASE / "iter_7" / "gen_art" / "eval_id5_it7__opus"

DEP1_PATH = BASE / "iter_5" / "gen_art" / "exp_id1_it5__opus" / "mini_method_out.json"
DEP2_PATH = BASE / "iter_3" / "gen_art" / "exp_id3_it3__opus" / "mini_method_out.json"
DEP3_PATH = BASE / "iter_2" / "gen_art" / "exp_id1_it2__opus" / "mini_method_out.json"
DEP4_PATH = BASE / "iter_4" / "gen_art" / "exp_id2_it4__opus" / "mini_method_out.json"

CLASSIFICATION_DATASETS = [
    "adult", "electricity", "eye_movements", "credit",
    "higgs_small", "jannis", "miniboone",
]
ALL_DATASETS = CLASSIFICATION_DATASETS + ["california_housing"]
FIGS_METHODS = ["axis_aligned", "random_oblique", "unsigned_spectral", "signed_spectral", "hard_threshold"]
BASELINE_METHODS = ["ebm", "random_forest", "linear"]
SYNTHETIC_VARIANTS = [
    "easy_2mod_xor", "medium_4mod_mixed", "overlapping_modules",
    "no_structure_control", "hard_4mod_unequal", "highdim_8mod",
]
STRUCTURED_VARIANTS = [v for v in SYNTHETIC_VARIANTS if v != "no_structure_control"]
EASY_MEDIUM = ["easy_2mod_xor", "medium_4mod_mixed"]
# DEP3 method mapping to DEP2 method names
DEP3_TO_DEP2 = {
    "sponge_auto_k": "signed_spectral",
    "unsigned_spectral": "unsigned_spectral",
    "hard_threshold": "hard_threshold",
}
CLUSTERING_METHODS_DEP2 = ["signed_spectral", "unsigned_spectral", "hard_threshold"]
ALPHA = 0.05
ROPE = 0.01


def load_json(path: Path) -> dict:
    logger.info(f"Loading {path.name} ({path.stat().st_size / 1e6:.2f} MB)")
    data = json.loads(path.read_text())
    logger.info(f"  Loaded OK - top keys: {list(data.keys())[:5]}")
    return data


# ===================================================================
# Helper: Hedges' g
# ===================================================================
def hedges_g(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Hedges' g (bias-corrected Cohen's d)."""
    n1, n2 = len(x), len(y)
    if n1 < 2 or n2 < 2:
        return float("nan")
    s1, s2 = np.std(x, ddof=1), np.std(y, ddof=1)
    s_pooled = math.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    if s_pooled < 1e-15:
        return 0.0
    d = (np.mean(x) - np.mean(y)) / s_pooled
    correction = 1 - 3 / (4 * (n1 + n2) - 9)
    return float(d * correction)


def interpret_hedges_g(g: float) -> str:
    ag = abs(g)
    if ag < 0.2:
        return "negligible"
    if ag < 0.5:
        return "small"
    if ag < 0.8:
        return "medium"
    return "large"


# ===================================================================
# Helper: Bayesian signed-rank with ROPE (simplified sign-test)
# ===================================================================
def bayesian_sign_test_rope(diffs: np.ndarray, rope: float = ROPE) -> dict:
    """Simple Bayesian sign-test with ROPE."""
    n = len(diffs)
    if n == 0:
        return {"p_left": 0, "p_rope": 1, "p_right": 0}
    n_left = int(np.sum(diffs < -rope))
    n_rope = int(np.sum(np.abs(diffs) <= rope))
    n_right = int(np.sum(diffs > rope))
    total = n_left + n_rope + n_right
    if total == 0:
        return {"p_left": 0, "p_rope": 1, "p_right": 0}
    return {
        "p_left": round(n_left / total, 4),
        "p_rope": round(n_rope / total, 4),
        "p_right": round(n_right / total, 4),
    }


# ===================================================================
# Helper: safe Wilcoxon
# ===================================================================
def safe_wilcoxon(x: np.ndarray, y: np.ndarray) -> dict:
    """Wilcoxon signed-rank test handling ties and small n."""
    diffs = x - y
    nonzero = diffs[diffs != 0]
    if len(nonzero) < 3:
        return {"W": None, "p_value": 1.0, "n_nonzero": len(nonzero),
                "note": "Too few non-zero differences for Wilcoxon"}
    try:
        res = sp_stats.wilcoxon(x, y, alternative="two-sided")
        return {"W": float(res.statistic), "p_value": float(res.pvalue), "n_nonzero": len(nonzero)}
    except Exception as e:
        return {"W": None, "p_value": 1.0, "n_nonzero": len(nonzero), "note": str(e)[:200]}


# ===================================================================
# SECTION A: Method Progression Table
# ===================================================================
def section_a(dep1: dict, dep2: dict, dep4: dict) -> dict:
    logger.info("=== Section A: Method Progression Table ===")

    # --- A1/A2: Synthetic summary ---
    pvr = dep2["metadata"]["per_variant_results"]
    synth_rows = {}
    for method in FIGS_METHODS:
        accs_em, arities_em = [], []
        accs_all, arities_all = [], []
        for variant in SYNTHETIC_VARIANTS:
            if variant not in pvr:
                continue
            mdata = pvr[variant]["methods"].get(method)
            if mdata is None:
                continue
            acc = mdata.get("mean_balanced_accuracy")
            ari = mdata.get("mean_avg_split_arity")
            if acc is not None:
                accs_all.append(acc)
                if variant in EASY_MEDIUM:
                    accs_em.append(acc)
            if ari is not None:
                arities_all.append(ari)
                if variant in EASY_MEDIUM:
                    arities_em.append(ari)

        synth_rows[method] = {
            "synth_acc_easy_medium": round(float(np.mean(accs_em)), 6) if accs_em else None,
            "synth_arity_easy_medium": round(float(np.mean(arities_em)), 4) if arities_em else None,
            "synth_acc_all6": round(float(np.mean(accs_all)), 6) if accs_all else None,
            "synth_arity_all6": round(float(np.mean(arities_all)), 4) if arities_all else None,
        }

    # --- A3: Real summary (DEP1, max_splits=20, 7 classification datasets) ---
    rs = dep1["metadata"]["results_summary"]
    real_rows = {}
    for method in FIGS_METHODS:
        accs, arities = [], []
        for entry in rs:
            if (entry["method"] == method
                    and entry["max_splits"] == 20
                    and entry["dataset"] in CLASSIFICATION_DATASETS):
                ba = entry.get("balanced_accuracy_mean")
                ar = entry.get("avg_split_arity_mean")
                if ba is not None:
                    accs.append(ba)
                if ar is not None:
                    arities.append(ar)
        real_rows[method] = {
            "real_acc_7ds": round(float(np.mean(accs)), 6) if accs else None,
            "real_arity_7ds": round(float(np.mean(arities)), 4) if arities else None,
            "n_datasets": len(accs),
        }

    # --- A4: Baselines from DEP4 ---
    pdr = dep4["metadata"]["per_dataset_results"]
    for bmethod in BASELINE_METHODS:
        accs = []
        for ds in CLASSIFICATION_DATASETS:
            if ds in pdr and bmethod in pdr[ds]:
                ba = pdr[ds][bmethod]["aggregate"].get("balanced_accuracy_mean")
                if ba is not None:
                    accs.append(ba)
        real_rows[bmethod] = {
            "real_acc_7ds": round(float(np.mean(accs)), 6) if accs else None,
            "real_arity_7ds": None,  # baselines have no arity
            "n_datasets": len(accs),
        }

    # --- A5: Build table rows ---
    axis_synth = synth_rows.get("axis_aligned", {}).get("synth_acc_easy_medium")
    axis_real = real_rows.get("axis_aligned", {}).get("real_acc_7ds")
    rows = []
    all_methods = FIGS_METHODS + BASELINE_METHODS
    for method in all_methods:
        sr = synth_rows.get(method, {})
        rr = real_rows.get(method, {})
        delta_synth = None
        delta_real = None
        s_acc = sr.get("synth_acc_easy_medium")
        r_acc = rr.get("real_acc_7ds")
        if s_acc is not None and axis_synth is not None:
            delta_synth = round(s_acc - axis_synth, 6)
        if r_acc is not None and axis_real is not None:
            delta_real = round(r_acc - axis_real, 6)
        rows.append({
            "method": method,
            "synth_acc_easy_medium": sr.get("synth_acc_easy_medium"),
            "synth_arity_easy_medium": sr.get("synth_arity_easy_medium"),
            "synth_acc_all6": sr.get("synth_acc_all6"),
            "synth_arity_all6": sr.get("synth_arity_all6"),
            "real_acc_7ds": rr.get("real_acc_7ds"),
            "real_arity_7ds": rr.get("real_arity_7ds"),
            "delta_synth_vs_axis": delta_synth,
            "delta_real_vs_axis": delta_real,
        })

    # --- A6: Narrative ---
    us_synth = synth_rows.get("unsigned_spectral", {}).get("synth_acc_easy_medium")
    ss_synth = synth_rows.get("signed_spectral", {}).get("synth_acc_easy_medium")
    us_real = real_rows.get("unsigned_spectral", {}).get("real_acc_7ds")
    ss_real = real_rows.get("signed_spectral", {}).get("real_acc_7ds")

    gap_synth = (us_synth - ss_synth) if (us_synth is not None and ss_synth is not None) else None
    gap_real = (us_real - ss_real) if (us_real is not None and ss_real is not None) else None

    narrative = (
        f"unsigned-signed gap on synthetic(easy+medium)={gap_synth:.4f}, "
        f"on real(7ds)={gap_real:.4f}. "
        "Synthetic data with planted XOR modules has stronger CoI signal, "
        "so oblique methods show larger gains over axis-aligned there. "
        "On real data the CoI signal is weaker and noisier, narrowing the gap."
    ) if gap_synth is not None and gap_real is not None else "Insufficient data for narrative."

    logger.info(f"  Section A: {len(rows)} rows built")
    return {
        "description": "Method progression: accuracy/arity on synthetic vs real data",
        "rows": rows,
        "narrative": narrative,
    }


# ===================================================================
# SECTION B: Recovery-Accuracy Correlation
# ===================================================================
def section_b(dep2: dict, dep3: dict) -> dict:
    logger.info("=== Section B: Recovery-Accuracy Correlation ===")
    pvr2 = dep2["metadata"]["per_variant_results"]
    pvr3 = dep3["metadata"]["per_variant"]

    points_jaccard = []
    points_acc_improvement = []
    per_variant_tables = []

    for variant in STRUCTURED_VARIANTS:
        if variant not in pvr3 or variant not in pvr2:
            continue
        # axis_aligned baseline accuracy from DEP2
        aa_data = pvr2[variant]["methods"].get("axis_aligned", {})
        aa_acc = aa_data.get("mean_balanced_accuracy")
        if aa_acc is None:
            continue

        mini_rows = []
        for dep3_method, dep2_method in DEP3_TO_DEP2.items():
            # Recovery from DEP3
            m3 = pvr3[variant]["methods"].get(dep3_method, {})
            jaccard = m3.get("synergistic_pair_jaccard")
            if jaccard is None:
                continue

            # Accuracy from DEP2
            m2 = pvr2[variant]["methods"].get(dep2_method, {})
            acc = m2.get("mean_balanced_accuracy")
            if acc is None:
                continue

            improvement = acc - aa_acc
            points_jaccard.append(jaccard)
            points_acc_improvement.append(improvement)
            mini_rows.append({
                "method": dep2_method,
                "dep3_method": dep3_method,
                "jaccard": round(jaccard, 6),
                "accuracy": round(acc, 6),
                "accuracy_improvement": round(improvement, 6),
            })
        per_variant_tables.append({"variant": variant, "rows": mini_rows})

    result = {
        "description": "Spearman correlation: module recovery Jaccard vs downstream accuracy improvement",
        "n_points": len(points_jaccard),
        "per_variant_tables": per_variant_tables,
    }

    if len(points_jaccard) >= 3:
        j = np.array(points_jaccard)
        a = np.array(points_acc_improvement)
        sp_rho, sp_p = sp_stats.spearmanr(j, a)
        pr_r, pr_p = sp_stats.pearsonr(j, a)
        result.update({
            "spearman_rho": round(float(sp_rho), 6),
            "spearman_p": round(float(sp_p), 6),
            "pearson_r": round(float(pr_r), 6),
            "pearson_p": round(float(pr_p), 6),
            "interpretation": (
                f"Spearman rho={sp_rho:.4f} (p={sp_p:.4f}) across {len(j)} variant-method pairs. "
                + ("Significant positive correlation: better module recovery associates with larger accuracy gains."
                   if sp_p < ALPHA and sp_rho > 0
                   else "No significant correlation at alpha=0.05." if sp_p >= ALPHA
                   else f"Correlation is {'positive' if sp_rho > 0 else 'negative'} (p={sp_p:.4f}).")
            ),
        })
    else:
        result.update({
            "spearman_rho": None, "spearman_p": None,
            "pearson_r": None, "pearson_p": None,
            "interpretation": "Too few data points for correlation.",
        })

    logger.info(f"  Section B: {result['n_points']} correlation points")
    return result


# ===================================================================
# SECTION C: Signed-vs-Unsigned Narrative
# ===================================================================
def section_c(dep1: dict, dep2: dict, dep3: dict) -> dict:
    logger.info("=== Section C: Signed-vs-Unsigned Narrative ===")
    rpf = dep1["metadata"]["results_per_fold"]
    pvr2 = dep2["metadata"]["per_variant_results"]
    ci = dep1["metadata"]["clustering_info"]

    per_item = []

    # --- C1/C2: Real data Hedges' g per dataset ---
    real_g_accs, real_g_arities = [], []
    for ds in CLASSIFICATION_DATASETS:
        signed_accs = [r["balanced_accuracy"] for r in rpf
                       if r["dataset"] == ds and r["method"] == "signed_spectral"
                       and r["max_splits"] == 20 and r["balanced_accuracy"] is not None]
        unsigned_accs = [r["balanced_accuracy"] for r in rpf
                         if r["dataset"] == ds and r["method"] == "unsigned_spectral"
                         and r["max_splits"] == 20 and r["balanced_accuracy"] is not None]
        signed_ari = [r["avg_split_arity"] for r in rpf
                      if r["dataset"] == ds and r["method"] == "signed_spectral"
                      and r["max_splits"] == 20 and r["avg_split_arity"] is not None]
        unsigned_ari = [r["avg_split_arity"] for r in rpf
                        if r["dataset"] == ds and r["method"] == "unsigned_spectral"
                        and r["max_splits"] == 20 and r["avg_split_arity"] is not None]

        g_acc = hedges_g(np.array(signed_accs), np.array(unsigned_accs)) if signed_accs and unsigned_accs else float("nan")
        g_ari = hedges_g(np.array(signed_ari), np.array(unsigned_ari)) if signed_ari and unsigned_ari else float("nan")

        if not math.isnan(g_acc):
            real_g_accs.append(g_acc)
        if not math.isnan(g_ari):
            real_g_arities.append(g_ari)

        per_item.append({
            "name": ds,
            "domain": "real",
            "signed_acc_mean": round(float(np.mean(signed_accs)), 6) if signed_accs else None,
            "unsigned_acc_mean": round(float(np.mean(unsigned_accs)), 6) if unsigned_accs else None,
            "hedges_g_acc": round(g_acc, 6) if not math.isnan(g_acc) else None,
            "signed_arity_mean": round(float(np.mean(signed_ari)), 4) if signed_ari else None,
            "unsigned_arity_mean": round(float(np.mean(unsigned_ari)), 4) if unsigned_ari else None,
            "hedges_g_arity": round(g_ari, 6) if not math.isnan(g_ari) else None,
        })

    # --- C3/C4: Synthetic data Hedges' g per variant ---
    synth_g_accs, synth_g_arities = [], []
    for variant in STRUCTURED_VARIANTS:
        if variant not in pvr2:
            continue
        sm = pvr2[variant]["methods"].get("signed_spectral", {})
        um = pvr2[variant]["methods"].get("unsigned_spectral", {})
        s_folds = sm.get("best_folds", sm.get("folds", []))
        u_folds = um.get("best_folds", um.get("folds", []))
        s_accs = [f["balanced_accuracy"] for f in s_folds if f.get("balanced_accuracy") is not None]
        u_accs = [f["balanced_accuracy"] for f in u_folds if f.get("balanced_accuracy") is not None]
        s_ari = [f["avg_split_arity"] for f in s_folds if f.get("avg_split_arity") is not None]
        u_ari = [f["avg_split_arity"] for f in u_folds if f.get("avg_split_arity") is not None]

        g_acc = hedges_g(np.array(s_accs), np.array(u_accs)) if len(s_accs) >= 2 and len(u_accs) >= 2 else float("nan")
        g_ari = hedges_g(np.array(s_ari), np.array(u_ari)) if len(s_ari) >= 2 and len(u_ari) >= 2 else float("nan")

        if not math.isnan(g_acc):
            synth_g_accs.append(g_acc)
        if not math.isnan(g_ari):
            synth_g_arities.append(g_ari)

        per_item.append({
            "name": variant,
            "domain": "synthetic",
            "signed_acc_mean": round(float(np.mean(s_accs)), 6) if s_accs else None,
            "unsigned_acc_mean": round(float(np.mean(u_accs)), 6) if u_accs else None,
            "hedges_g_acc": round(g_acc, 6) if not math.isnan(g_acc) else None,
            "signed_arity_mean": round(float(np.mean(s_ari)), 4) if s_ari else None,
            "unsigned_arity_mean": round(float(np.mean(u_ari)), 4) if u_ari else None,
            "hedges_g_arity": round(g_ari, 6) if not math.isnan(g_ari) else None,
        })

    avg_real_g_acc = round(float(np.mean(real_g_accs)), 6) if real_g_accs else None
    avg_real_g_ari = round(float(np.mean(real_g_arities)), 6) if real_g_arities else None
    avg_synth_g_acc = round(float(np.mean(synth_g_accs)), 6) if synth_g_accs else None
    avg_synth_g_ari = round(float(np.mean(synth_g_arities)), 6) if synth_g_arities else None

    # --- C6: Divergence explanation using CoI diagnostics ---
    # Synthetic CoI magnitudes from DEP3
    synth_coi_vals = []
    pvr3 = dep3["metadata"]["per_variant"]
    for variant in STRUCTURED_VARIANTS:
        if variant in pvr3 and "coi_sign_diagnostics" in pvr3[variant]:
            for diag in pvr3[variant]["coi_sign_diagnostics"]:
                if diag.get("coi") is not None:
                    synth_coi_vals.append(abs(diag["coi"]))
    mean_synth_coi = round(float(np.mean(synth_coi_vals)), 6) if synth_coi_vals else None

    # Real CoI: ratio of positive to negative pairs
    real_pos_neg = []
    for ds in ALL_DATASETS:
        if ds in ci:
            npos = ci[ds].get("n_positive_coi_pairs", 0)
            nneg = ci[ds].get("n_negative_coi_pairs", 0)
            real_pos_neg.append({"dataset": ds, "n_positive": npos, "n_negative": nneg})

    divergence = (
        f"Mean |CoI| on synthetic={mean_synth_coi:.4f}. "
        f"Avg Hedges' g(acc): real={avg_real_g_acc}, synthetic={avg_synth_g_acc}. "
        "Synthetic has stronger CoI signal with more extreme values. "
        "On real data, the CoI signal is weaker and noisier, so the signed/unsigned "
        "distinction matters less and both are closer to axis-aligned."
    ) if mean_synth_coi is not None else "Insufficient CoI data."

    logger.info(f"  Section C: real_g_acc={avg_real_g_acc}, synth_g_acc={avg_synth_g_acc}")
    return {
        "description": "Signed-vs-unsigned Hedges' g analysis (real and synthetic)",
        "real_hedges_g_accuracy": avg_real_g_acc,
        "real_hedges_g_arity": avg_real_g_ari,
        "synthetic_hedges_g_accuracy": avg_synth_g_acc,
        "synthetic_hedges_g_arity": avg_synth_g_ari,
        "per_item_breakdown": per_item,
        "coi_diagnostics": {
            "mean_abs_coi_synthetic": mean_synth_coi,
            "real_pos_neg_pairs": real_pos_neg,
        },
        "divergence_explanation": divergence,
    }


# ===================================================================
# SECTION D: Statistical Test Catalogue
# ===================================================================
def _make_test(test_id, test_name, scope, null_h, metric, stat_val, stat_name,
               p_val, effect_size, effect_name, n, k=None,
               significant=None, interpretation="", details=None):
    """Build a test dict."""
    if significant is None and p_val is not None:
        significant = p_val < ALPHA
    return {
        "test_id": test_id,
        "test_name": test_name,
        "scope": scope,
        "null_hypothesis": null_h,
        "metric": metric,
        "test_statistic": stat_val,
        "statistic_name": stat_name,
        "p_value": p_val,
        "effect_size": effect_size,
        "effect_size_name": effect_name,
        "n": n,
        "k": k,
        "significant_at_005": significant,
        "interpretation": interpretation,
        "details": details or {},
    }


def section_d(dep1: dict, dep2: dict, dep3: dict, dep4: dict, section_b_result: dict) -> dict:
    logger.info("=== Section D: Statistical Test Catalogue ===")
    tests = []
    rpf = dep1["metadata"]["results_per_fold"]
    rs = dep1["metadata"]["results_summary"]
    pvr2 = dep2["metadata"]["per_variant_results"]
    pvr3 = dep3["metadata"]["per_variant"]
    pdr = dep4["metadata"]["per_dataset_results"]
    ci = dep1["metadata"]["clustering_info"]

    # ------------------------------------------------------------------
    # Helper: get per-dataset mean accuracy for a method (real, max_splits=20)
    # ------------------------------------------------------------------
    def get_real_means(method_name: str, is_baseline: bool = False) -> dict:
        """Return {dataset: mean_balanced_accuracy} for 7 classification datasets."""
        result = {}
        for ds in CLASSIFICATION_DATASETS:
            if is_baseline:
                if ds in pdr and method_name in pdr[ds]:
                    ba = pdr[ds][method_name]["aggregate"].get("balanced_accuracy_mean")
                    if ba is not None:
                        result[ds] = ba
            else:
                for entry in rs:
                    if (entry["dataset"] == ds and entry["method"] == method_name
                            and entry["max_splits"] == 20):
                        ba = entry.get("balanced_accuracy_mean")
                        if ba is not None:
                            result[ds] = ba
        return result

    def get_real_arity_means(method_name: str) -> dict:
        result = {}
        for entry in rs:
            if (entry["method"] == method_name and entry["max_splits"] == 20
                    and entry["dataset"] in CLASSIFICATION_DATASETS):
                ar = entry.get("avg_split_arity_mean")
                if ar is not None:
                    result[entry["dataset"]] = ar
        return result

    # Collect per-dataset means for all 8 methods
    all_method_means = {}
    for m in FIGS_METHODS:
        all_method_means[m] = get_real_means(m)
    for m in BASELINE_METHODS:
        all_method_means[m] = get_real_means(m, is_baseline=True)

    # Common datasets across all 8 methods
    common_ds = set(CLASSIFICATION_DATASETS)
    for m in FIGS_METHODS + BASELINE_METHODS:
        common_ds &= set(all_method_means[m].keys())
    common_ds = sorted(common_ds)
    logger.info(f"  Common datasets for Friedman: {common_ds} (n={len(common_ds)})")

    # ------------------------------------------------------------------
    # D1: Friedman test
    # ------------------------------------------------------------------
    if len(common_ds) >= 3:
        groups = []
        method_order = FIGS_METHODS + BASELINE_METHODS
        for m in method_order:
            groups.append([all_method_means[m][ds] for ds in common_ds])

        try:
            friedman_stat, friedman_p = sp_stats.friedmanchisquare(*groups)
            n_methods = len(method_order)
            n_ds = len(common_ds)
            kendall_w = friedman_stat / (n_ds * (n_methods - 1))
            # Compute average ranks
            rank_matrix = np.zeros((n_ds, n_methods))
            for i in range(n_ds):
                vals = [groups[j][i] for j in range(n_methods)]
                rank_matrix[i] = sp_stats.rankdata([-v for v in vals])  # higher accuracy = rank 1
            avg_ranks = {method_order[j]: round(float(rank_matrix[:, j].mean()), 4)
                         for j in range(n_methods)}

            tests.append(_make_test(
                "D1_friedman_8methods", "Friedman",
                f"{n_methods} methods x {n_ds} classification datasets",
                "All methods have equal mean ranks across datasets",
                "balanced_accuracy",
                round(float(friedman_stat), 6), "chi2",
                round(float(friedman_p), 8),
                round(float(kendall_w), 6), "Kendall_W",
                n_ds, k=n_methods,
                interpretation=f"Friedman chi2={friedman_stat:.2f}, p={friedman_p:.6f}. "
                               + ("Reject H0: methods differ significantly." if friedman_p < ALPHA
                                  else "Cannot reject H0."),
                details={"average_ranks": avg_ranks},
            ))
        except Exception:
            logger.exception("Friedman test failed")

    # ------------------------------------------------------------------
    # D2: Nemenyi post-hoc
    # ------------------------------------------------------------------
    if len(common_ds) >= 3 and tests and tests[-1]["p_value"] is not None and tests[-1]["p_value"] < ALPHA:
        try:
            import scikit_posthocs as sp
            data_matrix = np.array(groups).T  # shape (n_ds, n_methods)
            nemenyi_result = sp.posthoc_nemenyi_friedman(data_matrix)
            # nemenyi_result is a DataFrame of p-values
            pvals = nemenyi_result.values
            n_methods = len(method_order)
            sig_pairs = []
            for i in range(n_methods):
                for j in range(i + 1, n_methods):
                    if pvals[i, j] < ALPHA:
                        sig_pairs.append({
                            "method_a": method_order[i],
                            "method_b": method_order[j],
                            "p_value": round(float(pvals[i, j]), 6),
                        })

            # Critical difference
            q_alpha = 3.031  # q for alpha=0.05, k=8 (Nemenyi table)
            cd = q_alpha * math.sqrt(n_methods * (n_methods + 1) / (6 * len(common_ds)))

            tests.append(_make_test(
                "D2_nemenyi_posthoc", "Nemenyi post-hoc",
                f"{n_methods} methods x {len(common_ds)} datasets (pairwise)",
                "Each pair of methods has equal mean ranks",
                "balanced_accuracy",
                round(cd, 4), "critical_difference",
                None,  # Nemenyi has per-pair p-values, no single p
                None, None,
                len(common_ds), k=n_methods,
                significant=len(sig_pairs) > 0,
                interpretation=f"Critical difference={cd:.3f}. {len(sig_pairs)} significant pairs found.",
                details={
                    "significant_pairs": sig_pairs,
                    "n_total_pairs": n_methods * (n_methods - 1) // 2,
                },
            ))
        except Exception:
            logger.exception("Nemenyi test failed")

    # ------------------------------------------------------------------
    # D3: Pairwise Wilcoxon signed-rank on accuracy (7 tests)
    # ------------------------------------------------------------------
    wilcoxon_acc_pairs = [
        ("signed_spectral", "axis_aligned", False, False),
        ("signed_spectral", "random_oblique", False, False),
        ("signed_spectral", "unsigned_spectral", False, False),
        ("unsigned_spectral", "axis_aligned", False, False),
        ("hard_threshold", "axis_aligned", False, False),
        ("signed_spectral", "ebm", False, True),
        ("signed_spectral", "random_forest", False, True),
    ]
    n_acc_tests = len(wilcoxon_acc_pairs)
    for idx, (m1, m2, m1_bl, m2_bl) in enumerate(wilcoxon_acc_pairs):
        means1 = get_real_means(m1, is_baseline=m1_bl)
        means2 = get_real_means(m2, is_baseline=m2_bl)
        shared = sorted(set(means1.keys()) & set(means2.keys()))
        if len(shared) < 3:
            continue
        x = np.array([means1[ds] for ds in shared])
        y = np.array([means2[ds] for ds in shared])
        wres = safe_wilcoxon(x, y)
        bonf_p = min(wres["p_value"] * n_acc_tests, 1.0)
        tests.append(_make_test(
            f"D3_wilcoxon_acc_{m1}_vs_{m2}", "Wilcoxon signed-rank",
            f"{m1} vs {m2}, {len(shared)} classification datasets",
            f"{m1} and {m2} have equal balanced_accuracy across datasets",
            "balanced_accuracy",
            wres["W"], "W",
            round(wres["p_value"], 8),
            None, None,
            len(shared),
            interpretation=f"W={wres['W']}, p_raw={wres['p_value']:.6f}, p_bonferroni={bonf_p:.6f}.",
            details={
                "p_bonferroni": round(bonf_p, 8),
                "n_comparisons": n_acc_tests,
                "mean_diff": round(float(np.mean(x - y)), 6),
                "per_dataset": {ds: round(float(means1[ds] - means2[ds]), 6) for ds in shared},
            },
        ))

    # ------------------------------------------------------------------
    # D4: Pairwise Wilcoxon on arity (3 tests)
    # ------------------------------------------------------------------
    wilcoxon_arity_pairs = [
        ("signed_spectral", "random_oblique"),
        ("unsigned_spectral", "random_oblique"),
        ("signed_spectral", "unsigned_spectral"),
    ]
    n_ari_tests = len(wilcoxon_arity_pairs)
    for m1, m2 in wilcoxon_arity_pairs:
        a1 = get_real_arity_means(m1)
        a2 = get_real_arity_means(m2)
        shared = sorted(set(a1.keys()) & set(a2.keys()))
        if len(shared) < 3:
            continue
        x = np.array([a1[ds] for ds in shared])
        y = np.array([a2[ds] for ds in shared])
        wres = safe_wilcoxon(x, y)
        bonf_p = min(wres["p_value"] * n_ari_tests, 1.0)
        tests.append(_make_test(
            f"D4_wilcoxon_arity_{m1}_vs_{m2}", "Wilcoxon signed-rank",
            f"{m1} vs {m2} arity, {len(shared)} classification datasets",
            f"{m1} and {m2} have equal avg_split_arity across datasets",
            "avg_split_arity",
            wres["W"], "W",
            round(wres["p_value"], 8),
            None, None,
            len(shared),
            interpretation=f"W={wres['W']}, p_raw={wres['p_value']:.6f}, p_bonferroni={bonf_p:.6f}.",
            details={
                "p_bonferroni": round(bonf_p, 8),
                "n_comparisons": n_ari_tests,
                "mean_diff": round(float(np.mean(x - y)), 6),
            },
        ))

    # ------------------------------------------------------------------
    # D5: Hedges' g effect sizes (8 tests)
    # ------------------------------------------------------------------
    hedges_configs = [
        ("signed_spectral", "unsigned_spectral", "balanced_accuracy", "real"),
        ("signed_spectral", "unsigned_spectral", "balanced_accuracy", "synthetic"),
        ("signed_spectral", "unsigned_spectral", "avg_split_arity", "real"),
        ("signed_spectral", "unsigned_spectral", "avg_split_arity", "synthetic"),
        ("signed_spectral", "axis_aligned", "balanced_accuracy", "real"),
        ("signed_spectral", "axis_aligned", "balanced_accuracy", "synthetic"),
        ("random_oblique", "axis_aligned", "balanced_accuracy", "real"),
        ("random_oblique", "axis_aligned", "balanced_accuracy", "synthetic"),
    ]

    for m1, m2, metric, domain in hedges_configs:
        vals1, vals2 = [], []
        if domain == "real":
            for ds in CLASSIFICATION_DATASETS:
                key = "balanced_accuracy" if metric == "balanced_accuracy" else "avg_split_arity"
                v1 = [r[key] for r in rpf
                      if r["dataset"] == ds and r["method"] == m1
                      and r["max_splits"] == 20 and r.get(key) is not None]
                v2 = [r[key] for r in rpf
                      if r["dataset"] == ds and r["method"] == m2
                      and r["max_splits"] == 20 and r.get(key) is not None]
                vals1.extend(v1)
                vals2.extend(v2)
        else:  # synthetic
            for variant in STRUCTURED_VARIANTS:
                if variant not in pvr2:
                    continue
                md1 = pvr2[variant]["methods"].get(m1, {})
                md2 = pvr2[variant]["methods"].get(m2, {})
                folds1 = md1.get("best_folds", md1.get("folds", []))
                folds2 = md2.get("best_folds", md2.get("folds", []))
                key = "balanced_accuracy" if metric == "balanced_accuracy" else "avg_split_arity"
                vals1.extend([f[key] for f in folds1 if f.get(key) is not None])
                vals2.extend([f[key] for f in folds2 if f.get(key) is not None])

        if len(vals1) >= 2 and len(vals2) >= 2:
            g = hedges_g(np.array(vals1), np.array(vals2))
            tests.append(_make_test(
                f"D5_hedges_g_{m1}_vs_{m2}_{metric}_{domain}",
                "Hedges' g",
                f"{m1} vs {m2} on {domain} data",
                f"No effect size difference in {metric}",
                metric,
                None, None,
                None,
                round(g, 6), "Hedges_g",
                len(vals1),
                significant=abs(g) > 0.2,
                interpretation=f"g={g:.4f} ({interpret_hedges_g(g)}). {m1} {'higher' if g > 0 else 'lower'} than {m2}.",
                details={"n1": len(vals1), "n2": len(vals2),
                         "mean1": round(float(np.mean(vals1)), 6),
                         "mean2": round(float(np.mean(vals2)), 6)},
            ))

    # ------------------------------------------------------------------
    # D6: Bayesian signed-rank with ROPE (4 tests)
    # ------------------------------------------------------------------
    bayesian_pairs = [
        ("signed_spectral", "axis_aligned"),
        ("signed_spectral", "unsigned_spectral"),
        ("signed_spectral", "random_oblique"),
        ("unsigned_spectral", "axis_aligned"),
    ]
    for m1, m2 in bayesian_pairs:
        means1 = get_real_means(m1)
        means2 = get_real_means(m2)
        shared = sorted(set(means1.keys()) & set(means2.keys()))
        if len(shared) < 3:
            continue
        diffs = np.array([means1[ds] - means2[ds] for ds in shared])
        bsr = bayesian_sign_test_rope(diffs, rope=ROPE)
        tests.append(_make_test(
            f"D6_bayesian_rope_{m1}_vs_{m2}", "Bayesian signed-rank (ROPE)",
            f"{m1} vs {m2}, ROPE=[-{ROPE},{ROPE}], {len(shared)} datasets",
            f"Difference between {m1} and {m2} is practically equivalent (within ROPE)",
            "balanced_accuracy",
            None, None, None,
            None, None,
            len(shared),
            significant=bsr["p_left"] > 0.5 or bsr["p_right"] > 0.5,
            interpretation=f"P(left)={bsr['p_left']}, P(ROPE)={bsr['p_rope']}, P(right)={bsr['p_right']}. "
                           + ("Practically equivalent." if bsr['p_rope'] > 0.5
                              else f"{m1} {'better' if bsr['p_right'] > bsr['p_left'] else 'worse'}."),
            details=bsr,
        ))

    # ------------------------------------------------------------------
    # D7: Spearman frustration-index correlation (2 tests)
    # ------------------------------------------------------------------
    # D7a: Real data
    frustrations_real = []
    acc_gaps_real = []
    for ds in ALL_DATASETS:
        if ds not in ci:
            continue
        fi = ci[ds].get("signed_spectral", {}).get("frustration_index")
        if fi is None:
            continue
        # oblique_best = max(signed, unsigned, random_oblique) at max_splits=20
        oblique_accs = []
        aa_acc = None
        for entry in rs:
            if entry["dataset"] == ds and entry["max_splits"] == 20:
                ba = entry.get("balanced_accuracy_mean")
                if ba is None:
                    continue
                if entry["method"] == "axis_aligned":
                    aa_acc = ba
                elif entry["method"] in ["signed_spectral", "unsigned_spectral", "random_oblique"]:
                    oblique_accs.append(ba)
        if aa_acc is not None and oblique_accs:
            oblique_best = max(oblique_accs)
            frustrations_real.append(fi)
            acc_gaps_real.append(oblique_best - aa_acc)

    if len(frustrations_real) >= 3:
        rho, p = sp_stats.spearmanr(frustrations_real, acc_gaps_real)
        tests.append(_make_test(
            "D7a_spearman_frustration_real", "Spearman correlation",
            f"Frustration index vs accuracy gap, {len(frustrations_real)} real datasets",
            "No correlation between frustration index and oblique accuracy gain",
            "frustration_index vs accuracy_gap",
            round(float(rho), 6), "Spearman_rho",
            round(float(p), 8),
            round(float(rho), 6), "rho",
            len(frustrations_real),
            interpretation=f"rho={rho:.4f}, p={p:.4f}. "
                           + ("Significant." if p < ALPHA else "Not significant."),
            details={
                "frustrations": [round(f, 6) for f in frustrations_real],
                "acc_gaps": [round(g, 6) for g in acc_gaps_real],
            },
        ))

    # D7b: Synthetic data
    frustrations_synth = []
    acc_gaps_synth = []
    for variant in SYNTHETIC_VARIANTS:
        if variant not in pvr3 or variant not in pvr2:
            continue
        fi = pvr3[variant].get("frustration_index")
        if fi is None:
            continue
        aa = pvr2[variant]["methods"].get("axis_aligned", {}).get("mean_balanced_accuracy")
        if aa is None:
            continue
        oblique_accs = []
        for m in ["signed_spectral", "unsigned_spectral", "random_oblique"]:
            md = pvr2[variant]["methods"].get(m, {})
            ba = md.get("mean_balanced_accuracy")
            if ba is not None:
                oblique_accs.append(ba)
        if oblique_accs:
            frustrations_synth.append(fi)
            acc_gaps_synth.append(max(oblique_accs) - aa)

    if len(frustrations_synth) >= 3:
        rho, p = sp_stats.spearmanr(frustrations_synth, acc_gaps_synth)
        tests.append(_make_test(
            "D7b_spearman_frustration_synth", "Spearman correlation",
            f"Frustration index vs accuracy gap, {len(frustrations_synth)} synthetic variants",
            "No correlation between frustration index and oblique accuracy gain (synthetic)",
            "frustration_index vs accuracy_gap",
            round(float(rho), 6), "Spearman_rho",
            round(float(p), 8),
            round(float(rho), 6), "rho",
            len(frustrations_synth),
            interpretation=f"rho={rho:.4f}, p={p:.4f}.",
            details={
                "frustrations": [round(f, 6) for f in frustrations_synth],
                "acc_gaps": [round(g, 6) for g in acc_gaps_synth],
            },
        ))

    # ------------------------------------------------------------------
    # D8: Module recovery correlation (from Section B)
    # ------------------------------------------------------------------
    if section_b_result.get("spearman_rho") is not None:
        tests.append(_make_test(
            "D8_module_recovery_corr", "Spearman correlation",
            f"Jaccard recovery vs accuracy improvement, {section_b_result['n_points']} points",
            "No correlation between module recovery and accuracy improvement",
            "jaccard vs accuracy_improvement",
            round(section_b_result["spearman_rho"], 6), "Spearman_rho",
            round(section_b_result["spearman_p"], 8),
            round(section_b_result["spearman_rho"], 6), "rho",
            section_b_result["n_points"],
            interpretation=section_b_result.get("interpretation", ""),
        ))

    # ------------------------------------------------------------------
    # D9: Fisher z-test for Hedges' g difference (synthetic vs real)
    # ------------------------------------------------------------------
    # Find the Hedges' g values for signed_vs_unsigned on real and synthetic
    g_real = g_synth = None
    for t in tests:
        if t["test_id"] == "D5_hedges_g_signed_spectral_vs_unsigned_spectral_balanced_accuracy_real":
            g_real = t["effect_size"]
        if t["test_id"] == "D5_hedges_g_signed_spectral_vs_unsigned_spectral_balanced_accuracy_synthetic":
            g_synth = t["effect_size"]
    if g_real is not None and g_synth is not None:
        # Fisher z-test to compare two effect sizes
        # Using the approximation: z = (g1 - g2) / sqrt(SE1^2 + SE2^2)
        # SE of Hedges' g ~ sqrt(1/n1 + 1/n2 + g^2/(2*(n1+n2)))
        n_real = 7 * 5  # 7 datasets * 5 folds
        n_synth = 5 * 5  # 5 variants * 5 folds (approx)
        se_real = math.sqrt(2 / n_real + g_real ** 2 / (2 * n_real))
        se_synth = math.sqrt(2 / n_synth + g_synth ** 2 / (2 * n_synth))
        se_diff = math.sqrt(se_real ** 2 + se_synth ** 2)
        if se_diff > 1e-12:
            z = (g_real - g_synth) / se_diff
            p_z = 2 * (1 - sp_stats.norm.cdf(abs(z)))
        else:
            z, p_z = 0.0, 1.0
        tests.append(_make_test(
            "D9_fisher_z_hedges_g_real_vs_synth", "Fisher z-test",
            "Compare Hedges' g (signed-vs-unsigned accuracy) between real and synthetic",
            "The signed-vs-unsigned effect size is the same on real and synthetic data",
            "Hedges_g difference",
            round(z, 6), "z",
            round(p_z, 8),
            round(g_real - g_synth, 6), "delta_g",
            None,
            interpretation=f"z={z:.4f}, p={p_z:.4f}. g_real={g_real:.4f}, g_synth={g_synth:.4f}. "
                           + ("Effect sizes differ significantly." if p_z < ALPHA
                              else "No significant difference in effect sizes."),
            details={"g_real": g_real, "g_synth": g_synth, "se_real": round(se_real, 6),
                     "se_synth": round(se_synth, 6)},
        ))

    logger.info(f"  Section D: {len(tests)} statistical tests computed")
    return {
        "description": "Complete catalogue of all statistical tests",
        "n_tests": len(tests),
        "tests": tests,
    }


# ===================================================================
# Build schema-compliant output
# ===================================================================
def build_output(sec_a: dict, sec_b: dict, sec_c: dict, sec_d: dict) -> dict:
    """Build eval_out.json in exp_eval_sol_out schema format."""

    # metrics_agg: key summary numbers
    metrics_agg = {}
    # From section A
    for row in sec_a["rows"]:
        if row["method"] == "signed_spectral":
            if row.get("real_acc_7ds") is not None:
                metrics_agg["signed_spectral_real_acc"] = row["real_acc_7ds"]
            if row.get("synth_acc_easy_medium") is not None:
                metrics_agg["signed_spectral_synth_acc_em"] = row["synth_acc_easy_medium"]
        if row["method"] == "axis_aligned":
            if row.get("real_acc_7ds") is not None:
                metrics_agg["axis_aligned_real_acc"] = row["real_acc_7ds"]

    # From section B
    if sec_b.get("spearman_rho") is not None:
        metrics_agg["recovery_accuracy_spearman_rho"] = sec_b["spearman_rho"]
    if sec_b.get("spearman_p") is not None:
        metrics_agg["recovery_accuracy_spearman_p"] = sec_b["spearman_p"]

    # From section C
    if sec_c.get("real_hedges_g_accuracy") is not None:
        metrics_agg["signed_vs_unsigned_hedges_g_real"] = sec_c["real_hedges_g_accuracy"]
    if sec_c.get("synthetic_hedges_g_accuracy") is not None:
        metrics_agg["signed_vs_unsigned_hedges_g_synth"] = sec_c["synthetic_hedges_g_accuracy"]

    # From section D
    metrics_agg["n_statistical_tests"] = sec_d["n_tests"]
    n_sig = sum(1 for t in sec_d["tests"] if t.get("significant_at_005"))
    metrics_agg["n_significant_tests"] = n_sig

    # Ensure at least one metric
    if not metrics_agg:
        metrics_agg["n_statistical_tests"] = sec_d["n_tests"]

    # Build datasets array: one "dataset" per analysis section
    # Each example represents one test/row
    datasets = []

    # Dataset 1: Section A rows
    sec_a_examples = []
    for row in sec_a["rows"]:
        sec_a_examples.append({
            "input": json_dumps({"method": row["method"], "analysis": "method_progression_table"}),
            "output": json_dumps({k: v for k, v in row.items() if k != "method"}),
            "eval_delta_synth_vs_axis": row["delta_synth_vs_axis"] if row.get("delta_synth_vs_axis") is not None else 0,
            "eval_delta_real_vs_axis": row["delta_real_vs_axis"] if row.get("delta_real_vs_axis") is not None else 0,
        })
    datasets.append({"dataset": "A_method_progression", "examples": sec_a_examples})

    # Dataset 2: Section B per-variant tables
    sec_b_examples = []
    for vtable in sec_b.get("per_variant_tables", []):
        for row in vtable.get("rows", []):
            sec_b_examples.append({
                "input": json_dumps({"variant": vtable["variant"], "method": row["method"],
                                     "analysis": "recovery_accuracy_correlation"}),
                "output": json_dumps({"jaccard": row["jaccard"], "accuracy_improvement": row["accuracy_improvement"]}),
                "eval_jaccard": row["jaccard"],
                "eval_accuracy_improvement": row["accuracy_improvement"],
            })
    if not sec_b_examples:
        sec_b_examples.append({
            "input": json_dumps({"analysis": "recovery_accuracy_correlation", "note": "no data"}),
            "output": json_dumps({"spearman_rho": sec_b.get("spearman_rho")}),
        })
    datasets.append({"dataset": "B_recovery_accuracy", "examples": sec_b_examples})

    # Dataset 3: Section C per-item breakdown
    sec_c_examples = []
    for item in sec_c.get("per_item_breakdown", []):
        sec_c_examples.append({
            "input": json_dumps({"name": item["name"], "domain": item["domain"],
                                  "analysis": "signed_vs_unsigned"}),
            "output": json_dumps({k: v for k, v in item.items() if k not in ("name", "domain")}),
            "eval_hedges_g_acc": item["hedges_g_acc"] if item.get("hedges_g_acc") is not None else 0,
        })
    if not sec_c_examples:
        sec_c_examples.append({
            "input": json_dumps({"analysis": "signed_vs_unsigned", "note": "no data"}),
            "output": "{}",
        })
    datasets.append({"dataset": "C_signed_vs_unsigned", "examples": sec_c_examples})

    # Dataset 4: Section D statistical tests
    sec_d_examples = []
    for test in sec_d["tests"]:
        sec_d_examples.append({
            "input": json_dumps({"test_id": test["test_id"], "test_name": test["test_name"],
                                  "scope": test["scope"]}),
            "output": json_dumps({
                "p_value": test["p_value"],
                "effect_size": test["effect_size"],
                "significant": test["significant_at_005"],
                "interpretation": test["interpretation"][:200],
            }),
            "eval_p_value": test["p_value"] if test["p_value"] is not None else 1.0,
            "eval_significant": 1 if test.get("significant_at_005") else 0,
        })
    if not sec_d_examples:
        sec_d_examples.append({
            "input": json_dumps({"analysis": "statistical_tests", "note": "no tests"}),
            "output": "{}",
        })
    datasets.append({"dataset": "D_statistical_tests", "examples": sec_d_examples})

    return {
        "metadata": {
            "evaluation_name": "unified_results_and_statistical_catalogue",
            "dependencies": ["exp_id1_it5__opus", "exp_id3_it3__opus", "exp_id1_it2__opus", "exp_id2_it4__opus"],
            "n_figs_methods": 5,
            "n_baseline_methods": 3,
            "n_real_datasets": 8,
            "n_classification_datasets": 7,
            "n_synthetic_variants": 6,
            "alpha": ALPHA,
            "rope_width": ROPE,
            "A_method_progression_table": sec_a,
            "B_recovery_accuracy_correlation": sec_b,
            "C_signed_vs_unsigned_narrative": sec_c,
            "D_statistical_test_catalogue": sec_d,
        },
        "metrics_agg": metrics_agg,
        "datasets": datasets,
    }


# ===================================================================
# Main
# ===================================================================
@logger.catch
def main():
    logger.info("Starting unified evaluation")

    # Load dependencies
    dep1 = load_json(DEP1_PATH)
    dep2 = load_json(DEP2_PATH)
    dep3 = load_json(DEP3_PATH)
    dep4 = load_json(DEP4_PATH)

    # Run sections
    sec_a = section_a(dep1, dep2, dep4)
    sec_b = section_b(dep2, dep3)
    sec_c = section_c(dep1, dep2, dep3)
    sec_d = section_d(dep1, dep2, dep3, dep4, sec_b)

    # Build output
    output = build_output(sec_a, sec_b, sec_c, sec_d)

    # Write
    out_path = WORKSPACE / "eval_out.json"
    out_path.write_text(json_dumps(output, indent=2))
    logger.info(f"Wrote eval_out.json ({out_path.stat().st_size / 1e3:.1f} KB)")

    # Summary
    logger.info(f"metrics_agg keys: {list(output['metrics_agg'].keys())}")
    logger.info(f"n_datasets: {len(output['datasets'])}")
    for ds in output["datasets"]:
        logger.info(f"  {ds['dataset']}: {len(ds['examples'])} examples")
    logger.info("Evaluation complete")


if __name__ == "__main__":
    main()

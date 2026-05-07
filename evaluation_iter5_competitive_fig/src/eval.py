#!/usr/bin/env python3
"""Competitive FIGS-vs-Baselines evaluation across 5 Grinsztajn benchmarks.

Sections:
  A) Friedman Test + Nemenyi Post-Hoc
  B) Pairwise Bayesian Signed-Rank Tests
  C) Cohen's d Effect Sizes
  D) Per-Dataset Results Table
  E) Interpretability Comparison
  F) Wall-Clock Time Comparison
  G) Winner Analysis and Pattern Identification
"""

import gc
import json
import math
import os
import resource
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from loguru import logger
from scipy import stats
from sklearn.metrics import balanced_accuracy_score, r2_score

# ──────────────────────────────────────────────────────────────────
# Setup
# ──────────────────────────────────────────────────────────────────
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
WORKSPACE = Path(__file__).parent
LOG_DIR = WORKSPACE / "logs"
LOG_DIR.mkdir(exist_ok=True)
logger.add(str(LOG_DIR / "run.log"), rotation="30 MB", level="DEBUG")


def _container_ram_gb() -> float | None:
    for p in ["/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"]:
        try:
            v = Path(p).read_text().strip()
            if v != "max" and int(v) < 1_000_000_000_000:
                return int(v) / 1e9
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
TOTAL_RAM_GB = _container_ram_gb() or 29.0
RAM_BUDGET = int(TOTAL_RAM_GB * 0.7 * 1e9)
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET * 3, RAM_BUDGET * 3))
logger.info(f"Hardware: {NUM_CPUS} CPUs, {TOTAL_RAM_GB:.1f} GB RAM, budget {RAM_BUDGET / 1e9:.1f} GB")

# ──────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────
FIGS_DIR = Path(
    "/ai-inventor/aii_pipeline/runs/jamnik-sgfigs-pid-v2/"
    "3_invention_loop/iter_2/gen_art/exp_id2_it2__opus"
)
BASELINES_DIR = Path(
    "/ai-inventor/aii_pipeline/runs/jamnik-sgfigs-pid-v2/"
    "3_invention_loop/iter_4/gen_art/exp_id2_it4__opus"
)
OVERLAPPING_DATASETS = ["electricity", "adult", "california_housing", "jannis", "higgs_small"]
CLASSIFICATION_DATASETS = ["electricity", "adult", "jannis", "higgs_small"]
REGRESSION_DATASETS = ["california_housing"]
FIGS_METHODS = ["axis_aligned_figs", "random_oblique_figs", "signed_spectral_figs"]
BASELINE_METHODS = ["ebm", "random_forest", "linear"]
ALL_METHODS = FIGS_METHODS + BASELINE_METHODS
N_FOLDS = 5

# Map each dataset to its split file index
_DS_TO_SPLIT = {
    "electricity": 1, "adult": 1, "california_housing": 1,
    "jannis": 2, "higgs_small": 3,
}


# ──────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────
def _json_safe(obj):
    """Recursively convert numpy / special types for JSON serialisation."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        v = float(obj)
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


# ──────────────────────────────────────────────────────────────────
# Data Loading
# ──────────────────────────────────────────────────────────────────
def load_figs_metadata() -> dict:
    """Load FIGS metadata (aggregated_results, best_max_splits, frustration)."""
    logger.info("Loading FIGS metadata from full_method_out.json ...")
    raw = json.loads((FIGS_DIR / "full_method_out.json").read_text())
    meta = raw["metadata"]
    del raw
    gc.collect()
    logger.info(f"  {len(meta['aggregated_results'])} aggregated results, "
                f"{len(meta['best_max_splits'])} best_max_splits entries")
    return meta


def load_figs_examples(dataset: str) -> list[dict]:
    """Load FIGS per-example predictions for *dataset* from split files."""
    split_idx = _DS_TO_SPLIT[dataset]
    path = FIGS_DIR / "method_out" / f"full_method_out_{split_idx}.json"
    logger.info(f"  Loading FIGS examples for {dataset} from {path.name} ...")
    raw = json.loads(path.read_text())
    for ds_block in raw["datasets"]:
        if ds_block["dataset"] == dataset:
            exs = ds_block["examples"]
            logger.info(f"    {dataset}: {len(exs)} examples")
            del raw
            gc.collect()
            return exs
    del raw
    gc.collect()
    logger.warning(f"    {dataset} not found in {path.name}")
    return []


def load_baselines() -> tuple[dict, dict[str, list[dict]]]:
    """Return (baselines_metadata, {dataset: [example, ...]})."""
    logger.info("Loading baselines full_method_out.json ...")
    raw = json.loads((BASELINES_DIR / "full_method_out.json").read_text())
    meta = raw["metadata"]
    examples: dict[str, list[dict]] = {}
    for ds_block in raw["datasets"]:
        if ds_block["dataset"] in OVERLAPPING_DATASETS:
            examples[ds_block["dataset"]] = ds_block["examples"]
    del raw
    gc.collect()
    logger.info(f"  {len(examples)} overlapping datasets loaded, "
                f"{sum(len(v) for v in examples.values())} examples total")
    return meta, examples


# ──────────────────────────────────────────────────────────────────
# Per-fold metric extraction
# ──────────────────────────────────────────────────────────────────
def compute_figs_fold_metrics(
    examples: list[dict], task_type: str
) -> dict[str, dict[int, float]]:
    """From per-example FIGS predictions, compute per-fold primary metric.

    Returns {method: {fold: metric_value}}.
    """
    by_fold: dict[int, list[dict]] = defaultdict(list)
    for ex in examples:
        by_fold[ex["metadata_fold"]].append(ex)

    out: dict[str, dict[int, float]] = {}
    for method in FIGS_METHODS:
        out[method] = {}
        pk = f"predict_{method}"
        for fold in sorted(by_fold):
            fexs = by_fold[fold]
            if task_type == "classification":
                yt = [int(float(e["output"])) for e in fexs]
                yp = [int(float(e[pk])) for e in fexs]
                out[method][fold] = balanced_accuracy_score(yt, yp)
            else:
                yt = [float(e["output"]) for e in fexs]
                yp = [float(e[pk]) for e in fexs]
                out[method][fold] = r2_score(yt, yp)
    return out


def extract_baselines_fold_data(
    meta: dict,
) -> dict[str, dict[str, dict[int, dict]]]:
    """Extract baselines per-fold results from metadata.per_dataset_results.

    Returns {dataset: {method: {fold: {balanced_accuracy, r2, auc, fit_time, ...}}}}.
    """
    pdr = meta["per_dataset_results"]
    out: dict = {}
    for ds in OVERLAPPING_DATASETS:
        out[ds] = {}
        for method in BASELINE_METHODS:
            out[ds][method] = {}
            for fr in pdr[ds][method]["fold_results"]:
                out[ds][method][fr["fold"]] = {
                    "balanced_accuracy": fr.get("balanced_accuracy"),
                    "r2": fr.get("r2"),
                    "auc": fr.get("auc"),
                    "fit_time": fr.get("fit_time"),
                    "n_terms": fr.get("n_terms"),
                    "n_interaction_terms": fr.get("n_interaction_terms"),
                    "n_main_effects": fr.get("n_main_effects"),
                }
    return out


# ──────────────────────────────────────────────────────────────────
# Section A  –  Friedman + Nemenyi
# ──────────────────────────────────────────────────────────────────
def section_a(
    per_ds_means: dict[str, dict[str, float]],
    per_fold: dict[str, dict[str, list[float]]],
) -> dict:
    logger.info("=== Section A: Friedman + Nemenyi ===")
    k = len(ALL_METHODS)
    n = len(OVERLAPPING_DATASETS)
    res: dict = {}

    # --- dataset-level Friedman ---
    perf = np.array([[per_ds_means[ds][m] for m in ALL_METHODS] for ds in OVERLAPPING_DATASETS])
    # ranks (1 = best)
    ranks = np.zeros_like(perf)
    for i in range(n):
        ranks[i] = stats.rankdata(-perf[i])
    avg_ranks = ranks.mean(axis=0)
    for j, m in enumerate(ALL_METHODS):
        res[f"avg_rank_{m}"] = float(avg_ranks[j])
        logger.info(f"  avg_rank {m}: {avg_ranks[j]:.3f}")

    try:
        chi2, pval = stats.friedmanchisquare(*[perf[:, j] for j in range(k)])
    except Exception:
        chi2, pval = 0.0, 1.0
    res["friedman_chi2"] = float(chi2)
    res["friedman_pvalue"] = float(pval)
    logger.info(f"  Friedman chi2={chi2:.4f}, p={pval:.6f}")

    # Nemenyi CD
    q_alpha = {3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031}.get(k, 2.850)
    cd = q_alpha * math.sqrt(k * (k + 1) / (6 * n))
    res["nemenyi_cd"] = float(cd)
    logger.info(f"  Nemenyi CD={cd:.4f}")

    # Pairwise significance
    nemenyi_pairs: dict = {}
    for i in range(k):
        for j in range(i + 1, k):
            diff = abs(avg_ranks[i] - avg_ranks[j])
            sig = diff > cd
            nemenyi_pairs[f"{ALL_METHODS[i]}_vs_{ALL_METHODS[j]}"] = {
                "rank_diff": float(diff),
                "significant": bool(sig),
            }
    res["nemenyi_pairwise"] = nemenyi_pairs

    # Cliques (non-significant groups)
    cliques: list[list[str]] = []
    for i in range(k):
        clique = sorted(
            [ALL_METHODS[j] for j in range(k) if abs(avg_ranks[i] - avg_ranks[j]) <= cd]
        )
        if clique not in cliques:
            cliques.append(clique)
    res["nemenyi_cliques"] = cliques

    # --- per-fold Friedman (25 blocks) ---
    fold_rows = []
    for ds in OVERLAPPING_DATASETS:
        for fold_idx in range(N_FOLDS):
            fold_rows.append([per_fold[ds][m][fold_idx] for m in ALL_METHODS])
    fold_mat = np.array(fold_rows)
    try:
        chi2f, pvalf = stats.friedmanchisquare(*[fold_mat[:, j] for j in range(k)])
    except Exception:
        chi2f, pvalf = 0.0, 1.0
    res["friedman_perfold_chi2"] = float(chi2f)
    res["friedman_perfold_pvalue"] = float(pvalf)
    logger.info(f"  Per-fold Friedman chi2={chi2f:.4f}, p={pvalf:.6f}")
    return res


# ──────────────────────────────────────────────────────────────────
# Section B  –  Bayesian Signed-Rank (bootstrap)
# ──────────────────────────────────────────────────────────────────
def _bayesian_bootstrap(diffs: np.ndarray, rope: float = 0.01,
                        n_boot: int = 10_000, seed: int = 42):
    rng = np.random.RandomState(seed)
    means = np.array([np.mean(rng.choice(diffs, size=len(diffs), replace=True))
                      for _ in range(n_boot)])
    p_left = float(np.mean(means > rope))       # A wins
    p_right = float(np.mean(means < -rope))      # B wins
    p_rope = float(np.mean((means >= -rope) & (means <= rope)))
    return p_left, p_rope, p_right


def section_b(
    figs_fold: dict[str, dict[str, dict[int, float]]],
    bl_fold: dict[str, dict[str, dict[int, float]]],
) -> dict:
    logger.info("=== Section B: Bayesian Signed-Rank Tests ===")
    res: dict = {}
    pairs = [
        ("signed_spectral_figs", "ebm"),
        ("random_oblique_figs", "ebm"),
        ("axis_aligned_figs", "ebm"),
    ]
    for fm, bm in pairs:
        diffs = []
        for ds in CLASSIFICATION_DATASETS:
            for fold in range(N_FOLDS):
                diffs.append(figs_fold[ds][fm][fold] - bl_fold[ds][bm][fold])
        diffs_arr = np.array(diffs)
        p_l, p_r_ope, p_r = _bayesian_bootstrap(diffs_arr)
        short = fm.replace("_figs", "")
        pfx = f"bayesian_{short}_vs_{bm}"
        res[f"{pfx}_prob_left"] = p_l
        res[f"{pfx}_prob_rope"] = p_r_ope
        res[f"{pfx}_prob_right"] = p_r
        res[f"{pfx}_mean_diff"] = float(np.mean(diffs_arr))
        res[f"{pfx}_std_diff"] = float(np.std(diffs_arr))
        if p_r_ope > 0.5:
            interp = "practically_equivalent"
        elif p_l > 0.95:
            interp = f"{fm}_wins"
        elif p_r > 0.95:
            interp = f"{bm}_wins"
        else:
            interp = "inconclusive"
        res[f"{pfx}_interpretation"] = interp
        logger.info(f"  {fm} vs {bm}: P(left)={p_l:.3f} P(ROPE)={p_r_ope:.3f} "
                    f"P(right)={p_r:.3f} -> {interp}")
    return res


# ──────────────────────────────────────────────────────────────────
# Section C  –  Cohen's d
# ──────────────────────────────────────────────────────────────────
def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    d = a - b
    sd = np.std(d, ddof=1)
    return 0.0 if sd == 0 else float(np.mean(d) / sd)


def _classify_d(d: float) -> str:
    ad = abs(d)
    if ad < 0.2:
        return "negligible"
    if ad < 0.5:
        return "small"
    if ad < 0.8:
        return "medium"
    return "large"


def section_c(
    figs_fold: dict, bl_fold: dict,
) -> dict:
    logger.info("=== Section C: Cohen's d Effect Sizes ===")
    res: dict = {}
    wlt: dict[str, dict[str, int]] = {fm: {"wins": 0, "losses": 0, "ties": 0} for fm in FIGS_METHODS}

    for fm in FIGS_METHODS:
        for bm in BASELINE_METHODS:
            fa, ba = [], []
            for ds in CLASSIFICATION_DATASETS:
                for fold in range(N_FOLDS):
                    fa.append(figs_fold[ds][fm][fold])
                    ba.append(bl_fold[ds][bm][fold])
            d = _cohens_d(np.array(fa), np.array(ba))
            eff = _classify_d(d)
            short = fm.replace("_figs", "")
            key = f"cohens_d_{short}_vs_{bm}"
            res[key] = d
            res[f"{key}_effect"] = eff
            logger.info(f"  {fm} vs {bm}: d={d:.4f} ({eff})")

            # per-dataset win/loss/tie
            for ds in CLASSIFICATION_DATASETS:
                fm_mean = np.mean([figs_fold[ds][fm][f] for f in range(N_FOLDS)])
                bm_mean = np.mean([bl_fold[ds][bm][f] for f in range(N_FOLDS)])
                if fm_mean > bm_mean + 1e-4:
                    wlt[fm]["wins"] += 1
                elif bm_mean > fm_mean + 1e-4:
                    wlt[fm]["losses"] += 1
                else:
                    wlt[fm]["ties"] += 1

    for fm in FIGS_METHODS:
        short = fm.replace("_figs", "")
        for tag in ("wins", "losses", "ties"):
            res[f"wlt_{short}_{tag}"] = wlt[fm][tag]
        logger.info(f"  {fm} W/L/T: {wlt[fm]}")
    return res


# ──────────────────────────────────────────────────────────────────
# Section D  –  Per-Dataset Results Table
# ──────────────────────────────────────────────────────────────────
def section_d(
    figs_agg: list[dict],
    bl_fold_full: dict,
    best_ms: dict,
) -> dict:
    logger.info("=== Section D: Per-Dataset Results Table ===")
    table: dict = {}
    for ds in OVERLAPPING_DATASETS:
        tt = "classification" if ds in CLASSIFICATION_DATASETS else "regression"
        pm = "balanced_accuracy" if tt == "classification" else "r2"
        table[ds] = {"task_type": tt, "primary_metric": pm, "methods": {}}

        # FIGS
        for method in FIGS_METHODS:
            bms = best_ms.get(f"{ds}__{method}")
            agg = next(
                (a for a in figs_agg
                 if a["dataset"] == ds and a["method"] == method and a["max_splits"] == bms),
                None,
            )
            if agg is None:
                continue
            entry: dict = {
                "primary_mean": agg.get(f"{pm}_mean"),
                "primary_std": agg.get(f"{pm}_std"),
                "best_max_splits": bms,
                "fit_time_mean": agg.get("fit_time_sec_mean"),
                "total_splits_mean": agg.get("total_splits_mean"),
                "avg_split_arity_mean": agg.get("avg_split_arity_mean"),
                "avg_path_length_mean": agg.get("avg_path_length_mean"),
            }
            if tt == "classification":
                entry["auc_mean"] = agg.get("auc_roc_mean")
                entry["auc_std"] = agg.get("auc_roc_std")
            table[ds]["methods"][method] = entry

        # Baselines
        for method in BASELINE_METHODS:
            folds = bl_fold_full[ds][method]
            pvals = [folds[f][pm] for f in range(N_FOLDS) if folds[f].get(pm) is not None]
            if not pvals:
                continue
            entry = {
                "primary_mean": float(np.mean(pvals)),
                "primary_std": float(np.std(pvals)),
                "fit_time_mean": float(np.mean([folds[f]["fit_time"] for f in range(N_FOLDS)])),
            }
            if tt == "classification":
                auc_vals = [folds[f]["auc"] for f in range(N_FOLDS)
                            if folds[f].get("auc") is not None]
                if auc_vals:
                    entry["auc_mean"] = float(np.mean(auc_vals))
                    entry["auc_std"] = float(np.std(auc_vals))
            if method == "ebm":
                entry["n_terms"] = float(np.mean(
                    [folds[f]["n_terms"] for f in range(N_FOLDS)
                     if folds[f].get("n_terms") is not None]
                ))
            table[ds]["methods"][method] = entry

        # Winner / second-best
        ranked = sorted(
            ((m, e["primary_mean"]) for m, e in table[ds]["methods"].items()
             if e.get("primary_mean") is not None),
            key=lambda x: x[1], reverse=True,
        )
        if ranked:
            table[ds]["winner"] = ranked[0][0]
            table[ds]["winner_value"] = ranked[0][1]
        if len(ranked) > 1:
            table[ds]["second_best"] = ranked[1][0]
            table[ds]["second_best_value"] = ranked[1][1]
        logger.info(f"  {ds} ({tt}): winner={table[ds].get('winner')}, "
                    f"val={table[ds].get('winner_value', 0):.4f}")
    return table


# ──────────────────────────────────────────────────────────────────
# Section E  –  Interpretability Comparison
# ──────────────────────────────────────────────────────────────────
def section_e(figs_agg: list[dict], bl_fold_full: dict, best_ms: dict) -> dict:
    logger.info("=== Section E: Interpretability ===")
    res: dict = {"per_dataset": {}}
    figs_splits_all, ebm_terms_all = [], []
    figs_arity_all, ebm_inter_all = [], []

    for ds in OVERLAPPING_DATASETS:
        dr: dict = {}
        # signed-spectral at best ms
        bms = best_ms.get(f"{ds}__signed_spectral_figs")
        agg = next(
            (a for a in figs_agg
             if a["dataset"] == ds and a["method"] == "signed_spectral_figs"
             and a["max_splits"] == bms),
            None,
        )
        if agg:
            dr["figs_total_splits"] = agg.get("total_splits_mean")
            dr["figs_avg_split_arity"] = agg.get("avg_split_arity_mean")
            dr["figs_avg_path_length"] = agg.get("avg_path_length_mean")
            if dr["figs_total_splits"] is not None:
                figs_splits_all.append(dr["figs_total_splits"])
            if dr["figs_avg_split_arity"] is not None:
                figs_arity_all.append(dr["figs_avg_split_arity"])

        # EBM
        folds = bl_fold_full[ds]["ebm"]
        nt = [folds[f]["n_terms"] for f in range(N_FOLDS) if folds[f].get("n_terms") is not None]
        ni = [folds[f]["n_interaction_terms"] for f in range(N_FOLDS)
              if folds[f].get("n_interaction_terms") is not None]
        nm = [folds[f]["n_main_effects"] for f in range(N_FOLDS)
              if folds[f].get("n_main_effects") is not None]
        if nt:
            dr["ebm_n_terms"] = float(np.mean(nt))
            ebm_terms_all.append(dr["ebm_n_terms"])
        if ni:
            dr["ebm_n_interaction_terms"] = float(np.mean(ni))
            ebm_inter_all.append(dr["ebm_n_interaction_terms"])
        if nm:
            dr["ebm_n_main_effects"] = float(np.mean(nm))
        if dr.get("figs_total_splits") and dr.get("ebm_n_terms"):
            dr["complexity_ratio"] = dr["figs_total_splits"] / dr["ebm_n_terms"]
        res["per_dataset"][ds] = dr
        logger.info(f"  {ds}: FIGS splits={dr.get('figs_total_splits')}, "
                    f"EBM terms={dr.get('ebm_n_terms')}, "
                    f"ratio={dr.get('complexity_ratio', 'N/A')}")

    ratios = [v["complexity_ratio"] for v in res["per_dataset"].values()
              if "complexity_ratio" in v]
    res["avg_complexity_ratio"] = float(np.mean(ratios)) if ratios else None

    # Wilcoxon: FIGS arity vs EBM interactions
    min_len = min(len(figs_arity_all), len(ebm_inter_all))
    if min_len >= 3:
        try:
            w, wp = stats.wilcoxon(figs_arity_all[:min_len], ebm_inter_all[:min_len])
            res["wilcoxon_arity_vs_interactions_stat"] = float(w)
            res["wilcoxon_arity_vs_interactions_pvalue"] = float(wp)
        except ValueError:
            res["wilcoxon_arity_vs_interactions_stat"] = None
            res["wilcoxon_arity_vs_interactions_pvalue"] = None
    else:
        # Fewer than 3 paired observations - note it but use a manual comparison
        res["wilcoxon_arity_vs_interactions_stat"] = None
        res["wilcoxon_arity_vs_interactions_pvalue"] = None
        if figs_arity_all and ebm_inter_all:
            res["mean_figs_arity"] = float(np.mean(figs_arity_all))
            res["mean_ebm_interactions"] = float(np.mean(ebm_inter_all))
    return res


# ──────────────────────────────────────────────────────────────────
# Section F  –  Wall-Clock Time
# ──────────────────────────────────────────────────────────────────
def section_f(figs_agg: list[dict], bl_fold_full: dict, best_ms: dict) -> dict:
    logger.info("=== Section F: Timing ===")
    res: dict = {"per_dataset": {}}

    for ds in OVERLAPPING_DATASETS:
        dr: dict = {}
        for method in FIGS_METHODS:
            bms = best_ms.get(f"{ds}__{method}")
            agg = next(
                (a for a in figs_agg
                 if a["dataset"] == ds and a["method"] == method and a["max_splits"] == bms),
                None,
            )
            if agg:
                dr[method] = agg.get("fit_time_sec_mean")
        for method in BASELINE_METHODS:
            folds = bl_fold_full[ds][method]
            dr[method] = float(np.mean([folds[f]["fit_time"] for f in range(N_FOLDS)]))
        aa = dr.get("axis_aligned_figs", 1.0)
        if aa and aa > 0:
            dr["time_ratios"] = {m: v / aa for m, v in dr.items()
                                 if isinstance(v, (int, float)) and m != "time_ratios"}
        res["per_dataset"][ds] = dr
        parts = ", ".join(f"{m}={dr.get(m, 0):.2f}s" for m in ALL_METHODS if m in dr)
        logger.info(f"  {ds}: {parts}")

    avg_ratios: dict[str, list[float]] = defaultdict(list)
    for ds in OVERLAPPING_DATASETS:
        tr = res["per_dataset"][ds].get("time_ratios", {})
        for m, v in tr.items():
            avg_ratios[m].append(v)
    res["avg_time_ratios"] = {m: float(np.mean(v)) for m, v in avg_ratios.items()}
    return res


# ──────────────────────────────────────────────────────────────────
# Section G  –  Winner Analysis
# ──────────────────────────────────────────────────────────────────
def section_g(
    per_ds_means: dict[str, dict[str, float]],
    frustration: dict,
) -> dict:
    logger.info("=== Section G: Winner Analysis ===")
    res: dict = {"per_dataset": {}}
    gaps = []
    n_figs_wins, n_bl_wins = 0, 0
    fi_vals, rel_perfs = [], []

    for ds in OVERLAPPING_DATASETS:
        ranked = sorted(per_ds_means[ds].items(), key=lambda x: x[1], reverse=True)
        rankings = {m: r + 1 for r, (m, _) in enumerate(ranked)}
        winner = ranked[0][0]
        best_figs = max(per_ds_means[ds][m] for m in FIGS_METHODS)
        best_bl = max(per_ds_means[ds][m] for m in BASELINE_METHODS)
        gap = best_figs - best_bl
        gaps.append(gap)
        if winner in FIGS_METHODS:
            n_figs_wins += 1
        else:
            n_bl_wins += 1

        res["per_dataset"][ds] = {
            "winner": winner,
            "winner_value": ranked[0][1],
            "rankings": rankings,
            "best_figs": best_figs,
            "best_baseline": best_bl,
            "figs_minus_baseline_gap": gap,
        }
        logger.info(f"  {ds}: winner={winner} ({ranked[0][1]:.4f}), gap={gap:+.4f}")

        # frustration cross-reference
        if ds in frustration:
            fi = frustration[ds].get("frustration_index", 0)
            fi_vals.append(fi)
            ss = per_ds_means[ds].get("signed_spectral_figs", 0)
            aa = per_ds_means[ds].get("axis_aligned_figs", 0)
            rel_perfs.append(ss - aa)

    res["n_datasets_figs_wins"] = n_figs_wins
    res["n_datasets_baseline_wins"] = n_bl_wins
    res["avg_figs_baseline_gap"] = float(np.mean(gaps))
    res["max_figs_baseline_gap"] = float(np.max(gaps))
    res["min_figs_baseline_gap"] = float(np.min(gaps))

    if len(fi_vals) >= 3:
        try:
            rho, rp = stats.spearmanr(fi_vals, rel_perfs)
            res["frustration_correlation"] = float(rho)
            res["frustration_correlation_pvalue"] = float(rp)
            logger.info(f"  Frustration-index corr: rho={rho:.4f}, p={rp:.4f}")
        except Exception:
            res["frustration_correlation"] = None
            res["frustration_correlation_pvalue"] = None
    logger.info(f"  FIGS wins {n_figs_wins}/{len(OVERLAPPING_DATASETS)}, "
                f"avg gap={np.mean(gaps):+.4f}")
    return res


# ──────────────────────────────────────────────────────────────────
# Build schema-compliant output
# ──────────────────────────────────────────────────────────────────
def build_output(
    sections: dict,
    results_table: dict,
    figs_pred_lookup: dict[str, dict[int, dict]],
    bl_examples: dict[str, list[dict]],
) -> dict:
    logger.info("Building output JSON ...")

    # ---- metrics_agg (flat numeric only) ----
    metrics_agg: dict[str, float] = {}

    def _add_flat(src: dict, skip_keys: set | None = None):
        for k, v in src.items():
            if skip_keys and k in skip_keys:
                continue
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                if not (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
                    metrics_agg[k] = v

    _add_flat(sections["A"], skip_keys={"nemenyi_pairwise", "nemenyi_cliques"})
    _add_flat(sections["B"])
    _add_flat(sections["C"])
    _add_flat(sections["G"])
    if sections["E"].get("avg_complexity_ratio") is not None:
        metrics_agg["avg_complexity_ratio"] = sections["E"]["avg_complexity_ratio"]
    for k in ("wilcoxon_arity_vs_interactions_stat", "wilcoxon_arity_vs_interactions_pvalue",
              "mean_figs_arity", "mean_ebm_interactions"):
        if sections["E"].get(k) is not None:
            metrics_agg[k] = sections["E"][k]
    for m, ratio in sections["F"].get("avg_time_ratios", {}).items():
        metrics_agg[f"avg_time_ratio_{m}"] = ratio

    # Remove string-valued items that slipped through
    metrics_agg = {k: v for k, v in metrics_agg.items()
                   if isinstance(v, (int, float)) and not isinstance(v, bool)}

    # ---- datasets with merged examples ----
    datasets_out: list[dict] = []
    for ds in OVERLAPPING_DATASETS:
        tt = "classification" if ds in CLASSIFICATION_DATASETS else "regression"
        bl_exs = bl_examples.get(ds, [])
        lookup = figs_pred_lookup.get(ds, {})
        merged: list[dict] = []
        for bex in bl_exs:
            ri = bex.get("metadata_row_index")
            ex: dict = {
                "input": bex["input"],
                "output": bex["output"],
                "metadata_fold": bex["metadata_fold"],
                "metadata_task_type": bex["metadata_task_type"],
                "metadata_row_index": ri,
            }
            # baseline predictions
            for bm in BASELINE_METHODS:
                pk = f"predict_{bm}"
                if pk in bex:
                    ex[pk] = bex[pk]
            # FIGS predictions
            figs_preds = lookup.get(ri, {})
            for fm in FIGS_METHODS:
                pk = f"predict_{fm}"
                if pk in figs_preds:
                    ex[pk] = figs_preds[pk]
            # eval fields
            if tt == "classification":
                true_lbl = int(float(bex["output"]))
                for m in ALL_METHODS:
                    pk = f"predict_{m}"
                    if pk in ex:
                        try:
                            ex[f"eval_correct_{m}"] = 1 if int(float(ex[pk])) == true_lbl else 0
                        except (ValueError, TypeError):
                            pass
            else:
                true_val = float(bex["output"])
                for m in ALL_METHODS:
                    pk = f"predict_{m}"
                    if pk in ex:
                        try:
                            ex[f"eval_abs_error_{m}"] = round(abs(float(ex[pk]) - true_val), 6)
                        except (ValueError, TypeError):
                            pass
            merged.append(ex)
        datasets_out.append({"dataset": ds, "examples": merged})
        logger.info(f"  {ds}: {len(merged)} merged examples")

    metadata = _json_safe({
        "evaluation_name": "FIGS_vs_Baselines_Competitive_Comparison",
        "description": (
            "Paper-ready comparison of 3 FIGS variants against 3 baselines "
            "across 5 Grinsztajn benchmarks (Friedman/Nemenyi, Bayesian, "
            "Cohen's d, interpretability, timing, winner analysis)"
        ),
        "date": "2026-03-19",
        "overlapping_datasets": OVERLAPPING_DATASETS,
        "classification_datasets": CLASSIFICATION_DATASETS,
        "regression_datasets": REGRESSION_DATASETS,
        "figs_methods": FIGS_METHODS,
        "baseline_methods": BASELINE_METHODS,
        "n_folds": N_FOLDS,
        "sections": {
            "A_friedman_nemenyi": sections["A"],
            "B_bayesian_signed_rank": sections["B"],
            "C_cohens_d": sections["C"],
            "D_results_table": results_table,
            "E_interpretability": sections["E"],
            "F_timing": sections["F"],
            "G_winner_analysis": sections["G"],
        },
    })

    return _json_safe({
        "metadata": metadata,
        "metrics_agg": metrics_agg,
        "datasets": datasets_out,
    })


# ──────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────
@logger.catch
def main():
    logger.info("=" * 60)
    logger.info("FIGS-vs-Baselines Competitive Evaluation")
    logger.info("=" * 60)

    # ── 1. Load data ──
    figs_meta = load_figs_metadata()
    bl_meta, bl_examples = load_baselines()

    figs_agg = figs_meta["aggregated_results"]
    best_ms = figs_meta["best_max_splits"]
    frustration = figs_meta.get("frustration_analysis", {})
    bl_fold_full = extract_baselines_fold_data(bl_meta)

    # ── 2. Compute FIGS per-fold metrics & build prediction lookup ──
    figs_fold: dict[str, dict[str, dict[int, float]]] = {}
    figs_pred_lookup: dict[str, dict[int, dict]] = {}

    for ds in OVERLAPPING_DATASETS:
        tt = "classification" if ds in CLASSIFICATION_DATASETS else "regression"
        examples = load_figs_examples(ds)
        fold_m = compute_figs_fold_metrics(examples, tt)

        pm = "balanced_accuracy" if tt == "classification" else "r2"
        figs_fold[ds] = {m: {f: fold_m[m][f] for f in range(N_FOLDS)} for m in FIGS_METHODS}

        # prediction lookup for output merging
        lk: dict[int, dict] = {}
        for ex in examples:
            ri = ex.get("metadata_row_index")
            if ri is not None:
                lk[ri] = {f"predict_{m}": ex.get(f"predict_{m}") for m in FIGS_METHODS}
        figs_pred_lookup[ds] = lk

        # Log & verify against aggregated
        for m in FIGS_METHODS:
            vals = [figs_fold[ds][m][f] for f in range(N_FOLDS)]
            computed_mean = np.mean(vals)
            bms = best_ms.get(f"{ds}__{m}")
            agg_entry = next(
                (a for a in figs_agg
                 if a["dataset"] == ds and a["method"] == m and a["max_splits"] == bms),
                None,
            )
            agg_mean = agg_entry.get(f"{pm}_mean") if agg_entry else None
            match_str = ""
            if agg_mean is not None:
                diff = abs(computed_mean - agg_mean)
                match_str = f" (agg={agg_mean:.4f}, diff={diff:.4f})"
            logger.info(f"  {ds}/{m}: per-fold {pm} = "
                        f"{[f'{v:.4f}' for v in vals]}, "
                        f"mean={computed_mean:.4f}{match_str}")

        del examples
        gc.collect()

    # ── 3. Baselines per-fold primary metric ──
    bl_fold_primary: dict[str, dict[str, dict[int, float]]] = {}
    for ds in OVERLAPPING_DATASETS:
        tt = "classification" if ds in CLASSIFICATION_DATASETS else "regression"
        pm = "balanced_accuracy" if tt == "classification" else "r2"
        bl_fold_primary[ds] = {}
        for m in BASELINE_METHODS:
            bl_fold_primary[ds][m] = {}
            for f in range(N_FOLDS):
                v = bl_fold_full[ds][m][f][pm]
                bl_fold_primary[ds][m][f] = v if v is not None else 0.0
            vals = [bl_fold_primary[ds][m][f] for f in range(N_FOLDS)]
            logger.info(f"  {ds}/{m}: per-fold {pm} = "
                        f"{[f'{v:.4f}' for v in vals]}, mean={np.mean(vals):.4f}")

    # ── 4. Merge all per-fold data ──
    all_fold: dict[str, dict[str, list[float]]] = {}
    per_ds_means: dict[str, dict[str, float]] = {}
    for ds in OVERLAPPING_DATASETS:
        all_fold[ds] = {}
        per_ds_means[ds] = {}
        for m in FIGS_METHODS:
            all_fold[ds][m] = [figs_fold[ds][m][f] for f in range(N_FOLDS)]
        for m in BASELINE_METHODS:
            all_fold[ds][m] = [bl_fold_primary[ds][m][f] for f in range(N_FOLDS)]
        for m in ALL_METHODS:
            per_ds_means[ds][m] = float(np.mean(all_fold[ds][m]))

    # ── 5. Run all sections ──
    secs: dict = {}
    secs["A"] = section_a(per_ds_means, all_fold)
    secs["B"] = section_b(figs_fold, bl_fold_primary)
    secs["C"] = section_c(figs_fold, bl_fold_primary)
    results_table = section_d(figs_agg, bl_fold_full, best_ms)
    secs["E"] = section_e(figs_agg, bl_fold_full, best_ms)
    secs["F"] = section_f(figs_agg, bl_fold_full, best_ms)
    secs["G"] = section_g(per_ds_means, frustration)

    # ── 6. Build & save output ──
    output = build_output(secs, results_table, figs_pred_lookup, bl_examples)

    out_path = WORKSPACE / "eval_out.json"
    out_path.write_text(json.dumps(output, indent=2, default=str))
    logger.info(f"Saved {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")
    logger.info(f"metrics_agg: {len(output['metrics_agg'])} metrics")
    logger.info(f"datasets: {len(output['datasets'])} datasets, "
                f"{sum(len(d['examples']) for d in output['datasets'])} examples")
    logger.info("=" * 60)
    logger.info("Evaluation complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

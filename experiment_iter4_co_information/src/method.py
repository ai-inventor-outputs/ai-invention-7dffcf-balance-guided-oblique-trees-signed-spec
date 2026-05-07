#!/usr/bin/env python3
"""Co-Information Estimator Bias Diagnosis & SPONGE Failure Mechanism Analysis.

Two-part diagnostic experiment:
  Part 1: Compare CoI estimation methods (NPEET KSG, raw NPEET, binning, sklearn)
          on synthetic datasets with planted synergy modules to determine whether
          universally negative CoI values are a KSG bias artifact or genuine.
  Part 2: Diagnose why SPONGE signed spectral clustering underperforms unsigned
          spectral on all-negative CoI graphs via eigenspectrum, condition number,
          and positive edge injection analysis.
"""

import gc
import json
import math
import os
import resource
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from loguru import logger
from scipy.linalg import eigh, eigvalsh
from scipy.special import digamma
from sklearn.cluster import KMeans
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import adjusted_rand_score, mutual_info_score
from sklearn.neighbors import KDTree, NearestNeighbors

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
WORKSPACE = Path(__file__).parent
LOG_DIR = WORKSPACE / "logs"
LOG_DIR.mkdir(exist_ok=True)
logger.add(LOG_DIR / "run.log", rotation="30 MB", level="DEBUG")

# ---------------------------------------------------------------------------
# Hardware detection (cgroup-aware)
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
TOTAL_RAM_GB = _container_ram_gb() or 29.0
logger.info(f"Hardware: {NUM_CPUS} CPUs, {TOTAL_RAM_GB:.1f} GB RAM")

RAM_BUDGET = int(min(14, TOTAL_RAM_GB * 0.45) * 1024**3)
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET * 3, RAM_BUDGET * 3))
resource.setrlimit(resource.RLIMIT_CPU, (3600, 3600))
logger.info(f"RAM budget: {RAM_BUDGET / 1e9:.1f} GB, CPU limit: 3600s")

NUM_WORKERS = max(1, NUM_CPUS - 1)
MASTER_SEED = 42

# ---------------------------------------------------------------------------
# Data generation helpers — import from dependency
# ---------------------------------------------------------------------------
DEP_DATA_DIR = str(WORKSPACE)
sys.path.insert(0, DEP_DATA_DIR)


def load_datasets() -> dict:
    """Generate datasets using data.py generators with deterministic seeds."""
    from data import (
        gen_easy_2mod_xor,
        gen_medium_4mod_mixed,
        gen_no_structure_control,
    )

    base_rng = np.random.default_rng(MASTER_SEED)
    variant_seeds = [int(base_rng.integers(0, 2**31)) for _ in range(6)]

    datasets = {}

    # Easy 2-module XOR (seed index 0)
    logger.info("Generating easy_2mod_xor...")
    r = gen_easy_2mod_xor(np.random.default_rng(variant_seeds[0]))
    datasets["easy_2mod_xor"] = {"X": r["X"], "y": r["y"], "meta": r["meta"]}

    # Medium 4-module mixed (seed index 1)
    logger.info("Generating medium_4mod_mixed...")
    r = gen_medium_4mod_mixed(np.random.default_rng(variant_seeds[1]))
    datasets["medium_4mod_mixed"] = {"X": r["X"], "y": r["y"], "meta": r["meta"]}

    # No structure control (seed index 4)
    logger.info("Generating no_structure_control...")
    r = gen_no_structure_control(np.random.default_rng(variant_seeds[4]))
    datasets["no_structure_control"] = {"X": r["X"], "y": r["y"], "meta": r["meta"]}

    # Calibration pure XOR (custom)
    logger.info("Generating calibration_pure_xor...")
    n_cal = 50000
    rng99 = np.random.default_rng(99)
    X_cal = rng99.standard_normal((n_cal, 5))
    X_cal[:, 3] = X_cal[:, 0] + np.random.default_rng(100).normal(0, 0.3, n_cal)
    # X_cal[:, 4] is already pure noise
    Y_cal = (X_cal[:, 0] * X_cal[:, 1] > 0).astype(int)
    datasets["calibration_pure_xor"] = {
        "X": X_cal, "y": Y_cal,
        "meta": {
            "n_samples": n_cal, "n_features": 5, "n_modules": 1,
            "ground_truth_modules": [[0, 1]],
            "module_types": ["xor"],
            "redundant_pairs": [[0, 3]],
            "noise_features": [4],
            "feature_names": [f"X{i}" for i in range(5)],
        }
    }

    return datasets


# ---------------------------------------------------------------------------
# MI Estimation Methods
# ---------------------------------------------------------------------------

def npeet_mi_cd(X_nd: np.ndarray, y: np.ndarray, k: int = 5) -> float:
    """Compute MI(X;Y) using NPEET's micd. X continuous, Y discrete.

    NPEET uses abs() internally so result is always >= 0.
    Accepts X as 1D or 2D numpy array, y as 1D numpy array.
    """
    import npeet.entropy_estimators as ee
    if X_nd.ndim == 1:
        X_nd = X_nd.reshape(-1, 1)
    y_2d = y.reshape(-1, 1) if y.ndim == 1 else y
    try:
        mi = ee.micd(X_nd, y_2d, k=k, base=np.e, warning=False)
        return float(mi)
    except Exception as e:
        logger.warning(f"NPEET micd failed: {e}")
        return 0.0


def raw_npeet_mi_cd(X_nd: np.ndarray, y: np.ndarray, k: int = 5) -> float:
    """Compute MI(X;Y) using NPEET entropy internals WITHOUT abs() clipping.

    This exposes the raw KSG estimate, which can be negative. Useful for
    diagnosing whether negative MI estimates contribute to all-negative CoI.
    """
    import npeet.entropy_estimators as ee
    if X_nd.ndim == 1:
        X_nd = X_nd.reshape(-1, 1)
    y_2d = y.reshape(-1, 1) if y.ndim == 1 else y
    try:
        entropy_x = ee.entropy(X_nd, k=k, base=np.e)
        y_unique, y_count = np.unique(y_2d, return_counts=True, axis=0)
        y_proba = y_count / len(y_2d)
        entropy_x_given_y = 0.0
        for yval, py in zip(y_unique, y_proba):
            mask = (y_2d == yval).all(axis=1)
            x_given_y = X_nd[mask]
            if k <= len(x_given_y) - 1:
                entropy_x_given_y += py * ee.entropy(x_given_y, k=k, base=np.e)
            else:
                entropy_x_given_y += py * entropy_x
        return float(entropy_x - entropy_x_given_y)  # RAW, no abs()
    except Exception as e:
        logger.warning(f"raw NPEET MI failed: {e}")
        return 0.0


def binned_mi(x_binned: np.ndarray, y: np.ndarray) -> float:
    """MI between discrete binned feature and discrete target (nats)."""
    return float(mutual_info_score(x_binned, y))


def bin_feature(x: np.ndarray, n_bins: int) -> np.ndarray:
    """Bin a continuous feature into n_bins discrete bins."""
    edges = np.linspace(x.min() - 1e-10, x.max() + 1e-10, n_bins + 1)
    return np.clip(np.digitize(x, edges[1:-1]), 0, n_bins - 1)


# ---------------------------------------------------------------------------
# Parallel worker functions
# ---------------------------------------------------------------------------

def _worker_npeet_pair(args: tuple) -> tuple:
    """Worker: compute joint MI for pair (i,j) using NPEET."""
    i, j, X_i, X_j, y, k = args
    X_2d = np.column_stack([X_i, X_j])
    jmi = npeet_mi_cd(X_2d, y, k=k)
    return (i, j, jmi)


def _worker_raw_npeet_pair(args: tuple) -> tuple:
    """Worker: compute joint MI for pair (i,j) using raw NPEET (no abs)."""
    i, j, X_i, X_j, y, k = args
    X_2d = np.column_stack([X_i, X_j])
    jmi = raw_npeet_mi_cd(X_2d, y, k=k)
    return (i, j, jmi)


# ---------------------------------------------------------------------------
# CoI Matrix computation
# ---------------------------------------------------------------------------

def compute_coi_npeet(
    X: np.ndarray, y: np.ndarray, k: int = 5,
    max_samples: int | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute CoI matrix using NPEET KSG (abs-clipped individual MI)."""
    n, d = X.shape
    if max_samples and n > max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, max_samples, replace=False)
        X, y = X[idx], y[idx]

    mi_ind = np.array([npeet_mi_cd(X[:, i], y, k=k) for i in range(d)])

    # Joint MI in parallel
    pairs = [(i, j, X[:, i].copy(), X[:, j].copy(), y.copy(), k)
             for i in range(d) for j in range(i + 1, d)]

    mi_jnt = np.zeros((d, d))
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as pool:
        futs = {pool.submit(_worker_npeet_pair, p): p[:2] for p in pairs}
        for fut in as_completed(futs):
            try:
                i, j, jmi = fut.result()
                mi_jnt[i, j] = mi_jnt[j, i] = jmi
            except Exception:
                logger.exception(f"NPEET pair failed")

    coi = np.zeros((d, d))
    for i in range(d):
        for j in range(i + 1, d):
            coi[i, j] = coi[j, i] = mi_ind[i] + mi_ind[j] - mi_jnt[i, j]
    return coi, mi_ind, mi_jnt


def compute_coi_raw_npeet(
    X: np.ndarray, y: np.ndarray, k: int = 5,
    max_samples: int | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute CoI using raw NPEET (no abs clipping) — diagnostic method."""
    n, d = X.shape
    if max_samples and n > max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, max_samples, replace=False)
        X, y = X[idx], y[idx]

    mi_ind = np.array([raw_npeet_mi_cd(X[:, i], y, k=k) for i in range(d)])

    pairs = [(i, j, X[:, i].copy(), X[:, j].copy(), y.copy(), k)
             for i in range(d) for j in range(i + 1, d)]

    mi_jnt = np.zeros((d, d))
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as pool:
        futs = {pool.submit(_worker_raw_npeet_pair, p): p[:2] for p in pairs}
        for fut in as_completed(futs):
            try:
                i, j, jmi = fut.result()
                mi_jnt[i, j] = mi_jnt[j, i] = jmi
            except Exception:
                logger.exception(f"raw NPEET pair failed")

    coi = np.zeros((d, d))
    for i in range(d):
        for j in range(i + 1, d):
            coi[i, j] = coi[j, i] = mi_ind[i] + mi_ind[j] - mi_jnt[i, j]
    return coi, mi_ind, mi_jnt


def compute_coi_binned(
    X: np.ndarray, y: np.ndarray, n_bins: int = 20
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute CoI matrix using discretization-based MI."""
    n, d = X.shape
    X_binned = np.zeros((n, d), dtype=int)
    for i in range(d):
        X_binned[:, i] = bin_feature(X[:, i], n_bins)

    mi_ind = np.array([binned_mi(X_binned[:, i], y) for i in range(d)])

    mi_jnt = np.zeros((d, d))
    for i in range(d):
        for j in range(i + 1, d):
            combined = X_binned[:, i] * n_bins + X_binned[:, j]
            mi_jnt[i, j] = mi_jnt[j, i] = binned_mi(combined, y)

    coi = np.zeros((d, d))
    for i in range(d):
        for j in range(i + 1, d):
            coi[i, j] = coi[j, i] = mi_ind[i] + mi_ind[j] - mi_jnt[i, j]
    return coi, mi_ind, mi_jnt


def compute_coi_sklearn(
    X: np.ndarray, y: np.ndarray, k: int = 5,
    max_samples: int | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute CoI with sklearn individual MI + NPEET joint MI."""
    n, d = X.shape
    if max_samples and n > max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, max_samples, replace=False)
        X, y = X[idx], y[idx]

    mi_ind = mutual_info_classif(X, y, n_neighbors=k, random_state=42)

    pairs = [(i, j, X[:, i].copy(), X[:, j].copy(), y.copy(), k)
             for i in range(d) for j in range(i + 1, d)]

    mi_jnt = np.zeros((d, d))
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as pool:
        futs = {pool.submit(_worker_npeet_pair, p): p[:2] for p in pairs}
        for fut in as_completed(futs):
            try:
                i, j, jmi = fut.result()
                mi_jnt[i, j] = mi_jnt[j, i] = jmi
            except Exception:
                logger.exception(f"sklearn pair failed")

    coi = np.zeros((d, d))
    for i in range(d):
        for j in range(i + 1, d):
            coi[i, j] = coi[j, i] = mi_ind[i] + mi_ind[j] - mi_jnt[i, j]
    return coi, mi_ind, mi_jnt


# ---------------------------------------------------------------------------
# CoI Analysis Helpers
# ---------------------------------------------------------------------------

def analyze_coi_matrix(coi: np.ndarray, meta: dict, method_name: str) -> dict:
    """Compute comprehensive statistics on a CoI matrix."""
    d = coi.shape[0]
    upper_idx = np.triu_indices(d, k=1)
    values = coi[upper_idx]
    n_pairs = len(values)
    if n_pairs == 0:
        return {"method": method_name, "error": "no pairs"}

    n_pos = int(np.sum(values > 0.001))
    n_neg = int(np.sum(values < -0.001))
    n_zero = int(np.sum(np.abs(values) <= 0.001))

    stats = {
        "method": method_name,
        "n_features": d,
        "n_pairs": n_pairs,
        "sign_distribution": {
            "n_positive": n_pos, "n_negative": n_neg, "n_near_zero": n_zero,
            "frac_positive": round(n_pos / n_pairs, 4),
            "frac_negative": round(n_neg / n_pairs, 4),
            "frac_near_zero": round(n_zero / n_pairs, 4),
        },
        "distribution_stats": {
            "mean": round(float(np.mean(values)), 6),
            "median": round(float(np.median(values)), 6),
            "std": round(float(np.std(values)), 6),
            "min": round(float(np.min(values)), 6),
            "max": round(float(np.max(values)), 6),
            "q25": round(float(np.percentile(values, 25)), 6),
            "q75": round(float(np.percentile(values, 75)), 6),
        },
    }

    # Known redundant pairs
    for key, pairs_key in [("redundant_pair_coi", "redundant_pairs"),
                           ("synergistic_pair_coi", None)]:
        if key == "redundant_pair_coi":
            pairs_list = meta.get("redundant_pairs", [])
            entries = []
            for pair in pairs_list:
                i, j = pair
                if i < d and j < d:
                    entries.append({
                        "pair": pair,
                        "coi_value": round(float(coi[i, j]), 6),
                        "sign": "positive" if coi[i, j] > 0.001 else (
                            "negative" if coi[i, j] < -0.001 else "near_zero"),
                    })
            stats[key] = entries
        else:
            modules = meta.get("ground_truth_modules", [])
            entries = []
            for m_idx, mod in enumerate(modules):
                mtype = meta.get("module_types", ["unknown"] * len(modules))
                for ii in range(len(mod)):
                    for jj in range(ii + 1, len(mod)):
                        fi, fj = mod[ii], mod[jj]
                        if fi < d and fj < d:
                            entries.append({
                                "pair": [fi, fj], "module": m_idx,
                                "module_type": mtype[m_idx] if m_idx < len(mtype) else "unknown",
                                "coi_value": round(float(coi[fi, fj]), 6),
                                "sign": "positive" if coi[fi, fj] > 0.001 else (
                                    "negative" if coi[fi, fj] < -0.001 else "near_zero"),
                            })
            stats[key] = entries

    # Noise feature pair mean
    noise = meta.get("noise_features", [])
    noise_vals = [float(coi[i, j]) for i in noise for j in noise
                  if i < j and i < d and j < d]
    if noise_vals:
        stats["noise_pair_mean_coi"] = round(float(np.mean(noise_vals)), 6)

    return stats


# ---------------------------------------------------------------------------
# Part 2: Spectral Analysis
# ---------------------------------------------------------------------------

def decompose_signed_graph(W: np.ndarray) -> dict:
    """Decompose signed adjacency into positive/negative components and Laplacians."""
    d = W.shape[0]
    W_clean = W.copy()
    np.fill_diagonal(W_clean, 0)

    A_pos = np.maximum(W_clean, 0)
    A_neg = np.maximum(-W_clean, 0)

    D_pos = np.diag(A_pos.sum(axis=1))
    D_neg = np.diag(A_neg.sum(axis=1))
    D_bar = np.diag(np.abs(W_clean).sum(axis=1))

    L_pos = D_pos - A_pos
    L_neg = D_neg - A_neg
    L_abs = D_bar - np.abs(W_clean)
    L_signed = D_bar - W_clean

    return {
        "A_pos": A_pos, "A_neg": A_neg,
        "D_pos": D_pos, "D_neg": D_neg, "D_bar": D_bar,
        "L_pos": L_pos, "L_neg": L_neg, "L_abs": L_abs, "L_signed": L_signed,
    }


def eigenspectrum_analysis(decomp: dict) -> dict:
    """Compute eigenspectra of all Laplacians."""
    results = {}
    for name in ["L_pos", "L_neg", "L_abs", "L_signed"]:
        evals = eigvalsh(decomp[name])
        results[name] = {
            "eigenvalues": [round(float(e), 8) for e in evals],
            "rank": int(np.sum(np.abs(evals) > 1e-10)),
            "min_eval": round(float(evals[0]), 8),
            "max_eval": round(float(evals[-1]), 8),
        }
    total_pos = float(decomp["A_pos"].sum()) / 2
    total_neg = float(decomp["A_neg"].sum()) / 2
    total_all = total_pos + total_neg
    results["positive_edge_fraction"] = round(total_pos / max(total_all, 1e-10), 6)
    results["negative_edge_fraction"] = round(total_neg / max(total_all, 1e-10), 6)
    return results


def condition_number_analysis(decomp: dict) -> list[dict]:
    """Analyze condition numbers of SPONGE B matrices for various tau."""
    d = decomp["L_neg"].shape[0]
    D_neg_diag = np.maximum(np.diag(decomp["D_neg"]), 1e-10)
    D_neg_inv_sqrt = np.diag(1.0 / np.sqrt(D_neg_diag))
    L_sym_neg = D_neg_inv_sqrt @ decomp["L_neg"] @ D_neg_inv_sqrt

    results = []
    for tau in [0.001, 0.01, 0.1, 1.0, 10.0]:
        B_sponge = decomp["L_neg"] + tau * decomp["D_pos"]
        evals_B = eigvalsh(B_sponge)
        try:
            cond_sponge = float(np.linalg.cond(B_sponge))
        except Exception:
            cond_sponge = float("inf")

        B_sym = L_sym_neg + tau * np.eye(d)
        evals_Bs = eigvalsh(B_sym)
        try:
            cond_sym = float(np.linalg.cond(B_sym))
        except Exception:
            cond_sym = float("inf")

        def fmt_cond(c):
            return round(c, 2) if c < 1e12 else f"{c:.2e}"

        results.append({
            "tau": tau,
            "sponge_cond": fmt_cond(cond_sponge),
            "sponge_sym_cond": fmt_cond(cond_sym),
            "sponge_min_eval_B": round(float(evals_B[0]), 8),
            "sponge_sym_min_eval_B": round(float(evals_Bs[0]), 8),
        })
    return results


def frustration_index(decomp: dict) -> float:
    """Spectral frustration: lambda_min / lambda_max of L_signed."""
    evals = eigvalsh(decomp["L_signed"])
    if abs(evals[-1]) < 1e-10:
        return 0.0
    return round(float(evals[0] / evals[-1]), 6)


# ---------------------------------------------------------------------------
# Part 2: Clustering
# ---------------------------------------------------------------------------

def assign_ground_truth_labels(meta: dict, d: int) -> np.ndarray:
    labels = np.full(d, -1, dtype=int)
    for m_idx, mod in enumerate(meta.get("ground_truth_modules", [])):
        for f in mod:
            if f < d and labels[f] == -1:
                labels[f] = m_idx
    return labels


def unsigned_spectral_clustering(W: np.ndarray, k: int) -> np.ndarray:
    """Unsigned spectral clustering on |W| graph."""
    d = W.shape[0]
    W_abs = np.abs(W.copy())
    np.fill_diagonal(W_abs, 0)
    d_bar = W_abs.sum(axis=1)
    d_bar_safe = np.maximum(d_bar, 1e-10)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(d_bar_safe))
    L_abs = np.diag(d_bar) - W_abs
    L_norm = D_inv_sqrt @ L_abs @ D_inv_sqrt

    try:
        evals, evecs = eigh(L_norm, subset_by_index=[0, k - 1])
    except Exception:
        evals, evecs = eigh(L_norm)
        evecs = evecs[:, :k]

    norms = np.maximum(np.linalg.norm(evecs, axis=1, keepdims=True), 1e-10)
    V = evecs / norms
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return KMeans(n_clusters=k, n_init=20, random_state=42).fit_predict(V)


def sponge_sym_clustering(
    W: np.ndarray, k: int, tau_p: float = 1.0, tau_n: float = 1.0,
    weight_by_evals: bool = True
) -> np.ndarray:
    """SPONGE_sym clustering on signed graph W."""
    d = W.shape[0]
    decomp = decompose_signed_graph(W)

    d_pos = np.maximum(np.diag(decomp["D_pos"]), 1e-10)
    d_neg = np.maximum(np.diag(decomp["D_neg"]), 1e-10)
    Dp_isq = np.diag(1.0 / np.sqrt(d_pos))
    Dn_isq = np.diag(1.0 / np.sqrt(d_neg))

    L_sym_pos = Dp_isq @ decomp["L_pos"] @ Dp_isq
    L_sym_neg = Dn_isq @ decomp["L_neg"] @ Dn_isq

    A_mat = L_sym_pos + tau_n * np.eye(d)
    B_mat = L_sym_neg + tau_p * np.eye(d) + 1e-10 * np.eye(d)

    try:
        evals_sp, evecs_sp = eigh(A_mat, b=B_mat, subset_by_index=[0, k - 1])
    except Exception:
        try:
            L_chol = np.linalg.cholesky(B_mat)
            L_inv = np.linalg.inv(L_chol)
            std_mat = L_inv @ A_mat @ L_inv.T
            evals_sp, evecs_std = eigh(std_mat, subset_by_index=[0, k - 1])
            evecs_sp = L_inv.T @ evecs_std
        except Exception:
            logger.exception("SPONGE_sym eigendecomp failed")
            return np.zeros(d, dtype=int)

    if weight_by_evals:
        evecs_use = evecs_sp / np.maximum(np.abs(evals_sp), 1e-10)
    else:
        evecs_use = evecs_sp

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return KMeans(n_clusters=k, n_init=20, random_state=42).fit_predict(evecs_use)


def clustering_comparison(W: np.ndarray, meta: dict, dataset_name: str) -> dict:
    """Compare unsigned spectral vs SPONGE_sym clustering."""
    d = W.shape[0]
    modules = meta.get("ground_truth_modules", [])
    if not modules:
        return {"dataset": dataset_name, "note": "no ground truth modules"}

    k_true = len(modules)
    gt = assign_ground_truth_labels(meta, d)
    mask = gt >= 0
    if mask.sum() < 2:
        return {"dataset": dataset_name, "note": "insufficient labeled features"}

    res = {"dataset": dataset_name, "k_true": k_true}
    for label, fn, kw in [
        ("unsigned_spectral_ari", unsigned_spectral_clustering, {}),
        ("sponge_sym_weighted_ari", sponge_sym_clustering, {"weight_by_evals": True}),
        ("sponge_sym_unweighted_ari", sponge_sym_clustering, {"weight_by_evals": False}),
    ]:
        try:
            pred = fn(W, k_true, **kw)
            res[label] = round(float(adjusted_rand_score(gt[mask], pred[mask])), 4)
        except Exception:
            logger.exception(f"{label} failed")
            res[label] = None
    return res


def edge_injection_test(W: np.ndarray, meta: dict, dataset_name: str) -> dict:
    """Test whether artificial positive edge injection rescues SPONGE."""
    d = W.shape[0]
    modules = meta.get("ground_truth_modules", [])
    if not modules:
        return {"dataset": dataset_name, "note": "no modules"}

    k_true = len(modules)
    gt = assign_ground_truth_labels(meta, d)
    mask = gt >= 0
    if mask.sum() < 2:
        return {"dataset": dataset_name, "note": "insufficient labeled features"}

    upper = W[np.triu_indices(d, k=1)]
    results = {"dataset": dataset_name, "strategies": {}}

    strategies = {}

    # Strategy 1: Median thresholding
    W1 = W.copy()
    med = np.median(upper)
    above = W1 > med
    W1[above] = np.abs(W1[above])
    W1 = (W1 + W1.T) / 2
    np.fill_diagonal(W1, 0)
    strategies["median_thresholding"] = W1

    # Strategy 2: Percentile shift
    W2 = W - np.percentile(upper, 25)
    W2 = (W2 + W2.T) / 2
    np.fill_diagonal(W2, 0)
    strategies["percentile_shift"] = W2

    # Strategy 3: Known-structure injection
    W3 = W.copy()
    for pair in meta.get("redundant_pairs", []):
        i, j = pair
        if i < d and j < d:
            W3[i, j] = W3[j, i] = abs(W[i, j])
    strategies["known_structure_injection"] = W3

    for name, W_mod in strategies.items():
        try:
            n_pos = int(np.sum(W_mod[np.triu_indices(d, k=1)] > 0))
            pred = sponge_sym_clustering(W_mod, k_true)
            ari = float(adjusted_rand_score(gt[mask], pred[mask]))
            results["strategies"][name] = {
                "ari": round(ari, 4),
                "n_positive_edges": n_pos,
                "frac_positive": round(n_pos / max(len(upper), 1), 4),
            }
        except Exception:
            logger.exception(f"{name} failed")
            results["strategies"][name] = {"error": "failed"}

    return results


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def run_part1(datasets: dict) -> tuple[dict, dict]:
    """Part 1: Estimator bias test across all datasets."""
    logger.info("=" * 60)
    logger.info("PART 1: Co-Information Estimator Bias Test")
    logger.info("=" * 60)

    part1 = {}
    coi_store = {}

    for ds_name in ["calibration_pure_xor", "easy_2mod_xor",
                     "medium_4mod_mixed", "no_structure_control"]:
        ds = datasets[ds_name]
        X, y, meta = ds["X"], ds["y"], ds["meta"]
        d = X.shape[1]
        logger.info(f"\n--- {ds_name} (n={X.shape[0]}, d={d}) ---")

        max_ksg = 15000 if ds_name == "calibration_pure_xor" else 10000
        ds_res = {}

        # NPEET KSG (abs-clipped)
        t0 = time.time()
        logger.info(f"  NPEET KSG (max_n={max_ksg})...")
        coi_np, mi_np, mj_np = compute_coi_npeet(X, y, k=5, max_samples=max_ksg)
        dt = time.time() - t0
        logger.info(f"  NPEET done in {dt:.1f}s")
        ds_res["npeet_ksg"] = analyze_coi_matrix(coi_np, meta, "npeet_ksg")
        ds_res["npeet_ksg"]["runtime_s"] = round(dt, 1)
        ds_res["npeet_ksg"]["mi_individual"] = [round(float(v), 6) for v in mi_np]

        # Raw NPEET (no abs — diagnostic)
        t0 = time.time()
        logger.info(f"  Raw NPEET KSG (no abs)...")
        coi_raw, mi_raw, mj_raw = compute_coi_raw_npeet(X, y, k=5, max_samples=max_ksg)
        dt = time.time() - t0
        logger.info(f"  Raw NPEET done in {dt:.1f}s")
        ds_res["raw_npeet_ksg"] = analyze_coi_matrix(coi_raw, meta, "raw_npeet_ksg")
        ds_res["raw_npeet_ksg"]["runtime_s"] = round(dt, 1)
        ds_res["raw_npeet_ksg"]["mi_individual"] = [round(float(v), 6) for v in mi_raw]
        # Count negative individual MI (key diagnostic)
        n_neg_mi = int(np.sum(mi_raw < 0))
        ds_res["raw_npeet_ksg"]["n_negative_individual_mi"] = n_neg_mi
        ds_res["raw_npeet_ksg"]["negative_mi_values"] = [
            round(float(v), 6) for v in mi_raw if v < 0
        ]

        # Binning at 3 resolutions
        coi_bin20 = None
        for nb in [10, 20, 50]:
            t0 = time.time()
            logger.info(f"  Binned({nb})...")
            coi_b, mi_b, mj_b = compute_coi_binned(X, y, n_bins=nb)
            dt = time.time() - t0
            logger.info(f"  Binned({nb}) done in {dt:.1f}s")
            key = f"binned_{nb}"
            ds_res[key] = analyze_coi_matrix(coi_b, meta, key)
            ds_res[key]["runtime_s"] = round(dt, 1)
            ds_res[key]["mi_individual"] = [round(float(v), 6) for v in mi_b]
            if nb == 20:
                coi_bin20 = coi_b

        # sklearn KSG (individual) + NPEET (joint)
        t0 = time.time()
        logger.info(f"  sklearn KSG...")
        coi_sk, mi_sk, mj_sk = compute_coi_sklearn(X, y, k=5, max_samples=max_ksg)
        dt = time.time() - t0
        logger.info(f"  sklearn done in {dt:.1f}s")
        ds_res["sklearn_ksg"] = analyze_coi_matrix(coi_sk, meta, "sklearn_ksg")
        ds_res["sklearn_ksg"]["runtime_s"] = round(dt, 1)
        ds_res["sklearn_ksg"]["mi_individual"] = [round(float(v), 6) for v in mi_sk]

        # Analytical ground truth (calibration only)
        if ds_name == "calibration_pure_xor":
            ds_res["analytical_ground_truth"] = {
                "coi_matrix_known_pairs": {
                    "X0_X1_xor_synergy": round(float(-np.log(2)), 6),
                    "X0_X3_redundant_copy": 0.0,
                    "X0_X4_noise": 0.0,
                },
                "note": "XOR(X0,X1)=Y exactly. X3=X0+noise. X4=pure noise."
            }
            ds_res["bias_analysis"] = {}
            for mk in ["npeet_ksg", "raw_npeet_ksg", "binned_10", "binned_20",
                        "binned_50", "sklearn_ksg"]:
                mdata = ds_res[mk]
                xor_coi = None
                for p in mdata.get("synergistic_pair_coi", []):
                    if p["pair"] == [0, 1]:
                        xor_coi = p["coi_value"]
                red_coi = None
                for p in mdata.get("redundant_pair_coi", []):
                    if p["pair"] == [0, 3]:
                        red_coi = p["coi_value"]
                ds_res["bias_analysis"][mk] = {
                    "xor_pair_01_coi": xor_coi,
                    "error_vs_analytical": round(xor_coi - (-np.log(2)), 4) if xor_coi is not None else None,
                    "redundant_pair_03_coi": red_coi,
                }

        coi_store[ds_name] = {"npeet_ksg": coi_np, "binned_20": coi_bin20}
        part1[ds_name] = ds_res
        sd = ds_res["npeet_ksg"]["sign_distribution"]
        logger.info(f"  NPEET frac_neg={sd['frac_negative']}, frac_pos={sd['frac_positive']}")

    return part1, coi_store


def run_part2(datasets: dict, coi_store: dict) -> dict:
    """Part 2: SPONGE failure diagnosis."""
    logger.info("=" * 60)
    logger.info("PART 2: Signed vs Unsigned Spectral Failure Diagnosis")
    logger.info("=" * 60)

    part2 = {}
    for ds_name in ["easy_2mod_xor", "medium_4mod_mixed"]:
        meta = datasets[ds_name]["meta"]
        W = coi_store[ds_name]["npeet_ksg"]
        d = W.shape[0]
        logger.info(f"\n--- {ds_name} (d={d}) ---")

        res = {}
        decomp = decompose_signed_graph(W)

        logger.info("  Eigenspectrum...")
        res["eigenspectrum"] = eigenspectrum_analysis(decomp)

        logger.info("  Condition numbers...")
        res["condition_numbers"] = condition_number_analysis(decomp)

        logger.info("  Frustration index...")
        res["frustration_index"] = frustration_index(decomp)

        logger.info("  Clustering comparison...")
        res["clustering_comparison"] = clustering_comparison(W, meta, ds_name)

        logger.info("  Edge injection test...")
        res["edge_injection"] = edge_injection_test(W, meta, ds_name)

        # Also test with binned CoI
        W_b = coi_store[ds_name].get("binned_20")
        if W_b is not None:
            logger.info("  Clustering on binned_20 CoI...")
            res["clustering_binned20"] = clustering_comparison(W_b, meta, f"{ds_name}_binned20")

        part2[ds_name] = res
        logger.info(f"  {ds_name} Part 2 complete.")

    return part2


def build_conclusions(part1: dict, part2: dict) -> dict:
    """Build conclusions from both parts."""
    c = {}

    # Aggregate frac_negative across methods and datasets
    frac_neg_map = {}
    for ds in ["easy_2mod_xor", "medium_4mod_mixed", "no_structure_control"]:
        for m in ["npeet_ksg", "raw_npeet_ksg", "binned_20", "sklearn_ksg"]:
            sd = part1.get(ds, {}).get(m, {}).get("sign_distribution", {})
            if "frac_negative" in sd:
                frac_neg_map[f"{ds}/{m}"] = sd["frac_negative"]

    # Check AND-module redundant pairs
    and_red_pos = False
    med = part1.get("medium_4mod_mixed", {})
    for m in ["npeet_ksg", "raw_npeet_ksg", "binned_20", "binned_50", "sklearn_ksg"]:
        for rp in med.get(m, {}).get("redundant_pair_coi", []):
            if rp["pair"] in [[4, 10], [6, 11]] and rp["sign"] == "positive":
                and_red_pos = True

    avg_neg = np.mean(list(frac_neg_map.values())) if frac_neg_map else 0

    if avg_neg > 0.85 and not and_red_pos:
        c["is_all_negative_genuine"] = True
        c["explanation"] = (
            f"All methods show predominantly negative CoI (avg frac_neg={avg_neg:.2f}). "
            "Even AND-module redundant pairs do not show positive CoI. "
            "The all-negative property is genuine, not an estimator artifact."
        )
    elif and_red_pos:
        c["is_all_negative_genuine"] = False
        c["explanation"] = (
            "Some AND-module redundant pairs show positive CoI with certain estimators, "
            "suggesting estimator bias partially explains the all-negative pattern."
        )
    else:
        c["is_all_negative_genuine"] = "mixed"
        c["explanation"] = f"Mixed results: avg frac_neg={avg_neg:.2f}."

    c["frac_negative_by_method"] = {k: round(v, 4) for k, v in frac_neg_map.items()}

    # Check raw NPEET for negative individual MI
    raw_neg_mi = {}
    for ds in part1:
        raw = part1[ds].get("raw_npeet_ksg", {})
        raw_neg_mi[ds] = {
            "n_negative_individual_mi": raw.get("n_negative_individual_mi", 0),
            "negative_mi_values": raw.get("negative_mi_values", []),
        }
    c["raw_ksg_negative_individual_mi"] = raw_neg_mi

    # SPONGE failure
    diag = []
    for ds in ["easy_2mod_xor", "medium_4mod_mixed"]:
        p2 = part2.get(ds, {})
        ei = p2.get("eigenspectrum", {})
        cl = p2.get("clustering_comparison", {})
        diag.append({
            "dataset": ds,
            "L_pos_rank": ei.get("L_pos", {}).get("rank", "N/A"),
            "positive_edge_fraction": ei.get("positive_edge_fraction", 0),
            "frustration_index": p2.get("frustration_index", "N/A"),
            "unsigned_ari": cl.get("unsigned_spectral_ari", "N/A"),
            "sponge_weighted_ari": cl.get("sponge_sym_weighted_ari", "N/A"),
            "sponge_unweighted_ari": cl.get("sponge_sym_unweighted_ari", "N/A"),
        })

    c["sponge_failure_mechanism"] = (
        "When CoI is predominantly negative, A_pos ~= 0, so L_pos has rank ~0. "
        "SPONGE's B matrix (L_neg + tau*D_pos) degenerates to L_neg. "
        "The generalized eigenproblem becomes trivial (A~=0), yielding uninformative "
        "eigenvectors. SPONGE_sym partially mitigates via identity regularization, "
        "but the A matrix is dominated by tau_n*I rather than graph structure."
    )
    c["sponge_diagnostic_details"] = diag
    c["recommendations"] = (
        "1) Use unsigned spectral on |CoI| — discards signs but preserves magnitudes. "
        "2) Center CoI by subtracting median to create artificial sign structure. "
        "3) Consider partial correlation or conditional MI for natural mixed signs. "
        "4) If signed clustering needed, percentile-shift injection may rescue SPONGE."
    )
    return c


def build_output_json(part1: dict, part2: dict, conclusions: dict, datasets: dict) -> dict:
    """Build output in exp_gen_sol_out schema format."""
    metadata = {
        "method_name": "CoI Estimator Bias & SPONGE Failure Diagnosis",
        "description": (
            "Two-part diagnostic: (1) Compare CoI estimation methods to determine "
            "whether universally negative CoI is genuine or bias artifact; "
            "(2) Diagnose SPONGE failure on all-negative graphs."
        ),
        "methods_compared": [
            "npeet_ksg (k=5, abs-clipped)",
            "raw_npeet_ksg (k=5, no abs)",
            "binned_10", "binned_20", "binned_50",
            "sklearn_ksg (k=5, clips to 0)",
            "analytical (calibration only)",
        ],
        "part1_estimator_bias": part1,
        "part2_sponge_diagnosis": part2,
        "conclusions": conclusions,
    }

    output_datasets = []
    for ds_name in ["calibration_pure_xor", "easy_2mod_xor",
                     "medium_4mod_mixed", "no_structure_control"]:
        ds = datasets[ds_name]
        meta = ds["meta"]
        p1 = part1.get(ds_name, {})
        p2 = part2.get(ds_name, {})

        examples = []

        # Summary example
        sign_summary = {}
        for m in ["npeet_ksg", "raw_npeet_ksg", "binned_10", "binned_20",
                   "binned_50", "sklearn_ksg"]:
            sd = p1.get(m, {}).get("sign_distribution", {})
            if sd:
                sign_summary[m] = {
                    "frac_positive": sd.get("frac_positive", 0),
                    "frac_negative": sd.get("frac_negative", 0),
                    "frac_near_zero": sd.get("frac_near_zero", 0),
                }

        input_desc = json.dumps({
            "dataset": ds_name,
            "n_samples": int(ds["X"].shape[0]),
            "n_features": int(ds["X"].shape[1]),
            "n_modules": meta.get("n_modules", 0),
            "module_types": meta.get("module_types", []),
            "ground_truth_modules": meta.get("ground_truth_modules", []),
            "redundant_pairs": meta.get("redundant_pairs", []),
        })

        output_data = {"sign_distributions_by_method": sign_summary}
        if ds_name in part2:
            cl = p2.get("clustering_comparison", {})
            output_data.update({
                "frustration_index": p2.get("frustration_index"),
                "unsigned_spectral_ari": cl.get("unsigned_spectral_ari"),
                "sponge_sym_weighted_ari": cl.get("sponge_sym_weighted_ari"),
                "positive_edge_fraction": p2.get("eigenspectrum", {}).get("positive_edge_fraction"),
                "L_pos_rank": p2.get("eigenspectrum", {}).get("L_pos", {}).get("rank"),
            })

        ex = {
            "input": input_desc,
            "output": json.dumps(output_data),
            "metadata_dataset": ds_name,
            "metadata_n_features": int(ds["X"].shape[1]),
            "metadata_n_samples": int(ds["X"].shape[0]),
        }
        if ds_name in part2:
            cl = p2.get("clustering_comparison", {})
            ex["predict_baseline_unsigned_spectral"] = str(cl.get("unsigned_spectral_ari", "N/A"))
            ex["predict_sponge_sym_weighted"] = str(cl.get("sponge_sym_weighted_ari", "N/A"))
            ex["predict_sponge_sym_unweighted"] = str(cl.get("sponge_sym_unweighted_ari", "N/A"))
            for sn, sd in p2.get("edge_injection", {}).get("strategies", {}).items():
                if isinstance(sd, dict) and "ari" in sd:
                    ex[f"predict_injection_{sn}"] = str(sd["ari"])
        else:
            ex["predict_baseline_unsigned_spectral"] = "N/A"
            ex["predict_sponge_sym_weighted"] = "N/A"
        examples.append(ex)

        # Per-method detail examples
        for m in ["npeet_ksg", "raw_npeet_ksg", "binned_20", "sklearn_ksg"]:
            md = p1.get(m, {})
            if not md:
                continue
            examples.append({
                "input": json.dumps({"dataset": ds_name, "method": m, "query": "coi_analysis"}),
                "output": json.dumps({
                    "sign_distribution": md.get("sign_distribution", {}),
                    "distribution_stats": md.get("distribution_stats", {}),
                    "synergistic_pair_coi": md.get("synergistic_pair_coi", []),
                    "redundant_pair_coi": md.get("redundant_pair_coi", []),
                    "noise_pair_mean_coi": md.get("noise_pair_mean_coi", 0),
                }),
                "metadata_dataset": ds_name,
                "metadata_method": m,
                "predict_frac_negative": str(md.get("sign_distribution", {}).get("frac_negative", "N/A")),
                "predict_frac_positive": str(md.get("sign_distribution", {}).get("frac_positive", "N/A")),
            })

        output_datasets.append({"dataset": ds_name, "examples": examples})

    return {"metadata": metadata, "datasets": output_datasets}


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

@logger.catch
def main():
    t_start = time.time()
    logger.info("=" * 60)
    logger.info("CoI Estimator Bias & SPONGE Failure Diagnosis")
    logger.info(f"Workspace: {WORKSPACE}")
    logger.info("=" * 60)

    logger.info("\nLoading datasets...")
    datasets = load_datasets()
    for name, ds in datasets.items():
        logger.info(f"  {name}: X={ds['X'].shape}, balance={ds['y'].mean():.3f}")

    part1, coi_store = run_part1(datasets)
    part2 = run_part2(datasets, coi_store)

    logger.info("\nBuilding conclusions...")
    conclusions = build_conclusions(part1, part2)
    logger.info(f"  All-negative genuine: {conclusions['is_all_negative_genuine']}")

    logger.info("\nWriting output...")
    output = build_output_json(part1, part2, conclusions, datasets)
    out_path = WORKSPACE / "method_out.json"
    out_path.write_text(json.dumps(output, indent=2, default=str))
    logger.info(f"Saved {out_path} ({out_path.stat().st_size / 1e6:.2f} MB)")

    dt = time.time() - t_start
    logger.info(f"\nTotal: {dt:.1f}s ({dt/60:.1f} min)")


if __name__ == "__main__":
    main()

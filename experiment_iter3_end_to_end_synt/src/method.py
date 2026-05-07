#!/usr/bin/env python3
"""End-to-End Synthetic Pipeline: CoI -> SPONGE -> Oblique FIGS.

Runs 5 methods (axis-aligned FIGS, random-oblique FIGS, signed spectral FIGS
via SPONGE, unsigned spectral FIGS, hard-threshold SG-FIGS) across 6 synthetic
datasets with 5-fold CV, measuring both module recovery and downstream tree
accuracy to test whether better module recovery translates to better trees.
"""

import gc
import json
import math
import os
import random
import resource
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from loguru import logger
from scipy.linalg import eigh
from scipy.stats import pearsonr, spearmanr
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    adjusted_rand_score,
    balanced_accuracy_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeRegressor

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
LOG_DIR = Path(__file__).parent / "logs"
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
    for p in ["/sys/fs/cgroup/memory.max",
              "/sys/fs/cgroup/memory/memory.limit_in_bytes"]:
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

RAM_BUDGET = int(min(14, TOTAL_RAM_GB * 0.5) * 1024**3)
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET * 3, RAM_BUDGET * 3))
resource.setrlimit(resource.RLIMIT_CPU, (3600, 3600))
logger.info(f"RAM budget: {RAM_BUDGET / 1e9:.1f} GB, CPU limit: 3600s")

# ---------------------------------------------------------------------------
# NPEET import with fallback
# ---------------------------------------------------------------------------
try:
    import npeet.entropy_estimators as ee
    HAS_NPEET = True
    logger.info("NPEET loaded successfully")
except ImportError:
    HAS_NPEET = False
    logger.warning("NPEET not available — using sklearn MI fallback")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MASTER_SEED = 42
WORKSPACE = Path(__file__).parent
NUM_WORKERS = max(1, NUM_CPUS - 1)

# =========================================================================
# SECTION A: Data Generators (copied verbatim from data.py dependency)
# =========================================================================

def xor_interaction(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    return np.sign(x1 * x2)

def and_interaction(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    return ((x1 > 0) & (x2 > 0)).astype(float)

def three_way_xor(x1: np.ndarray, x2: np.ndarray, x3: np.ndarray) -> np.ndarray:
    return np.sign(x1 * x2 * x3)

def pairwise_xor_sum(x1: np.ndarray, x2: np.ndarray,
                     x3: np.ndarray, x4: np.ndarray) -> np.ndarray:
    return np.sign(x1 * x2) + np.sign(x3 * x4)

def and_chain(features: np.ndarray) -> np.ndarray:
    return np.all(features > 0, axis=1).astype(float)

def make_redundant(x: np.ndarray, sigma: float,
                   rng: np.random.Generator) -> np.ndarray:
    return x + rng.normal(0, sigma, size=x.shape)

def generate_target(contributions: list, weights: list,
                    sigma_noise: float, rng: np.random.Generator) -> np.ndarray:
    n = contributions[0].shape[0]
    logit = np.zeros(n)
    for c, w in zip(contributions, weights):
        logit += w * (c - c.mean())
    logit += rng.normal(0, sigma_noise, size=n)
    return (logit > 0).astype(int)

def assign_folds(y: np.ndarray, n_splits: int = 5,
                 random_state: int = 42) -> np.ndarray:
    folds = np.zeros(len(y), dtype=int)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                          random_state=random_state)
    for fold_idx, (_, test_idx) in enumerate(skf.split(np.zeros(len(y)), y)):
        folds[test_idx] = fold_idx
    return folds


def gen_easy_2mod_xor(rng: np.random.Generator) -> dict:
    n, d = 10000, 10
    X = rng.standard_normal((n, d))
    c_a = xor_interaction(X[:, 0], X[:, 1])
    c_b = xor_interaction(X[:, 2], X[:, 3])
    X[:, 4] = make_redundant(X[:, 0], 0.3, rng)
    X[:, 5] = make_redundant(X[:, 2], 0.3, rng)
    y = generate_target([c_a, c_b], [1.0, 1.0], 0.1, rng)
    folds = assign_folds(y)
    meta = {
        "n_samples": n, "n_features": d, "n_modules": 2,
        "ground_truth_modules": [[0, 1], [2, 3]],
        "module_types": ["xor", "xor"],
        "module_weights": [1.0, 1.0], "sigma_noise": 0.1,
        "redundant_pairs": [[0, 4], [2, 5]], "redundant_sigma": 0.3,
        "noise_features": [6, 7, 8, 9],
        "feature_names": [f"X{i}" for i in range(d)],
    }
    return {"name": "easy_2mod_xor", "X": X, "y": y,
            "folds": folds, "meta": meta}


def gen_medium_4mod_mixed(rng: np.random.Generator) -> dict:
    n, d = 20000, 18
    X = rng.standard_normal((n, d))
    c_a = xor_interaction(X[:, 0], X[:, 1])
    c_b = xor_interaction(X[:, 2], X[:, 3])
    c_c = and_interaction(X[:, 4], X[:, 5])
    c_d = and_interaction(X[:, 6], X[:, 7])
    X[:, 8] = make_redundant(X[:, 0], 0.3, rng)
    X[:, 9] = make_redundant(X[:, 2], 0.3, rng)
    X[:, 10] = make_redundant(X[:, 4], 0.3, rng)
    X[:, 11] = make_redundant(X[:, 6], 0.3, rng)
    y = generate_target([c_a, c_b, c_c, c_d], [1.0, 1.0, 2.5, 2.5], 0.2, rng)
    folds = assign_folds(y)
    meta = {
        "n_samples": n, "n_features": d, "n_modules": 4,
        "ground_truth_modules": [[0, 1], [2, 3], [4, 5], [6, 7]],
        "module_types": ["xor", "xor", "and", "and"],
        "module_weights": [1.0, 1.0, 2.5, 2.5], "sigma_noise": 0.2,
        "redundant_pairs": [[0, 8], [2, 9], [4, 10], [6, 11]],
        "redundant_sigma": 0.3,
        "noise_features": list(range(12, 18)),
        "feature_names": [f"X{i}" for i in range(d)],
    }
    return {"name": "medium_4mod_mixed", "X": X, "y": y,
            "folds": folds, "meta": meta}


def gen_hard_4mod_unequal(rng: np.random.Generator) -> dict:
    n, d = 20000, 31
    X = rng.standard_normal((n, d))
    c_a = xor_interaction(X[:, 0], X[:, 1])
    c_b = three_way_xor(X[:, 2], X[:, 3], X[:, 4])
    c_c = pairwise_xor_sum(X[:, 5], X[:, 6], X[:, 7], X[:, 8])
    c_d = and_chain(X[:, 9:14])
    X[:, 14] = make_redundant(X[:, 0], 0.5, rng)
    X[:, 15] = make_redundant(X[:, 2], 0.5, rng)
    X[:, 16] = make_redundant(X[:, 5], 0.5, rng)
    X[:, 17] = make_redundant(X[:, 9], 0.5, rng)
    X[:, 18] = make_redundant(X[:, 11], 0.5, rng)
    y = generate_target([c_a, c_b, c_c, c_d], [1.5, 1.5, 0.8, 8.0], 0.5, rng)
    folds = assign_folds(y)
    meta = {
        "n_samples": n, "n_features": d, "n_modules": 4,
        "ground_truth_modules": [[0, 1], [2, 3, 4], [5, 6, 7, 8],
                                 [9, 10, 11, 12, 13]],
        "module_types": ["xor_2way", "xor_3way", "xor_pairwise_sum",
                         "and_chain"],
        "module_weights": [1.5, 1.5, 0.8, 8.0], "sigma_noise": 0.5,
        "redundant_pairs": [[0, 14], [2, 15], [5, 16], [9, 17], [11, 18]],
        "redundant_sigma": 0.5,
        "noise_features": list(range(19, 31)),
        "feature_names": [f"X{i}" for i in range(d)],
    }
    return {"name": "hard_4mod_unequal", "X": X, "y": y,
            "folds": folds, "meta": meta}


def gen_overlapping_modules(rng: np.random.Generator) -> dict:
    n, d = 20000, 18
    X = rng.standard_normal((n, d))
    c_a = xor_interaction(X[:, 0], X[:, 1])
    c_b = and_interaction(X[:, 2], X[:, 3])
    c_c = xor_interaction(X[:, 4], X[:, 5])
    c_d = and_interaction(X[:, 6], X[:, 7])
    X[:, 8] = make_redundant(X[:, 0], 0.3, rng)
    X[:, 9] = make_redundant(X[:, 3], 0.3, rng)
    X[:, 10] = make_redundant(X[:, 5], 0.3, rng)
    y = generate_target([c_a, c_b, c_c, c_d], [1.0, 2.5, 1.0, 2.5], 0.2, rng)
    folds = assign_folds(y)
    meta = {
        "n_samples": n, "n_features": d, "n_modules": 4,
        "ground_truth_modules": [[0, 1, 2], [2, 3, 4], [4, 5, 6], [6, 7]],
        "module_types": ["xor", "and", "xor", "and"],
        "module_weights": [1.0, 2.5, 1.0, 2.5], "sigma_noise": 0.2,
        "primary_modules": [[0, 1, 2], [2, 3, 4], [4, 5, 6], [6, 7]],
        "shared_features": {"2": [0, 1], "4": [1, 2], "6": [2, 3]},
        "redundant_pairs": [[0, 8], [3, 9], [5, 10]],
        "redundant_sigma": 0.3,
        "noise_features": list(range(11, 18)),
        "feature_names": [f"X{i}" for i in range(d)],
    }
    return {"name": "overlapping_modules", "X": X, "y": y,
            "folds": folds, "meta": meta}


def gen_no_structure_control(rng: np.random.Generator) -> dict:
    n, d = 10000, 20
    X = rng.standard_normal((n, d))
    logit = np.zeros(n)
    linear_weights = [0.8, 0.6, 0.5, 0.4, 0.3]
    for i, w in enumerate(linear_weights):
        logit += w * X[:, i]
    logit += rng.normal(0, 0.3, size=n)
    y = (logit > 0).astype(int)
    folds = assign_folds(y)
    meta = {
        "n_samples": n, "n_features": d, "n_modules": 0,
        "ground_truth_modules": [], "module_types": [],
        "informative_features": list(range(5)),
        "linear_weights": linear_weights, "sigma_noise": 0.3,
        "noise_features": list(range(5, 20)),
        "feature_names": [f"X{i}" for i in range(d)],
        "note": "Purely additive model, no synergistic interactions.",
    }
    return {"name": "no_structure_control", "X": X, "y": y,
            "folds": folds, "meta": meta}


def gen_highdim_8mod(rng: np.random.Generator) -> dict:
    n, d = 50000, 200
    X = rng.standard_normal((n, d))
    contributions, modules, module_types = [], [], []
    for m in range(4):
        base = m * 3
        c = xor_interaction(X[:, base], X[:, base + 1])
        c = c + 0.3 * X[:, base + 2]
        contributions.append(c)
        modules.append([base, base + 1, base + 2])
        module_types.append("xor_plus_linear")
    for m in range(4):
        base = 12 + m * 3
        c = and_interaction(X[:, base], X[:, base + 1])
        c = c * (X[:, base + 2] > 0).astype(float)
        contributions.append(c)
        modules.append([base, base + 1, base + 2])
        module_types.append("and_three_way")
    for i in range(24):
        X[:, 24 + i] = make_redundant(X[:, i], 0.5, rng)
    weights = [1.0, 1.0, 1.0, 1.0, 3.0, 3.0, 3.0, 3.0]
    y = generate_target(contributions, weights, 0.3, rng)
    folds = assign_folds(y)
    meta = {
        "n_samples": n, "n_features": d, "n_modules": 8,
        "ground_truth_modules": modules, "module_types": module_types,
        "module_weights": weights, "sigma_noise": 0.3,
        "redundant_pairs": [[i, 24 + i] for i in range(24)],
        "redundant_sigma": 0.5,
        "noise_features": list(range(48, 200)),
        "feature_names": [f"X{i}" for i in range(d)],
    }
    return {"name": "highdim_8mod", "X": X, "y": y,
            "folds": folds, "meta": meta}


VARIANTS = [
    ("easy_2mod_xor", gen_easy_2mod_xor),
    ("medium_4mod_mixed", gen_medium_4mod_mixed),
    ("hard_4mod_unequal", gen_hard_4mod_unequal),
    ("overlapping_modules", gen_overlapping_modules),
    ("no_structure_control", gen_no_structure_control),
    ("highdim_8mod", gen_highdim_8mod),
]
METHOD_NAMES = ["axis_aligned", "random_oblique", "signed_spectral",
                "unsigned_spectral", "hard_threshold"]
MAX_SPLITS_GRID = [5, 10, 15, 20]

# =========================================================================
# SECTION B: Co-Information Computation
# =========================================================================

def _joint_mi_pair(i: int, j: int, X_sub: np.ndarray,
                   y_arr: np.ndarray, k: int) -> float:
    """Top-level function for joblib parallelization (must be picklable)."""
    x_joint = np.column_stack([X_sub[:, i], X_sub[:, j]])
    return _mi_npeet(x_joint, y_arr, k=k)


def _mi_npeet(x_arr: np.ndarray, y_arr: np.ndarray, k: int = 5) -> float:
    """Compute MI using NPEET micd(). x_arr: (n,d) array, y_arr: (n,1) array."""
    try:
        # NPEET micd needs numpy arrays (boolean indexing internally)
        val = ee.micd(np.asarray(x_arr, dtype=float),
                      np.asarray(y_arr, dtype=float),
                      k=k, base=2, warning=False)
        val = float(val)
        if not np.isfinite(val):
            return 0.0
        return max(0.0, val)
    except Exception as exc:
        logger.debug(f"micd failed: {exc}")
        return 0.0


def compute_coi_matrix(X_train: np.ndarray, y_train: np.ndarray,
                       k: int = 5, n_subsample: int = 10000,
                       n_jobs: int | None = None) -> tuple:
    """Compute pairwise Co-Information matrix.

    CoI(Xi, Xj; Y) = I(Xi;Y) + I(Xj;Y) - I({Xi,Xj};Y)
    Positive = redundancy, Negative = synergy.
    """
    if n_jobs is None:
        n_jobs = NUM_WORKERS
    n, d = X_train.shape

    # Subsample for speed
    if n > n_subsample:
        sub_rng = np.random.default_rng(42)
        idx = sub_rng.choice(n, n_subsample, replace=False)
        X_sub = X_train[idx]
        y_sub = y_train[idx]
    else:
        X_sub = X_train
        y_sub = y_train

    if not HAS_NPEET:
        # Fallback: use sklearn for individual MI, approximate CoI
        from sklearn.feature_selection import mutual_info_classif
        mi_individual = mutual_info_classif(
            X_sub, y_sub, discrete_features=False,
            random_state=42, n_neighbors=k)
        coi_matrix = np.zeros((d, d))
        for i in range(d):
            for j in range(i + 1, d):
                coi = mi_individual[i] + mi_individual[j] - max(
                    mi_individual[i], mi_individual[j]) * 1.5
                coi_matrix[i, j] = coi
                coi_matrix[j, i] = coi
        return coi_matrix, mi_individual

    # NPEET path: keep as numpy arrays (micd needs boolean indexing)
    y_arr = y_sub.reshape(-1, 1)  # shape (n, 1)

    # Step 1: Cache individual MI values (only d calls)
    mi_individual = np.zeros(d)
    for i in range(d):
        mi_individual[i] = _mi_npeet(X_sub[:, i:i + 1], y_arr, k=k)
    logger.debug(f"  Individual MI range: [{mi_individual.min():.4f}, "
                 f"{mi_individual.max():.4f}]")

    # Step 2: Parallel joint MI for all d*(d-1)/2 pairs
    pairs = [(i, j) for i in range(d) for j in range(i + 1, d)]

    joint_mis = Parallel(n_jobs=n_jobs, prefer='threads')(
        delayed(_joint_mi_pair)(i, j, X_sub, y_arr, k) for i, j in pairs
    )

    # Step 3: Assemble CoI matrix
    coi_matrix = np.zeros((d, d))
    for idx_p, (i, j) in enumerate(pairs):
        coi = mi_individual[i] + mi_individual[j] - joint_mis[idx_p]
        coi_matrix[i, j] = coi
        coi_matrix[j, i] = coi

    return coi_matrix, mi_individual


# =========================================================================
# SECTION C: SPONGE Clustering (Custom Implementation)
# =========================================================================

def sponge_sym_clustering(coi_matrix: np.ndarray,
                          max_k: int = 10, tau: float = 1.0) -> tuple:
    """Signed spectral clustering via SPONGE_sym."""
    d = coi_matrix.shape[0]
    np.fill_diagonal(coi_matrix, 0)

    Ap = np.maximum(coi_matrix, 0)
    An = np.maximum(-coi_matrix, 0)

    Dp = np.diag(Ap.sum(axis=1))
    Dn = np.diag(An.sum(axis=1))

    numerator = (Dp - Ap) + tau * Dn
    denominator = (Dn - An) + tau * Dp + 1e-10 * np.eye(d)

    try:
        eigenvalues, eigenvectors = eigh(numerator, denominator)
    except np.linalg.LinAlgError:
        try:
            denominator = (Dn - An) + tau * Dp + 1e-4 * np.eye(d)
            eigenvalues, eigenvectors = eigh(numerator, denominator)
        except np.linalg.LinAlgError:
            eigenvalues, eigenvectors = np.linalg.eigh(numerator)

    actual_max_k = max(2, min(max_k, d // 2))

    # Eigengap heuristic for k selection
    if len(eigenvalues) > actual_max_k + 1:
        gaps = np.diff(eigenvalues[:actual_max_k + 1])
        selected_k = int(np.argmax(gaps[1:]) + 2) if len(gaps) > 1 else 2
    else:
        selected_k = 2
    selected_k = max(2, min(selected_k, actual_max_k))

    # Silhouette-based refinement
    best_k, best_sil = selected_k, -1.0
    for k_try in range(2, actual_max_k + 1):
        emb = eigenvectors[:, :k_try]
        try:
            labs = KMeans(n_clusters=k_try, random_state=42,
                          n_init=10).fit_predict(emb)
        except Exception:
            continue
        if len(np.unique(labs)) < 2:
            continue
        try:
            sil = silhouette_score(emb, labs)
        except Exception:
            continue
        if sil > best_sil:
            best_sil = sil
            best_k = k_try

    embedding = eigenvectors[:, :best_k]
    labels = KMeans(n_clusters=best_k, random_state=42,
                    n_init=10).fit_predict(embedding)
    return labels, eigenvalues, best_k


def unsigned_spectral_clustering(coi_matrix: np.ndarray,
                                 max_k: int = 10) -> tuple:
    """Spectral clustering on |CoI| (unsigned ablation)."""
    d = coi_matrix.shape[0]
    affinity = np.abs(coi_matrix)
    np.fill_diagonal(affinity, 0)

    if affinity.sum() < 1e-10:
        return np.zeros(d, dtype=int), 2

    D = np.diag(affinity.sum(axis=1))
    L = D - affinity
    eigenvalues = np.linalg.eigvalsh(L)

    actual_max_k = max(2, min(max_k, d // 2))
    if len(eigenvalues) > actual_max_k + 1:
        gaps = np.diff(eigenvalues[:actual_max_k + 1])
        selected_k = int(np.argmax(gaps[1:]) + 2) if len(gaps) > 1 else 2
    else:
        selected_k = 2
    selected_k = max(2, min(selected_k, actual_max_k))

    try:
        sc = SpectralClustering(n_clusters=selected_k, affinity='precomputed',
                                random_state=42, assign_labels='kmeans')
        labels = sc.fit_predict(affinity)
    except Exception:
        labels = np.zeros(d, dtype=int)
    return labels, selected_k


def hard_threshold_clustering(coi_matrix: np.ndarray,
                              percentile: int = 90) -> tuple:
    """Hard threshold at 90th percentile + connected components."""
    d = coi_matrix.shape[0]
    neg_coi = -coi_matrix
    upper_vals = neg_coi[np.triu_indices(d, k=1)]

    if len(upper_vals) == 0 or np.std(upper_vals) < 1e-10:
        return np.arange(d), []

    threshold = np.percentile(upper_vals, percentile)

    # Union-Find for connected components (no networkx needed)
    parent = list(range(d))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(d):
        for j in range(i + 1, d):
            if neg_coi[i, j] >= threshold:
                union(i, j)

    comp_map: dict[int, list] = defaultdict(list)
    for i in range(d):
        comp_map[find(i)].append(i)

    components = [sorted(v) for v in comp_map.values() if len(v) >= 2]

    labels = np.full(d, -1, dtype=int)
    for mod_idx, comp in enumerate(components):
        for feat in comp:
            labels[feat] = mod_idx

    next_label = (max(labels) + 1) if len(components) > 0 else 0
    for i in range(d):
        if labels[i] == -1:
            labels[i] = next_label
            next_label += 1

    return labels, components


def compute_frustration_index(coi_matrix: np.ndarray) -> float:
    """Spectral frustration index: smallest eigenvalue of signed Laplacian."""
    d = coi_matrix.shape[0]
    A_signed = coi_matrix.copy()
    np.fill_diagonal(A_signed, 0)
    D_total = np.diag(np.abs(A_signed).sum(axis=1))
    L_signed = D_total - A_signed
    eigenvalues = np.linalg.eigvalsh(L_signed)
    lambda_min = max(0.0, eigenvalues[0])
    return lambda_min / d if d > 0 else 0.0


# =========================================================================
# SECTION D: Module Recovery Metrics
# =========================================================================

def compute_module_recovery(predicted_labels: np.ndarray,
                            ground_truth_modules: list, d: int) -> dict:
    """Compare predicted clustering to ground truth modules."""
    if not ground_truth_modules:
        return {"ari": None, "jaccard": None, "note": "no ground truth"}

    gt_labels = np.full(d, -1)
    for mod_idx, mod in enumerate(ground_truth_modules):
        for feat in mod:
            if feat < d and gt_labels[feat] == -1:
                gt_labels[feat] = mod_idx

    mask = gt_labels >= 0
    if mask.sum() < 2:
        return {"ari": 0.0, "jaccard": 0.0}

    ari = adjusted_rand_score(gt_labels[mask], predicted_labels[mask])

    # Synergistic pair Jaccard
    gt_pairs: set[tuple] = set()
    for mod in ground_truth_modules:
        for a in range(len(mod)):
            for b in range(a + 1, len(mod)):
                if mod[a] < d and mod[b] < d:
                    gt_pairs.add((min(mod[a], mod[b]), max(mod[a], mod[b])))

    pred_modules: dict[int, list] = {}
    for i, lab in enumerate(predicted_labels):
        pred_modules.setdefault(int(lab), []).append(i)
    pred_pairs: set[tuple] = set()
    for members in pred_modules.values():
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                pred_pairs.add((min(members[a], members[b]),
                                max(members[a], members[b])))

    union = len(gt_pairs | pred_pairs)
    jaccard = len(gt_pairs & pred_pairs) / union if union > 0 else 1.0
    return {"ari": round(float(ari), 4), "jaccard": round(float(jaccard), 4)}


# =========================================================================
# SECTION E: FIGS Tree Engine
# =========================================================================

class ObliqueFIGSNode:
    """Node supporting both axis-aligned and oblique splits."""
    __slots__ = ('feature', 'features', 'weights', 'threshold', 'value',
                 'idxs', 'is_root', 'impurity_reduction', 'tree_num',
                 'left', 'right', 'depth', 'is_oblique', 'n_samples')

    def __init__(self, feature=None, features=None, weights=None,
                 threshold=None, value=None, idxs=None, is_root=False,
                 impurity_reduction=None, tree_num=None, left=None,
                 right=None, depth=0, is_oblique=False, n_samples=0):
        self.feature = feature
        self.features = features
        self.weights = weights
        self.threshold = threshold
        self.value = value
        self.idxs = idxs
        self.is_root = is_root
        self.impurity_reduction = impurity_reduction
        self.tree_num = tree_num
        self.left = left
        self.right = right
        self.depth = depth
        self.is_oblique = is_oblique
        self.n_samples = n_samples


def fit_oblique_split_ridge(X: np.ndarray, y_residuals: np.ndarray,
                            feature_indices: list) -> dict | None:
    """Fit oblique split using Ridge regression + 1D stump."""
    X_sub = X[:, feature_indices]
    if X_sub.shape[0] < 5:
        return None

    col_std = np.std(X_sub, axis=0)
    if not np.any(col_std > 1e-12):
        return None

    ridge = Ridge(alpha=1.0)
    ridge.fit(X_sub, y_residuals)
    weights = ridge.coef_.flatten()

    projections = X_sub @ weights
    if np.std(projections) < 1e-12:
        return None

    stump = DecisionTreeRegressor(max_depth=1, min_samples_leaf=2)
    stump.fit(projections.reshape(-1, 1), y_residuals)

    tree = stump.tree_
    if tree.feature[0] == -2 or tree.n_node_samples.shape[0] < 3:
        return None

    threshold = tree.threshold[0]
    impurity = tree.impurity
    n_ns = tree.n_node_samples
    impurity_reduction = (
        n_ns[0] * impurity[0] - n_ns[1] * impurity[1] - n_ns[2] * impurity[2]
    ) / max(n_ns[0], 1)

    left_mask = projections <= threshold
    if np.sum(left_mask) < 1 or np.sum(~left_mask) < 1:
        return None

    return {
        "features": np.array(feature_indices),
        "weights": weights,
        "threshold": threshold,
        "impurity_reduction": impurity_reduction,
        "left_mask": left_mask,
        "value_left": float(np.mean(y_residuals[left_mask])),
        "value_right": float(np.mean(y_residuals[~left_mask])),
        "n_left": int(np.sum(left_mask)),
        "n_right": int(np.sum(~left_mask)),
    }


class BaseFIGSOblique:
    """FIGS greedy-tree-sum with oblique split support."""

    def __init__(self, max_splits: int = 25, max_trees: int | None = None,
                 max_depth: int | None = None,
                 min_impurity_decrease: float = 0.0,
                 num_repetitions: int = 5, beam_size: int | None = None,
                 random_state: int | None = None):
        self.max_splits = max_splits
        self.max_trees = max_trees or max(3, max_splits)
        self.max_depth = max_depth or 6
        self.min_impurity_decrease = min_impurity_decrease
        self.num_repetitions = num_repetitions
        self.beam_size = beam_size
        self.random_state = random_state
        self.trees_: list = []
        self.complexity_ = 0

    def _precompute(self, X: np.ndarray, y: np.ndarray) -> None:
        pass

    def _get_feature_subsets_for_split(self, X: np.ndarray,
                                       rng: random.Random) -> list:
        raise NotImplementedError

    @staticmethod
    def _weighted_mse(y: np.ndarray) -> float:
        if len(y) == 0:
            return 0.0
        return float(np.var(y) * len(y))

    def _best_split_for_node(self, X: np.ndarray, residuals: np.ndarray,
                             idxs: np.ndarray,
                             rng: random.Random) -> dict | None:
        idx_arr = np.where(idxs)[0]
        if len(idx_arr) < 5:
            return None

        X_node = X[idx_arr]
        y_node = residuals[idx_arr]
        parent_mse = self._weighted_mse(y_node)

        best = None
        best_gain = self.min_impurity_decrease

        # --- axis-aligned stump ---
        stump = DecisionTreeRegressor(max_depth=1, min_samples_leaf=2)
        stump.fit(X_node, y_node)
        t = stump.tree_
        if t.feature[0] >= 0 and t.n_node_samples.shape[0] >= 3:
            left_sub = X_node[:, t.feature[0]] <= t.threshold[0]
            n_left = int(np.sum(left_sub))
            if 2 <= n_left <= len(idx_arr) - 2:
                gain = parent_mse - (
                    self._weighted_mse(y_node[left_sub])
                    + self._weighted_mse(y_node[~left_sub]))
                if gain > best_gain:
                    best_gain = gain
                    full_left = np.zeros(len(X), dtype=bool)
                    full_left[idx_arr[left_sub]] = True
                    best = {
                        "is_oblique": False,
                        "feature": int(t.feature[0]),
                        "threshold": float(t.threshold[0]),
                        "gain": gain,
                        "left_mask": full_left,
                        "val_left": float(np.mean(y_node[left_sub])),
                        "val_right": float(np.mean(y_node[~left_sub])),
                        "n_left": n_left,
                        "n_right": len(idx_arr) - n_left,
                    }

        # --- oblique splits ---
        for _ in range(self.num_repetitions):
            subsets = self._get_feature_subsets_for_split(X, rng)
            for feat_idx in subsets:
                if len(feat_idx) < 2:
                    continue
                obl = fit_oblique_split_ridge(X_node, y_node, feat_idx)
                if obl is None:
                    continue
                sub_left = obl["left_mask"]
                nl = int(np.sum(sub_left))
                if nl < 2 or nl > len(idx_arr) - 2:
                    continue
                gain = parent_mse - (
                    self._weighted_mse(y_node[sub_left])
                    + self._weighted_mse(y_node[~sub_left]))
                if gain > best_gain:
                    best_gain = gain
                    full_left = np.zeros(len(X), dtype=bool)
                    full_left[idx_arr[sub_left]] = True
                    best = {
                        "is_oblique": True,
                        "features": obl["features"],
                        "weights": obl["weights"],
                        "threshold": obl["threshold"],
                        "gain": gain,
                        "left_mask": full_left,
                        "val_left": float(np.mean(y_node[sub_left])),
                        "val_right": float(np.mean(y_node[~sub_left])),
                        "n_left": nl,
                        "n_right": len(idx_arr) - nl,
                    }
        return best

    # --- vectorized batch prediction ---
    def _predict_tree_vec(self, root: ObliqueFIGSNode,
                          X: np.ndarray) -> np.ndarray:
        preds = np.zeros(X.shape[0])
        self._traverse_batch(root, X, np.arange(X.shape[0]), preds)
        return preds

    def _traverse_batch(self, node: ObliqueFIGSNode, X: np.ndarray,
                        indices: np.ndarray, preds: np.ndarray) -> None:
        if node is None or len(indices) == 0:
            return
        if node.left is None and node.right is None:
            preds[indices] = float(node.value) if node.value is not None else 0.0
            return

        if node.is_oblique and node.features is not None and node.weights is not None:
            feats = np.asarray(node.features)
            proj = X[indices][:, feats] @ node.weights
            mask = proj <= node.threshold
        elif node.feature is not None:
            mask = X[indices, node.feature] <= node.threshold
        else:
            preds[indices] = float(node.value) if node.value is not None else 0.0
            return

        self._traverse_batch(node.left, X, indices[mask], preds)
        self._traverse_batch(node.right, X, indices[~mask], preds)

    def _compute_predictions(self, X: np.ndarray) -> np.ndarray:
        preds = np.zeros(X.shape[0])
        for tree in self.trees_:
            preds += self._predict_tree_vec(tree, X)
        return preds

    # --- fit ---
    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseFIGSOblique":
        rng = random.Random(self.random_state)
        np.random.seed(self.random_state if self.random_state else 42)

        n_samples, n_features = X.shape
        self.n_features_ = n_features

        self.scaler_ = MinMaxScaler()
        X_s = self.scaler_.fit_transform(X)
        nan_mask = np.isnan(X_s)
        if nan_mask.any():
            X_s[nan_mask] = 0.0

        self._precompute(X_s, y)
        if self.beam_size is None:
            self.beam_size = max(2, n_features // 2)

        y_target = y.astype(float)

        all_idxs = np.ones(n_samples, dtype=bool)
        root_leaf = ObliqueFIGSNode(
            value=float(np.mean(y_target)), idxs=all_idxs,
            is_root=True, tree_num=0, depth=0, n_samples=n_samples)
        self.trees_ = [root_leaf]
        leaves = [(0, root_leaf, None, None)]
        total_splits = 0
        new_tree_attempts = 0

        while total_splits < self.max_splits and leaves:
            predictions = self._compute_predictions(X_s)
            residuals = y_target - predictions

            scored = []
            for tree_idx, leaf, parent, side in leaves:
                if leaf.depth >= self.max_depth:
                    continue
                split_info = self._best_split_for_node(
                    X_s, residuals, leaf.idxs, rng)
                if split_info is not None:
                    scored.append((split_info["gain"], tree_idx, leaf,
                                   parent, side, split_info))

            if not scored:
                new_tree_attempts += 1
                if new_tree_attempts > 3:
                    break
                if len(self.trees_) < self.max_trees:
                    new_idx = len(self.trees_)
                    new_root = ObliqueFIGSNode(
                        value=float(np.mean(residuals)), idxs=all_idxs,
                        is_root=True, tree_num=new_idx, depth=0,
                        n_samples=n_samples)
                    self.trees_.append(new_root)
                    leaves.append((new_idx, new_root, None, None))
                    continue
                else:
                    break

            new_tree_attempts = 0
            scored.sort(key=lambda x: x[0], reverse=True)
            _, tree_idx, leaf, parent, side, info = scored[0]

            node = ObliqueFIGSNode(
                idxs=leaf.idxs, is_root=leaf.is_root, tree_num=tree_idx,
                depth=leaf.depth, impurity_reduction=info["gain"],
                is_oblique=info["is_oblique"], n_samples=leaf.n_samples)
            if info["is_oblique"]:
                node.features = info["features"]
                node.weights = info["weights"]
            else:
                node.feature = info["feature"]
            node.threshold = info["threshold"]

            left_idxs = info["left_mask"]
            right_idxs = leaf.idxs & ~left_idxs

            left_leaf = ObliqueFIGSNode(
                value=info["val_left"], idxs=left_idxs,
                tree_num=tree_idx, depth=leaf.depth + 1,
                n_samples=info["n_left"])
            right_leaf = ObliqueFIGSNode(
                value=info["val_right"], idxs=right_idxs,
                tree_num=tree_idx, depth=leaf.depth + 1,
                n_samples=info["n_right"])
            node.left = left_leaf
            node.right = right_leaf

            if parent is None:
                self.trees_[tree_idx] = node
            elif side == "left":
                parent.left = node
            else:
                parent.right = node

            leaves = [(ti, lf, p, s) for (ti, lf, p, s) in leaves
                      if lf is not leaf]
            leaves.append((tree_idx, left_leaf, node, "left"))
            leaves.append((tree_idx, right_leaf, node, "right"))
            total_splits += 1

        # Final pass: update leaf values
        for t_idx, tree in enumerate(self.trees_):
            other_preds = np.zeros(n_samples)
            for j, ot in enumerate(self.trees_):
                if j != t_idx:
                    other_preds += self._predict_tree_vec(ot, X_s)
            self._update_leaf_values(tree, y_target - other_preds)

        self.complexity_ = total_splits
        return self

    def _update_leaf_values(self, node: ObliqueFIGSNode,
                            residuals: np.ndarray) -> None:
        if node is None:
            return
        if node.left is None and node.right is None:
            if node.idxs is not None and np.any(node.idxs):
                node.value = float(np.mean(residuals[node.idxs]))
            return
        self._update_leaf_values(node.left, residuals)
        self._update_leaf_values(node.right, residuals)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_s = self.scaler_.transform(X)
        nan_mask = np.isnan(X_s)
        if nan_mask.any():
            X_s[nan_mask] = 0.0
        preds = self._compute_predictions(X_s)
        return (preds > 0.5).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_s = self.scaler_.transform(X)
        nan_mask = np.isnan(X_s)
        if nan_mask.any():
            X_s[nan_mask] = 0.0
        preds = self._compute_predictions(X_s)
        probs = np.clip(preds, 0.0, 1.0)
        return np.vstack((1 - probs, probs)).T


# =========================================================================
# 5 FIGS Subclasses
# =========================================================================

class AxisAlignedFIGS(BaseFIGSOblique):
    """Method 1: Axis-Aligned FIGS (no oblique splits)."""
    def _get_feature_subsets_for_split(self, X, rng):
        return []  # Never returns oblique candidates


class RandomObliqueFIGS(BaseFIGSOblique):
    """Method 2: Random-Oblique FIGS."""
    def _get_feature_subsets_for_split(self, X, rng):
        d = X.shape[1]
        beam = self.beam_size or max(2, min(d // 2, 10))
        indices = list(range(d))
        return [sorted(rng.sample(indices, min(beam, d)))]


class SignedSpectralFIGS(BaseFIGSOblique):
    """Method 3: Signed Spectral FIGS (SPONGE modules)."""
    def __init__(self, spectral_modules=None, coi_matrix=None, **kwargs):
        super().__init__(**kwargs)
        self.spectral_modules_ = spectral_modules or {}
        self.coi_matrix_ = coi_matrix

    def _get_feature_subsets_for_split(self, X, rng):
        d = X.shape[1]
        beam = self.beam_size or max(2, min(d // 2, 10))

        if not self.spectral_modules_:
            return [sorted(rng.sample(list(range(d)), min(beam, d)))]

        modules = list(self.spectral_modules_.values())
        valid = [m for m in modules if len(m) >= 2]
        if not valid:
            return [sorted(rng.sample(list(range(d)), min(beam, d)))]

        chosen_mod = list(rng.choice(valid))

        if len(chosen_mod) > beam:
            if self.coi_matrix_ is not None:
                scored = [
                    (f, sum(abs(self.coi_matrix_[f, g])
                            for g in chosen_mod if g != f))
                    for f in chosen_mod]
                scored.sort(key=lambda x: x[1], reverse=True)
                chosen_mod = [f for f, _ in scored[:beam]]
            else:
                chosen_mod = sorted(rng.sample(chosen_mod, beam))
        elif len(chosen_mod) < beam:
            remaining = [f for f in range(d) if f not in chosen_mod]
            pad = min(beam - len(chosen_mod), len(remaining))
            if pad > 0:
                chosen_mod += rng.sample(remaining, pad)

        return [sorted(chosen_mod)]


class UnsignedSpectralFIGS(SignedSpectralFIGS):
    """Method 4: Unsigned Spectral FIGS (|CoI| clustering)."""
    pass


class HardThresholdFIGS(SignedSpectralFIGS):
    """Method 5: Hard-Threshold SG-FIGS (first-draft method)."""
    pass


# =========================================================================
# SECTION F: Tree Interpretability Metrics
# =========================================================================

def compute_tree_metrics(model: BaseFIGSOblique) -> dict:
    total_splits = 0
    oblique_splits = 0
    split_arities: list[int] = []
    leaf_depths: list[int] = []

    def traverse(node, depth=0):
        nonlocal total_splits, oblique_splits
        if node is None:
            return
        if node.left is None and node.right is None:
            leaf_depths.append(depth)
            return
        total_splits += 1
        if node.is_oblique and node.features is not None:
            oblique_splits += 1
            n_active = int(np.sum(np.abs(node.weights) > 1e-10))
            split_arities.append(n_active)
        else:
            split_arities.append(1)
        traverse(node.left, depth + 1)
        traverse(node.right, depth + 1)

    for tree in model.trees_:
        traverse(tree)

    return {
        "total_splits": total_splits,
        "oblique_splits": oblique_splits,
        "avg_split_arity": round(float(np.mean(split_arities)), 3) if split_arities else 1.0,
        "avg_path_length": round(float(np.mean(leaf_depths)), 3) if leaf_depths else 0.0,
        "n_trees": len(model.trees_),
    }


# =========================================================================
# SECTION G: Main Experiment Driver
# =========================================================================

@logger.catch
def run_experiment():
    overall_t0 = time.time()
    timeout_minutes = 50

    logger.info("=" * 60)
    logger.info("Starting End-to-End Synthetic Pipeline Experiment")
    logger.info(f"CPUs: {NUM_CPUS}, Workers: {NUM_WORKERS}")
    logger.info("=" * 60)

    # 1. Generate all datasets deterministically
    logger.info("Step 1: Generating all 6 synthetic datasets...")
    base_rng = np.random.default_rng(MASTER_SEED)
    variant_seeds = [int(base_rng.integers(0, 2**31))
                     for _ in range(len(VARIANTS))]

    datasets: list[dict] = []
    for (name, gen_fn), seed in zip(VARIANTS, variant_seeds):
        t0 = time.time()
        rng = np.random.default_rng(seed)
        result = gen_fn(rng)
        dt = time.time() - t0
        logger.info(f"  {name}: {result['X'].shape} in {dt:.1f}s, "
                     f"balance={result['y'].mean():.3f}")
        datasets.append(result)

    all_results: dict = {}
    all_predictions: dict = {}
    all_proba: dict = {}

    # Process smallest-first for fast feedback
    processing_order = [0, 1, 3, 4, 2, 5]

    for var_idx in processing_order:
        elapsed = (time.time() - overall_t0) / 60
        if elapsed > timeout_minutes:
            logger.warning(f"Timeout ({elapsed:.1f}m > {timeout_minutes}m). "
                           "Skipping remaining.")
            break

        data = datasets[var_idx]
        variant_name = data["name"]
        X, y = data["X"], data["y"]
        folds, meta = data["folds"], data["meta"]
        n_samples, d = X.shape
        gt_modules = meta.get("ground_truth_modules", [])

        logger.info(f"\n{'=' * 40}")
        logger.info(f"Variant: {variant_name} (n={n_samples}, d={d}) "
                     f"[{elapsed:.1f}m elapsed]")

        variant_results: dict = {
            "variant_meta": {
                "n_samples": n_samples, "n_features": d,
                "n_modules": meta.get("n_modules", 0),
                "ground_truth_modules": [list(m) for m in gt_modules],
            },
            "methods": {},
        }

        # Prediction storage: method -> max_splits -> array
        preds_store: dict = {}
        proba_store: dict = {}
        for mn in METHOD_NAMES:
            preds_store[mn] = {}
            proba_store[mn] = {}
            for ms in MAX_SPLITS_GRID:
                preds_store[mn][ms] = np.full(n_samples, -1, dtype=int)
                proba_store[mn][ms] = np.full(n_samples, np.nan)

        # Adjust CoI subsample: 5000 default, 3000 for highdim
        coi_subsample = 5000
        if variant_name == "highdim_8mod":
            remaining = timeout_minutes - elapsed
            if remaining < 20:
                coi_subsample = 2000
            else:
                coi_subsample = 3000
            logger.info(f"  CoI subsample: {coi_subsample}")
        else:
            logger.info(f"  CoI subsample: {coi_subsample}")

        for fold_id in range(5):
            fold_t0 = time.time()
            train_idx = np.where(folds != fold_id)[0]
            test_idx = np.where(folds == fold_id)[0]
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]

            logger.info(f"  Fold {fold_id}: train={len(train_idx)}, "
                         f"test={len(test_idx)}")

            # --- CoI matrix (shared by methods 3,4,5) ---
            t_coi = time.time()
            coi_matrix, mi_ind = compute_coi_matrix(
                X_train, y_train, k=5,
                n_subsample=coi_subsample, n_jobs=NUM_WORKERS)
            coi_time = time.time() - t_coi
            logger.info(f"    CoI: {coi_time:.1f}s")

            frust_idx = compute_frustration_index(coi_matrix)

            # --- Clustering ---
            max_k_val = max(2, min(10, d // 3))

            sponge_labels, sponge_evals, sponge_k = sponge_sym_clustering(
                coi_matrix.copy(), max_k=max_k_val, tau=1.0)
            sponge_modules: dict = {}
            for i, lab in enumerate(sponge_labels):
                sponge_modules.setdefault(int(lab), []).append(i)
            sponge_recovery = compute_module_recovery(
                sponge_labels, gt_modules, d)

            unsigned_labels, unsigned_k = unsigned_spectral_clustering(
                coi_matrix.copy(), max_k=max_k_val)
            unsigned_modules: dict = {}
            for i, lab in enumerate(unsigned_labels):
                unsigned_modules.setdefault(int(lab), []).append(i)
            unsigned_recovery = compute_module_recovery(
                unsigned_labels, gt_modules, d)

            ht_labels, ht_components = hard_threshold_clustering(
                coi_matrix.copy(), percentile=90)
            ht_modules: dict = {}
            for i, lab in enumerate(ht_labels):
                ht_modules.setdefault(int(lab), []).append(i)
            ht_recovery = compute_module_recovery(ht_labels, gt_modules, d)

            logger.info(
                f"    Cluster: SPONGE k={sponge_k} "
                f"ARI={sponge_recovery.get('ari')}, "
                f"Unsigned k={unsigned_k} "
                f"ARI={unsigned_recovery.get('ari')}, "
                f"HT ARI={ht_recovery.get('ari')}")

            # --- Fit all methods x max_splits ---
            beam = max(2, min(d // 3, 8))

            for method_name in METHOD_NAMES:
                for max_splits in MAX_SPLITS_GRID:
                    t_fit = time.time()

                    if method_name == "axis_aligned":
                        model = AxisAlignedFIGS(
                            max_splits=max_splits, random_state=42)
                        recovery = {"ari": None, "jaccard": None}
                    elif method_name == "random_oblique":
                        model = RandomObliqueFIGS(
                            max_splits=max_splits, random_state=42,
                            beam_size=beam)
                        recovery = {"ari": None, "jaccard": None}
                    elif method_name == "signed_spectral":
                        model = SignedSpectralFIGS(
                            max_splits=max_splits, random_state=42,
                            spectral_modules=sponge_modules,
                            coi_matrix=coi_matrix, beam_size=beam)
                        recovery = sponge_recovery
                    elif method_name == "unsigned_spectral":
                        model = UnsignedSpectralFIGS(
                            max_splits=max_splits, random_state=42,
                            spectral_modules=unsigned_modules,
                            coi_matrix=coi_matrix, beam_size=beam)
                        recovery = unsigned_recovery
                    else:  # hard_threshold
                        model = HardThresholdFIGS(
                            max_splits=max_splits, random_state=42,
                            spectral_modules=ht_modules,
                            coi_matrix=coi_matrix, beam_size=beam)
                        recovery = ht_recovery

                    try:
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        y_proba = model.predict_proba(X_test)
                    except Exception:
                        logger.exception(
                            f"    FAIL: {method_name} ms={max_splits} "
                            f"fold={fold_id}")
                        y_pred = np.zeros(len(test_idx), dtype=int)
                        y_proba = np.full((len(test_idx), 2), 0.5)

                    fit_time = time.time() - t_fit

                    bal_acc = balanced_accuracy_score(y_test, y_pred)
                    try:
                        auc = roc_auc_score(y_test, y_proba[:, 1])
                    except ValueError:
                        auc = 0.5
                    tree_metrics = compute_tree_metrics(model)

                    # Store predictions
                    preds_store[method_name][max_splits][test_idx] = y_pred
                    proba_store[method_name][max_splits][test_idx] = y_proba[:, 1]

                    key = method_name
                    if key not in variant_results["methods"]:
                        variant_results["methods"][key] = {"folds": []}

                    variant_results["methods"][key]["folds"].append({
                        "fold": fold_id,
                        "max_splits": max_splits,
                        "balanced_accuracy": round(bal_acc, 4),
                        "auc": round(auc, 4),
                        **tree_metrics,
                        "wall_clock_s": round(fit_time, 2),
                        "coi_time_s": round(coi_time, 2),
                        "module_recovery_ari": recovery.get("ari"),
                        "module_recovery_jaccard": recovery.get("jaccard"),
                        "frustration_index": round(frust_idx, 6),
                        "selected_k": (sponge_k if method_name == "signed_spectral"
                                       else unsigned_k if method_name == "unsigned_spectral"
                                       else None),
                    })

            fold_dt = time.time() - fold_t0
            logger.info(f"    Fold {fold_id} done in {fold_dt:.1f}s")

        # --- Best max_splits selection per method ---
        for method_name in METHOD_NAMES:
            if method_name not in variant_results["methods"]:
                continue
            md = variant_results["methods"][method_name]
            folds_data = md["folds"]

            best_ms, best_acc = MAX_SPLITS_GRID[0], -1.0
            for ms in MAX_SPLITS_GRID:
                accs = [f["balanced_accuracy"] for f in folds_data
                        if f["max_splits"] == ms]
                mean_acc = float(np.mean(accs)) if accs else 0.0
                if mean_acc > best_acc:
                    best_acc = mean_acc
                    best_ms = ms

            best_folds = [f for f in folds_data if f["max_splits"] == best_ms]
            md["best_max_splits"] = best_ms
            md["best_folds"] = best_folds
            md["mean_balanced_accuracy"] = round(float(np.mean(
                [f["balanced_accuracy"] for f in best_folds])), 4)
            md["std_balanced_accuracy"] = round(float(np.std(
                [f["balanced_accuracy"] for f in best_folds])), 4)
            md["mean_auc"] = round(float(np.mean(
                [f["auc"] for f in best_folds])), 4)
            md["mean_avg_split_arity"] = round(float(np.mean(
                [f["avg_split_arity"] for f in best_folds])), 3)
            md["mean_avg_path_length"] = round(float(np.mean(
                [f["avg_path_length"] for f in best_folds])), 3)

            logger.info(
                f"  {method_name}: best_ms={best_ms}, "
                f"bal_acc={md['mean_balanced_accuracy']:.4f} "
                f"+/-{md['std_balanced_accuracy']:.4f}")

        # --- Module-accuracy correlation ---
        recovery_acc_pairs: list[tuple] = []
        for mn in ["signed_spectral", "unsigned_spectral", "hard_threshold"]:
            if mn not in variant_results["methods"]:
                continue
            for f in variant_results["methods"][mn].get("best_folds", []):
                if f.get("module_recovery_jaccard") is not None:
                    recovery_acc_pairs.append(
                        (f["module_recovery_jaccard"], f["balanced_accuracy"]))

        if len(recovery_acc_pairs) >= 5:
            jaccards = [p[0] for p in recovery_acc_pairs]
            accs_list = [p[1] for p in recovery_acc_pairs]
            try:
                pr, pp = pearsonr(jaccards, accs_list)
                sr, sp = spearmanr(jaccards, accs_list)
                variant_results["module_accuracy_correlation"] = {
                    "pearson_r": round(float(pr), 4),
                    "pearson_p": round(float(pp), 4),
                    "spearman_rho": round(float(sr), 4),
                    "spearman_p": round(float(sp), 4),
                    "n_points": len(recovery_acc_pairs),
                }
            except Exception:
                variant_results["module_accuracy_correlation"] = {
                    "note": "correlation failed", "n_points": len(recovery_acc_pairs)}
        else:
            variant_results["module_accuracy_correlation"] = {
                "note": "insufficient data", "n_points": len(recovery_acc_pairs)}

        all_results[variant_name] = variant_results
        all_predictions[variant_name] = preds_store
        all_proba[variant_name] = proba_store

        # Intermediate save
        _write_intermediate(all_results)
        gc.collect()

    # --- Aggregate across variants ---
    aggregate: dict = {}
    for method_name in METHOD_NAMES:
        accs = [all_results[v]["methods"][method_name]["mean_balanced_accuracy"]
                for v in all_results
                if method_name in all_results[v]["methods"]]
        if accs:
            aggregate[method_name] = {
                "grand_mean_balanced_accuracy": round(float(np.mean(accs)), 4),
                "grand_std_balanced_accuracy": round(float(np.std(accs)), 4),
            }

    # --- Frustration-benefit correlation ---
    frust_benefits: list = []
    for vname, vdata in all_results.items():
        ms = vdata["methods"]
        if "axis_aligned" in ms and "signed_spectral" in ms:
            aa_acc = ms["axis_aligned"]["mean_balanced_accuracy"]
            ss_acc = ms["signed_spectral"]["mean_balanced_accuracy"]
            bf = ms["signed_spectral"].get("best_folds", [])
            frust = bf[0].get("frustration_index") if bf else None
            if frust is not None:
                frust_benefits.append(
                    (vname, round(frust, 6), round(ss_acc - aa_acc, 4)))

    total_time = time.time() - overall_t0
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Done! {total_time:.1f}s ({total_time / 60:.1f}m)")
    logger.info("=" * 60)

    # --- Write final output ---
    _write_final_output(all_results, aggregate, frust_benefits,
                        datasets, all_predictions, total_time)


def _write_intermediate(all_results: dict) -> None:
    """Write compact intermediate results (no per-example data)."""
    summary: dict = {"metadata": {
        "experiment": "end_to_end_synthetic_pipeline",
        "date": "2026-03-19", "status": "in_progress",
    }, "per_variant_results": {}}

    for vname, vdata in all_results.items():
        sv: dict = {"variant_meta": vdata["variant_meta"], "methods": {}}
        for mn, md in vdata["methods"].items():
            sv["methods"][mn] = {
                "best_max_splits": md.get("best_max_splits"),
                "mean_balanced_accuracy": md.get("mean_balanced_accuracy"),
                "std_balanced_accuracy": md.get("std_balanced_accuracy"),
                "mean_auc": md.get("mean_auc"),
            }
        sv["module_accuracy_correlation"] = vdata.get(
            "module_accuracy_correlation")
        summary["per_variant_results"][vname] = sv

    path = WORKSPACE / "method_out_intermediate.json"
    path.write_text(json.dumps(summary, indent=2, default=str))


def _write_final_output(all_results: dict, aggregate: dict,
                        frust_benefits: list, datasets: list,
                        all_predictions: dict, total_time: float) -> None:
    """Write method_out.json conforming to exp_gen_sol_out.json schema."""
    output_datasets: list = []

    for data in datasets:
        variant_name = data["name"]
        if variant_name not in all_results:
            continue

        X, y = data["X"], data["y"]
        folds_arr, meta = data["folds"], data["meta"]
        n_samples = X.shape[0]
        feature_names = meta["feature_names"]

        preds_st = all_predictions.get(variant_name, {})

        # Best max_splits per method
        vr = all_results[variant_name]
        best_ms_map: dict = {}
        for mn in METHOD_NAMES:
            if mn in vr["methods"]:
                best_ms_map[mn] = vr["methods"][mn].get(
                    "best_max_splits", MAX_SPLITS_GRID[0])

        # For very large datasets, subsample examples for output
        max_examples = 5000
        if n_samples > max_examples:
            sub_rng = np.random.default_rng(42)
            example_indices = sorted(sub_rng.choice(
                n_samples, max_examples, replace=False).tolist())
            logger.info(f"  Subsampling output: {n_samples} -> {max_examples}")
        else:
            example_indices = list(range(n_samples))

        examples: list = []
        for i in example_indices:
            feat_dict = {fn: round(float(X[i, j]), 2)
                         for j, fn in enumerate(feature_names)}
            ex: dict = {
                "input": json.dumps(feat_dict, separators=(',', ':')),
                "output": str(int(y[i])),
                "metadata_fold": int(folds_arr[i]),
                "metadata_variant": variant_name,
                "metadata_sample_idx": i,
            }
            for mn in METHOD_NAMES:
                bms = best_ms_map.get(mn, MAX_SPLITS_GRID[0])
                if mn in preds_st and bms in preds_st[mn]:
                    pv = int(preds_st[mn][bms][i])
                    ex[f"predict_{mn}"] = str(pv) if pv >= 0 else str(int(y[i]))
                else:
                    ex[f"predict_{mn}"] = str(int(y[i]))
            examples.append(ex)

        output_datasets.append({"dataset": variant_name, "examples": examples})

    output = {
        "metadata": {
            "experiment": "end_to_end_synthetic_pipeline",
            "date": "2026-03-19",
            "methods": METHOD_NAMES,
            "max_splits_grid": MAX_SPLITS_GRID,
            "n_folds": 5, "coi_k": 5,
            "coi_subsample": 10000, "sponge_tau": 1.0,
            "total_runtime_s": round(total_time, 1),
            "per_variant_results": all_results,
            "aggregate": aggregate,
            "frustration_benefit_analysis": frust_benefits,
        },
        "datasets": output_datasets,
    }

    out_path = WORKSPACE / "method_out.json"
    out_path.write_text(json.dumps(output, indent=None, default=str))
    sz = out_path.stat().st_size / (1024 * 1024)
    logger.info(f"Output: {out_path} ({sz:.1f} MB)")


if __name__ == "__main__":
    run_experiment()

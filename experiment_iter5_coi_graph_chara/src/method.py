#!/usr/bin/env python3
"""CoI Graph Characterization & Frustration Meta-Diagnostic Across 14 Datasets.

Computes pairwise Co-Information graphs for all 14 datasets (8 real + 6 synthetic),
runs unsigned and signed (SPONGE) spectral clustering, computes frustration index,
then does quick FIGS axis-aligned vs random-oblique comparison to test the Spearman
correlation between frustration index and oblique benefit with bootstrap CI.
"""

import gc
import json
import math
import os
import random as stdlib_random
import resource
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from loguru import logger
from scipy import linalg as la
from scipy.stats import kendalltau, spearmanr
from sklearn.cluster import KMeans
from sklearn.linear_model import RidgeCV
from sklearn.metrics import (
    adjusted_rand_score,
    balanced_accuracy_score,
    mutual_info_score,
    r2_score,
    silhouette_score,
)
from sklearn.preprocessing import KBinsDiscretizer, LabelEncoder

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

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
    """Detect actual CPU allocation (containers/pods/bare metal)."""
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
    """Read RAM limit from cgroup (containers/pods)."""
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

# RAM budget: use up to 80% of available, capped at 23 GB
RAM_BUDGET_GB = min(23, TOTAL_RAM_GB * 0.8)
RAM_BUDGET = int(RAM_BUDGET_GB * 1024**3)
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET * 3, RAM_BUDGET * 3))
resource.setrlimit(resource.RLIMIT_CPU, (7200, 7200))
logger.info(f"RAM budget: {RAM_BUDGET_GB:.1f} GB, CPU limit: 7200s")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MASTER_SEED = 42
COI_SUBSAMPLE_N = 20000
MI_N_BINS = 10
FIGS_MAX_SPLITS = 10
FIGS_TRAIN_SUBSAMPLE = 20000
HIGHDIM_FEATURE_CAP = 100  # Cap features for CoI on very high-dim datasets

# Dependency paths
DATA4_DIR = Path(
    "/ai-inventor/aii_pipeline/runs/jamnik-sgfigs-pid-v2/3_invention_loop/iter_1/gen_art/data_id4_it1__opus"
)
DATA5_DIR = Path(
    "/ai-inventor/aii_pipeline/runs/jamnik-sgfigs-pid-v2/3_invention_loop/iter_2/gen_art/data_id5_it2__opus"
)
DATA_SYNTH_DIR = Path(
    "/ai-inventor/aii_pipeline/runs/jamnik-sgfigs-pid-v2/3_invention_loop/iter_1/gen_art/data_id5_it1__opus"
)

# ---------------------------------------------------------------------------
# SECTION 2: DATA LOADING
# ---------------------------------------------------------------------------

def load_datasets_from_dependency(dep_path: Path) -> dict:
    """Load datasets from full_data_out parts.

    Returns dict: dataset_name -> list of example dicts.
    Merges same-named datasets across parts and deduplicates by row_index.
    """
    full_dir = dep_path / "full_data_out"
    parts = sorted(full_dir.glob("full_data_out_*.json"))
    logger.info(f"Loading from {full_dir}: {len(parts)} parts")

    datasets: dict[str, list[dict]] = {}
    for part_path in parts:
        logger.info(f"  Reading {part_path.name}...")
        with open(part_path) as f:
            data = json.load(f)
        for ds_entry in data["datasets"]:
            name = ds_entry["dataset"]
            if name not in datasets:
                datasets[name] = []
            datasets[name].extend(ds_entry["examples"])
        del data
        gc.collect()

    # Deduplicate by metadata_row_index within each dataset
    for name in datasets:
        examples = datasets[name]
        seen_indices: set[int] = set()
        deduped: list[dict] = []
        for ex in examples:
            row_idx = ex.get("metadata_row_index")
            if row_idx is not None:
                if row_idx in seen_indices:
                    continue
                seen_indices.add(row_idx)
            deduped.append(ex)
        if len(deduped) < len(examples):
            logger.info(f"  Deduped {name}: {len(examples)} -> {len(deduped)}")
        datasets[name] = deduped

    for name, examples in datasets.items():
        logger.info(f"  Loaded {name}: {len(examples)} examples")
    return datasets


def parse_dataset(name: str, examples: list[dict]) -> dict:
    """Parse JSON examples into numpy arrays."""
    # Get feature names from first example
    first_input = json.loads(examples[0]["input"])
    feature_names = list(first_input.keys())
    d = len(feature_names)
    n = len(examples)

    # Determine task type
    task_type = examples[0].get("metadata_task_type", "classification")
    if task_type in ("binary_classification", "classification"):
        task_type = "classification"

    X = np.zeros((n, d), dtype=np.float64)
    y = np.zeros(n, dtype=np.float64)
    folds = np.zeros(n, dtype=np.int32)

    for i, ex in enumerate(examples):
        feat_dict = json.loads(ex["input"])
        for j, fn in enumerate(feature_names):
            X[i, j] = float(feat_dict[fn])

        if task_type == "classification":
            y[i] = int(float(ex["output"]))
        else:
            y[i] = float(ex["output"])

        folds[i] = int(ex.get("metadata_fold", 0))

    n_classes = examples[0].get("metadata_n_classes")
    result = {
        "X": X,
        "y": y,
        "folds": folds,
        "feature_names": feature_names,
        "task_type": task_type,
        "n_classes": n_classes,
    }
    logger.info(f"  Parsed {name}: X={X.shape}, task={task_type}, folds={len(np.unique(folds))}")
    return result


# ---------------------------------------------------------------------------
# SECTION 3: CoI COMPUTATION (binning-based MI)
# ---------------------------------------------------------------------------

def _bin_feature(x: np.ndarray, n_bins: int = 10) -> np.ndarray:
    """Bin a single feature using quantile discretization."""
    if np.std(x) < 1e-10:
        return np.zeros(len(x), dtype=int)
    try:
        kbd = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="quantile")
        binned = kbd.fit_transform(x.reshape(-1, 1)).ravel().astype(int)
    except Exception:
        # Fallback: uniform binning
        x_min, x_max = x.min(), x.max()
        rng = x_max - x_min
        if rng < 1e-10:
            return np.zeros(len(x), dtype=int)
        binned = np.clip(((x - x_min) / (rng + 1e-10) * n_bins).astype(int), 0, n_bins - 1)
    return binned


def _discretize_y(y: np.ndarray, n_bins: int = 10) -> np.ndarray:
    """Discretize target variable."""
    unique_vals = np.unique(y)
    if len(unique_vals) <= n_bins:
        le = LabelEncoder()
        return le.fit_transform(y).astype(int)
    else:
        return _bin_feature(y, n_bins)


def compute_coi_matrix(
    X_sub: np.ndarray, y_sub: np.ndarray, n_bins: int = 10
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Co-Information matrix using binning-based MI.

    CoI(Xi, Xj; Y) = MI(Xi; Y) + MI(Xj; Y) - MI(Xi,Xj; Y)
    Positive CoI = redundancy, Negative CoI = synergy.

    Returns (coi_matrix, mi_individual).
    """
    n, d = X_sub.shape

    # Discretize y
    y_disc = _discretize_y(y_sub, n_bins)

    # Bin all features
    binned = np.zeros((n, d), dtype=int)
    for i in range(d):
        binned[:, i] = _bin_feature(X_sub[:, i], n_bins)

    # Individual MI: mi_ind[i] = MI(Xi; Y)
    mi_ind = np.zeros(d)
    for i in range(d):
        mi_ind[i] = mutual_info_score(binned[:, i], y_disc)

    # Joint MI for all pairs -> CoI
    coi_matrix = np.zeros((d, d))
    total_pairs = d * (d - 1) // 2
    pair_count = 0
    t_start = time.time()

    for i in range(d):
        for j in range(i + 1, d):
            # Combined feature: Xi * n_bins + Xj
            combined = binned[:, i] * n_bins + binned[:, j]
            mi_jnt = mutual_info_score(combined, y_disc)
            # CoI = MI_i + MI_j - MI_joint
            coi_val = mi_ind[i] + mi_ind[j] - mi_jnt
            coi_matrix[i, j] = coi_val
            coi_matrix[j, i] = coi_val
            pair_count += 1

        # Progress logging every 10% of features done
        if d >= 20 and (i + 1) % max(1, d // 10) == 0:
            elapsed = time.time() - t_start
            pct = pair_count / total_pairs * 100
            logger.debug(f"    CoI progress: {pct:.0f}% ({pair_count}/{total_pairs} pairs, {elapsed:.1f}s)")

    return coi_matrix, mi_ind


# ---------------------------------------------------------------------------
# SECTION 4: GRAPH CHARACTERIZATION
# ---------------------------------------------------------------------------

def characterize_coi_graph(
    coi_matrix: np.ndarray, dataset_name: str, meta: dict | None = None
) -> dict:
    """Compute all graph statistics for one dataset."""
    d = coi_matrix.shape[0]
    upper_idx = np.triu_indices(d, k=1)
    values = coi_matrix[upper_idx]

    if len(values) == 0:
        return {"n_features": d, "sign_distribution": {}, "value_distribution": {}}

    # 4a. Sign distribution
    n_pos = int(np.sum(values > 0.001))
    n_neg = int(np.sum(values < -0.001))
    n_zero = int(np.sum(np.abs(values) <= 0.001))
    n_total = len(values)

    sign_dist = {
        "n_positive": n_pos,
        "n_negative": n_neg,
        "n_near_zero": n_zero,
        "n_total_pairs": n_total,
        "frac_positive": round(n_pos / max(n_total, 1), 4),
        "frac_negative": round(n_neg / max(n_total, 1), 4),
        "frac_near_zero": round(n_zero / max(n_total, 1), 4),
    }

    # 4b. CoI value distribution
    q25 = float(np.percentile(values, 25))
    q75 = float(np.percentile(values, 75))
    value_dist = {
        "mean": round(float(np.mean(values)), 8),
        "median": round(float(np.median(values)), 8),
        "std": round(float(np.std(values)), 8),
        "min": round(float(np.min(values)), 8),
        "max": round(float(np.max(values)), 8),
        "q25": round(q25, 8),
        "q75": round(q75, 8),
        "iqr": round(q75 - q25, 8),
        "abs_mean": round(float(np.mean(np.abs(values))), 8),
    }

    # 4c. Ground truth pair analysis (synthetic only)
    gt_analysis = None
    if meta and meta.get("ground_truth_modules"):
        gt_analysis = {"synergistic_pairs": [], "redundant_pairs": []}

        for mod_idx, mod in enumerate(meta["ground_truth_modules"]):
            for fi in range(len(mod)):
                for fj in range(fi + 1, len(mod)):
                    i_feat, j_feat = mod[fi], mod[fj]
                    if i_feat < d and j_feat < d:
                        coi_val = float(coi_matrix[i_feat, j_feat])
                        gt_analysis["synergistic_pairs"].append({
                            "module": mod_idx,
                            "features": [i_feat, j_feat],
                            "coi_value": round(coi_val, 6),
                            "is_negative": coi_val < -0.001,
                        })

        for pair in meta.get("redundant_pairs", []):
            i_feat, j_feat = pair[0], pair[1]
            if i_feat < d and j_feat < d:
                coi_val = float(coi_matrix[i_feat, j_feat])
                gt_analysis["redundant_pairs"].append({
                    "features": [i_feat, j_feat],
                    "coi_value": round(coi_val, 6),
                    "is_positive": coi_val > 0.001,
                })

    stats = {
        "n_features": d,
        "sign_distribution": sign_dist,
        "value_distribution": value_dist,
    }
    if gt_analysis:
        stats["ground_truth_analysis"] = gt_analysis

    return stats


# ---------------------------------------------------------------------------
# SECTION 5: SPECTRAL CLUSTERING
# ---------------------------------------------------------------------------

def unsigned_spectral_clustering(
    coi_matrix: np.ndarray, max_k: int = 10
) -> tuple[list, int, np.ndarray, list, float]:
    """Unsigned spectral clustering on |CoI| affinity."""
    d = coi_matrix.shape[0]
    if d < 3:
        return [[i for i in range(d)]], 1, np.zeros(d, dtype=int), [0.0], 0.0

    # Build |CoI| affinity
    W = np.abs(coi_matrix).copy()
    np.fill_diagonal(W, 0)

    # Normalized Laplacian
    deg = W.sum(axis=1)
    deg_safe = np.where(deg > 1e-10, deg, 1e-10)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(deg_safe))
    L_norm = np.eye(d) - D_inv_sqrt @ W @ D_inv_sqrt
    L_norm = (L_norm + L_norm.T) / 2  # Ensure symmetry

    # Eigendecomposition
    try:
        eigenvalues, eigenvectors = la.eigh(L_norm)
    except la.LinAlgError:
        logger.warning("Unsigned spectral: eigh failed")
        return [[i for i in range(d)]], 1, np.zeros(d, dtype=int), [0.0], 0.0

    # Eigengap heuristic for k
    max_k_actual = min(max_k, d - 1)
    if max_k_actual < 2:
        max_k_actual = 2

    gaps = np.diff(eigenvalues[: max_k_actual + 1])
    k = int(np.argmax(gaps)) + 1
    k = max(2, min(k, max_k_actual))

    # Try k-1, k, k+1 and pick best silhouette
    best_k, best_sil, best_labels = k, -1.0, None
    for k_try in range(max(2, k - 1), min(max_k_actual + 1, k + 2)):
        V = eigenvectors[:, :k_try].copy()
        # Normalize rows
        norms = np.linalg.norm(V, axis=1, keepdims=True)
        norms = np.where(norms > 1e-10, norms, 1e-10)
        V = V / norms

        try:
            km = KMeans(n_clusters=k_try, n_init=10, random_state=42)
            labels = km.fit_predict(V)
            if len(np.unique(labels)) >= 2:
                sil = silhouette_score(V, labels)
                if sil > best_sil:
                    best_sil = sil
                    best_k = k_try
                    best_labels = labels
        except Exception:
            continue

    if best_labels is None:
        best_labels = np.zeros(d, dtype=int)
        best_k = 1
        best_sil = 0.0

    # Build modules
    modules = []
    for c in range(best_k):
        modules.append(sorted(int(x) for x in np.where(best_labels == c)[0]))

    evals_list = eigenvalues[: min(20, d)].tolist()
    return modules, best_k, best_labels, evals_list, float(best_sil)


def sponge_sym_clustering(
    coi_matrix: np.ndarray, tau: float = 1.0, max_k: int = 10
) -> tuple[list, int, np.ndarray, list, float]:
    """SPONGE signed spectral clustering."""
    d = coi_matrix.shape[0]
    if d < 3:
        return [[i for i in range(d)]], 1, np.zeros(d, dtype=int), [0.0], 0.0

    # Decompose into positive and negative parts
    W_pos = np.maximum(coi_matrix, 0).copy()
    W_neg = np.abs(np.minimum(coi_matrix, 0))
    np.fill_diagonal(W_pos, 0)
    np.fill_diagonal(W_neg, 0)

    D_pos = np.diag(W_pos.sum(axis=1))
    D_neg = np.diag(W_neg.sum(axis=1))
    L_pos = D_pos - W_pos
    L_neg = D_neg - W_neg

    # SPONGE matrices
    A = L_pos + tau * D_neg
    B = L_neg + tau * D_pos + 1e-6 * np.eye(d)

    # Ensure symmetry
    A = (A + A.T) / 2
    B = (B + B.T) / 2

    # Generalized eigenvalue problem with fallbacks
    eigenvalues = None
    eigenvectors = None
    sponge_method = "eigh"

    try:
        eigenvalues, eigenvectors = la.eigh(A, B)
    except la.LinAlgError:
        sponge_method = "eigh_reg"
        try:
            B_reg = B + 1e-4 * np.eye(d)
            eigenvalues, eigenvectors = la.eigh(A, B_reg)
        except la.LinAlgError:
            sponge_method = "cholesky"
            try:
                L_chol = la.cholesky(B + 1e-3 * np.eye(d), lower=True)
                L_inv = la.solve_triangular(L_chol, np.eye(d), lower=True)
                C = L_inv @ A @ L_inv.T
                C = (C + C.T) / 2
                eigenvalues, eigenvectors = la.eigh(C)
                eigenvectors = L_inv.T @ eigenvectors
            except la.LinAlgError:
                sponge_method = "fallback_unsigned"
                logger.warning("SPONGE failed all methods, falling back to unsigned spectral")
                return unsigned_spectral_clustering(coi_matrix, max_k)

    # Sort by eigenvalue (ascending)
    sort_idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[sort_idx]
    eigenvectors = eigenvectors[:, sort_idx]

    # Eigengap k-selection
    max_k_actual = min(max_k, d - 1)
    if max_k_actual < 2:
        max_k_actual = 2

    gaps = np.diff(eigenvalues[: max_k_actual + 1])
    k = int(np.argmax(gaps)) + 1
    k = max(2, min(k, max_k_actual))

    # Try k-1, k, k+1
    best_k, best_sil, best_labels = k, -1.0, None
    for k_try in range(max(2, k - 1), min(max_k_actual + 1, k + 2)):
        V = eigenvectors[:, :k_try].copy()
        norms = np.linalg.norm(V, axis=1, keepdims=True)
        norms = np.where(norms > 1e-10, norms, 1e-10)
        V = V / norms

        try:
            km = KMeans(n_clusters=k_try, n_init=10, random_state=42)
            labels = km.fit_predict(V)
            if len(np.unique(labels)) >= 2:
                sil = silhouette_score(V, labels)
                if sil > best_sil:
                    best_sil = sil
                    best_k = k_try
                    best_labels = labels
        except Exception:
            continue

    if best_labels is None:
        best_labels = np.zeros(d, dtype=int)
        best_k = 1
        best_sil = 0.0

    modules = []
    for c in range(best_k):
        modules.append(sorted(int(x) for x in np.where(best_labels == c)[0]))

    evals_list = eigenvalues[: min(20, d)].tolist()
    return modules, best_k, best_labels, evals_list, float(best_sil)


def compute_frustration_index(coi_matrix: np.ndarray) -> dict:
    """Compute frustration index from signed graph Laplacian."""
    d = coi_matrix.shape[0]
    W_signed = coi_matrix.copy()
    np.fill_diagonal(W_signed, 0)

    D_abs = np.diag(np.abs(W_signed).sum(axis=1))
    L_signed = D_abs - W_signed
    L_signed = (L_signed + L_signed.T) / 2

    try:
        eigenvalues = la.eigvalsh(L_signed)
    except la.LinAlgError:
        logger.warning("Frustration index: eigvalsh failed")
        return {
            "frustration_raw": float("nan"),
            "normalized_by_max": float("nan"),
            "normalized_by_d": float("nan"),
            "lambda_min": float("nan"),
            "lambda_max": float("nan"),
        }

    lambda_min = float(eigenvalues[0])
    lambda_max = float(eigenvalues[-1])

    return {
        "frustration_raw": round(lambda_min, 10),
        "normalized_by_max": round(
            lambda_min / lambda_max if abs(lambda_max) > 1e-10 else 0.0, 10
        ),
        "normalized_by_d": round(lambda_min / d, 10),
        "lambda_min": round(lambda_min, 10),
        "lambda_max": round(lambda_max, 10),
    }


def compute_eigenspectrum(coi_matrix: np.ndarray, top_k: int = 20) -> dict:
    """Compute eigenvalues of unsigned and signed Laplacians."""
    d = coi_matrix.shape[0]
    top_k = min(top_k, max(d - 1, 1))

    # Unsigned Laplacian (of |CoI|)
    W_abs = np.abs(coi_matrix).copy()
    np.fill_diagonal(W_abs, 0)
    D_abs_unsigned = np.diag(W_abs.sum(axis=1))
    L_abs = D_abs_unsigned - W_abs
    L_abs = (L_abs + L_abs.T) / 2

    # Signed Laplacian
    W_signed = coi_matrix.copy()
    np.fill_diagonal(W_signed, 0)
    D_abs_signed = np.diag(np.abs(W_signed).sum(axis=1))
    L_signed = D_abs_signed - W_signed
    L_signed = (L_signed + L_signed.T) / 2

    try:
        evals_unsigned = [round(float(v), 10) for v in la.eigvalsh(L_abs)[:top_k]]
    except la.LinAlgError:
        evals_unsigned = []

    try:
        evals_signed = [round(float(v), 10) for v in la.eigvalsh(L_signed)[:top_k]]
    except la.LinAlgError:
        evals_signed = []

    return {
        "unsigned_laplacian_eigenvalues": evals_unsigned,
        "signed_laplacian_eigenvalues": evals_signed,
    }


# ---------------------------------------------------------------------------
# SECTION 6: FIGS (Quick axis-aligned vs random-oblique comparison)
# ---------------------------------------------------------------------------

class QuickFIGS:
    """Minimal FIGS for axis-aligned vs random-oblique comparison."""

    def __init__(
        self,
        max_splits: int = 10,
        strategy: str = "axis_aligned",
        random_state: int = 42,
    ):
        self.max_splits = max_splits
        self.strategy = strategy
        self.random_state = random_state
        self.nodes: list[dict] = []
        self.leaf_indices: list[int] = []
        self.splits_used = 0
        self.split_arities: list[int] = []
        self.task_type: str | None = None
        self._X: np.ndarray | None = None
        self._y: np.ndarray | None = None

    def _get_candidates(self, d: int) -> list[list[int]]:
        """Get candidate feature subsets for splitting."""
        if self.strategy == "axis_aligned":
            return [[i] for i in range(d)]
        else:  # random_oblique
            rng = stdlib_random.Random(self.random_state)
            cands: list[list[int]] = []
            for _ in range(min(50, d * 2)):
                size = rng.randint(2, min(5, d))
                cands.append(sorted(rng.sample(range(d), size)))
            # Also include single-feature candidates
            cands.extend([[i] for i in range(d)])
            return cands

    def _evaluate_split(
        self,
        indices: np.ndarray,
        feat_subset: list[int],
    ) -> tuple[float, object, np.ndarray | None, np.ndarray | None]:
        """Evaluate a split candidate on given indices.

        Returns (improvement, split_info, left_mask, right_mask).
        """
        if len(indices) < 4:
            return -np.inf, None, None, None

        X_sub = self._X[indices]
        y_sub = self._y[indices]

        # Current impurity
        if self.task_type == "classification":
            p = np.clip(y_sub.mean(), 1e-10, 1 - 1e-10)
            current_impurity = 2 * p * (1 - p)  # Gini
        else:
            current_impurity = np.var(y_sub)

        if current_impurity < 1e-10:
            return -np.inf, None, None, None

        if len(feat_subset) == 1:
            return self._eval_single_feature(
                X_sub, y_sub, feat_subset[0], current_impurity
            )
        else:
            return self._eval_multi_feature(
                X_sub, y_sub, feat_subset, current_impurity
            )

    def _eval_single_feature(
        self,
        X_sub: np.ndarray,
        y_sub: np.ndarray,
        f_idx: int,
        current_impurity: float,
    ) -> tuple[float, object, np.ndarray | None, np.ndarray | None]:
        """Find best threshold for a single feature."""
        x_feat = X_sub[:, f_idx]
        unique_vals = np.unique(x_feat)
        if len(unique_vals) < 2:
            return -np.inf, None, None, None

        thresholds = np.percentile(x_feat, [20, 30, 40, 50, 60, 70, 80])
        thresholds = np.unique(thresholds)

        best_imp = -np.inf
        best_thresh = None
        best_left = None
        best_right = None

        for thresh in thresholds:
            left = x_feat <= thresh
            right = ~left
            n_left, n_right = int(left.sum()), int(right.sum())
            if n_left < 2 or n_right < 2:
                continue

            if self.task_type == "classification":
                p_l = np.clip(y_sub[left].mean(), 1e-10, 1 - 1e-10)
                p_r = np.clip(y_sub[right].mean(), 1e-10, 1 - 1e-10)
                imp_l = 2 * p_l * (1 - p_l)
                imp_r = 2 * p_r * (1 - p_r)
            else:
                imp_l = np.var(y_sub[left])
                imp_r = np.var(y_sub[right])

            weighted = (n_left * imp_l + n_right * imp_r) / len(y_sub)
            improvement = current_impurity - weighted

            if improvement > best_imp:
                best_imp = improvement
                best_thresh = thresh
                best_left = left
                best_right = right

        if best_thresh is None:
            return -np.inf, None, None, None

        return best_imp, ("single", f_idx, float(best_thresh)), best_left, best_right

    def _eval_multi_feature(
        self,
        X_sub: np.ndarray,
        y_sub: np.ndarray,
        feat_subset: list[int],
        current_impurity: float,
    ) -> tuple[float, object, np.ndarray | None, np.ndarray | None]:
        """Multi-feature: RidgeCV projection then threshold."""
        X_multi = X_sub[:, feat_subset]

        # Standardize
        mu = X_multi.mean(axis=0)
        std = X_multi.std(axis=0)
        std = np.where(std > 1e-10, std, 1.0)
        X_std = (X_multi - mu) / std

        try:
            ridge = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0])
            ridge.fit(X_std, y_sub)
            projection = X_std @ ridge.coef_
        except Exception:
            return -np.inf, None, None, None

        thresholds = np.percentile(projection, [20, 30, 40, 50, 60, 70, 80])
        thresholds = np.unique(thresholds)

        best_imp = -np.inf
        best_thresh = None
        best_left = None
        best_right = None

        for thresh in thresholds:
            left = projection <= thresh
            right = ~left
            n_left, n_right = int(left.sum()), int(right.sum())
            if n_left < 2 or n_right < 2:
                continue

            if self.task_type == "classification":
                p_l = np.clip(y_sub[left].mean(), 1e-10, 1 - 1e-10)
                p_r = np.clip(y_sub[right].mean(), 1e-10, 1 - 1e-10)
                imp_l = 2 * p_l * (1 - p_l)
                imp_r = 2 * p_r * (1 - p_r)
            else:
                imp_l = np.var(y_sub[left])
                imp_r = np.var(y_sub[right])

            weighted = (n_left * imp_l + n_right * imp_r) / len(y_sub)
            improvement = current_impurity - weighted

            if improvement > best_imp:
                best_imp = improvement
                best_thresh = thresh
                best_left = left
                best_right = right

        if best_thresh is None:
            return -np.inf, None, None, None

        return (
            best_imp,
            ("multi", feat_subset, ridge.coef_.tolist(), mu.tolist(), std.tolist(), float(best_thresh)),
            best_left,
            best_right,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "QuickFIGS":
        """Greedy tree growing with FIGS."""
        n, d = X.shape
        self.task_type = "classification" if len(np.unique(y)) <= 10 else "regression"
        self._X = X
        self._y = y

        candidates = self._get_candidates(d)

        # Initialize: single leaf with all data
        self.nodes = [
            {
                "indices": np.arange(n),
                "prediction": float(y.mean()),
                "split": None,
                "left": None,
                "right": None,
            }
        ]
        self.leaf_indices = [0]
        self.splits_used = 0
        self.split_arities = []

        for split_round in range(self.max_splits):
            best_improvement = -np.inf
            best_leaf_pos = None
            best_split_info = None
            best_left_mask = None
            best_right_mask = None

            for leaf_pos, node_idx in enumerate(self.leaf_indices):
                node = self.nodes[node_idx]
                indices = node["indices"]
                if len(indices) < 4:
                    continue

                for feat_subset in candidates:
                    imp, split_info, left_mask, right_mask = self._evaluate_split(
                        indices, feat_subset
                    )
                    if imp > best_improvement:
                        best_improvement = imp
                        best_leaf_pos = leaf_pos
                        best_split_info = split_info
                        best_left_mask = left_mask
                        best_right_mask = right_mask

            if best_improvement <= 1e-10 or best_split_info is None or best_leaf_pos is None:
                break

            # Apply split
            node_idx = self.leaf_indices[best_leaf_pos]
            node = self.nodes[node_idx]
            indices = node["indices"]

            left_indices = indices[best_left_mask]
            right_indices = indices[best_right_mask]

            left_pred = float(y[left_indices].mean())
            right_pred = float(y[right_indices].mean())

            left_node_idx = len(self.nodes)
            self.nodes.append(
                {
                    "indices": left_indices,
                    "prediction": left_pred,
                    "split": None,
                    "left": None,
                    "right": None,
                }
            )
            right_node_idx = len(self.nodes)
            self.nodes.append(
                {
                    "indices": right_indices,
                    "prediction": right_pred,
                    "split": None,
                    "left": None,
                    "right": None,
                }
            )

            node["split"] = best_split_info
            node["left"] = left_node_idx
            node["right"] = right_node_idx

            # Update leaf list
            self.leaf_indices[best_leaf_pos] = left_node_idx
            self.leaf_indices.append(right_node_idx)

            self.splits_used += 1
            if best_split_info[0] == "multi":
                self.split_arities.append(len(best_split_info[1]))
            else:
                self.split_arities.append(1)

        # Free references to training data
        self._X = None
        self._y = None

        return self

    def _predict_one(self, x: np.ndarray) -> float:
        """Predict for a single sample by tree traversal."""
        node_idx = 0
        max_depth = 100  # Safety limit
        depth = 0
        while self.nodes[node_idx]["split"] is not None and depth < max_depth:
            depth += 1
            split = self.nodes[node_idx]["split"]
            if split[0] == "single":
                _, feat_idx, thresh = split
                if x[feat_idx] <= thresh:
                    node_idx = self.nodes[node_idx]["left"]
                else:
                    node_idx = self.nodes[node_idx]["right"]
            else:  # multi
                _, feat_subset, coef, mu, std, thresh = split
                x_sub = np.array(
                    [(x[f] - mu[i]) / std[i] for i, f in enumerate(feat_subset)]
                )
                projection = float(np.dot(x_sub, coef))
                if projection <= thresh:
                    node_idx = self.nodes[node_idx]["left"]
                else:
                    node_idx = self.nodes[node_idx]["right"]
        return self.nodes[node_idx]["prediction"]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict for multiple samples."""
        return np.array([self._predict_one(X[i]) for i in range(len(X))])


def quick_figs_comparison(
    X: np.ndarray,
    y: np.ndarray,
    folds: np.ndarray,
    task_type: str,
    n_classes: int | None = None,
) -> dict:
    """Run axis-aligned vs random-oblique FIGS on fold 0 only.

    Returns oblique_benefit = metric_random_oblique - metric_axis_aligned.
    Uses balanced_accuracy for classification, r2 for regression.
    """
    train_mask = folds != 0
    test_mask = folds == 0

    if test_mask.sum() < 2 or train_mask.sum() < 2:
        return {
            "metric_axis_aligned": 0.0,
            "metric_random_oblique": 0.0,
            "oblique_benefit": 0.0,
            "metric_name": "balanced_accuracy" if task_type == "classification" else "r2",
            "n_train": int(train_mask.sum()),
            "n_test": int(test_mask.sum()),
            "aa_splits": 0,
            "ro_splits": 0,
            "ro_mean_arity": 1.0,
        }

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    # Subsample training data if too large (for speed)
    if len(X_train) > FIGS_TRAIN_SUBSAMPLE:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X_train), FIGS_TRAIN_SUBSAMPLE, replace=False)
        X_train_fit = X_train[idx]
        y_train_fit = y_train[idx]
    else:
        X_train_fit = X_train
        y_train_fit = y_train

    # Fit axis-aligned
    aa = QuickFIGS(max_splits=FIGS_MAX_SPLITS, strategy="axis_aligned", random_state=42)
    aa.fit(X_train_fit, y_train_fit)
    pred_aa = aa.predict(X_test)
    if task_type == "classification":
        y_pred_aa = (pred_aa > 0.5).astype(int)
        metric_aa = float(balanced_accuracy_score(y_test, y_pred_aa))
    else:
        metric_aa = float(r2_score(y_test, pred_aa))

    # Fit random-oblique
    ro = QuickFIGS(max_splits=FIGS_MAX_SPLITS, strategy="random_oblique", random_state=42)
    ro.fit(X_train_fit, y_train_fit)
    pred_ro = ro.predict(X_test)
    if task_type == "classification":
        y_pred_ro = (pred_ro > 0.5).astype(int)
        metric_ro = float(balanced_accuracy_score(y_test, y_pred_ro))
    else:
        metric_ro = float(r2_score(y_test, pred_ro))

    oblique_benefit = metric_ro - metric_aa

    return {
        "metric_axis_aligned": round(metric_aa, 6),
        "metric_random_oblique": round(metric_ro, 6),
        "oblique_benefit": round(oblique_benefit, 6),
        "metric_name": "balanced_accuracy" if task_type == "classification" else "r2",
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
        "aa_splits": aa.splits_used,
        "ro_splits": ro.splits_used,
        "ro_mean_arity": round(float(np.mean(ro.split_arities)) if ro.split_arities else 1.0, 2),
    }


# ---------------------------------------------------------------------------
# Ground truth label assignment for ARI
# ---------------------------------------------------------------------------

def assign_ground_truth_labels(meta: dict, d: int) -> np.ndarray:
    """Assign ground truth module labels (-1 for unassigned features)."""
    labels = np.full(d, -1, dtype=int)
    for mod_idx, mod in enumerate(meta.get("ground_truth_modules", [])):
        for feat in mod:
            if feat < d and labels[feat] == -1:
                labels[feat] = mod_idx
    return labels


# ---------------------------------------------------------------------------
# SECTION 7: MAIN PIPELINE
# ---------------------------------------------------------------------------

@logger.catch
def main():
    t_start = time.time()

    logger.info("=" * 60)
    logger.info("CoI Graph Characterization & Frustration Meta-Diagnostic")
    logger.info("14 datasets (8 real + 6 synthetic)")
    logger.info("=" * 60)

    # ---- STEP 1: Load all datasets ----
    logger.info("\n--- Step 1: Loading datasets ---")
    all_datasets: dict[str, dict] = {}
    synth_dataset_names: set[str] = set()

    # 1a. Load real datasets from data_id4
    logger.info("Loading real datasets from data_id4...")
    raw4 = load_datasets_from_dependency(DATA4_DIR)
    for name, examples in raw4.items():
        try:
            parsed = parse_dataset(name, examples)
            all_datasets[name] = parsed
        except Exception:
            logger.exception(f"Failed to parse {name}")
    del raw4
    gc.collect()

    # 1b. Load real datasets from data_id5
    logger.info("Loading real datasets from data_id5...")
    raw5 = load_datasets_from_dependency(DATA5_DIR)
    for name, examples in raw5.items():
        try:
            parsed = parse_dataset(name, examples)
            all_datasets[name] = parsed
        except Exception:
            logger.exception(f"Failed to parse {name}")
    del raw5
    gc.collect()

    logger.info(f"Loaded {len(all_datasets)} real datasets")

    # 1c. Generate synthetic datasets
    logger.info("\nGenerating synthetic datasets...")
    import shutil

    synth_data_path = WORKSPACE / "synth_data.py"
    shutil.copy(DATA_SYNTH_DIR / "data.py", synth_data_path)

    sys.path.insert(0, str(WORKSPACE))
    from synth_data import (
        gen_easy_2mod_xor,
        gen_hard_4mod_unequal,
        gen_highdim_8mod,
        gen_medium_4mod_mixed,
        gen_no_structure_control,
        gen_overlapping_modules,
    )

    GENERATORS = [
        ("easy_2mod_xor", gen_easy_2mod_xor, 0),
        ("medium_4mod_mixed", gen_medium_4mod_mixed, 1),
        ("hard_4mod_unequal", gen_hard_4mod_unequal, 2),
        ("overlapping_modules", gen_overlapping_modules, 3),
        ("no_structure_control", gen_no_structure_control, 4),
        ("highdim_8mod", gen_highdim_8mod, 5),
    ]

    base_rng = np.random.default_rng(MASTER_SEED)
    variant_seeds = [int(base_rng.integers(0, 2**31)) for _ in range(6)]

    for name, gen_fn, seed_idx in GENERATORS:
        t0 = time.time()
        result = gen_fn(np.random.default_rng(variant_seeds[seed_idx]))
        dt = time.time() - t0
        all_datasets[name] = {
            "X": result["X"],
            "y": result["y"],
            "meta": result["meta"],
            "folds": result["folds"],
            "task_type": "classification",
            "feature_names": result["meta"]["feature_names"],
        }
        synth_dataset_names.add(name)
        logger.info(f"  Generated {name}: {result['X'].shape} in {dt:.1f}s")

    logger.info(f"\nTotal datasets: {len(all_datasets)}")

    # ---- STEP 2: Per-dataset processing ----
    logger.info("\n--- Step 2: Per-dataset processing ---")

    dataset_order = [
        "credit",
        "adult",
        "electricity",
        "california_housing",
        "easy_2mod_xor",
        "eye_movements",
        "no_structure_control",
        "medium_4mod_mixed",
        "overlapping_modules",
        "higgs_small",
        "hard_4mod_unequal",
        "miniboone",
        "jannis",
        "highdim_8mod",
    ]

    results_per_dataset: dict[str, dict] = {}
    schema_datasets: list[dict] = []

    for ds_idx, ds_name in enumerate(dataset_order):
        if ds_name not in all_datasets:
            logger.warning(f"Dataset {ds_name} not found, skipping")
            continue

        ds_t0 = time.time()
        logger.info(f"\n[{ds_idx + 1}/{len(dataset_order)}] Processing {ds_name}...")
        ds = all_datasets[ds_name]
        X, y = ds["X"], ds["y"]
        n, d = X.shape
        logger.info(f"  Shape: ({n}, {d}), task: {ds['task_type']}")

        # 2a. Subsample for CoI computation
        if n > COI_SUBSAMPLE_N:
            rng = np.random.default_rng(42)
            idx = rng.choice(n, COI_SUBSAMPLE_N, replace=False)
            X_sub, y_sub = X[idx], y[idx]
        else:
            X_sub, y_sub = X.copy(), y.copy()

        # For very high-dim datasets, cap features for CoI
        d_coi = d
        X_coi = X_sub
        if d > HIGHDIM_FEATURE_CAP:
            d_coi = HIGHDIM_FEATURE_CAP
            X_coi = X_sub[:, :d_coi]
            logger.info(f"  Capping features from {d} to {d_coi} for CoI computation")
        else:
            X_coi = X_sub

        # 2b. Compute CoI matrix
        t0 = time.time()
        coi_matrix, mi_individual = compute_coi_matrix(X_coi, y_sub, n_bins=MI_N_BINS)
        coi_time = time.time() - t0
        logger.info(f"  CoI matrix: ({d_coi},{d_coi}) in {coi_time:.1f}s")

        # 2c. Graph characterization
        graph_stats = characterize_coi_graph(
            coi_matrix, ds_name, meta=ds.get("meta")
        )

        # 2d. Unsigned spectral clustering
        t0 = time.time()
        us_modules, us_k, us_labels, us_evals, us_sil = unsigned_spectral_clustering(
            coi_matrix
        )
        us_time = time.time() - t0
        logger.info(f"  Unsigned spectral: k={us_k}, sil={us_sil:.3f} in {us_time:.2f}s")

        # 2e. SPONGE signed spectral clustering
        t0 = time.time()
        ss_modules, ss_k, ss_labels, ss_evals, ss_sil = sponge_sym_clustering(
            coi_matrix
        )
        ss_time = time.time() - t0
        logger.info(f"  SPONGE: k={ss_k}, sil={ss_sil:.3f} in {ss_time:.2f}s")

        # 2f. Frustration index
        frust = compute_frustration_index(coi_matrix)
        logger.info(
            f"  Frustration: raw={frust['frustration_raw']:.6f}, "
            f"norm_max={frust['normalized_by_max']:.6f}"
        )

        # 2g. Eigenspectrum
        eigenspec = compute_eigenspectrum(coi_matrix, top_k=min(20, d_coi - 1))

        # 2h. Quick FIGS comparison (fold 0 only)
        t0 = time.time()
        try:
            figs_result = quick_figs_comparison(
                X, y, ds["folds"], ds["task_type"], ds.get("n_classes")
            )
        except Exception:
            logger.exception(f"  FIGS failed for {ds_name}")
            figs_result = {
                "metric_axis_aligned": 0.0,
                "metric_random_oblique": 0.0,
                "oblique_benefit": 0.0,
                "metric_name": "balanced_accuracy"
                if ds["task_type"] == "classification"
                else "r2",
                "n_train": 0,
                "n_test": 0,
                "aa_splits": 0,
                "ro_splits": 0,
                "ro_mean_arity": 1.0,
            }
        figs_time = time.time() - t0
        logger.info(
            f"  FIGS: aa={figs_result['metric_axis_aligned']:.4f}, "
            f"ro={figs_result['metric_random_oblique']:.4f}, "
            f"benefit={figs_result['oblique_benefit']:.4f} in {figs_time:.1f}s"
        )

        # 2i. Record results
        mi_top10 = sorted(
            [(int(i), round(float(mi_individual[i]), 6)) for i in range(len(mi_individual))],
            key=lambda x: -x[1],
        )[:10]

        ds_result = {
            "n_samples": n,
            "n_features": d,
            "task_type": ds["task_type"],
            "is_synthetic": ds_name in synth_dataset_names,
            "coi_computation": {
                "subsample_n": len(X_sub),
                "n_bins": MI_N_BINS,
                "features_used": d_coi,
                "time_s": round(coi_time, 2),
            },
            "mi_individual_top10": mi_top10,
            "graph_characterization": graph_stats,
            "unsigned_spectral": {
                "k": us_k,
                "module_sizes": [len(m) for m in us_modules],
                "modules": us_modules,
                "silhouette": round(us_sil, 6),
                "time_s": round(us_time, 3),
                "eigenvalues_top20": [round(float(v), 8) for v in us_evals[:20]],
            },
            "signed_spectral_sponge": {
                "k": ss_k,
                "module_sizes": [len(m) for m in ss_modules],
                "modules": ss_modules,
                "silhouette": round(ss_sil, 6),
                "time_s": round(ss_time, 3),
                "eigenvalues_top20": [round(float(v), 8) for v in ss_evals[:20]],
            },
            "frustration_index": frust,
            "eigenspectrum": eigenspec,
            "figs_comparison": figs_result,
        }

        # 2j. Ground truth recovery (synthetic only)
        if ds.get("meta", {}).get("ground_truth_modules"):
            gt_labels = assign_ground_truth_labels(ds["meta"], d_coi)
            mask = gt_labels >= 0
            if mask.sum() >= 2:
                try:
                    ari_us = float(adjusted_rand_score(gt_labels[mask], us_labels[mask]))
                    ari_ss = float(adjusted_rand_score(gt_labels[mask], ss_labels[mask]))
                    ds_result["ground_truth_recovery"] = {
                        "unsigned_ari": round(ari_us, 4),
                        "sponge_ari": round(ari_ss, 4),
                        "n_assigned_features": int(mask.sum()),
                    }
                    logger.info(
                        f"  GT recovery: unsigned_ari={ari_us:.4f}, sponge_ari={ari_ss:.4f}"
                    )
                except Exception as e:
                    logger.warning(f"  GT recovery failed: {e}")

        results_per_dataset[ds_name] = ds_result

        # Build schema-compliant example for this dataset
        schema_example = {
            "input": json.dumps(
                {
                    "dataset": ds_name,
                    "n_samples": n,
                    "n_features": d,
                    "task_type": ds["task_type"],
                    "is_synthetic": ds_name in synth_dataset_names,
                }
            ),
            "output": json.dumps(
                {
                    "frustration_normalized": round(frust["normalized_by_max"], 6),
                    "unsigned_k": us_k,
                    "sponge_k": ss_k,
                    "oblique_benefit": round(figs_result["oblique_benefit"], 6),
                    "frac_negative_coi": graph_stats["sign_distribution"].get(
                        "frac_negative", 0
                    ),
                    "frac_positive_coi": graph_stats["sign_distribution"].get(
                        "frac_positive", 0
                    ),
                }
            ),
            "predict_axis_aligned": str(
                round(figs_result["metric_axis_aligned"], 6)
            ),
            "predict_random_oblique": str(
                round(figs_result["metric_random_oblique"], 6)
            ),
            "metadata_frustration_raw": round(frust["frustration_raw"], 8),
            "metadata_frustration_normalized": round(
                frust["normalized_by_max"], 8
            ),
            "metadata_unsigned_k": us_k,
            "metadata_sponge_k": ss_k,
            "metadata_unsigned_silhouette": round(us_sil, 6),
            "metadata_sponge_silhouette": round(ss_sil, 6),
            "metadata_coi_frac_negative": graph_stats["sign_distribution"].get(
                "frac_negative", 0
            ),
            "metadata_coi_frac_positive": graph_stats["sign_distribution"].get(
                "frac_positive", 0
            ),
            "metadata_oblique_benefit": round(figs_result["oblique_benefit"], 6),
        }
        schema_datasets.append(
            {"dataset": ds_name, "examples": [schema_example]}
        )

        # Free memory
        del X_sub, y_sub, coi_matrix, X_coi
        gc.collect()

        ds_total_time = time.time() - ds_t0
        logger.info(
            f"  {ds_name} total: {ds_total_time:.1f}s | "
            f"frust={frust['normalized_by_max']:.4f}, "
            f"oblique_benefit={figs_result['oblique_benefit']:.4f}"
        )

    # ---- STEP 3: Cross-dataset correlation analysis ----
    logger.info("\n--- Step 3: Cross-dataset correlation analysis ---")

    frustration_values: list[float] = []
    oblique_benefits: list[float] = []
    dataset_names_ordered: list[str] = []

    for ds_name in dataset_order:
        if ds_name in results_per_dataset:
            r = results_per_dataset[ds_name]
            f_val = r["frustration_index"]["normalized_by_max"]
            o_val = r["figs_comparison"]["oblique_benefit"]
            if np.isfinite(f_val) and np.isfinite(o_val):
                frustration_values.append(f_val)
                oblique_benefits.append(o_val)
                dataset_names_ordered.append(ds_name)

    frust_arr = np.array(frustration_values)
    obliq_arr = np.array(oblique_benefits)

    # Spearman correlation
    if len(frust_arr) >= 3:
        rho, p_value = spearmanr(frust_arr, obliq_arr)
    else:
        rho, p_value = 0.0, 1.0
    logger.info(f"Spearman rho={rho:.4f}, p={p_value:.4f} (n={len(frust_arr)})")

    # Bootstrap 95% CI (2000 resamples)
    n_boot = 2000
    rng_boot = np.random.default_rng(42)
    boot_rhos: list[float] = []
    for _ in range(n_boot):
        idx = rng_boot.choice(len(frust_arr), size=len(frust_arr), replace=True)
        if len(np.unique(frust_arr[idx])) < 2 or len(np.unique(obliq_arr[idx])) < 2:
            continue
        r_boot, _ = spearmanr(frust_arr[idx], obliq_arr[idx])
        if np.isfinite(r_boot):
            boot_rhos.append(float(r_boot))

    boot_arr = np.array(boot_rhos) if boot_rhos else np.array([0.0])
    ci_lower = float(np.percentile(boot_arr, 2.5))
    ci_upper = float(np.percentile(boot_arr, 97.5))
    logger.info(
        f"Bootstrap CI: [{ci_lower:.4f}, {ci_upper:.4f}] ({len(boot_rhos)} valid resamples)"
    )

    correlation_analysis = {
        "spearman_rho": round(float(rho), 6),
        "p_value": round(float(p_value), 6),
        "n_datasets": len(frust_arr),
        "bootstrap_ci_95": {
            "lower": round(ci_lower, 6),
            "upper": round(ci_upper, 6),
            "n_resamples": n_boot,
            "n_valid": len(boot_rhos),
        },
        "dataset_values": [
            {
                "dataset": name,
                "frustration_index": round(float(f), 6),
                "oblique_benefit": round(float(o), 6),
            }
            for name, f, o in zip(dataset_names_ordered, frust_arr, obliq_arr)
        ],
        "hypothesis_test": {
            "h0": "No correlation between frustration index and oblique benefit",
            "h1": "Negative correlation: lower frustration (cleaner modules) -> more oblique benefit",
            "significant_at_0.05": bool(p_value < 0.05),
            "direction": "negative" if rho < 0 else "positive",
        },
    }

    # ---- STEP 4: Subset analyses ----
    logger.info("\n--- Step 4: Subset analyses ---")
    for subset_name, subset_filter in [("synthetic_only", True), ("real_only", False)]:
        sub_f: list[float] = []
        sub_o: list[float] = []
        for ds_name, f_val, o_val in zip(
            dataset_names_ordered, frust_arr, obliq_arr
        ):
            is_synth = results_per_dataset[ds_name]["is_synthetic"]
            if is_synth == subset_filter:
                sub_f.append(f_val)
                sub_o.append(o_val)
        if len(sub_f) >= 4:
            sub_rho, sub_p = spearmanr(sub_f, sub_o)
            correlation_analysis[f"{subset_name}_spearman"] = {
                "rho": round(float(sub_rho), 6),
                "p_value": round(float(sub_p), 6),
                "n": len(sub_f),
            }
            logger.info(
                f"  {subset_name}: rho={sub_rho:.4f}, p={sub_p:.4f}, n={len(sub_f)}"
            )

    # Kendall's tau as complementary statistic
    if len(frust_arr) >= 3:
        tau_val, tau_p = kendalltau(frust_arr, obliq_arr)
        correlation_analysis["kendall_tau"] = {
            "tau": round(float(tau_val), 6),
            "p_value": round(float(tau_p), 6),
        }
        logger.info(f"Kendall's tau={tau_val:.4f}, p={tau_p:.4f}")

    # ---- STEP 5: Summary ----
    total_time = round(time.time() - t_start, 1)
    key_finding = (
        "CONFIRM"
        if (rho < 0 and p_value < 0.05)
        else "DISCONFIRM/INCONCLUSIVE"
    )

    summary = {
        "total_datasets": len(results_per_dataset),
        "total_time_s": total_time,
        "mean_frustration": round(float(np.mean(frust_arr)), 6),
        "std_frustration": round(float(np.std(frust_arr)), 6),
        "mean_oblique_benefit": round(float(np.mean(obliq_arr)), 6),
        "std_oblique_benefit": round(float(np.std(obliq_arr)), 6),
        "key_finding": key_finding,
    }

    logger.info(f"\n{'=' * 60}")
    logger.info(f"Key finding: {key_finding}")
    logger.info(f"Spearman rho={rho:.4f}, p={p_value:.4f}")
    logger.info(f"Total time: {total_time}s")
    logger.info(f"{'=' * 60}")

    # ---- STEP 6: Write schema-compliant output ----
    all_metadata = {
        "experiment": "CoI Graph Characterization & Frustration Meta-Diagnostic",
        "description": (
            "Computes pairwise Co-Information graphs for 14 datasets, runs unsigned "
            "and signed (SPONGE) spectral clustering, computes frustration index, "
            "then tests Spearman correlation between frustration index and oblique benefit."
        ),
        "n_datasets_total": len(results_per_dataset),
        "coi_method": "binning_based_mi",
        "n_bins": MI_N_BINS,
        "subsample_n": COI_SUBSAMPLE_N,
        "figs_max_splits": FIGS_MAX_SPLITS,
        "master_seed": MASTER_SEED,
        "per_dataset_results": results_per_dataset,
        "correlation_analysis": correlation_analysis,
        "summary": summary,
    }

    schema_output = {
        "metadata": all_metadata,
        "datasets": schema_datasets,
    }

    out_path = WORKSPACE / "method_out.json"
    out_path.write_text(json.dumps(schema_output, indent=2, default=str))
    file_size_mb = out_path.stat().st_size / 1e6
    logger.info(f"Output written to {out_path} ({file_size_mb:.1f} MB)")


if __name__ == "__main__":
    main()

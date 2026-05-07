#!/usr/bin/env python3
"""Real Benchmark: 5 FIGS Variants x 8 Datasets x 5-fold CV — Output Size Fix.

Replicates exp_id1_it4__opus (all 5 FIGS variants across 8 Grinsztajn datasets
with 5-fold CV) with a critical fix: output only aggregate metrics per
(dataset x method x max_splits x fold) plus clustering metadata and 5
representative examples per dataset for schema compliance.

Prior run produced correct results but failed validation because per-example
predictions for 344K examples x 5 methods made output hundreds of MB.
This version outputs ~300KB total.

Output: method_out.json (exp_gen_sol_out schema)
"""

import gc
import json
import math
import os

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"

import resource
import sys
import time
import warnings
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from loguru import logger
from scipy.linalg import eigh
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.linear_model import RidgeCV
from sklearn.metrics import (
    balanced_accuracy_score,
    mutual_info_score,
    r2_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.preprocessing import KBinsDiscretizer

# ============================================================
# LOGGING SETUP
# ============================================================
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
logger.add(LOG_DIR / "run.log", rotation="30 MB", level="DEBUG")


# ============================================================
# HARDWARE DETECTION & MEMORY LIMITS
# ============================================================
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
TOTAL_RAM_GB = _container_ram_gb() or 16.0
RAM_BUDGET = int(TOTAL_RAM_GB * 0.7 * 1e9)
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET * 3, RAM_BUDGET * 3))
resource.setrlimit(resource.RLIMIT_CPU, (3600, 3600))
logger.info(f"Hardware: {NUM_CPUS} CPUs, {TOTAL_RAM_GB:.1f} GB RAM, budget={RAM_BUDGET / 1e9:.1f} GB")


# ============================================================
# CONSTANTS
# ============================================================
WORKSPACE = Path(__file__).parent
DATA4_PATH = Path(
    "/ai-inventor/aii_pipeline/runs/jamnik-sgfigs-pid-v2/"
    "3_invention_loop/iter_1/gen_art/data_id4_it1__opus"
)
DATA5_PATH = Path(
    "/ai-inventor/aii_pipeline/runs/jamnik-sgfigs-pid-v2/"
    "3_invention_loop/iter_2/gen_art/data_id5_it2__opus"
)

METHODS = [
    "axis_aligned",
    "random_oblique",
    "unsigned_spectral",
    "signed_spectral",
    "hard_threshold",
]
MAX_SPLITS_VALUES = [5, 10, 20]
PREDICT_MAX_SPLITS = 10
COI_SUBSAMPLE_N = 20000
MI_N_BINS = 10

# Dataset processing order: validation datasets first, then by ascending cost
DATASET_ORDER = [
    "adult", "electricity",  # Validation datasets (checkpoint after these)
    "eye_movements", "credit", "california_housing",
    "higgs_small", "jannis", "miniboone",
]

# Try NPEET
try:
    from npeet import entropy_estimators as ee

    HAS_NPEET = True
    logger.info("NPEET available - will use KSG MI estimator")
except ImportError:
    HAS_NPEET = False
    logger.info("NPEET not available - using binning-based MI")


# ============================================================
# DATA LOADING
# ============================================================
def load_datasets_from_dependency(dep_path: Path) -> dict[str, list]:
    """Load datasets from a dependency directory.

    Returns dict: dataset_name -> list of example dicts.
    """
    full_data_dir = dep_path / "full_data_out"
    parts = sorted(full_data_dir.glob("full_data_out_*.json"))

    if not parts:
        logger.warning(f"No full_data_out parts in {full_data_dir}, trying mini")
        mini = dep_path / "mini_data_out.json"
        if mini.exists():
            parts = [mini]
        else:
            raise FileNotFoundError(f"No data files in {dep_path}")

    by_dataset: dict[str, list] = defaultdict(list)

    for part_path in parts:
        logger.info(f"  Loading {part_path.name}...")
        data = json.loads(part_path.read_text())

        if isinstance(data, dict) and "datasets" in data:
            for ds_entry in data["datasets"]:
                by_dataset[ds_entry["dataset"]].extend(ds_entry["examples"])
        elif isinstance(data, list) and data and "input" in data[0]:
            for ex in data:
                src = ex.get("metadata_source", "unknown")
                name = src.rsplit("_", 1)[-1] if "_" in src else src
                by_dataset[name].append(ex)
        else:
            logger.warning(f"Unrecognized format in {part_path.name}")

    return dict(by_dataset)


def parse_dataset(name: str, examples: list) -> dict:
    """Parse raw examples into numpy arrays and keep originals for output."""
    if not examples:
        raise ValueError(f"No examples for {name}")

    # Get feature names from first example's input JSON
    first_input = (
        json.loads(examples[0]["input"])
        if isinstance(examples[0]["input"], str)
        else examples[0]["input"]
    )
    feature_names = list(first_input.keys())

    n = len(examples)
    d = len(feature_names)
    X = np.zeros((n, d), dtype=np.float64)
    y = np.zeros(n, dtype=np.float64)
    folds = np.zeros(n, dtype=int)

    task_type = examples[0].get("metadata_task_type", "classification")
    n_classes = examples[0].get("metadata_n_classes", 2)

    for i, ex in enumerate(examples):
        inp = json.loads(ex["input"]) if isinstance(ex["input"], str) else ex["input"]
        for j, fn in enumerate(feature_names):
            val = inp.get(fn, 0.0)
            try:
                X[i, j] = float(val)
            except (ValueError, TypeError):
                X[i, j] = 0.0

        if task_type == "regression":
            y[i] = float(ex["output"])
        else:
            y[i] = int(float(ex["output"]))

        folds[i] = int(ex.get("metadata_fold", 0))

    # Check for and handle NaN/Inf
    nan_mask = ~np.isfinite(X)
    if nan_mask.any():
        n_nan = nan_mask.sum()
        logger.warning(f"  {name}: replacing {n_nan} NaN/Inf values with 0")
        X[nan_mask] = 0.0

    logger.info(
        f"  {name}: n={n}, d={d}, task={task_type}, "
        f"classes={n_classes}, folds={list(np.unique(folds).astype(int))}"
    )

    return {
        "X": X,
        "y": y,
        "folds": folds,
        "feature_names": feature_names,
        "task_type": task_type,
        "n_classes": n_classes,
        "examples": examples,
    }


# ============================================================
# MI / COI COMPUTATION
# ============================================================
def _discretize_y(y: np.ndarray, n_bins: int = 10) -> np.ndarray:
    """Discretize continuous y into bins; pass through discrete y."""
    unique = np.unique(y)
    if len(unique) > 20:
        try:
            kbd = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="quantile")
            return kbd.fit_transform(y.reshape(-1, 1)).ravel().astype(int)
        except ValueError:
            edges = np.linspace(y.min(), y.max(), n_bins + 1)
            return np.digitize(y, edges[1:-1]).astype(int)
    return y.astype(int)


def _bin_feature(x: np.ndarray, n_bins: int = 10) -> np.ndarray:
    """Discretize a single feature into quantile bins."""
    if np.std(x) < 1e-10:
        return np.zeros(len(x), dtype=int)
    try:
        kbd = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="quantile")
        return kbd.fit_transform(x.reshape(-1, 1)).ravel().astype(int)
    except ValueError:
        return np.zeros(len(x), dtype=int)


def _mi_npeet_single(x_col: np.ndarray, y_list: list, k: int = 5) -> float:
    """MI(Xi; Y) using NPEET micd()."""
    try:
        mi = ee.micd(x_col.reshape(-1, 1), y_list, k=k)
        return max(0.0, float(mi))
    except Exception:
        return 0.0


def _mi_npeet_joint(x_pair: np.ndarray, y_list: list, k: int = 5) -> float:
    """MI([Xi,Xj]; Y) using NPEET micd()."""
    try:
        mi = ee.micd(x_pair, y_list, k=k)
        return max(0.0, float(mi))
    except Exception:
        return 0.0


def compute_coi_matrix(
    X_sub: np.ndarray,
    y_sub: np.ndarray,
    n_bins: int = 10,
    use_npeet: bool = False,
    n_jobs: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute d x d Co-Information matrix.

    CoI(Xi,Xj;Y) = MI(Xi;Y) + MI(Xj;Y) - MI(Xi,Xj;Y)
    Positive = redundancy, Negative = synergy.

    Returns (CoI_matrix, mi_individual).
    """
    n, d = X_sub.shape
    pairs = [(i, j) for i in range(d) for j in range(i + 1, d)]

    if use_npeet and HAS_NPEET:
        logger.debug(f"  CoI via NPEET (d={d}, n={n}, pairs={len(pairs)})")
        y_disc = _discretize_y(y_sub, n_bins)
        y_list = y_disc.tolist()

        from joblib import Parallel, delayed

        mi_individual = np.array(
            Parallel(n_jobs=n_jobs)(
                delayed(_mi_npeet_single)(X_sub[:, i], y_list, 5) for i in range(d)
            )
        )
        mi_joint_vals = Parallel(n_jobs=n_jobs)(
            delayed(_mi_npeet_joint)(X_sub[:, [i, j]], y_list, 5) for i, j in pairs
        )
    else:
        logger.debug(f"  CoI via binning (d={d}, n={n}, pairs={len(pairs)})")
        y_disc = _discretize_y(y_sub, n_bins)

        # Pre-bin all features
        binned = np.zeros((n, d), dtype=int)
        for i in range(d):
            binned[:, i] = _bin_feature(X_sub[:, i], n_bins)

        # Individual MI
        mi_individual = np.array(
            [mutual_info_score(binned[:, i], y_disc) for i in range(d)]
        )

        # Joint MI for all pairs
        mi_joint_vals = []
        for i, j in pairs:
            combined = binned[:, i] * n_bins + binned[:, j]
            mi_joint_vals.append(mutual_info_score(combined, y_disc))

    # Assemble CoI matrix
    CoI = np.zeros((d, d))
    for idx, (i, j) in enumerate(pairs):
        coi_val = mi_individual[i] + mi_individual[j] - mi_joint_vals[idx]
        CoI[i, j] = coi_val
        CoI[j, i] = coi_val

    return CoI, mi_individual


# ============================================================
# CLUSTERING METHODS
# ============================================================
def unsigned_spectral_clustering(
    CoI_matrix: np.ndarray, max_k: int = 10
) -> tuple[list, int, np.ndarray]:
    """Spectral clustering on |CoI| affinity. Returns (modules, k, labels)."""
    d = CoI_matrix.shape[0]
    if d < 3:
        return [list(range(d))], 1, np.zeros(d, dtype=int)

    max_k = min(max_k, d - 1)
    affinity = np.abs(CoI_matrix.copy())
    np.fill_diagonal(affinity, 0)

    # Eigengap on normalized Laplacian
    degree = affinity.sum(axis=1)
    degree_safe = np.maximum(degree, 1e-10)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(degree_safe))
    L = np.diag(degree) - affinity
    L_norm = D_inv_sqrt @ L @ D_inv_sqrt

    try:
        eigenvalues = np.sort(np.real(eigh(L_norm, eigvals_only=True)))
    except Exception:
        logger.warning("  Unsigned spectral eigdecomp failed, returning single module")
        return [list(range(d))], 1, np.zeros(d, dtype=int)

    n_eig = min(max_k + 1, len(eigenvalues))
    gaps = np.diff(eigenvalues[1:n_eig])
    if len(gaps) == 0:
        return [list(range(d))], 1, np.zeros(d, dtype=int)

    k_eigengap = int(np.argmax(gaps) + 2)
    k_eigengap = max(2, min(k_eigengap, max_k))

    # Silhouette-validated k
    best_k, best_score, best_labels = k_eigengap, -1.0, None
    for k_try in range(max(2, k_eigengap - 1), min(max_k + 1, k_eigengap + 2)):
        try:
            sc = SpectralClustering(
                n_clusters=k_try, affinity="precomputed", random_state=42, n_init=10
            )
            labels = sc.fit_predict(affinity)
            if len(np.unique(labels)) > 1:
                sil = silhouette_score(affinity, labels, metric="precomputed")
                if sil > best_score:
                    best_k, best_score, best_labels = k_try, sil, labels
        except Exception:
            continue

    if best_labels is None:
        try:
            sc = SpectralClustering(
                n_clusters=k_eigengap,
                affinity="precomputed",
                random_state=42,
                n_init=10,
            )
            best_labels = sc.fit_predict(affinity)
            best_k = k_eigengap
        except Exception:
            return [list(range(d))], 1, np.zeros(d, dtype=int)

    modules = []
    for c in range(best_k):
        module = list(np.where(best_labels == c)[0])
        if len(module) >= 2:
            modules.append(module)

    if not modules:
        modules = [list(range(d))]
    return modules, best_k, best_labels


def sponge_sym_clustering(
    CoI_matrix: np.ndarray, tau: float = 1.0, max_k: int = 10
) -> tuple[list, int, np.ndarray, float]:
    """SPONGE_sym signed spectral clustering.

    Returns (modules, k, labels, frustration_index).
    """
    d = CoI_matrix.shape[0]
    if d < 3:
        return [list(range(d))], 1, np.zeros(d, dtype=int), 0.0

    max_k = min(max_k, d - 1)

    W_pos = np.maximum(CoI_matrix, 0).copy()
    W_neg = np.abs(np.minimum(CoI_matrix, 0))
    np.fill_diagonal(W_pos, 0)
    np.fill_diagonal(W_neg, 0)

    D_pos = np.diag(W_pos.sum(axis=1))
    D_neg = np.diag(W_neg.sum(axis=1))
    L_pos = D_pos - W_pos
    L_neg = D_neg - W_neg

    A = L_pos + tau * D_neg
    B = L_neg + tau * D_pos + 1e-6 * np.eye(d)

    try:
        eigenvalues, eigenvectors = eigh(A, B)
    except Exception:
        try:
            B += 1e-4 * np.eye(d)
            eigenvalues, eigenvectors = eigh(A, B)
        except Exception:
            logger.warning("  SPONGE failed, falling back to unsigned spectral")
            modules, k, labels = unsigned_spectral_clustering(CoI_matrix, max_k)
            return modules, k, labels, 0.0

    sorted_idx = np.argsort(eigenvalues)
    sorted_evals = eigenvalues[sorted_idx]

    n_eig = min(max_k + 1, len(sorted_evals))
    gaps = np.diff(sorted_evals[1:n_eig])
    k_eigengap = int(np.argmax(gaps) + 2) if len(gaps) > 0 else 2
    k_eigengap = max(2, min(k_eigengap, max_k))

    V = eigenvectors[:, sorted_idx[:k_eigengap]]
    norms = np.linalg.norm(V, axis=1, keepdims=True)
    V_norm = V / np.maximum(norms, 1e-10)

    km = KMeans(n_clusters=k_eigengap, random_state=42, n_init=10)
    labels = km.fit_predict(V_norm)

    # Frustration index: smallest eigenvalue of signed Laplacian
    W_signed = W_pos - W_neg
    D_abs = np.diag(np.abs(W_signed).sum(axis=1))
    L_signed = D_abs - W_signed
    try:
        frust_evals = np.sort(np.real(eigh(L_signed, eigvals_only=True)))
        frustration = float(frust_evals[0])
    except Exception:
        frustration = 0.0

    modules = []
    for c in range(k_eigengap):
        module = list(np.where(labels == c)[0])
        if len(module) >= 2:
            modules.append(module)

    if not modules:
        modules = [list(range(d))]
    return modules, k_eigengap, labels, frustration


def hard_threshold_clustering(
    CoI_matrix: np.ndarray, percentile: int = 90
) -> tuple[list, int, np.ndarray]:
    """Hard threshold |CoI| at percentile. Returns (modules, n_comp, labels)."""
    d = CoI_matrix.shape[0]
    if d < 3:
        return [list(range(d))], 1, np.zeros(d, dtype=int)

    abs_coi = np.abs(CoI_matrix.copy())
    np.fill_diagonal(abs_coi, 0)
    nonzero = abs_coi[abs_coi > 0]

    if len(nonzero) == 0:
        return [list(range(d))], 1, np.zeros(d, dtype=int)

    threshold = np.percentile(nonzero, percentile)
    adj = (abs_coi >= threshold).astype(float)
    n_comp, labels = connected_components(csr_matrix(adj), directed=False)

    modules = []
    for c in range(n_comp):
        module = list(np.where(labels == c)[0])
        if len(module) >= 2:
            modules.append(module)

    if not modules:
        modules = [list(range(d))]
    return modules, n_comp, labels


# ============================================================
# FIGS IMPLEMENTATION
# ============================================================
class FIGSNode:
    """Single node in a FIGS tree."""

    __slots__ = [
        "feature_indices",
        "coefs",
        "threshold",
        "left",
        "right",
        "value",
        "is_leaf",
        "n_samples",
    ]

    def __init__(self):
        self.feature_indices = None
        self.coefs = None
        self.threshold = None
        self.left = None
        self.right = None
        self.value = 0.0
        self.is_leaf = True
        self.n_samples = 0


class FIGSModel:
    """FIGS greedy-tree with pluggable split selection strategy.

    Strategies:
    - axis_aligned: standard single-feature splits (baseline)
    - random_oblique: random multi-feature subsets with Ridge projection
    - unsigned_spectral: CoI-module-guided oblique splits (PRIMARY)
    - signed_spectral: SPONGE-module-guided oblique splits (ABLATION)
    - hard_threshold: threshold-graph-guided oblique splits
    """

    def __init__(
        self,
        max_splits: int = 10,
        split_strategy: str = "axis_aligned",
        feature_modules: list | None = None,
        task_type: str = "classification",
        random_state: int = 42,
    ):
        self.max_splits = max_splits
        self.split_strategy = split_strategy
        self.feature_modules = feature_modules or []
        self.task_type = task_type
        self.random_state = random_state
        self.root = None
        self.splits_used = 0
        self.split_arities = []

    def _get_candidates(self, d: int) -> list[list[int]]:
        """Get deduplicated candidate feature subsets based on strategy."""
        raw = []

        if self.split_strategy == "axis_aligned":
            raw = [[i] for i in range(d)]

        elif self.split_strategy == "random_oblique":
            import random

            rng = random.Random(self.random_state)
            n_cands = min(50, d * 2)
            for _ in range(n_cands):
                size = rng.randint(2, min(5, d))
                raw.append(sorted(rng.sample(range(d), size)))
            # Single-feature fallbacks
            raw.extend([[i] for i in range(d)])

        elif self.split_strategy in (
            "unsigned_spectral",
            "signed_spectral",
            "hard_threshold",
        ):
            for module in self.feature_modules:
                if len(module) >= 2:
                    raw.append(list(module))
                    if len(module) <= 6:
                        for pair in combinations(module, 2):
                            raw.append(list(pair))
            # Single-feature fallbacks
            raw.extend([[i] for i in range(d)])
        else:
            raw = [[i] for i in range(d)]

        # Deduplicate
        seen = set()
        unique = []
        for c in raw:
            key = tuple(c)
            if key not in seen:
                seen.add(key)
                unique.append(c)
        return unique

    def _evaluate_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        indices: np.ndarray,
        feat_subset: list[int],
    ) -> tuple:
        """Evaluate a split candidate.

        Returns (coefs, threshold, impurity_reduction).
        """
        n = len(indices)
        if n < 10:
            return None, None, -np.inf

        y_sub = y[indices]
        total_var = np.var(y_sub) * n
        if total_var < 1e-12:
            return None, None, -np.inf

        if len(feat_subset) == 1:
            proj = X[indices, feat_subset[0]]
            coefs = np.array([1.0])
        else:
            X_feat = X[np.ix_(indices, feat_subset)]
            feat_std = X_feat.std(axis=0)
            if np.all(feat_std < 1e-10):
                return None, None, -np.inf
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    ridge = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0])
                    ridge.fit(X_feat, y_sub)
                    coefs = ridge.coef_.ravel()
            except Exception:
                return None, None, -np.inf
            if np.all(np.abs(coefs) < 1e-10):
                return None, None, -np.inf
            proj = X_feat @ coefs

        if np.std(proj) < 1e-10:
            return None, None, -np.inf

        # Try percentile thresholds
        thresholds = np.unique(np.percentile(proj, np.arange(10, 100, 10)))
        best_red = -np.inf
        best_thresh = None

        for thresh in thresholds:
            left_mask = proj <= thresh
            n_left = left_mask.sum()
            n_right = n - n_left
            if n_left < 5 or n_right < 5:
                continue
            red = total_var - (
                np.var(y_sub[left_mask]) * n_left
                + np.var(y_sub[~left_mask]) * n_right
            )
            if red > best_red:
                best_red = red
                best_thresh = thresh

        return coefs, best_thresh, best_red

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FIGSModel":
        """Grow tree greedily."""
        n, d = X.shape
        y_work = y.astype(np.float64)

        self.root = FIGSNode()
        self.root.value = float(np.mean(y_work))
        self.root.n_samples = n
        self.splits_used = 0
        self.split_arities = []

        candidates = self._get_candidates(d)

        # Active leaves: list of (node, indices)
        active: list[tuple[FIGSNode, np.ndarray]] = [
            (self.root, np.arange(n, dtype=np.int64))
        ]

        for _ in range(self.max_splits):
            if not active:
                break

            best_red = -np.inf
            best_info = None

            for leaf_idx, (leaf, indices) in enumerate(active):
                if len(indices) < 10:
                    continue
                for feat_sub in candidates:
                    coefs, thresh, red = self._evaluate_split(
                        X, y_work, indices, feat_sub
                    )
                    if red > best_red and thresh is not None:
                        best_red = red
                        best_info = (leaf_idx, feat_sub, coefs, thresh)

            if best_info is None or best_red <= 1e-10:
                break

            leaf_idx, feat_sub, coefs, thresh = best_info
            leaf, indices = active[leaf_idx]

            # Apply split
            leaf.is_leaf = False
            leaf.feature_indices = list(feat_sub)
            leaf.coefs = coefs
            leaf.threshold = thresh
            self.split_arities.append(len(feat_sub))
            self.splits_used += 1

            # Compute projection
            if len(feat_sub) == 1:
                proj = X[indices, feat_sub[0]]
            else:
                proj = X[np.ix_(indices, feat_sub)] @ coefs

            left_mask = proj <= thresh
            left_idx = indices[left_mask]
            right_idx = indices[~left_mask]

            # Create children with absolute y-mean values
            leaf.left = FIGSNode()
            leaf.left.value = (
                float(np.mean(y_work[left_idx])) if len(left_idx) > 0 else leaf.value
            )
            leaf.left.n_samples = len(left_idx)

            leaf.right = FIGSNode()
            leaf.right.value = (
                float(np.mean(y_work[right_idx]))
                if len(right_idx) > 0
                else leaf.value
            )
            leaf.right.n_samples = len(right_idx)

            # Update active leaves
            active = [a for i, a in enumerate(active) if i != leaf_idx]
            if len(left_idx) >= 10:
                active.append((leaf.left, left_idx))
            if len(right_idx) >= 10:
                active.append((leaf.right, right_idx))

        return self

    def _predict_batch(
        self, node: FIGSNode, X: np.ndarray, indices: np.ndarray, preds: np.ndarray
    ):
        """Recursive batch prediction."""
        if len(indices) == 0:
            return
        if node.is_leaf or node.left is None or node.right is None:
            preds[indices] = node.value
            return

        if len(node.feature_indices) == 1:
            proj = X[indices, node.feature_indices[0]]
        else:
            proj = X[np.ix_(indices, node.feature_indices)] @ node.coefs

        left_mask = proj <= node.threshold
        self._predict_batch(node.left, X, indices[left_mask], preds)
        self._predict_batch(node.right, X, indices[~left_mask], preds)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict for batch of samples."""
        n = X.shape[0]
        preds = np.full(n, self.root.value if self.root else 0.0)
        if self.root is not None:
            self._predict_batch(self.root, X, np.arange(n, dtype=np.int64), preds)
        if self.task_type in ("classification", "binary_classification"):
            return (preds >= 0.5).astype(int)
        return preds

    def predict_raw(self, X: np.ndarray) -> np.ndarray:
        """Raw predictions (probabilities for clf, values for reg)."""
        n = X.shape[0]
        preds = np.full(n, self.root.value if self.root else 0.0)
        if self.root is not None:
            self._predict_batch(self.root, X, np.arange(n, dtype=np.int64), preds)
        return preds

    def get_metrics(self) -> dict:
        """Return model complexity metrics."""
        depths = []
        self._collect_depths(self.root, 0, depths)
        return {
            "total_splits": self.splits_used,
            "avg_split_arity": (
                round(float(np.mean(self.split_arities)), 4)
                if self.split_arities
                else 1.0
            ),
            "avg_path_length": (
                round(float(np.mean(depths)), 4) if depths else 0.0
            ),
        }

    def _collect_depths(self, node, depth, depths):
        if node is None:
            return
        if node.is_leaf:
            depths.append(depth)
            return
        self._collect_depths(node.left, depth + 1, depths)
        self._collect_depths(node.right, depth + 1, depths)


# ============================================================
# CHECKPOINT HELPER
# ============================================================
def _write_checkpoint(
    all_results: list,
    summary_results: list,
    clustering_info: dict,
    rep_examples: dict,
    t_start: float,
) -> None:
    """Write intermediate checkpoint to verify output format and size."""
    checkpoint = {
        "metadata": {
            "checkpoint": True,
            "n_results_per_fold": len(all_results),
            "n_summaries": len(summary_results),
            "n_datasets_done": len(clustering_info),
            "elapsed_s": round(time.time() - t_start, 1),
            "results_per_fold": all_results,
            "results_summary": summary_results,
            "clustering_info": clustering_info,
        },
        "datasets": [
            {"dataset": ds_name, "examples": exs}
            for ds_name, exs in rep_examples.items()
            if exs  # Only include datasets with examples
        ],
    }

    out_path = WORKSPACE / "method_out_intermediate.json"
    out_path.write_text(json.dumps(checkpoint, default=str))
    fsize_mb = out_path.stat().st_size / (1024 * 1024)
    logger.info(f"  Checkpoint: {fsize_mb:.3f} MB ({len(all_results)} fold results)")
    if fsize_mb > 1.0:
        logger.warning(f"  Checkpoint is {fsize_mb:.1f} MB - larger than expected!")


# ============================================================
# MAIN EXPERIMENT
# ============================================================
@logger.catch
def main():
    t_start = time.time()

    # ---- Phase 1: Load Data ----
    logger.info("=" * 60)
    logger.info("PHASE 1: Loading datasets")
    logger.info("=" * 60)

    datasets: dict[str, dict] = {}
    for dep_name, dep_path in [("data_id4", DATA4_PATH), ("data_id5", DATA5_PATH)]:
        logger.info(f"Loading {dep_name}...")
        try:
            raw = load_datasets_from_dependency(dep_path)
            for ds_name, examples in raw.items():
                datasets[ds_name] = parse_dataset(ds_name, examples)
        except Exception:
            logger.exception(f"Failed loading {dep_name}")
            raise

    all_ds_names = sorted(datasets.keys())
    logger.info(f"Loaded {len(datasets)} datasets: {all_ds_names}")

    # Determine processing order
    ordered_names = [name for name in DATASET_ORDER if name in datasets]
    # Add any extra datasets not in the predefined order
    for name in all_ds_names:
        if name not in ordered_names:
            ordered_names.append(name)
    logger.info(f"Processing order: {ordered_names}")

    # ---- Phase 2: Main Experiment Loop ----
    logger.info("=" * 60)
    logger.info("PHASE 2: Running experiments")
    logger.info("=" * 60)

    all_results: list[dict] = []        # Per-fold results
    summary_results: list[dict] = []    # Aggregated per (dataset, method, max_splits)
    clustering_info: dict[str, dict] = {}
    rep_examples: dict[str, list] = {}  # Representative examples per dataset

    for ds_idx, ds_name in enumerate(ordered_names):
        ds = datasets[ds_name]
        X, y, folds = ds["X"], ds["y"], ds["folds"]
        task_type = ds["task_type"]
        n, d = X.shape

        logger.info(f"\n{'=' * 50}")
        logger.info(f"Dataset {ds_idx + 1}/{len(ordered_names)}: {ds_name} (n={n}, d={d}, task={task_type})")
        logger.info(f"{'=' * 50}")

        # --- CoI Computation ---
        t_coi = time.time()
        n_sub = min(n, COI_SUBSAMPLE_N)
        rng = np.random.RandomState(42)
        sub_idx = rng.choice(n, n_sub, replace=False)

        # Filter constant features for CoI computation
        feat_std = X[sub_idx].std(axis=0)
        valid_feats = np.where(feat_std > 1e-10)[0]

        if len(valid_feats) < 2:
            logger.warning(f"  Only {len(valid_feats)} non-constant features")
            CoI_full = np.zeros((d, d))
            mi_full = np.zeros(d)
        else:
            CoI_sub, mi_sub = compute_coi_matrix(
                X[sub_idx][:, valid_feats],
                y[sub_idx],
                n_bins=MI_N_BINS,
                use_npeet=HAS_NPEET,
                n_jobs=NUM_CPUS,
            )
            # Map back to full feature space
            CoI_full = np.zeros((d, d))
            mi_full = np.zeros(d)
            for i_idx, vi in enumerate(valid_feats):
                mi_full[vi] = mi_sub[i_idx]
                for j_idx, vj in enumerate(valid_feats):
                    CoI_full[vi, vj] = CoI_sub[i_idx, j_idx]

        coi_time = time.time() - t_coi
        logger.info(
            f"  CoI: {coi_time:.1f}s (n_sub={n_sub}, valid_d={len(valid_feats)})"
        )

        # --- Clustering ---
        t_cl = time.time()
        if len(valid_feats) >= 2:
            CoI_v = CoI_full[np.ix_(valid_feats, valid_feats)]
        else:
            CoI_v = CoI_full

        us_mod_sub, us_k, us_lab = unsigned_spectral_clustering(CoI_v)
        ss_mod_sub, ss_k, ss_lab, frust = sponge_sym_clustering(CoI_v)
        ht_mod_sub, ht_k, ht_lab = hard_threshold_clustering(CoI_v)

        # Map module indices back to full feature space
        if len(valid_feats) >= 2:
            us_modules = [[int(valid_feats[i]) for i in m] for m in us_mod_sub]
            ss_modules = [[int(valid_feats[i]) for i in m] for m in ss_mod_sub]
            ht_modules = [[int(valid_feats[i]) for i in m] for m in ht_mod_sub]
        else:
            us_modules = [list(range(d))]
            ss_modules = [list(range(d))]
            ht_modules = [list(range(d))]

        cl_time = time.time() - t_cl
        logger.info(f"  Clustering: {cl_time:.1f}s")
        logger.info(
            f"    Unsigned: k={us_k}, sizes={[len(m) for m in us_modules]}"
        )
        logger.info(
            f"    Signed:   k={ss_k}, sizes={[len(m) for m in ss_modules]}, "
            f"frust={frust:.4f}"
        )
        logger.info(
            f"    HardThresh: k={ht_k}, sizes={[len(m) for m in ht_modules]}"
        )

        n_pos_coi = int(np.sum(CoI_full > 0) // 2)
        n_neg_coi = int(np.sum(CoI_full < 0) // 2)
        clustering_info[ds_name] = {
            "unsigned_spectral": {
                "k": us_k,
                "module_sizes": [len(m) for m in us_modules],
            },
            "signed_spectral": {
                "k": ss_k,
                "module_sizes": [len(m) for m in ss_modules],
                "frustration_index": round(frust, 6),
            },
            "hard_threshold": {
                "k": ht_k,
                "module_sizes": [len(m) for m in ht_modules],
            },
            "coi_time_s": round(coi_time, 2),
            "coi_subsample_n": n_sub,
            "n_valid_features": int(len(valid_feats)),
            "n_positive_coi_pairs": n_pos_coi,
            "n_negative_coi_pairs": n_neg_coi,
        }

        # --- Prepare representative examples ---
        unique_folds = sorted(np.unique(folds).astype(int))
        module_map = {
            "unsigned_spectral": us_modules,
            "signed_spectral": ss_modules,
            "hard_threshold": ht_modules,
            "axis_aligned": None,
            "random_oblique": None,
        }

        # Pick representative example indices: first test example per fold
        rep_indices: dict[int, int] = {}
        for fold_id in unique_folds:
            test_indices = np.where(folds == fold_id)[0]
            if len(test_indices) > 0:
                rep_indices[fold_id] = int(test_indices[0])

        # Storage for representative predictions: fold_id -> {method: pred_str}
        rep_preds: dict[int, dict[str, str]] = {fid: {} for fid in unique_folds}

        # --- 5-fold CV ---
        for method in METHODS:
            modules = module_map[method]

            for max_splits in MAX_SPLITS_VALUES:
                fold_metrics: list[dict] = []
                t_method_start = time.time()

                for fold_id in unique_folds:
                    test_mask = folds == fold_id
                    train_mask = ~test_mask
                    X_tr, y_tr = X[train_mask], y[train_mask]
                    X_te, y_te = X[test_mask], y[test_mask]

                    t_fit = time.time()
                    try:
                        model = FIGSModel(
                            max_splits=max_splits,
                            split_strategy=method,
                            feature_modules=modules,
                            task_type=task_type,
                            random_state=int(42 + fold_id),
                        )
                        model.fit(X_tr, y_tr)
                        fit_time = time.time() - t_fit

                        y_pred = model.predict(X_te)

                        # Record representative prediction (only at PREDICT_MAX_SPLITS)
                        if max_splits == PREDICT_MAX_SPLITS and fold_id in rep_indices:
                            pred_val = y_pred[0]  # First test example
                            if task_type in ("classification", "binary_classification"):
                                rep_preds[fold_id][method] = str(int(pred_val))
                            else:
                                rep_preds[fold_id][method] = f"{pred_val:.4f}"

                        # Compute metrics
                        fold_result: dict = {
                            "dataset": ds_name,
                            "method": method,
                            "max_splits": max_splits,
                            "fold": int(fold_id),
                            "n_train": int(len(X_tr)),
                            "n_test": int(len(X_te)),
                            "n_features": d,
                            "task_type": task_type,
                        }

                        if task_type in ("classification", "binary_classification"):
                            fold_result["balanced_accuracy"] = round(
                                float(balanced_accuracy_score(y_te, y_pred)), 6
                            )
                            try:
                                y_raw = model.predict_raw(X_te)
                                if len(np.unique(y_te)) == 2:
                                    fold_result["auc"] = round(
                                        float(roc_auc_score(y_te, y_raw)), 6
                                    )
                            except Exception:
                                pass
                        else:
                            fold_result["r2"] = round(
                                float(r2_score(y_te, y_pred)), 6
                            )

                        tree_met = model.get_metrics()
                        fold_result["total_splits"] = tree_met["total_splits"]
                        fold_result["avg_split_arity"] = tree_met["avg_split_arity"]
                        fold_result["avg_path_length"] = tree_met["avg_path_length"]
                        fold_result["fit_time_s"] = round(fit_time, 3)

                        all_results.append(fold_result)
                        fold_metrics.append(fold_result)

                    except Exception:
                        logger.exception(
                            f"  FIGS failed: {method} s={max_splits} fold={fold_id}"
                        )
                        err_result = {
                            "dataset": ds_name,
                            "method": method,
                            "max_splits": max_splits,
                            "fold": int(fold_id),
                            "error": True,
                        }
                        all_results.append(err_result)
                        fold_metrics.append(err_result)
                    finally:
                        try:
                            del model
                        except NameError:
                            pass
                        gc.collect()

                method_time = time.time() - t_method_start

                # Compute summary for this (dataset, method, max_splits)
                summary: dict = {
                    "dataset": ds_name,
                    "method": method,
                    "max_splits": max_splits,
                    "n_samples": n,
                    "n_features": d,
                    "task_type": task_type,
                }
                for key in [
                    "balanced_accuracy",
                    "auc",
                    "r2",
                    "total_splits",
                    "avg_split_arity",
                    "avg_path_length",
                    "fit_time_s",
                ]:
                    vals = [
                        fr[key]
                        for fr in fold_metrics
                        if key in fr and fr.get(key) is not None
                    ]
                    if vals:
                        summary[f"{key}_mean"] = round(float(np.mean(vals)), 6)
                        summary[f"{key}_std"] = round(float(np.std(vals)), 6)

                summary["method_total_time_s"] = round(method_time, 2)
                summary_results.append(summary)

                prim = summary.get("balanced_accuracy_mean", summary.get("r2_mean"))
                if prim is not None:
                    logger.info(
                        f"  {method} s={max_splits}: "
                        f"{prim:.4f} ({method_time:.1f}s)"
                    )

        # Assemble representative examples for this dataset
        ds_rep_examples = []
        for fold_id in unique_folds:
            if fold_id not in rep_indices:
                continue
            orig_idx = rep_indices[fold_id]
            orig_ex = ds["examples"][orig_idx]

            entry: dict = {
                "input": (
                    orig_ex["input"]
                    if isinstance(orig_ex["input"], str)
                    else json.dumps(orig_ex["input"])
                ),
                "output": str(orig_ex["output"]),
                "metadata_fold": int(fold_id),
                "metadata_task_type": task_type,
            }
            # Add predictions from all methods
            for method in METHODS:
                pred_key = f"predict_{method}"
                if method in rep_preds.get(fold_id, {}):
                    entry[pred_key] = rep_preds[fold_id][method]
                else:
                    # Fallback: "0" for classification, "0.0" for regression
                    if task_type in ("classification", "binary_classification"):
                        entry[pred_key] = "0"
                    else:
                        entry[pred_key] = "0.0"
            ds_rep_examples.append(entry)

        rep_examples[ds_name] = ds_rep_examples

        elapsed = time.time() - t_start
        logger.info(f"  Dataset {ds_name} done. Elapsed total: {elapsed:.0f}s")

        # Write checkpoint after validation datasets (adult + electricity)
        if ds_name == "electricity":
            _write_checkpoint(
                all_results, summary_results, clustering_info, rep_examples, t_start
            )
            # Estimate remaining time
            n_done = ds_idx + 1
            n_remaining = len(ordered_names) - n_done
            if n_done > 0:
                avg_time = elapsed / n_done
                est_remaining = avg_time * n_remaining * 1.5  # 1.5x safety for larger ds
                logger.info(
                    f"  Checkpoint: {n_done}/{len(ordered_names)} datasets done, "
                    f"elapsed={elapsed:.0f}s, est_remaining={est_remaining:.0f}s"
                )

        gc.collect()

    # ---- Phase 3: Save Output ----
    logger.info("=" * 60)
    logger.info("PHASE 3: Saving output")
    logger.info("=" * 60)

    elapsed = round(time.time() - t_start, 1)

    output = {
        "metadata": {
            "experiment_name": "balance_guided_oblique_trees_real_benchmark",
            "description": (
                "5 FIGS variants (axis-aligned, random-oblique, unsigned-spectral, "
                "signed-spectral, hard-threshold) on 8 Grinsztajn tabular datasets "
                "with CoI-guided feature clustering. Output contains aggregate "
                "metrics only (no per-example predictions) plus 5 representative "
                "examples per dataset for schema compliance."
            ),
            "methods": METHODS,
            "max_splits_for_predictions": PREDICT_MAX_SPLITS,
            "max_splits_values_tested": MAX_SPLITS_VALUES,
            "coi_subsample_strategy": f"min(n, {COI_SUBSAMPLE_N})",
            "mi_estimator": (
                "NPEET micd() k=5"
                if HAS_NPEET
                else "binning (10 quantile bins, sklearn mutual_info_score)"
            ),
            "n_datasets": len(datasets),
            "hardware": {"n_cpus": NUM_CPUS, "ram_gb": round(TOTAL_RAM_GB, 1)},
            "total_time_s": elapsed,
            "results_per_fold": all_results,
            "results_summary": summary_results,
            "clustering_info": clustering_info,
        },
        "datasets": [],
    }

    # Build datasets with representative examples for schema compliance
    for ds_name in ordered_names:
        exs = rep_examples.get(ds_name, [])
        if not exs:
            # Fallback: create minimal example from first data point
            ds = datasets[ds_name]
            orig_ex = ds["examples"][0]
            minimal = {
                "input": (
                    orig_ex["input"]
                    if isinstance(orig_ex["input"], str)
                    else json.dumps(orig_ex["input"])
                ),
                "output": str(orig_ex["output"]),
                "metadata_fold": int(orig_ex.get("metadata_fold", 0)),
                "metadata_task_type": ds["task_type"],
            }
            for method in METHODS:
                if ds["task_type"] in ("classification", "binary_classification"):
                    minimal[f"predict_{method}"] = "0"
                else:
                    minimal[f"predict_{method}"] = "0.0"
            exs = [minimal]

        output["datasets"].append({"dataset": ds_name, "examples": exs})

    # Write method_out.json
    out_path = WORKSPACE / "method_out.json"
    out_text = json.dumps(output, default=str)
    out_path.write_text(out_text)
    file_size_mb = out_path.stat().st_size / (1024 * 1024)
    logger.info(f"Output: method_out.json = {file_size_mb:.2f} MB")

    if file_size_mb > 50:
        logger.error(f"Output too large: {file_size_mb:.1f} MB > 50 MB limit!")
        raise RuntimeError(f"Output too large: {file_size_mb:.1f} MB")

    # Write full_method_out.json (identical since output is small)
    full_out_path = WORKSPACE / "full_method_out.json"
    full_out_path.write_text(out_text)
    logger.info(f"Output: full_method_out.json = {file_size_mb:.2f} MB")

    # Write preview_method_out.json
    preview = {
        "metadata": {
            "experiment_name": output["metadata"]["experiment_name"],
            "description": output["metadata"]["description"],
            "methods": METHODS,
            "n_datasets": output["metadata"]["n_datasets"],
            "total_time_s": elapsed,
            "results_summary": output["metadata"]["results_summary"][:20],
            "clustering_info": {
                k: v for k, v in list(clustering_info.items())[:3]
            },
        },
        "datasets": output["datasets"][:3],
    }
    preview_path = WORKSPACE / "preview_method_out.json"
    preview_path.write_text(json.dumps(preview, default=str))
    logger.info(
        f"Output: preview_method_out.json = "
        f"{preview_path.stat().st_size / (1024 * 1024):.2f} MB"
    )

    # Write mini_method_out.json
    mini = {
        "metadata": {
            "experiment_name": output["metadata"]["experiment_name"],
            "description": output["metadata"]["description"],
            "methods": METHODS,
            "n_datasets": output["metadata"]["n_datasets"],
            "total_time_s": elapsed,
            "results_summary": output["metadata"]["results_summary"][:5],
        },
        "datasets": output["datasets"][:3],
    }
    mini_path = WORKSPACE / "mini_method_out.json"
    mini_path.write_text(json.dumps(mini, default=str))
    logger.info(
        f"Output: mini_method_out.json = "
        f"{mini_path.stat().st_size / (1024 * 1024):.2f} MB"
    )

    # Summary stats
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total datasets: {len(ordered_names)}")
    logger.info(f"Total per-fold results: {len(all_results)}")
    logger.info(f"Total summary entries: {len(summary_results)}")
    logger.info(f"Representative examples: {sum(len(v) for v in rep_examples.values())}")
    logger.info(f"Total time: {elapsed}s")

    # Verify expected counts
    expected_fold = len(ordered_names) * len(METHODS) * len(MAX_SPLITS_VALUES) * 5
    expected_summary = len(ordered_names) * len(METHODS) * len(MAX_SPLITS_VALUES)
    logger.info(f"Expected per-fold: {expected_fold}, got: {len(all_results)}")
    logger.info(f"Expected summaries: {expected_summary}, got: {len(summary_results)}")

    return output


if __name__ == "__main__":
    main()

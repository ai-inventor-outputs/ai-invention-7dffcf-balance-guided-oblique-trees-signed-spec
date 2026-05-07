#!/usr/bin/env python3
"""Balance-Guided Oblique Trees: End-to-End Benchmark Experiment.

Compares Signed-Spectral FIGS against axis-aligned FIGS and random-oblique FIGS
across 5 Grinsztajn benchmark datasets, 5 CV folds, and 4 complexity levels.
Produces method_out.json conforming to exp_gen_sol_out.json schema.
"""

import gc
import json
import math
import os
import resource
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger

warnings.filterwarnings("ignore")

# =============================================================================
# Logging
# =============================================================================
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add(
    Path(__file__).parent / "logs" / "run.log",
    rotation="30 MB",
    level="DEBUG",
)

# =============================================================================
# Hardware detection and memory limits
# =============================================================================
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


def _container_ram_gb() -> float:
    for p in [
        "/sys/fs/cgroup/memory.max",
        "/sys/fs/cgroup/memory/memory.limit_in_bytes",
    ]:
        try:
            v = Path(p).read_text().strip()
            if v != "max" and int(v) < 1_000_000_000_000:
                return int(v) / 1e9
        except (FileNotFoundError, ValueError):
            pass
    return 29.0


NUM_CPUS = _detect_cpus()
TOTAL_RAM_GB = _container_ram_gb()
RAM_BUDGET_BYTES = int(TOTAL_RAM_GB * 0.7 * 1e9)
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET_BYTES * 3, RAM_BUDGET_BYTES * 3))
resource.setrlimit(resource.RLIMIT_CPU, (3500, 3500))

# =============================================================================
# Constants
# =============================================================================
WORKSPACE = Path(__file__).parent
DATA_DIR = Path("/ai-inventor/aii_pipeline/runs/jamnik-sgfigs-pid-v2/3_invention_loop/iter_1/gen_art/data_id4_it1__opus/full_data_out")
DATA_PARTS = [f"full_data_out_{i}.json" for i in range(1, 5)]
DATASETS = ["electricity", "adult", "california_housing", "jannis", "higgs_small"]
N_FOLDS = 5
MAX_SPLITS_GRID = [5, 10, 15, 20]
PREDICT_MAX_SPLITS = 10
COI_K_NEIGHBORS = 5
COI_SUBSAMPLE_N = 10000
RIDGE_ALPHA = 1.0
SPONGE_TAU_P = 1.0
SPONGE_TAU_N = 1.0
SPONGE_K_MAX = 10
SPONGE_EPS = 1e-10
RANDOM_STATE = 42
N_JOBS = max(1, NUM_CPUS - 1)
METHODS = ["axis_aligned_figs", "random_oblique_figs", "signed_spectral_figs"]
MIN_SAMPLES_LEAF = 5
NUM_REPETITIONS = 5
BEAM_SIZE_DEFAULT = 5

# =============================================================================
# NPEET import with fallback
# =============================================================================
try:
    import npeet.entropy_estimators as ee

    HAS_NPEET = True
    logger.info("NPEET loaded successfully")
except ImportError:
    HAS_NPEET = False
    logger.warning("NPEET not available, using sklearn fallback for MI")

# Lazy imports done at module level for clarity
from joblib import Parallel, delayed
from scipy.linalg import eigh, eigvalsh
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.linear_model import Ridge
from sklearn.metrics import balanced_accuracy_score, r2_score, roc_auc_score, silhouette_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeRegressor

# =============================================================================
# Data Loading
# =============================================================================
def load_all_datasets() -> dict:
    """Load all 5 datasets from the 4 JSON part files."""
    datasets: dict = {}

    for part_file in DATA_PARTS:
        path = DATA_DIR / part_file
        logger.info(f"Loading {part_file} ...")
        with open(path) as f:
            part = json.load(f)

        for ds_block in part["datasets"]:
            ds_name = ds_block["dataset"]
            examples = ds_block["examples"]
            # Get full feature names from first example's input keys
            feature_names_full = list(json.loads(examples[0]["input"]).keys())
            task_type = examples[0]["metadata_task_type"]

            X_rows, y_vals, fold_ids, row_indices, raw_examples = [], [], [], [], []
            for ex in examples:
                feat_dict = json.loads(ex["input"])
                X_rows.append([feat_dict[fn] for fn in feature_names_full])
                y_vals.append(ex["output"])
                fold_ids.append(ex["metadata_fold"])
                row_indices.append(ex["metadata_row_index"])
                raw_examples.append(ex)

            X = np.array(X_rows, dtype=np.float64)
            folds = np.array(fold_ids, dtype=int)
            row_idx = np.array(row_indices, dtype=int)

            if task_type == "classification":
                y = np.array([int(v) for v in y_vals], dtype=int)
                n_classes = examples[0].get("metadata_n_classes", len(set(y_vals)))
            else:
                y = np.array([float(v) for v in y_vals], dtype=np.float64)
                n_classes = None

            if ds_name in datasets:
                datasets[ds_name]["X"] = np.vstack([datasets[ds_name]["X"], X])
                datasets[ds_name]["y"] = np.concatenate([datasets[ds_name]["y"], y])
                datasets[ds_name]["folds"] = np.concatenate([datasets[ds_name]["folds"], folds])
                datasets[ds_name]["row_index"] = np.concatenate([datasets[ds_name]["row_index"], row_idx])
                datasets[ds_name]["raw_examples"].extend(raw_examples)
            else:
                datasets[ds_name] = {
                    "X": X, "y": y, "folds": folds,
                    "task_type": task_type,
                    "feature_names": feature_names_full,
                    "n_classes": n_classes,
                    "source": examples[0]["metadata_source"],
                    "row_index": row_idx,
                    "raw_examples": raw_examples,
                }

    # Sort each dataset by row_index
    for ds_name, ds in datasets.items():
        order = np.argsort(ds["row_index"])
        ds["X"] = ds["X"][order]
        ds["y"] = ds["y"][order]
        ds["folds"] = ds["folds"][order]
        ds["row_index"] = ds["row_index"][order]
        ds["raw_examples"] = [ds["raw_examples"][i] for i in order]
        # Replace NaN with 0
        nan_count = np.isnan(ds["X"]).sum()
        if nan_count > 0:
            logger.warning(f"  {ds_name}: replacing {nan_count} NaN values with 0")
            ds["X"] = np.nan_to_num(ds["X"], nan=0.0)
        logger.info(
            f"  {ds_name}: n={len(ds['y'])}, d={ds['X'].shape[1]}, "
            f"task={ds['task_type']}, classes={ds.get('n_classes')}, "
            f"folds={sorted(np.unique(ds['folds']).tolist())}"
        )

    return datasets


# =============================================================================
# Co-Information Matrix Computation
# =============================================================================
def _mi_individual(Xi: np.ndarray, y: np.ndarray, task_type: str, k: int = 5) -> float:
    """MI between single feature Xi and target y."""
    if Xi.ndim == 1:
        Xi = Xi.reshape(-1, 1)
    if task_type == "classification":
        if HAS_NPEET:
            # NPEET micd needs y as 2D for numpy>=2.0 compatibility
            val = ee.micd(Xi, y.reshape(-1, 1), k=k, warning=False)
        else:
            from sklearn.feature_selection import mutual_info_classif
            val = mutual_info_classif(Xi, y, n_neighbors=k, random_state=RANDOM_STATE)[0]
    else:
        if HAS_NPEET:
            val = ee.mi(Xi.tolist(), y.reshape(-1, 1).tolist(), k=k)
        else:
            from sklearn.feature_selection import mutual_info_regression
            val = mutual_info_regression(Xi, y, n_neighbors=k, random_state=RANDOM_STATE)[0]
    return max(0.0, float(val))


def _mi_joint(Xi: np.ndarray, Xj: np.ndarray, y: np.ndarray, task_type: str, k: int = 5) -> float:
    """Compute I({Xi,Xj}; Y) for one feature pair."""
    X_joint = np.column_stack([Xi, Xj])
    if task_type == "classification":
        if HAS_NPEET:
            return float(ee.micd(X_joint, y.reshape(-1, 1), k=k, warning=False))
        else:
            from sklearn.feature_selection._mutual_info import _compute_mi_cd
            return max(0.0, float(_compute_mi_cd(X_joint, y, n_neighbors=k)))
    else:
        if HAS_NPEET:
            return float(ee.mi(X_joint.tolist(), y.reshape(-1, 1).tolist(), k=k))
        else:
            from sklearn.feature_selection import mutual_info_regression
            from sklearn.decomposition import PCA
            proj = PCA(n_components=1).fit_transform(X_joint)
            return float(mutual_info_regression(proj, y, n_neighbors=k, random_state=RANDOM_STATE)[0])


def compute_coi_matrix(
    X: np.ndarray, y: np.ndarray, task_type: str, k: int = 5, n_jobs: int = -1
) -> tuple:
    """Compute d x d Co-Information matrix.

    CoI(Xi, Xj; Y) = I(Xi;Y) + I(Xj;Y) - I(Xi,Xj;Y)
    Positive = redundancy, Negative = synergy.
    """
    n, d = X.shape
    logger.debug(f"Computing CoI matrix: n={n}, d={d}, pairs={d*(d-1)//2}")

    # Step 1: Individual MI (cached)
    individual_mi = np.zeros(d)
    for i in range(d):
        individual_mi[i] = _mi_individual(X[:, i], y, task_type, k)
    logger.debug(f"Individual MI: min={individual_mi.min():.4f}, max={individual_mi.max():.4f}")

    # Step 2: All-pairs joint MI (parallelised)
    pairs = [(i, j) for i in range(d) for j in range(i + 1, d)]
    eff_jobs = min(n_jobs if n_jobs > 0 else N_JOBS, len(pairs))

    if len(pairs) <= 20 or eff_jobs <= 1:
        joint_mi_values = [
            _mi_joint(X[:, i], X[:, j], y, task_type, k)
            for i, j in pairs
        ]
    else:
        joint_mi_values = Parallel(n_jobs=eff_jobs, backend="loky")(
            delayed(_mi_joint)(X[:, i], X[:, j], y, task_type, k)
            for i, j in pairs
        )

    # Step 3: Assemble CoI matrix
    coi_matrix = np.zeros((d, d))
    for idx, (i, j) in enumerate(pairs):
        coi = individual_mi[i] + individual_mi[j] - joint_mi_values[idx]
        coi_matrix[i, j] = coi
        coi_matrix[j, i] = coi

    return coi_matrix, individual_mi


# =============================================================================
# SPONGE_sym Signed Spectral Clustering
# =============================================================================
def _safe_sqrt_inv_diag(D: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    dv = np.diag(D)
    inv_s = np.zeros_like(dv)
    nz = dv > eps
    inv_s[nz] = 1.0 / np.sqrt(dv[nz])
    return np.diag(inv_s)


def sponge_sym(
    coi_matrix: np.ndarray, k: int,
    tau_p: float = 1.0, tau_n: float = 1.0, eps: float = 1e-10,
) -> tuple:
    """SPONGE_sym via generalized eigenvalue problem with scipy.linalg.eigh."""
    d = coi_matrix.shape[0]
    A_pos = np.maximum(coi_matrix, 0)
    A_neg = np.maximum(-coi_matrix, 0)
    D_pos = np.diag(A_pos.sum(axis=1))
    D_neg = np.diag(A_neg.sum(axis=1))
    L_pos = D_pos - A_pos
    L_neg = D_neg - A_neg

    Dp_isq = _safe_sqrt_inv_diag(D_pos, eps)
    Dn_isq = _safe_sqrt_inv_diag(D_neg, eps)
    L_sym_pos = Dp_isq @ L_pos @ Dp_isq
    L_sym_neg = Dn_isq @ L_neg @ Dn_isq

    A_mat = L_sym_pos + tau_n * np.eye(d)
    B_mat = L_sym_neg + tau_p * np.eye(d) + eps * np.eye(d)

    eigenvalues, eigenvectors = eigh(A_mat, b=B_mat, subset_by_index=[0, k - 1])

    V = eigenvectors
    norms = np.linalg.norm(V, axis=1, keepdims=True)
    norms[norms < eps] = 1.0
    V_norm = V / norms

    labels = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE).fit_predict(V_norm)
    return labels, eigenvalues


def select_k_and_cluster(coi_matrix: np.ndarray, k_max: int = 10) -> tuple:
    """Select optimal k via eigengap + silhouette, then run SPONGE_sym."""
    d = coi_matrix.shape[0]
    k_max = min(k_max, max(2, d // 3), d - 1)
    if k_max < 2:
        k_max = 2

    A_pos = np.maximum(coi_matrix, 0)
    A_neg = np.maximum(-coi_matrix, 0)
    D_pos = np.diag(A_pos.sum(axis=1))
    D_neg = np.diag(A_neg.sum(axis=1))
    L_pos = D_pos - A_pos
    L_neg = D_neg - A_neg

    eps = SPONGE_EPS
    Dp_isq = _safe_sqrt_inv_diag(D_pos, eps)
    Dn_isq = _safe_sqrt_inv_diag(D_neg, eps)
    L_sym_pos = Dp_isq @ L_pos @ Dp_isq
    L_sym_neg = Dn_isq @ L_neg @ Dn_isq

    A_mat = L_sym_pos + SPONGE_TAU_N * np.eye(d)
    B_mat = L_sym_neg + SPONGE_TAU_P * np.eye(d) + eps * np.eye(d)

    all_evals, all_evecs = eigh(A_mat, b=B_mat, subset_by_index=[0, k_max - 1])

    gaps = np.diff(all_evals)
    if len(gaps) == 0:
        labels, evals = sponge_sym(coi_matrix, k=2)
        return labels, evals, 2, -1.0

    top3 = np.argsort(gaps)[-3:]
    candidates = sorted(set(idx + 1 for idx in top3 if idx + 1 >= 2))
    if not candidates:
        candidates = [2]

    best_k, best_sil = candidates[0], -1.0
    for kc in candidates:
        V = all_evecs[:, :kc]
        norms = np.linalg.norm(V, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1.0
        V_n = V / norms
        labs = KMeans(n_clusters=kc, n_init=20, random_state=RANDOM_STATE).fit_predict(V_n)
        if len(set(labs)) >= 2:
            sil = silhouette_score(V_n, labs)
            if sil > best_sil:
                best_sil = sil
                best_k = kc

    if best_sil < 0.0:
        best_k = max(2, int(np.ceil(np.sqrt(d / 2))))
        best_k = min(best_k, k_max)

    labels, eigenvalues = sponge_sym(coi_matrix, k=best_k)
    return labels, eigenvalues, best_k, best_sil


def compute_frustration_index(coi_matrix: np.ndarray) -> float:
    """Frustration = lambda_min / lambda_max of signed Laplacian."""
    D_bar = np.diag(np.sum(np.abs(coi_matrix), axis=1))
    L_sigma = D_bar - coi_matrix
    evals = eigvalsh(L_sigma)
    lam_min, lam_max = evals[0], evals[-1]
    if lam_max < 1e-12:
        return 0.0
    return float(max(0.0, lam_min) / lam_max)


def extract_modules(labels: np.ndarray, d: int) -> list:
    modules: dict = {}
    for i in range(d):
        modules.setdefault(int(labels[i]), []).append(i)
    return list(modules.values())


# =============================================================================
# Oblique FIGS Tree Framework
# =============================================================================
@dataclass
class ObliqueFIGSNode:
    feature: int = -1
    features: list = field(default_factory=list)
    weights: np.ndarray = field(default_factory=lambda: np.array([]))
    threshold: float = 0.0
    is_oblique: bool = False
    value: float = 0.0
    left: Optional["ObliqueFIGSNode"] = None
    right: Optional["ObliqueFIGSNode"] = None
    n_samples: int = 0

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


def _predict_single(node: ObliqueFIGSNode, x: np.ndarray) -> float:
    while not node.is_leaf:
        if node.is_oblique:
            proj = np.dot(x[node.features], node.weights)
        else:
            proj = x[node.feature]
        node = node.left if proj <= node.threshold else node.right
    return node.value


def _predict_tree(node: ObliqueFIGSNode, X: np.ndarray) -> np.ndarray:
    return np.array([_predict_single(node, X[i]) for i in range(len(X))])


def _get_leaves_and_masks(node: ObliqueFIGSNode, X: np.ndarray):
    """Return (leaves, masks) lists by DFS traversal."""
    leaves, masks = [], []

    def _recurse(nd, mask):
        if nd.is_leaf:
            leaves.append(nd)
            masks.append(mask)
            return
        if nd.is_oblique:
            proj = X[:, nd.features] @ nd.weights
        else:
            proj = X[:, nd.feature]
        _recurse(nd.left, mask & (proj <= nd.threshold))
        _recurse(nd.right, mask & (proj > nd.threshold))

    _recurse(node, np.ones(len(X), dtype=bool))
    return leaves, masks


def _fit_axis_aligned(X_leaf: np.ndarray, res: np.ndarray):
    """Best axis-aligned stump for a leaf's residuals."""
    if len(res) < 2 * MIN_SAMPLES_LEAF:
        return None
    stump = DecisionTreeRegressor(max_depth=1, min_samples_leaf=MIN_SAMPLES_LEAF, random_state=RANDOM_STATE)
    stump.fit(X_leaf, res)
    t = stump.tree_
    if t.feature[0] < 0 or t.n_leaves < 2:
        return None
    feat, thr = int(t.feature[0]), float(t.threshold[0])
    left_m = X_leaf[:, feat] <= thr
    n_l, n_r = int(left_m.sum()), int((~left_m).sum())
    if n_l < MIN_SAMPLES_LEAF or n_r < MIN_SAMPLES_LEAF:
        return None
    lv = float(np.mean(res[left_m]))
    rv = float(np.mean(res[~left_m]))
    gain = float(np.sum(res ** 2) - np.sum((res[left_m] - lv) ** 2) - np.sum((res[~left_m] - rv) ** 2))
    return {"gain": gain, "feature": feat, "threshold": thr, "is_oblique": False,
            "left_value": lv, "right_value": rv, "n_left": n_l, "n_right": n_r}


def _fit_oblique(X_leaf: np.ndarray, res: np.ndarray, feat_idx: list):
    """Best oblique split via Ridge + stump on a leaf's residuals."""
    if len(res) < 2 * MIN_SAMPLES_LEAF or len(feat_idx) < 2:
        return None
    X_sub = X_leaf[:, feat_idx]
    if np.var(X_sub, axis=0).max() < 1e-12:
        return None
    try:
        ridge = Ridge(alpha=RIDGE_ALPHA, fit_intercept=True)
        ridge.fit(X_sub, res)
        proj = X_sub @ ridge.coef_
    except Exception:
        return None
    if np.var(proj) < 1e-12:
        return None
    stump = DecisionTreeRegressor(max_depth=1, min_samples_leaf=MIN_SAMPLES_LEAF, random_state=RANDOM_STATE)
    stump.fit(proj.reshape(-1, 1), res)
    t = stump.tree_
    if t.feature[0] < 0 or t.n_leaves < 2:
        return None
    thr = float(t.threshold[0])
    left_m = proj <= thr
    n_l, n_r = int(left_m.sum()), int((~left_m).sum())
    if n_l < MIN_SAMPLES_LEAF or n_r < MIN_SAMPLES_LEAF:
        return None
    lv = float(np.mean(res[left_m]))
    rv = float(np.mean(res[~left_m]))
    gain = float(np.sum(res ** 2) - np.sum((res[left_m] - lv) ** 2) - np.sum((res[~left_m] - rv) ** 2))
    return {"gain": gain, "features": list(feat_idx), "weights": ridge.coef_.copy(),
            "threshold": thr, "is_oblique": True,
            "left_value": lv, "right_value": rv, "n_left": n_l, "n_right": n_r}


class BaseFIGSOblique:
    """Greedy tree-sum ensemble with oblique split support."""

    def __init__(self, max_splits: int = 10, beam_size: int = BEAM_SIZE_DEFAULT,
                 num_repetitions: int = NUM_REPETITIONS, random_state: int = RANDOM_STATE):
        self.max_splits = max_splits
        self.beam_size = beam_size
        self.num_repetitions = num_repetitions
        self.random_state = random_state
        self.trees_: list = []
        self.complexity_: int = 0

    def _get_feature_subsets(self, d: int, rng) -> list:
        raise NotImplementedError

    def fit(self, X: np.ndarray, y: np.ndarray):
        n, d = X.shape
        rng = np.random.default_rng(self.random_state)
        self.trees_ = []
        self.complexity_ = 0

        # Initialise with a single root leaf
        root = ObliqueFIGSNode(value=float(np.mean(y)), n_samples=n)
        self.trees_ = [root]

        for _ in range(self.max_splits):
            preds = np.zeros(n)
            for tree in self.trees_:
                preds += _predict_tree(tree, X)
            residuals = y - preds

            best_gain = 1e-10
            best_action = None  # ('split', leaf, info, idxs) | ('new', None, info, arange)

            # Option A: split existing leaves
            for tree in self.trees_:
                leaves, masks = _get_leaves_and_masks(tree, X)
                for leaf, mask in zip(leaves, masks):
                    idxs = np.where(mask)[0]
                    if len(idxs) < 2 * MIN_SAMPLES_LEAF:
                        continue
                    Xl, rl = X[idxs], residuals[idxs]

                    aa = _fit_axis_aligned(Xl, rl)
                    if aa and aa["gain"] > best_gain:
                        best_gain = aa["gain"]
                        best_action = ("split", leaf, aa, idxs)

                    for subset in self._get_feature_subsets(d, rng):
                        ob = _fit_oblique(Xl, rl, subset)
                        if ob and ob["gain"] > best_gain:
                            best_gain = ob["gain"]
                            best_action = ("split", leaf, ob, idxs)

            # Option B: new tree
            aa = _fit_axis_aligned(X, residuals)
            if aa and aa["gain"] > best_gain:
                best_gain = aa["gain"]
                best_action = ("new", None, aa, np.arange(n))

            for subset in self._get_feature_subsets(d, rng):
                ob = _fit_oblique(X, residuals, subset)
                if ob and ob["gain"] > best_gain:
                    best_gain = ob["gain"]
                    best_action = ("new", None, ob, np.arange(n))

            if best_action is None:
                break

            act, leaf, info, _ = best_action
            if act == "new":
                node = self._make_node(info)
                self.trees_.append(node)
            else:
                self._split_leaf(leaf, info)
            self.complexity_ += 1

        self._recompute_leaves(X, y)
        return self

    @staticmethod
    def _make_node(info: dict) -> ObliqueFIGSNode:
        node = ObliqueFIGSNode()
        if info["is_oblique"]:
            node.is_oblique = True
            node.features = info["features"]
            node.weights = info["weights"]
        else:
            node.feature = info["feature"]
        node.threshold = info["threshold"]
        node.left = ObliqueFIGSNode(value=info["left_value"], n_samples=info["n_left"])
        node.right = ObliqueFIGSNode(value=info["right_value"], n_samples=info["n_right"])
        node.n_samples = info["n_left"] + info["n_right"]
        return node

    @staticmethod
    def _split_leaf(leaf: ObliqueFIGSNode, info: dict):
        if info["is_oblique"]:
            leaf.is_oblique = True
            leaf.features = info["features"]
            leaf.weights = info["weights"]
        else:
            leaf.is_oblique = False
            leaf.feature = info["feature"]
        leaf.threshold = info["threshold"]
        leaf.left = ObliqueFIGSNode(value=info["left_value"], n_samples=info["n_left"])
        leaf.right = ObliqueFIGSNode(value=info["right_value"], n_samples=info["n_right"])

    def _recompute_leaves(self, X: np.ndarray, y: np.ndarray):
        n = len(y)
        for ti in range(len(self.trees_)):
            other = np.zeros(n)
            for tj, t in enumerate(self.trees_):
                if tj != ti:
                    other += _predict_tree(t, X)
            res = y - other
            leaves, masks = _get_leaves_and_masks(self.trees_[ti], X)
            for leaf, mask in zip(leaves, masks):
                idxs = np.where(mask)[0]
                leaf.value = float(np.mean(res[idxs])) if len(idxs) > 0 else 0.0

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.trees_:
            return np.zeros(len(X))
        return sum(_predict_tree(t, X) for t in self.trees_)

    def predict_class(self, X: np.ndarray) -> np.ndarray:
        raw = self.predict(X)
        return (raw > 0.5).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raw = np.clip(self.predict(X), 0.0, 1.0)
        return np.column_stack([1 - raw, raw])


class SignedSpectralFIGS(BaseFIGSOblique):
    """Our method: draws feature subsets from SPONGE spectral modules."""

    def __init__(self, spectral_modules: list, **kwargs):
        super().__init__(**kwargs)
        self.spectral_modules = spectral_modules

    def _get_feature_subsets(self, d: int, rng) -> list:
        valid = [m for m in self.spectral_modules if len(m) >= 2]
        if not valid:
            return [sorted(rng.choice(d, size=min(self.beam_size, d), replace=False).tolist())
                    for _ in range(self.num_repetitions)]
        subsets = []
        for _ in range(self.num_repetitions):
            mod = valid[rng.integers(len(valid))]
            if len(mod) <= self.beam_size:
                subsets.append(sorted(mod))
            else:
                subsets.append(sorted(rng.choice(mod, size=self.beam_size, replace=False).tolist()))
        return subsets


class RandomObliqueFIGS(BaseFIGSOblique):
    """Baseline: random feature subsets of matched size."""

    def _get_feature_subsets(self, d: int, rng) -> list:
        return [sorted(rng.choice(d, size=min(self.beam_size, d), replace=False).tolist())
                for _ in range(self.num_repetitions)]


class AxisAlignedFIGSCustom(BaseFIGSOblique):
    """Fallback axis-aligned FIGS (no oblique subsets)."""

    def _get_feature_subsets(self, d: int, rng) -> list:
        return []


class MultiClassOblique:
    """One-vs-Rest wrapper for multi-class with oblique FIGS."""

    def __init__(self, base_cls, n_classes: int, **kwargs):
        self.n_classes = n_classes
        ms = kwargs.get("max_splits", 10)
        kwargs["max_splits"] = max(1, ms // n_classes)
        self.models = [base_cls(**kwargs) for _ in range(n_classes)]
        self.trees_: list = []

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.trees_ = []
        for c in range(self.n_classes):
            self.models[c].fit(X, (y == c).astype(float))
            self.trees_.extend(self.models[c].trees_)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = np.column_stack([m.predict(X) for m in self.models])
        return np.argmax(scores, axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        scores = np.column_stack([m.predict(X) for m in self.models])
        exp_s = np.exp(scores - scores.max(axis=1, keepdims=True))
        return exp_s / exp_s.sum(axis=1, keepdims=True)


# =============================================================================
# Metrics
# =============================================================================
def compute_oblique_metrics(
    model, X_test: np.ndarray, y_test: np.ndarray,
    task_type: str, n_classes: Optional[int],
) -> tuple:
    """Compute accuracy + interpretability metrics; return (metrics_dict, y_pred)."""
    metrics: dict = {}

    if task_type == "classification":
        if n_classes and n_classes > 2:
            y_pred = model.predict(X_test)
        else:
            y_pred = model.predict_class(X_test) if hasattr(model, "predict_class") else model.predict(X_test)
        metrics["balanced_accuracy"] = float(balanced_accuracy_score(y_test, y_pred))
        if n_classes == 2:
            try:
                yp = model.predict_proba(X_test)[:, 1]
                metrics["auc_roc"] = float(roc_auc_score(y_test, yp)) if len(np.unique(yp)) > 1 else None
            except Exception:
                metrics["auc_roc"] = None
        else:
            metrics["auc_roc"] = None
    else:
        y_pred = model.predict(X_test)
        metrics["r2"] = float(r2_score(y_test, y_pred))

    # Interpretability
    total_splits, arities, depths = 0, [], []
    for root in (model.trees_ if hasattr(model, "trees_") else []):
        stack = [(root, 0)]
        while stack:
            nd, dep = stack.pop()
            if nd.is_leaf:
                depths.append(dep)
            else:
                total_splits += 1
                if nd.is_oblique and nd.weights is not None and len(nd.weights) > 0:
                    arities.append(max(int(np.sum(np.abs(nd.weights) > 1e-10)), 1))
                else:
                    arities.append(1)
                if nd.left:
                    stack.append((nd.left, dep + 1))
                if nd.right:
                    stack.append((nd.right, dep + 1))

    metrics["total_splits"] = total_splits
    metrics["n_trees"] = len(model.trees_) if hasattr(model, "trees_") else 0
    metrics["avg_split_arity"] = float(np.mean(arities)) if arities else 1.0
    metrics["max_split_arity"] = int(max(arities)) if arities else 1
    metrics["avg_path_length"] = float(np.mean(depths)) if depths else 0.0
    return metrics, y_pred


# =============================================================================
# Axis-aligned FIGS baseline (imodels)
# =============================================================================
def run_axis_aligned_figs(
    X_train, y_train, X_test, y_test, task_type, max_splits, n_classes
):
    """imodels FIGS baseline with fallback to custom implementation."""
    try:
        from imodels import FIGSClassifier, FIGSRegressor

        if task_type == "classification":
            model = FIGSClassifier(max_rules=max_splits, random_state=RANDOM_STATE)
        else:
            model = FIGSRegressor(max_rules=max_splits, random_state=RANDOM_STATE)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics: dict = {}
        if task_type == "classification":
            metrics["balanced_accuracy"] = float(balanced_accuracy_score(y_test, y_pred))
            if n_classes == 2:
                try:
                    yp = model.predict_proba(X_test)[:, 1]
                    metrics["auc_roc"] = float(roc_auc_score(y_test, yp)) if len(np.unique(yp)) > 1 else None
                except Exception:
                    metrics["auc_roc"] = None
            else:
                metrics["auc_roc"] = None
        else:
            metrics["r2"] = float(r2_score(y_test, y_pred))

        metrics["total_splits"] = int(getattr(model, "complexity_", max_splits))
        metrics["n_trees"] = len(model.trees_) if hasattr(model, "trees_") else 1
        metrics["avg_split_arity"] = 1.0
        metrics["max_split_arity"] = 1

        # Path length from tree structure
        path_lens = []
        for tree_root in (model.trees_ if hasattr(model, "trees_") else []):
            stack = [(tree_root, 0)]
            while stack:
                nd, dep = stack.pop()
                l = getattr(nd, "left", None)
                r = getattr(nd, "right", None)
                if l is None and r is None:
                    path_lens.append(dep)
                else:
                    if l is not None:
                        stack.append((l, dep + 1))
                    if r is not None:
                        stack.append((r, dep + 1))
        metrics["avg_path_length"] = float(np.mean(path_lens)) if path_lens else 0.0
        return metrics, y_pred

    except Exception as e:
        logger.warning(f"imodels FIGS failed ({e}), using custom fallback")
        model = AxisAlignedFIGSCustom(max_splits=max_splits)
        model.fit(X_train, y_train)
        return compute_oblique_metrics(model, X_test, y_test, task_type, n_classes)


# =============================================================================
# Main Experiment Loop
# =============================================================================
@logger.catch
def run_experiment():
    all_results: list = []
    all_predictions: dict = {}  # (ds, method, ms, fold) -> array

    datasets = load_all_datasets()
    t_total = time.time()

    for ds_idx, ds_name in enumerate(DATASETS):
        ds = datasets[ds_name]
        X_full, y_full = ds["X"], ds["y"]
        folds = ds["folds"]
        task_type = ds["task_type"]
        n_classes = ds.get("n_classes")
        d = X_full.shape[1]

        logger.info(f"{'='*60}")
        logger.info(f"Dataset {ds_idx+1}/5: {ds_name}, n={len(y_full)}, d={d}, task={task_type}")
        logger.info(f"{'='*60}")
        t_ds = time.time()

        for fold_id in range(N_FOLDS):
            t_fold = time.time()
            test_mask = folds == fold_id
            train_mask = ~test_mask
            X_train, y_train = X_full[train_mask], y_full[train_mask]
            X_test, y_test = X_full[test_mask], y_full[test_mask]
            logger.info(f"  Fold {fold_id}: train={len(y_train)}, test={len(y_test)}")

            # ---- Phase 1: CoI ----
            t0 = time.time()
            n_tr = len(y_train)
            if n_tr > COI_SUBSAMPLE_N:
                rng_sub = np.random.default_rng(RANDOM_STATE + fold_id)
                if task_type == "classification":
                    sss = StratifiedShuffleSplit(
                        n_splits=1, train_size=COI_SUBSAMPLE_N,
                        random_state=RANDOM_STATE + fold_id,
                    )
                    sub_idx, _ = next(sss.split(X_train, y_train))
                else:
                    sub_idx = rng_sub.choice(n_tr, size=COI_SUBSAMPLE_N, replace=False)
                X_sub, y_sub = X_train[sub_idx], y_train[sub_idx]
            else:
                X_sub, y_sub = X_train, y_train

            coi_matrix, individual_mi = compute_coi_matrix(X_sub, y_sub, task_type, k=COI_K_NEIGHBORS, n_jobs=N_JOBS)
            coi_time = time.time() - t0
            logger.info(f"    CoI: {coi_time:.1f}s, shape={coi_matrix.shape}, "
                        f"min={coi_matrix.min():.4f}, max={coi_matrix.max():.4f}")

            # ---- Phase 2: SPONGE ----
            t1 = time.time()
            nz = coi_matrix[np.triu_indices(d, k=1)]
            nz = nz[np.abs(nz) > 1e-12]
            degenerate = True
            if len(nz) > 0:
                frac_pos = float(np.mean(nz > 0))
                degenerate = frac_pos > 0.95 or frac_pos < 0.05

            if degenerate:
                try:
                    abs_coi = np.abs(coi_matrix)
                    np.fill_diagonal(abs_coi, 0)
                    k_fb = min(3, d - 1)
                    sc = SpectralClustering(n_clusters=k_fb, affinity="precomputed", random_state=RANDOM_STATE)
                    cluster_labels = sc.fit_predict(abs_coi)
                except Exception:
                    cluster_labels = np.zeros(d, dtype=int)
                    cluster_labels[d // 2:] = 1
                k_chosen = int(len(np.unique(cluster_labels)))
                silhouette_val = -1.0
            else:
                try:
                    cluster_labels, _, k_chosen, silhouette_val = select_k_and_cluster(coi_matrix, k_max=SPONGE_K_MAX)
                except Exception as e:
                    logger.warning(f"    SPONGE failed: {e}, fallback")
                    cluster_labels = np.zeros(d, dtype=int)
                    cluster_labels[d // 2:] = 1
                    k_chosen = 2
                    silhouette_val = -1.0

            frustration = compute_frustration_index(coi_matrix)
            modules = extract_modules(cluster_labels, d)
            sponge_time = time.time() - t1
            module_sizes = [len(m) for m in modules]
            logger.info(f"    SPONGE: k={k_chosen}, modules={module_sizes}, "
                        f"frust={frustration:.4f}, sil={silhouette_val:.3f}, degen={degenerate}")

            # ---- Phase 3: Scale ----
            scaler = MinMaxScaler()
            X_tr_s = scaler.fit_transform(X_train)
            X_te_s = scaler.transform(X_test)

            beam_size = max(2, int(np.median(module_sizes)))

            # ---- Phase 4: Train all methods ----
            for ms in MAX_SPLITS_GRID:
                base = {
                    "dataset": ds_name, "fold": fold_id, "max_splits": ms,
                    "n_train": len(y_train), "n_test": len(y_test), "d": d,
                    "task_type": task_type, "n_classes": n_classes,
                    "frustration_index": float(frustration), "k_chosen": k_chosen,
                    "module_sizes": module_sizes, "coi_time_sec": float(coi_time),
                    "sponge_time_sec": float(sponge_time), "degenerate_coi": degenerate,
                }

                # --- M1: axis-aligned FIGS ---
                t_m = time.time()
                try:
                    aa_met, aa_pred = run_axis_aligned_figs(X_tr_s, y_train, X_te_s, y_test, task_type, ms, n_classes)
                except Exception as e:
                    logger.exception(f"AA FIGS failed: {e}")
                    aa_met = {"balanced_accuracy": 0.5} if task_type == "classification" else {"r2": 0.0}
                    aa_met.update({"total_splits": 0, "n_trees": 0, "avg_split_arity": 1.0,
                                   "max_split_arity": 1, "avg_path_length": 0.0})
                    aa_pred = np.full(len(y_test), np.mean(y_train))
                aa_met["fit_time_sec"] = float(time.time() - t_m)
                all_results.append({**base, "method": "axis_aligned_figs", **aa_met})
                all_predictions[(ds_name, "axis_aligned_figs", ms, fold_id)] = aa_pred

                # --- M2: random oblique FIGS ---
                t_m = time.time()
                try:
                    if task_type == "classification" and n_classes and n_classes > 2:
                        ro = MultiClassOblique(RandomObliqueFIGS, n_classes=n_classes,
                                               max_splits=ms, beam_size=beam_size)
                    else:
                        ro = RandomObliqueFIGS(max_splits=ms, beam_size=beam_size)
                    ro.fit(X_tr_s, y_train)
                    ro_met, ro_pred = compute_oblique_metrics(ro, X_te_s, y_test, task_type, n_classes)
                except Exception as e:
                    logger.exception(f"RO FIGS failed: {e}")
                    ro_met = {"balanced_accuracy": 0.5} if task_type == "classification" else {"r2": 0.0}
                    ro_met.update({"total_splits": 0, "n_trees": 0, "avg_split_arity": 1.0,
                                   "max_split_arity": 1, "avg_path_length": 0.0})
                    ro_pred = np.full(len(y_test), np.mean(y_train))
                ro_met["fit_time_sec"] = float(time.time() - t_m)
                all_results.append({**base, "method": "random_oblique_figs", **ro_met})
                all_predictions[(ds_name, "random_oblique_figs", ms, fold_id)] = ro_pred

                # --- M3: signed spectral FIGS (ours) ---
                t_m = time.time()
                try:
                    if task_type == "classification" and n_classes and n_classes > 2:
                        ss = MultiClassOblique(SignedSpectralFIGS, n_classes=n_classes,
                                               spectral_modules=modules, max_splits=ms, beam_size=beam_size)
                    else:
                        ss = SignedSpectralFIGS(spectral_modules=modules, max_splits=ms, beam_size=beam_size)
                    ss.fit(X_tr_s, y_train)
                    ss_met, ss_pred = compute_oblique_metrics(ss, X_te_s, y_test, task_type, n_classes)
                except Exception as e:
                    logger.exception(f"SS FIGS failed: {e}")
                    ss_met = {"balanced_accuracy": 0.5} if task_type == "classification" else {"r2": 0.0}
                    ss_met.update({"total_splits": 0, "n_trees": 0, "avg_split_arity": 1.0,
                                   "max_split_arity": 1, "avg_path_length": 0.0})
                    ss_pred = np.full(len(y_test), np.mean(y_train))
                ss_met["fit_time_sec"] = float(time.time() - t_m)
                ss_met["total_pipeline_time_sec"] = float(coi_time + sponge_time + ss_met["fit_time_sec"])
                all_results.append({**base, "method": "signed_spectral_figs", **ss_met})
                all_predictions[(ds_name, "signed_spectral_figs", ms, fold_id)] = ss_pred

                # Quick log
                def _sc(m):
                    return m.get("balanced_accuracy", m.get("r2", "?"))
                logger.info(f"    ms={ms:2d}: AA={_sc(aa_met):.4f}  RO={_sc(ro_met):.4f}  SS={_sc(ss_met):.4f}")

            logger.info(f"    Fold {fold_id} done in {time.time()-t_fold:.1f}s")

        logger.info(f"  Dataset {ds_name} done in {time.time()-t_ds:.1f}s")
        gc.collect()

    logger.info(f"Total experiment: {time.time()-t_total:.1f}s")
    return all_results, all_predictions, datasets


# =============================================================================
# Output formatting (exp_gen_sol_out.json schema)
# =============================================================================
def build_output(all_results, all_predictions, datasets) -> dict:
    """Build method_out.json conforming to exp_gen_sol_out.json schema."""

    # --- Best max_splits per (dataset, method) ---
    best_ms: dict = {}
    for ds_name in DATASETS:
        for method in METHODS:
            scores_by_ms: dict = {}
            for r in all_results:
                if r["dataset"] == ds_name and r["method"] == method:
                    sc = r.get("balanced_accuracy", r.get("r2", 0))
                    scores_by_ms.setdefault(r["max_splits"], []).append(sc)
            if scores_by_ms:
                best_ms[(ds_name, method)] = max(scores_by_ms, key=lambda m: np.mean(scores_by_ms[m]))

    # --- Aggregated results ---
    agg_results = []
    for ds_name in DATASETS:
        for method in METHODS:
            for ms in MAX_SPLITS_GRID:
                rows = [r for r in all_results if r["dataset"] == ds_name and r["method"] == method and r["max_splits"] == ms]
                if not rows:
                    continue
                agg = {"dataset": ds_name, "method": method, "max_splits": ms,
                       "n_folds": len(rows), "task_type": rows[0]["task_type"]}
                if rows[0]["task_type"] == "classification":
                    accs = [r["balanced_accuracy"] for r in rows]
                    agg["balanced_accuracy_mean"] = float(np.mean(accs))
                    agg["balanced_accuracy_std"] = float(np.std(accs))
                    aucs = [r["auc_roc"] for r in rows if r.get("auc_roc") is not None]
                    if aucs:
                        agg["auc_roc_mean"] = float(np.mean(aucs))
                        agg["auc_roc_std"] = float(np.std(aucs))
                else:
                    r2s = [r["r2"] for r in rows]
                    agg["r2_mean"] = float(np.mean(r2s))
                    agg["r2_std"] = float(np.std(r2s))
                agg["total_splits_mean"] = float(np.mean([r["total_splits"] for r in rows]))
                agg["avg_split_arity_mean"] = float(np.mean([r["avg_split_arity"] for r in rows]))
                agg["avg_path_length_mean"] = float(np.mean([r["avg_path_length"] for r in rows]))
                agg["fit_time_sec_mean"] = float(np.mean([r["fit_time_sec"] for r in rows]))
                agg["frustration_index_mean"] = float(np.mean([r["frustration_index"] for r in rows]))
                agg_results.append(agg)

    # --- Frustration analysis ---
    frust_analysis = {}
    for ds_name in DATASETS:
        ds_rows = [r for r in all_results if r["dataset"] == ds_name]
        if not ds_rows:
            continue
        frust = float(np.mean([r["frustration_index"] for r in ds_rows]))
        ss = [r.get("balanced_accuracy", r.get("r2", 0)) for r in ds_rows if r["method"] == "signed_spectral_figs"]
        aa = [r.get("balanced_accuracy", r.get("r2", 0)) for r in ds_rows if r["method"] == "axis_aligned_figs"]
        diff = float(np.mean(ss) - np.mean(aa)) if ss and aa else 0.0
        frust_analysis[ds_name] = {"frustration_index": frust, "ss_minus_aa": diff}

    # --- Precompute test-set index maps ---
    test_idx_map: dict = {}
    for ds_name in DATASETS:
        folds_arr = datasets[ds_name]["folds"]
        for fid in range(N_FOLDS):
            test_idx_map[(ds_name, fid)] = np.where(folds_arr == fid)[0]

    # --- Build per-dataset examples ---
    output_datasets = []
    for ds_name in DATASETS:
        ds = datasets[ds_name]
        examples_out = []
        for i, ex in enumerate(ds["raw_examples"]):
            fold_id = ds["folds"][i]
            entry = {
                "input": ex["input"],
                "output": str(ex["output"]),
                "metadata_fold": ex["metadata_fold"],
                "metadata_task_type": ex["metadata_task_type"],
                "metadata_source": ex["metadata_source"],
                "metadata_row_index": ex["metadata_row_index"],
            }
            if "metadata_n_classes" in ex:
                entry["metadata_n_classes"] = ex["metadata_n_classes"]

            test_indices = test_idx_map[(ds_name, fold_id)]
            local_idx = np.searchsorted(test_indices, i)
            for method in METHODS:
                ms_use = best_ms.get((ds_name, method), PREDICT_MAX_SPLITS)
                key = (ds_name, method, ms_use, fold_id)
                if key in all_predictions and local_idx < len(all_predictions[key]) and test_indices[local_idx] == i:
                    pv = all_predictions[key][local_idx]
                    if ds["task_type"] == "classification":
                        entry[f"predict_{method}"] = str(int(round(float(pv))))
                    else:
                        entry[f"predict_{method}"] = str(round(float(pv), 6))
                else:
                    entry[f"predict_{method}"] = ""

            examples_out.append(entry)
        output_datasets.append({"dataset": ds_name, "examples": examples_out})

    return {
        "metadata": {
            "experiment": "balance_guided_oblique_trees_real_benchmarks",
            "hypothesis": "Signed spectral clustering of Co-Information graph improves oblique tree construction",
            "datasets": DATASETS,
            "methods": METHODS,
            "max_splits_grid": MAX_SPLITS_GRID,
            "best_max_splits": {f"{k[0]}__{k[1]}": v for k, v in best_ms.items()},
            "aggregated_results": agg_results,
            "frustration_analysis": frust_analysis,
            "n_folds": N_FOLDS,
            "coi_k_neighbors": COI_K_NEIGHBORS,
            "coi_subsample_n": COI_SUBSAMPLE_N,
        },
        "datasets": output_datasets,
    }


# =============================================================================
# Main
# =============================================================================
@logger.catch
def main():
    Path(WORKSPACE / "logs").mkdir(exist_ok=True)
    logger.info(f"Hardware: {NUM_CPUS} CPUs, {TOTAL_RAM_GB:.1f} GB RAM")
    logger.info(f"N_JOBS={N_JOBS}, RAM_BUDGET={RAM_BUDGET_BYTES/1e9:.1f} GB")

    all_results, all_predictions, datasets = run_experiment()

    output = build_output(all_results, all_predictions, datasets)

    out_path = WORKSPACE / "method_out.json"
    out_path.write_text(json.dumps(output, indent=2, default=str))
    logger.info(f"Saved {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")

    # Print summary table
    logger.info("=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    for agg in output["metadata"]["aggregated_results"]:
        if agg["task_type"] == "classification":
            metric = f"bal_acc={agg['balanced_accuracy_mean']:.4f}+/-{agg['balanced_accuracy_std']:.4f}"
        else:
            metric = f"r2={agg['r2_mean']:.4f}+/-{agg['r2_std']:.4f}"
        logger.info(f"  {agg['dataset']:20s} {agg['method']:25s} ms={agg['max_splits']:2d}: {metric}")


if __name__ == "__main__":
    main()

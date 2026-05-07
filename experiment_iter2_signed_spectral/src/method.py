#!/usr/bin/env python3
"""Signed Spectral Clustering Recovery on Synthetic Planted-Synergy Data.

Validates that SPONGE signed spectral clustering of the pairwise Co-Information
graph recovers planted synergistic modules in 6 synthetic datasets. Compares
against hard thresholding, unsigned spectral clustering, and random partition.
"""

import gc
import json
import math
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np
import scipy.linalg
from joblib import Parallel, delayed
from loguru import logger
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from scipy.special import digamma
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.model_selection import StratifiedShuffleSplit

# ═══════════════════════════════════════════════════════════════════════════════
# SETUP
# ═══════════════════════════════════════════════════════════════════════════════

WORKSPACE = Path(__file__).parent
DATA_DIR = Path(
    "/ai-inventor/aii_pipeline/runs/jamnik-sgfigs-pid-v2"
    "/3_invention_loop/iter_1/gen_art/data_id5_it1__opus"
)
LOG_DIR = WORKSPACE / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add(LOG_DIR / "run.log", rotation="30 MB", level="DEBUG")


# ── Hardware detection (cgroup-aware) ─────────────────────────────────────────

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
NUM_WORKERS = max(1, NUM_CPUS - 1)

# RAM budget: 50 % of container limit
RAM_BUDGET = int(min(14, TOTAL_RAM_GB * 0.5) * 1024**3)
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET * 3, RAM_BUDGET * 3))

logger.info(f"Hardware: {NUM_CPUS} CPUs, {TOTAL_RAM_GB:.1f} GB RAM, "
            f"{NUM_WORKERS} workers")
logger.info(f"RAM budget: {RAM_BUDGET / 1e9:.1f} GB")

# ── Constants ─────────────────────────────────────────────────────────────────

MASTER_SEED = 42
COI_K = 5
SUBSAMPLE_N = 10000
RANDOM_STATE = 42

VARIANT_ORDER = [
    "easy_2mod_xor",
    "medium_4mod_mixed",
    "hard_4mod_unequal",
    "overlapping_modules",
    "no_structure_control",
    "highdim_8mod",
]

# ═══════════════════════════════════════════════════════════════════════════════
# MI ESTIMATION  (Custom KSG / Ross-2014 estimator)
# ═══════════════════════════════════════════════════════════════════════════════

_MI_JITTER_RNG = np.random.RandomState(42)


def custom_micd(X_cont: np.ndarray, y_disc: np.ndarray, k: int = 5) -> float:
    """MI between continuous X (n, d) and discrete y (n,).

    Uses the KSG-type estimator for continuous-discrete MI (Ross 2014).
    Handles duplicates via tiny jitter and vectorises the radius counting
    with ``cKDTree.query_ball_point(..., return_length=True)``.
    """
    if X_cont.ndim == 1:
        X_cont = X_cont.reshape(-1, 1)

    n = X_cont.shape[0]
    if n <= k + 1:
        return 0.0

    # Constant features → MI = 0
    if np.all(np.ptp(X_cont, axis=0) < 1e-12):
        return 0.0

    classes, counts = np.unique(y_disc, return_counts=True)
    if len(classes) < 2:
        return 0.0

    # Tiny jitter to break ties (deterministic per call via fixed RNG copy)
    rng = np.random.RandomState(42)
    jitter = rng.randn(*X_cont.shape) * 1e-10
    X_j = X_cont + jitter

    tree_full = cKDTree(X_j)

    sum_psi_m = 0.0
    sum_psi_nc = 0.0

    for cls, cnt in zip(classes, counts):
        mask = y_disc == cls
        X_cls = X_j[mask]
        n_cls = int(cnt)

        if n_cls <= k:
            continue

        tree_cls = cKDTree(X_cls)
        dists, _ = tree_cls.query(X_cls, k=k + 1)
        eps = dists[:, -1]
        eps = np.maximum(eps, 1e-15)

        # Vectorised: count all neighbours within eps in full dataset
        m_counts = tree_full.query_ball_point(X_cls, eps, return_length=True)
        m_values = np.asarray(m_counts, dtype=np.float64) - 1  # exclude self
        m_values = np.maximum(m_values, 1)

        sum_psi_m += np.sum(digamma(m_values))
        sum_psi_nc += n_cls * digamma(n_cls)

    mi = digamma(k) + digamma(n) - sum_psi_m / n - sum_psi_nc / n
    return max(float(mi), 0.0)


# ── Try NPEET if available (fallback to custom) ──────────────────────────────

USE_NPEET = False
try:
    import npeet.entropy_estimators as ee
    _tx = np.random.randn(50, 1)
    _ty = (_tx[:, 0] > 0).astype(int).reshape(-1, 1)
    _tr = ee.micd(_tx, _ty, k=3, warning=False)
    if isinstance(_tr, (int, float)) and np.isfinite(_tr):
        USE_NPEET = True
        logger.info("Using NPEET for MI estimation")
    del _tx, _ty, _tr
except Exception as exc:
    logger.info(f"NPEET unavailable ({type(exc).__name__}), "
                "using custom KSG MI estimator")


def compute_mi(X_cont: np.ndarray, y_disc: np.ndarray, k: int = 5) -> float:
    """Compute MI(X; Y) for continuous X and discrete Y."""
    if USE_NPEET:
        if X_cont.ndim == 1:
            X_cont = X_cont.reshape(-1, 1)
        try:
            val = ee.micd(X_cont, y_disc.reshape(-1, 1), k=k, warning=False)
            return max(float(val), 0.0)
        except Exception:
            return custom_micd(X_cont, y_disc, k=k)
    return custom_micd(X_cont, y_disc, k=k)


# ═══════════════════════════════════════════════════════════════════════════════
# CO-INFORMATION MATRIX
# ═══════════════════════════════════════════════════════════════════════════════

def compute_coi_matrix(
    X: np.ndarray,
    y: np.ndarray,
    k: int = 5,
    n_subsample: int = 10000,
    n_jobs: int = -1,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute d x d Co-Information matrix.

    CoI[i,j] = I(Xi;Y) + I(Xj;Y) - I({Xi,Xj};Y)
    Negative → synergy,  Positive → redundancy.
    """
    n, d = X.shape
    if n_jobs == -1:
        n_jobs = NUM_WORKERS

    # ── Stratified subsample ──────────────────────────────────────────────
    if n > n_subsample:
        sss = StratifiedShuffleSplit(
            n_splits=1, train_size=n_subsample, random_state=RANDOM_STATE
        )
        idx, _ = next(sss.split(X, y))
        X_sub = X[idx].copy()
        y_sub = y[idx].copy()
        logger.info(f"  Subsampled {n} -> {n_subsample}")
    else:
        X_sub = X.copy()
        y_sub = y.copy()

    # ── Phase A: individual MI (d calls) ──────────────────────────────────
    logger.info(f"  Phase A: individual MI for {d} features ...")
    mi_individual = np.zeros(d)
    for i in range(d):
        mi_individual[i] = compute_mi(X_sub[:, i:i + 1], y_sub, k=k)
    logger.info(f"  Individual MI  [{mi_individual.min():.4f} .. "
                f"{mi_individual.max():.4f}]")

    # ── Phase B: joint MI (d*(d-1)/2 calls, parallelised) ─────────────────
    pairs = [(i, j) for i in range(d) for j in range(i + 1, d)]
    n_pairs = len(pairs)
    logger.info(f"  Phase B: joint MI for {n_pairs} pairs  "
                f"({n_jobs} workers) ...")

    def _pair_mi(i: int, j: int) -> tuple[int, int, float]:
        X_joint = np.column_stack([X_sub[:, i], X_sub[:, j]])
        return (i, j, compute_mi(X_joint, y_sub, k=k))

    t0 = time.time()
    results = Parallel(n_jobs=n_jobs, verbose=0, backend="loky")(
        delayed(_pair_mi)(i, j) for i, j in pairs
    )
    logger.info(f"  Phase B done in {time.time() - t0:.1f}s")

    # ── Phase C: assemble CoI matrix ──────────────────────────────────────
    CoI = np.zeros((d, d))
    for i, j, jmi in results:
        coi_val = mi_individual[i] + mi_individual[j] - jmi
        CoI[i, j] = coi_val
        CoI[j, i] = coi_val
    np.fill_diagonal(CoI, 0)

    return CoI, mi_individual


# ═══════════════════════════════════════════════════════════════════════════════
# SPONGE_SYM  SIGNED  SPECTRAL  CLUSTERING
# ═══════════════════════════════════════════════════════════════════════════════

def _safe_inv_sqrt(d_vec: np.ndarray) -> np.ndarray:
    out = np.zeros_like(d_vec)
    mask = d_vec > 1e-10
    out[mask] = 1.0 / np.sqrt(d_vec[mask])
    return out


def _build_sponge_matrices(
    CoI: np.ndarray,
    tau_p: float = 1.0,
    tau_n: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Build SPONGE_sym generalised eigenvalue matrices.

    SIGN CONVENTION: We negate CoI so that synergistic features
    (negative CoI) become *positive* edges in the SPONGE framework
    and are therefore kept **within** the same cluster.
    """
    d = CoI.shape[0]

    # Negate → synergy becomes the positive sub-graph
    signed_adj = -CoI

    A_pos = np.maximum(signed_adj, 0)   # synergy edges
    A_neg = np.maximum(-signed_adj, 0)  # redundancy edges

    d_pos = A_pos.sum(axis=1)
    d_neg = A_neg.sum(axis=1)

    D_pos_inv = np.diag(_safe_inv_sqrt(d_pos))
    D_neg_inv = np.diag(_safe_inv_sqrt(d_neg))

    L_pos = np.diag(d_pos) - A_pos
    L_neg = np.diag(d_neg) - A_neg

    L_sym_pos = D_pos_inv @ L_pos @ D_pos_inv
    L_sym_neg = D_neg_inv @ L_neg @ D_neg_inv

    A_mat = L_sym_pos + tau_n * np.eye(d)
    B_mat = L_sym_neg + tau_p * np.eye(d)
    B_mat += 1e-10 * np.eye(d)          # ridge for numerical stability

    return A_mat, B_mat


def sponge_sym(
    CoI: np.ndarray,
    k: int,
    tau_p: float = 1.0,
    tau_n: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """SPONGE_sym signed spectral clustering.

    Returns (labels, eigenvalues) for *k* clusters.
    """
    d = CoI.shape[0]
    k = max(2, min(k, d - 1))

    A_mat, B_mat = _build_sponge_matrices(CoI, tau_p, tau_n)

    try:
        eigenvalues, eigenvectors = scipy.linalg.eigh(
            A_mat, B_mat, subset_by_index=[0, k - 1]
        )
    except scipy.linalg.LinAlgError:
        logger.warning("SPONGE eigh failed — retrying with tau * 5")
        A2, B2 = _build_sponge_matrices(CoI, tau_p * 5, tau_n * 5)
        eigenvalues, eigenvectors = scipy.linalg.eigh(
            A2, B2, subset_by_index=[0, k - 1]
        )

    # Row-normalise eigenvectors (optional, helps KMeans)
    norms = np.linalg.norm(eigenvectors, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    eigenvectors_normed = eigenvectors / norms

    try:
        labels = KMeans(
            n_clusters=k, n_init=20, random_state=RANDOM_STATE
        ).fit_predict(eigenvectors_normed)
    except Exception:
        logger.warning("KMeans failed in SPONGE — falling back to modular split")
        labels = np.arange(d) % k

    return labels, eigenvalues


# ═══════════════════════════════════════════════════════════════════════════════
# AUTOMATIC  k  SELECTION  (eigengap + silhouette)
# ═══════════════════════════════════════════════════════════════════════════════

def select_k(
    CoI: np.ndarray,
    k_max: int | None = None,
    tau_p: float = 1.0,
    tau_n: float = 1.0,
) -> int:
    d = CoI.shape[0]
    if k_max is None:
        k_max = min(20, d // 3)
    k_max = max(k_max, 3)

    A_mat, B_mat = _build_sponge_matrices(CoI, tau_p, tau_n)
    n_eigs = min(k_max, d - 1)

    try:
        eigenvalues, eigenvectors = scipy.linalg.eigh(
            A_mat, B_mat, subset_by_index=[0, n_eigs - 1]
        )
    except scipy.linalg.LinAlgError:
        logger.warning("select_k eigh failed — using sqrt heuristic")
        return max(2, int(np.ceil(np.sqrt(d / 2))))

    gaps = np.diff(eigenvalues)
    if len(gaps) == 0:
        return 2

    top_idxs = np.argsort(gaps)[-min(3, len(gaps)):][::-1]
    top3_k = [int(idx + 2) for idx in top_idxs if 2 <= idx + 2 <= k_max]
    if not top3_k:
        top3_k = [2]

    best_k, best_sil = top3_k[0], -1.0
    for k_cand in top3_k:
        if k_cand > eigenvectors.shape[1]:
            continue
        try:
            lbl = KMeans(
                n_clusters=k_cand, n_init=20, random_state=RANDOM_STATE
            ).fit_predict(eigenvectors[:, :k_cand])
            if len(set(lbl)) < 2:
                continue
            sil = silhouette_score(eigenvectors[:, :k_cand], lbl)
            if sil > best_sil:
                best_k, best_sil = k_cand, sil
        except Exception:
            continue

    # Fallback when eigengap is ambiguous
    if len(gaps) > 0 and gaps.max() < 2 * np.median(gaps) and best_sil < 0.1:
        best_k = max(2, int(np.ceil(np.sqrt(d / 2))))

    logger.info(f"  Auto-k: k={best_k}  (silhouette={best_sil:.3f})")
    return best_k


# ═══════════════════════════════════════════════════════════════════════════════
# FRUSTRATION  INDEX
# ═══════════════════════════════════════════════════════════════════════════════

def compute_frustration_index(CoI: np.ndarray) -> float:
    """lambda_min / lambda_max of the signed Laplacian.

    Low → nearly balanced (clean modules).
    High → frustrated (no clean partition).
    """
    D_bar = np.diag(np.abs(CoI).sum(axis=1))
    L_sigma = D_bar - CoI

    try:
        evals = scipy.linalg.eigvalsh(L_sigma)
    except scipy.linalg.LinAlgError:
        return 0.0

    lmin = max(evals[0], 0.0)
    lmax = evals[-1]
    if lmax < 1e-10:
        return 0.0
    return float(lmin / lmax)


# ═══════════════════════════════════════════════════════════════════════════════
# BASELINES
# ═══════════════════════════════════════════════════════════════════════════════

def hard_threshold_baseline(
    CoI: np.ndarray,
    quantile: float = 0.9,
) -> np.ndarray:
    """Baseline A: threshold synergy at *quantile*, connected components."""
    d = CoI.shape[0]
    synergy = np.maximum(-CoI, 0)

    upper = synergy[np.triu_indices(d, k=1)]
    nz = upper[upper > 1e-10]

    if len(nz) == 0:
        return np.arange(d)

    thr = np.quantile(nz, quantile)
    adj = (synergy >= thr).astype(np.int32)
    np.fill_diagonal(adj, 0)

    _, labels = connected_components(adj, directed=False)
    return labels


def unsigned_spectral_baseline(CoI: np.ndarray, k: int) -> np.ndarray:
    """Baseline B: standard spectral clustering on |CoI|."""
    d = CoI.shape[0]
    k = max(2, min(k, d - 1))

    W = np.abs(CoI)
    np.fill_diagonal(W, 0)

    dv = W.sum(axis=1)
    Di = np.diag(_safe_inv_sqrt(dv))
    L = np.diag(dv) - W
    Ls = Di @ L @ Di

    try:
        _, vecs = scipy.linalg.eigh(Ls, subset_by_index=[0, k - 1])
    except scipy.linalg.LinAlgError:
        return np.arange(d) % k

    try:
        return KMeans(
            n_clusters=k, n_init=20, random_state=RANDOM_STATE
        ).fit_predict(vecs)
    except Exception:
        return np.arange(d) % k


def random_partition_baseline(d: int, k: int, seed: int = 42) -> np.ndarray:
    """Baseline C: random assignment."""
    return np.random.default_rng(seed).integers(0, k, size=d)


# ═══════════════════════════════════════════════════════════════════════════════
# EVALUATION  METRICS
# ═══════════════════════════════════════════════════════════════════════════════

def construct_ground_truth_labels(meta: dict) -> np.ndarray:
    """Feature-level ground-truth labels.

    * Synergistic feature → module index (first assignment wins for overlaps)
    * Redundant copy      → same label as its source
    * Noise / other       → unique singleton label
    """
    d = meta["n_features"]
    labels = np.full(d, -1, dtype=int)
    gt_modules = meta.get("ground_truth_modules", [])

    for mod_idx, mod in enumerate(gt_modules):
        for feat in mod:
            if labels[feat] == -1:
                labels[feat] = mod_idx

    for src, cpy in meta.get("redundant_pairs", []):
        if labels[src] >= 0 and labels[cpy] == -1:
            labels[cpy] = labels[src]

    next_lbl = len(gt_modules)
    for feat in range(d):
        if labels[feat] == -1:
            labels[feat] = next_lbl
            next_lbl += 1

    return labels


def compute_synergistic_pair_jaccard(
    gt_modules: list[list[int]],
    pred_labels: np.ndarray,
) -> float:
    """Jaccard over same-module feature pairs."""
    S_true: set[tuple[int, int]] = set()
    for mod in gt_modules:
        for ii in range(len(mod)):
            for jj in range(ii + 1, len(mod)):
                S_true.add((min(mod[ii], mod[jj]), max(mod[ii], mod[jj])))

    if not S_true:
        return 1.0

    gt_feats = sorted({f for mod in gt_modules for f in mod})
    S_pred: set[tuple[int, int]] = set()
    for ii in range(len(gt_feats)):
        for jj in range(ii + 1, len(gt_feats)):
            fi, fj = gt_feats[ii], gt_feats[jj]
            if pred_labels[fi] == pred_labels[fj]:
                S_pred.add((min(fi, fj), max(fi, fj)))

    inter = len(S_true & S_pred)
    union = len(S_true | S_pred)
    return inter / union if union else 1.0


def compute_xor_recovery(
    gt_modules: list[list[int]],
    module_types: list[str],
    pred_labels: np.ndarray,
) -> float | None:
    """Fraction of XOR modules whose features all land in ONE cluster."""
    xor_mods = [m for m, t in zip(gt_modules, module_types) if "xor" in t]
    if not xor_mods:
        return None
    recovered = sum(
        1 for m in xor_mods
        if len({int(pred_labels[f]) for f in m}) == 1
    )
    return recovered / len(xor_mods)


def compute_module_focused_ari(
    gt_labels: np.ndarray,
    pred_labels: np.ndarray,
    meta: dict,
) -> float | None:
    """ARI computed only on module + redundant features (ignoring noise).

    This avoids the inflation of singleton noise labels that depresses
    global ARI even when modules are recovered perfectly.
    """
    module_feats: set[int] = set()
    for mod in meta.get("ground_truth_modules", []):
        module_feats.update(mod)
    for src, cpy in meta.get("redundant_pairs", []):
        module_feats.add(src)
        module_feats.add(cpy)

    if len(module_feats) < 2:
        return None

    idx = sorted(module_feats)
    return float(adjusted_rand_score(gt_labels[idx], pred_labels[idx]))


# ═══════════════════════════════════════════════════════════════════════════════
# DATA  LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_data() -> list[dict]:
    """Load data by importing generator functions from data.py."""
    logger.info("Loading data from generators ...")

    sys.path.insert(0, str(DATA_DIR))
    try:
        from data import GENERATORS  # type: ignore[import-untyped]
    except ImportError:
        logger.warning("data.py import failed — loading from JSON files")
        return _load_data_from_json()

    # Restore our logger (data.py may modify it)
    logger.remove()
    logger.add(sys.stdout, level="INFO",
               format="{time:HH:mm:ss}|{level:<7}|{message}")
    logger.add(LOG_DIR / "run.log", rotation="30 MB", level="DEBUG")

    base_rng = np.random.default_rng(MASTER_SEED)
    variant_seeds = [
        int(base_rng.integers(0, 2**31)) for _ in range(len(GENERATORS))
    ]

    results = []
    for (name, gen_fn), seed in zip(GENERATORS, variant_seeds):
        rng = np.random.default_rng(seed)
        t0 = time.time()
        result = gen_fn(rng)
        logger.info(f"  {name}: {result['X'].shape} in {time.time()-t0:.2f}s")
        results.append(result)

    return results


def _load_data_from_json() -> list[dict]:
    """Fallback: parse the full_data_out JSON files."""
    logger.info("Loading data from JSON files ...")
    preview = json.loads((DATA_DIR / "preview_data_out.json").read_text())
    metadata = preview["metadata"]

    full_dir = DATA_DIR / "full_data_out"
    json_files = sorted(full_dir.glob("full_data_out_*.json"))

    all_ex: dict[str, list[dict]] = {}
    for jf in json_files:
        data = json.loads(jf.read_text())
        for ds in data.get("datasets", []):
            all_ex.setdefault(ds["dataset"], []).extend(ds["examples"])

    results = []
    for name in VARIANT_ORDER:
        if name not in all_ex:
            continue
        examples = all_ex[name]
        meta = metadata["variants"][name]
        d = meta["n_features"]
        n = len(examples)

        X = np.zeros((n, d))
        y = np.zeros(n, dtype=int)
        folds = np.zeros(n, dtype=int)

        for i, ex in enumerate(examples):
            fd = json.loads(ex["input"])
            for j in range(d):
                X[i, j] = fd[f"X{j}"]
            y[i] = int(ex["output"])
            folds[i] = ex.get("metadata_fold", 0)

        results.append(
            {"name": name, "X": X, "y": y, "folds": folds, "meta": meta}
        )
        logger.info(f"  {name}: ({n}, {d})")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN  EXPERIMENT  LOOP
# ═══════════════════════════════════════════════════════════════════════════════

@logger.catch
def main() -> None:
    overall_t0 = time.time()
    logger.info("=" * 60)
    logger.info("Signed Spectral Clustering Recovery Experiment")
    logger.info("=" * 60)

    # ── 1. Load data ──────────────────────────────────────────────────────
    results = load_data()

    # ── 2. Per-variant processing ─────────────────────────────────────────
    all_variant_results: dict[str, dict] = {}
    output_datasets: list[dict] = []

    for variant_result in results:
        name = variant_result["name"]
        X = variant_result["X"]
        y = variant_result["y"]
        meta = variant_result["meta"]
        d = X.shape[1]

        logger.info(f"\n{'─'*55}")
        logger.info(f"  {name}  (n={X.shape[0]}, d={d})")
        logger.info(f"{'─'*55}")

        # ── CoI matrix ────────────────────────────────────────────────────
        # For highdim: subsample more aggressively if needed
        subsample_n = SUBSAMPLE_N
        if d >= 100:
            subsample_n = min(SUBSAMPLE_N, 8000)
            logger.info(f"  High-dim: subsample reduced to {subsample_n}")

        t0 = time.time()
        CoI, mi_individual = compute_coi_matrix(
            X, y, k=COI_K, n_subsample=subsample_n, n_jobs=NUM_WORKERS,
        )
        t_coi = time.time() - t0
        logger.info(f"  CoI matrix: {t_coi:.1f}s")

        # ── Diagnostics ──────────────────────────────────────────────────
        gt_modules = meta.get("ground_truth_modules", [])
        module_types = meta.get("module_types", [])

        xor_mi_report: list[dict] = []
        for mod, mtype in zip(gt_modules, module_types):
            if "xor" in mtype:
                for feat in mod:
                    mi_val = float(mi_individual[feat])
                    xor_mi_report.append(
                        {"feature": int(feat), "mi": round(mi_val, 6)}
                    )
                    if mi_val > 0.05:
                        logger.warning(
                            f"  XOR feat {feat} MI={mi_val:.4f} (expected ~0)"
                        )

        coi_sign_report: list[dict] = []
        for mod, mtype in zip(gt_modules, module_types):
            if len(mod) >= 2:
                cv = float(CoI[mod[0], mod[1]])
                coi_sign_report.append({
                    "pair": [int(mod[0]), int(mod[1])],
                    "coi": round(cv, 6),
                    "is_synergistic": cv < 0,
                    "module_type": mtype,
                })
                if "xor" in mtype and cv >= 0:
                    logger.warning(
                        f"  Expected synergy for {mtype} pair "
                        f"{mod[:2]}, got CoI={cv:.4f}"
                    )

        # ── Frustration index ─────────────────────────────────────────────
        frustration = compute_frustration_index(CoI)
        logger.info(f"  Frustration index: {frustration:.4f}")

        # ── Ground truth ──────────────────────────────────────────────────
        gt_k = len(gt_modules) if gt_modules else 0
        gt_labels = construct_ground_truth_labels(meta)

        # ── SPONGE (auto k) ──────────────────────────────────────────────
        t0 = time.time()
        auto_k = select_k(CoI) if d >= 4 else max(gt_k, 2)
        sponge_auto_lbl, sponge_auto_ev = sponge_sym(CoI, k=auto_k)
        t_sponge_auto = time.time() - t0
        logger.info(f"  SPONGE auto  k={auto_k}: {t_sponge_auto:.2f}s")

        # ── SPONGE (oracle k) ────────────────────────────────────────────
        t0 = time.time()
        if gt_k >= 2:
            sponge_orc_lbl, sponge_orc_ev = sponge_sym(CoI, k=gt_k)
        else:
            sponge_orc_lbl = np.arange(d)
            sponge_orc_ev = np.array([])
        t_sponge_orc = time.time() - t0
        logger.info(f"  SPONGE oracle k={gt_k}: {t_sponge_orc:.2f}s")

        # ── Baselines ─────────────────────────────────────────────────────
        k_bl = gt_k if gt_k >= 2 else auto_k

        t0 = time.time()
        ht_lbl = hard_threshold_baseline(CoI)
        t_ht = time.time() - t0

        t0 = time.time()
        us_lbl = unsigned_spectral_baseline(CoI, k=k_bl)
        t_us = time.time() - t0

        t0 = time.time()
        rn_lbl = random_partition_baseline(d, k=k_bl)
        t_rn = time.time() - t0

        # ── Evaluate ──────────────────────────────────────────────────────
        methods = {
            "sponge_auto_k": {
                "labels": sponge_auto_lbl, "k": int(auto_k),
                "time": t_sponge_auto,
            },
            "sponge_oracle_k": {
                "labels": sponge_orc_lbl, "k": int(gt_k),
                "time": t_sponge_orc,
            },
            "hard_threshold": {
                "labels": ht_lbl,
                "k": int(len(set(ht_lbl.tolist()))),
                "time": t_ht,
            },
            "unsigned_spectral": {
                "labels": us_lbl, "k": int(k_bl), "time": t_us,
            },
            "random_partition": {
                "labels": rn_lbl, "k": int(k_bl), "time": t_rn,
            },
        }

        variant_metrics: dict = {
            "name": name,
            "n_samples": int(X.shape[0]),
            "n_features": int(d),
            "gt_n_modules": int(gt_k),
            "frustration_index": round(float(frustration), 6),
            "coi_computation_time_sec": round(float(t_coi), 2),
            "xor_marginal_mi": xor_mi_report,
            "coi_sign_diagnostics": coi_sign_report,
            "methods": {},
        }

        for mname, mdata in methods.items():
            pred = mdata["labels"]

            if gt_k >= 1:
                ari = float(adjusted_rand_score(gt_labels, pred))
                mf_ari = compute_module_focused_ari(
                    gt_labels, pred, meta,
                )
                jac = float(
                    compute_synergistic_pair_jaccard(gt_modules, pred)
                )
                xrf = compute_xor_recovery(gt_modules, module_types, pred)
            else:
                ari = mf_ari = jac = None
                xrf = None

            csizes = []
            if len(pred) > 0:
                for lbl in sorted(set(pred.tolist())):
                    csizes.append(int(np.sum(pred == lbl)))

            variant_metrics["methods"][mname] = {
                "k_used": mdata["k"],
                "adjusted_rand_index": (
                    round(ari, 6) if ari is not None else None
                ),
                "module_focused_ari": (
                    round(mf_ari, 6) if mf_ari is not None else None
                ),
                "synergistic_pair_jaccard": (
                    round(jac, 6) if jac is not None else None
                ),
                "xor_recovery_fraction": (
                    round(float(xrf), 6) if xrf is not None else None
                ),
                "time_sec": round(float(mdata["time"]), 3),
                "cluster_sizes": csizes,
            }

            a_s = f"ARI={ari:.3f}" if ari is not None else "ARI=N/A"
            m_s = f"mfARI={mf_ari:.3f}" if mf_ari is not None else ""
            j_s = f"Jac={jac:.3f}" if jac is not None else "Jac=N/A"
            x_s = f"XOR={xrf:.3f}" if xrf is not None else "XOR=N/A"
            logger.info(f"    {mname}: {a_s} {m_s} {j_s}  {x_s}")

        all_variant_results[name] = variant_metrics

        # ── Build output dataset (feature-level examples) ────────────────
        examples: list[dict] = []
        for fi in range(d):
            is_syn = any(fi in mod for mod in gt_modules)
            is_red = any(fi == p[1] for p in meta.get("redundant_pairs", []))
            is_noi = fi in meta.get("noise_features", [])

            ex: dict = {
                "input": json.dumps({
                    "variant": name,
                    "feature_index": fi,
                    "feature_name": f"X{fi}",
                    "n_samples": int(X.shape[0]),
                    "n_features": int(d),
                }),
                "output": str(int(gt_labels[fi])),
                "metadata_variant": name,
                "metadata_feature_index": fi,
                "metadata_individual_mi": round(float(mi_individual[fi]), 6),
                "metadata_is_synergistic": is_syn,
                "metadata_is_redundant": is_red,
                "metadata_is_noise": is_noi,
            }
            for mname, mdata in methods.items():
                ex[f"predict_{mname}"] = str(int(mdata["labels"][fi]))
            examples.append(ex)

        output_datasets.append({"dataset": name, "examples": examples})

        # free memory
        del X, y, CoI, mi_individual
        gc.collect()

    # ── 3. Summary ────────────────────────────────────────────────────────
    structured = [
        v for v in all_variant_results.values() if v["gt_n_modules"] >= 1
    ]

    def _mean(method: str, metric: str) -> float | None:
        vals = [
            v["methods"][method][metric]
            for v in structured
            if v["methods"].get(method, {}).get(metric) is not None
        ]
        return round(sum(vals) / len(vals), 6) if vals else None

    summary: dict = {
        # ── ARI (affected by singleton noise labels) ──
        "sponge_oracle_mean_ari": _mean("sponge_oracle_k",
                                        "adjusted_rand_index"),
        "sponge_auto_mean_ari": _mean("sponge_auto_k",
                                      "adjusted_rand_index"),
        "hard_threshold_mean_ari": _mean("hard_threshold",
                                         "adjusted_rand_index"),
        "unsigned_spectral_mean_ari": _mean("unsigned_spectral",
                                            "adjusted_rand_index"),
        "random_mean_ari": _mean("random_partition",
                                 "adjusted_rand_index"),
        # ── Module-focused ARI (noise features excluded) ──
        "sponge_oracle_mean_mfari": _mean("sponge_oracle_k",
                                          "module_focused_ari"),
        "sponge_auto_mean_mfari": _mean("sponge_auto_k",
                                        "module_focused_ari"),
        "hard_threshold_mean_mfari": _mean("hard_threshold",
                                           "module_focused_ari"),
        "unsigned_spectral_mean_mfari": _mean("unsigned_spectral",
                                              "module_focused_ari"),
        "random_mean_mfari": _mean("random_partition",
                                   "module_focused_ari"),
        # ── Jaccard (most informative for module recovery) ──
        "sponge_oracle_mean_jaccard": _mean("sponge_oracle_k",
                                            "synergistic_pair_jaccard"),
        "sponge_auto_mean_jaccard": _mean("sponge_auto_k",
                                          "synergistic_pair_jaccard"),
        "hard_threshold_mean_jaccard": _mean("hard_threshold",
                                             "synergistic_pair_jaccard"),
        "unsigned_spectral_mean_jaccard": _mean("unsigned_spectral",
                                                "synergistic_pair_jaccard"),
        "random_mean_jaccard": _mean("random_partition",
                                     "synergistic_pair_jaccard"),
    }

    # ── Comparison flags (use Jaccard — most meaningful metric) ──
    so_j = summary.get("sponge_oracle_mean_jaccard") or 0
    ht_j = summary.get("hard_threshold_mean_jaccard") or 0
    us_j = summary.get("unsigned_spectral_mean_jaccard") or 0
    rn_j = summary.get("random_mean_jaccard") or 0
    summary["sponge_beats_hard_threshold"] = so_j > ht_j
    summary["sponge_beats_unsigned"] = so_j > us_j
    summary["sponge_beats_random"] = so_j > rn_j

    easy_xor = (
        all_variant_results
        .get("easy_2mod_xor", {})
        .get("methods", {})
        .get("sponge_oracle_k", {})
        .get("xor_recovery_fraction")
    )
    summary["xor_features_recovered"] = (
        easy_xor == 1.0 if easy_xor is not None else False
    )

    # ── Frustration diagnostic ──
    ctrl_frust = (
        all_variant_results
        .get("no_structure_control", {})
        .get("frustration_index", 0)
    )
    easy_frust = (
        all_variant_results
        .get("easy_2mod_xor", {})
        .get("frustration_index", 0)
    )
    # Check if control has higher frustration than the easiest structured
    # variant (the one with cleanest module separation)
    summary["frustration_diagnostic_works"] = ctrl_frust > easy_frust

    total_wall = time.time() - overall_t0

    # ── 4. Write output ──────────────────────────────────────────────────
    output = {
        "metadata": {
            "experiment": "signed_spectral_recovery_synthetic",
            "hypothesis_test":
                "Does SPONGE recover planted synergistic modules?",
            "methods_compared": [
                "sponge_auto_k", "sponge_oracle_k",
                "hard_threshold", "unsigned_spectral", "random_partition",
            ],
            "summary": summary,
            "per_variant": all_variant_results,
            "total_wallclock_sec": round(total_wall, 2),
        },
        "datasets": output_datasets,
    }

    out_path = WORKSPACE / "method_out.json"
    out_path.write_text(json.dumps(output, indent=2))
    logger.info(f"\nOutput → {out_path}")
    logger.info(f"Total wall-clock: {total_wall:.1f}s")

    logger.info("\n=== SUMMARY ===")
    for k, v in summary.items():
        logger.info(f"  {k}: {v}")


# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    main()

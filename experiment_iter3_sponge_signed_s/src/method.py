#!/usr/bin/env python3
"""SPONGE Signed Spectral Diagnostic: Tau Sensitivity, K-Selection,
and Signed vs Unsigned Decomposition Analysis.

Systematic diagnostic experiment to understand why unsigned spectral clustering
matches signed SPONGE on synthetic data. Investigates tau regularization
sensitivity, k-selection strategies, spectral structure of the CoI matrix,
and high-dimensional failure modes.
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

# RAM budget: 50% of container limit
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

STRUCTURED_VARIANTS = [
    "easy_2mod_xor",
    "medium_4mod_mixed",
    "hard_4mod_unequal",
    "overlapping_modules",
    "highdim_8mod",
]

TAU_VALUES = [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 100.0]


# ═══════════════════════════════════════════════════════════════════════════════
# MI ESTIMATION (Custom KSG / Ross-2014 estimator)
# ═══════════════════════════════════════════════════════════════════════════════

def custom_micd(X_cont: np.ndarray, y_disc: np.ndarray, k: int = 5) -> float:
    """MI between continuous X (n, d) and discrete y (n,).

    Uses the KSG-type estimator for continuous-discrete MI (Ross 2014).
    """
    if X_cont.ndim == 1:
        X_cont = X_cont.reshape(-1, 1)

    n = X_cont.shape[0]
    if n <= k + 1:
        return 0.0

    if np.all(np.ptp(X_cont, axis=0) < 1e-12):
        return 0.0

    classes, counts = np.unique(y_disc, return_counts=True)
    if len(classes) < 2:
        return 0.0

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

        m_counts = tree_full.query_ball_point(X_cls, eps, return_length=True)
        m_values = np.asarray(m_counts, dtype=np.float64) - 1
        m_values = np.maximum(m_values, 1)

        sum_psi_m += np.sum(digamma(m_values))
        sum_psi_nc += n_cls * digamma(n_cls)

    mi = digamma(k) + digamma(n) - sum_psi_m / n - sum_psi_nc / n
    return max(float(mi), 0.0)


# ── Try NPEET if available ────────────────────────────────────────────────────

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
    Negative -> synergy, Positive -> redundancy.
    """
    n, d = X.shape
    if n_jobs == -1:
        n_jobs = NUM_WORKERS

    # Stratified subsample
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

    # Phase A: individual MI
    logger.info(f"  Phase A: individual MI for {d} features ...")
    mi_individual = np.zeros(d)
    for i in range(d):
        mi_individual[i] = compute_mi(X_sub[:, i:i + 1], y_sub, k=k)
    logger.info(f"  Individual MI  [{mi_individual.min():.4f} .. "
                f"{mi_individual.max():.4f}]")

    # Phase B: joint MI (parallelised)
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

    # Phase C: assemble CoI matrix
    CoI = np.zeros((d, d))
    for i, j, jmi in results:
        coi_val = mi_individual[i] + mi_individual[j] - jmi
        CoI[i, j] = coi_val
        CoI[j, i] = coi_val
    np.fill_diagonal(CoI, 0)

    return CoI, mi_individual


# ═══════════════════════════════════════════════════════════════════════════════
# SPONGE_SYM SIGNED SPECTRAL CLUSTERING
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

    Negates CoI so synergistic features (negative CoI) become positive edges.
    """
    d = CoI.shape[0]
    signed_adj = -CoI

    A_pos = np.maximum(signed_adj, 0)
    A_neg = np.maximum(-signed_adj, 0)

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
    B_mat += 1e-10 * np.eye(d)

    return A_mat, B_mat


def sponge_sym(
    CoI: np.ndarray,
    k: int,
    tau_p: float = 1.0,
    tau_n: float = 1.0,
    ridge: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray]:
    """SPONGE_sym signed spectral clustering.

    Returns (labels, eigenvalues) for k clusters.
    """
    d = CoI.shape[0]
    k = max(2, min(k, d - 1))

    A_mat, B_mat = _build_sponge_matrices(CoI, tau_p, tau_n)
    # Override ridge if specified
    if ridge > 1e-10:
        B_mat += (ridge - 1e-10) * np.eye(d)

    try:
        eigenvalues, eigenvectors = scipy.linalg.eigh(
            A_mat, B_mat, subset_by_index=[0, k - 1]
        )
    except scipy.linalg.LinAlgError:
        # Retry with larger ridge
        B_mat += 1e-6 * np.eye(d)
        try:
            eigenvalues, eigenvectors = scipy.linalg.eigh(
                A_mat, B_mat, subset_by_index=[0, k - 1]
            )
        except scipy.linalg.LinAlgError:
            return np.arange(d) % k, np.full(k, np.nan)

    # Row-normalise eigenvectors
    norms = np.linalg.norm(eigenvectors, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    eigenvectors_normed = eigenvectors / norms

    try:
        labels = KMeans(
            n_clusters=k, n_init=20, random_state=RANDOM_STATE
        ).fit_predict(eigenvectors_normed)
    except Exception:
        labels = np.arange(d) % k

    return labels, eigenvalues


def sponge_sym_embedding(
    CoI: np.ndarray,
    k: int,
    tau_p: float = 1.0,
    tau_n: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return eigenvectors embedding (not labels) for SPONGE_sym."""
    d = CoI.shape[0]
    k = max(2, min(k, d - 1))

    A_mat, B_mat = _build_sponge_matrices(CoI, tau_p, tau_n)

    try:
        eigenvalues, eigenvectors = scipy.linalg.eigh(
            A_mat, B_mat, subset_by_index=[0, k - 1]
        )
    except scipy.linalg.LinAlgError:
        B_mat += 1e-6 * np.eye(d)
        eigenvalues, eigenvectors = scipy.linalg.eigh(
            A_mat, B_mat, subset_by_index=[0, k - 1]
        )

    norms = np.linalg.norm(eigenvectors, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return eigenvectors / norms, eigenvalues


# ═══════════════════════════════════════════════════════════════════════════════
# UNSIGNED SPECTRAL CLUSTERING
# ═══════════════════════════════════════════════════════════════════════════════

def unsigned_spectral(CoI: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Standard spectral clustering on |CoI|. Returns (labels, eigenvectors)."""
    d = CoI.shape[0]
    k = max(2, min(k, d - 1))

    W = np.abs(CoI)
    np.fill_diagonal(W, 0)

    dv = W.sum(axis=1)
    Di = np.diag(_safe_inv_sqrt(dv))
    L = np.diag(dv) - W
    Ls = Di @ L @ Di

    try:
        evals, vecs = scipy.linalg.eigh(Ls, subset_by_index=[0, k - 1])
    except scipy.linalg.LinAlgError:
        return np.arange(d) % k, np.zeros((d, k))

    try:
        labels = KMeans(
            n_clusters=k, n_init=20, random_state=RANDOM_STATE
        ).fit_predict(vecs)
    except Exception:
        labels = np.arange(d) % k

    return labels, vecs


# ═══════════════════════════════════════════════════════════════════════════════
# EVALUATION METRICS
# ═══════════════════════════════════════════════════════════════════════════════

def construct_ground_truth_labels(meta: dict) -> np.ndarray:
    """Feature-level ground-truth labels."""
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


def compute_module_focused_ari(
    gt_labels: np.ndarray,
    pred_labels: np.ndarray,
    meta: dict,
) -> float | None:
    """ARI computed only on module + redundant features."""
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


def evaluate_clustering(
    pred_labels: np.ndarray,
    gt_labels: np.ndarray,
    gt_modules: list[list[int]],
    meta: dict,
) -> dict:
    """Compute all evaluation metrics for a clustering."""
    ari = float(adjusted_rand_score(gt_labels, pred_labels))
    mfari = compute_module_focused_ari(gt_labels, pred_labels, meta)
    jaccard = float(compute_synergistic_pair_jaccard(gt_modules, pred_labels))

    csizes = []
    if len(pred_labels) > 0:
        for lbl in sorted(set(pred_labels.tolist())):
            csizes.append(int(np.sum(pred_labels == lbl)))

    return {
        "ari": round(ari, 6),
        "mfari": round(mfari, 6) if mfari is not None else None,
        "jaccard": round(jaccard, 6),
        "cluster_sizes": csizes,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# FRUSTRATION INDEX
# ═══════════════════════════════════════════════════════════════════════════════

def compute_frustration_index(CoI: np.ndarray) -> float:
    """lambda_min / lambda_max of the signed Laplacian."""
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
# DATA LOADING (regenerate from generators)
# ═══════════════════════════════════════════════════════════════════════════════

def load_data() -> list[dict]:
    """Load data by importing generator functions from data.py."""
    logger.info("Loading data from generators ...")

    sys.path.insert(0, str(DATA_DIR))
    try:
        from data import GENERATORS  # type: ignore[import-untyped]
    except ImportError:
        logger.exception("data.py import failed")
        raise

    # Restore our logger
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


# ═══════════════════════════════════════════════════════════════════════════════
# INVESTIGATION 1: TAU SENSITIVITY SWEEP
# ═══════════════════════════════════════════════════════════════════════════════

def investigation_1_tau_sensitivity(
    coi_matrices: dict[str, np.ndarray],
    variant_metas: dict[str, dict],
    gt_labels_map: dict[str, np.ndarray],
) -> dict:
    """Sweep tau_p x tau_n grid for all structured variants."""
    logger.info("\n" + "=" * 60)
    logger.info("INVESTIGATION 1: TAU SENSITIVITY SWEEP")
    logger.info("=" * 60)

    tau_results = {}
    n_tau = len(TAU_VALUES)
    failed_combos_total = 0

    for vname in STRUCTURED_VARIANTS:
        if vname not in coi_matrices:
            continue
        CoI = coi_matrices[vname]
        meta = variant_metas[vname]
        gt_labels = gt_labels_map[vname]
        gt_modules = meta.get("ground_truth_modules", [])
        gt_k = len(gt_modules)

        if gt_k < 2:
            continue

        logger.info(f"\n  {vname}: {n_tau}x{n_tau} = {n_tau**2} tau combos, oracle k={gt_k}")

        heatmap_jaccard = np.zeros((n_tau, n_tau))
        heatmap_ari = np.zeros((n_tau, n_tau))
        heatmap_mfari = np.zeros((n_tau, n_tau))
        failed_count = 0

        t0 = time.time()
        for ti, tau_p in enumerate(TAU_VALUES):
            for tj, tau_n in enumerate(TAU_VALUES):
                try:
                    labels, _ = sponge_sym(CoI, k=gt_k, tau_p=tau_p, tau_n=tau_n)
                    metrics = evaluate_clustering(labels, gt_labels, gt_modules, meta)
                    heatmap_jaccard[ti, tj] = metrics["jaccard"]
                    heatmap_ari[ti, tj] = metrics["ari"]
                    heatmap_mfari[ti, tj] = metrics["mfari"] if metrics["mfari"] is not None else 0.0
                except Exception:
                    heatmap_jaccard[ti, tj] = np.nan
                    heatmap_ari[ti, tj] = np.nan
                    heatmap_mfari[ti, tj] = np.nan
                    failed_count += 1

        dt = time.time() - t0
        logger.info(f"    Done in {dt:.1f}s, {failed_count} failures")

        # Find best tau combo
        valid_mask = ~np.isnan(heatmap_jaccard)
        if valid_mask.any():
            best_idx = np.unravel_index(
                np.nanargmax(heatmap_jaccard), heatmap_jaccard.shape
            )
            best_tau_p = TAU_VALUES[best_idx[0]]
            best_tau_n = TAU_VALUES[best_idx[1]]
            best_jaccard = float(heatmap_jaccard[best_idx])
        else:
            best_tau_p, best_tau_n, best_jaccard = 1.0, 1.0, 0.0

        # Unsigned baseline for comparison
        us_labels, _ = unsigned_spectral(CoI, k=gt_k)
        us_metrics = evaluate_clustering(us_labels, gt_labels, gt_modules, meta)

        # Default tau=1 result
        default_idx = TAU_VALUES.index(1.0)
        default_jaccard = float(heatmap_jaccard[default_idx, default_idx])

        signed_beats = best_jaccard > us_metrics["jaccard"]

        logger.info(f"    Best tau: p={best_tau_p}, n={best_tau_n} -> Jac={best_jaccard:.4f}")
        logger.info(f"    Default tau=1: Jac={default_jaccard:.4f}")
        logger.info(f"    Unsigned: Jac={us_metrics['jaccard']:.4f}")
        logger.info(f"    Signed beats unsigned: {signed_beats}")

        tau_results[vname] = {
            "heatmap_jaccard": [[round(float(v), 6) if not np.isnan(v) else None
                                 for v in row] for row in heatmap_jaccard],
            "heatmap_ari": [[round(float(v), 6) if not np.isnan(v) else None
                             for v in row] for row in heatmap_ari],
            "heatmap_mfari": [[round(float(v), 6) if not np.isnan(v) else None
                               for v in row] for row in heatmap_mfari],
            "tau_values": TAU_VALUES,
            "best_tau_p": best_tau_p,
            "best_tau_n": best_tau_n,
            "best_jaccard": round(best_jaccard, 6),
            "default_tau1_jaccard": round(default_jaccard, 6),
            "unsigned_jaccard": round(us_metrics["jaccard"], 6),
            "unsigned_ari": round(us_metrics["ari"], 6),
            "signed_beats_unsigned": signed_beats,
            "failed_combos": failed_count,
        }
        failed_combos_total += failed_count

    logger.info(f"\n  Total failed tau combos: {failed_combos_total}")
    return tau_results


# ═══════════════════════════════════════════════════════════════════════════════
# INVESTIGATION 2: K-SELECTION ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def _eigengap_k(CoI: np.ndarray, tau_p: float = 1.0, tau_n: float = 1.0) -> int:
    """Eigengap heuristic for k selection."""
    d = CoI.shape[0]
    k_max = min(20, d // 3)
    k_max = max(k_max, 3)

    A_mat, B_mat = _build_sponge_matrices(CoI, tau_p, tau_n)
    n_eigs = min(k_max, d - 1)

    try:
        eigenvalues, _ = scipy.linalg.eigh(
            A_mat, B_mat, subset_by_index=[0, n_eigs - 1]
        )
    except scipy.linalg.LinAlgError:
        return max(2, int(np.ceil(np.sqrt(d / 2))))

    gaps = np.diff(eigenvalues)
    if len(gaps) == 0:
        return 2

    # Top eigengap
    best_idx = int(np.argmax(gaps))
    k_eigengap = best_idx + 2

    # Fallback when eigengap is ambiguous
    if gaps.max() < 2 * np.median(gaps):
        k_eigengap = max(2, int(np.ceil(np.sqrt(d / 2))))

    return min(k_eigengap, k_max)


def _silhouette_k(
    CoI: np.ndarray,
    tau_p: float = 1.0,
    tau_n: float = 1.0,
) -> int:
    """Silhouette-based k selection on SPONGE embedding."""
    d = CoI.shape[0]
    k_max = min(20, d // 3)
    k_max = max(k_max, 3)

    try:
        embedding, evals = sponge_sym_embedding(CoI, k=k_max, tau_p=tau_p, tau_n=tau_n)
    except Exception:
        return max(2, int(np.ceil(np.sqrt(d / 2))))

    best_k, best_sil = 2, -1.0
    for k_cand in range(2, min(k_max + 1, d)):
        if k_cand > embedding.shape[1]:
            break
        try:
            lbl = KMeans(
                n_clusters=k_cand, n_init=10, random_state=RANDOM_STATE
            ).fit_predict(embedding[:, :k_cand])
            if len(set(lbl)) < 2:
                continue
            sil = silhouette_score(embedding[:, :k_cand], lbl)
            if sil > best_sil:
                best_k, best_sil = k_cand, sil
        except Exception:
            continue

    return best_k


def _stability_k(
    CoI: np.ndarray,
    tau_p: float = 1.0,
    tau_n: float = 1.0,
    n_bootstrap: int = 10,
    timeout_sec: float = 300.0,
) -> int:
    """Stability-based k selection via bootstrap resampling."""
    d = CoI.shape[0]
    k_max = min(15, d // 3)
    k_max = max(k_max, 3)
    rng = np.random.default_rng(RANDOM_STATE)

    t0 = time.time()
    best_k, best_stability = 2, -1.0

    for k_cand in range(2, k_max + 1):
        if time.time() - t0 > timeout_sec:
            logger.warning(f"    Stability timeout at k={k_cand}")
            break

        labels_list = []
        for b in range(n_bootstrap):
            if time.time() - t0 > timeout_sec:
                break
            # 80% feature subsample
            n_sub = max(k_cand + 1, int(0.8 * d))
            feat_idx = rng.choice(d, size=n_sub, replace=False)
            feat_idx = np.sort(feat_idx)
            CoI_sub = CoI[np.ix_(feat_idx, feat_idx)]

            try:
                lbl_sub, _ = sponge_sym(CoI_sub, k=k_cand, tau_p=tau_p, tau_n=tau_n)
            except Exception:
                continue

            # Map back to full labels (-1 for excluded features)
            full_lbl = np.full(d, -1, dtype=int)
            full_lbl[feat_idx] = lbl_sub
            labels_list.append(full_lbl)

        if len(labels_list) < 2:
            continue

        # Pairwise ARI among bootstrap runs (only on shared features)
        aris = []
        for i in range(len(labels_list)):
            for j in range(i + 1, len(labels_list)):
                shared = (labels_list[i] >= 0) & (labels_list[j] >= 0)
                if shared.sum() < k_cand + 1:
                    continue
                a = adjusted_rand_score(labels_list[i][shared],
                                        labels_list[j][shared])
                aris.append(a)

        if aris:
            mean_ari = float(np.mean(aris))
            if mean_ari > best_stability:
                best_k, best_stability = k_cand, mean_ari

    return best_k


def investigation_2_k_selection(
    coi_matrices: dict[str, np.ndarray],
    variant_metas: dict[str, dict],
    gt_labels_map: dict[str, np.ndarray],
    best_taus: dict[str, tuple[float, float]],
) -> dict:
    """Compare 4 k-selection strategies across variants."""
    logger.info("\n" + "=" * 60)
    logger.info("INVESTIGATION 2: K-SELECTION ANALYSIS")
    logger.info("=" * 60)

    k_results = {}

    for vname in STRUCTURED_VARIANTS:
        if vname not in coi_matrices:
            continue
        CoI = coi_matrices[vname]
        meta = variant_metas[vname]
        gt_labels = gt_labels_map[vname]
        gt_modules = meta.get("ground_truth_modules", [])
        gt_k = len(gt_modules)
        d = CoI.shape[0]

        if gt_k < 2:
            continue

        bt_p, bt_n = best_taus.get(vname, (1.0, 1.0))
        logger.info(f"\n  {vname}: true k={gt_k}, d={d}")

        variant_k_results = {}

        # Strategy (a): Eigengap
        t0 = time.time()
        k_eig = _eigengap_k(CoI, tau_p=1.0, tau_n=1.0)
        lbl_eig, _ = sponge_sym(CoI, k=k_eig)
        m_eig = evaluate_clustering(lbl_eig, gt_labels, gt_modules, meta)
        dt = time.time() - t0
        variant_k_results["eigengap"] = {
            "k_selected": k_eig,
            "k_correct": k_eig == gt_k,
            "ari": m_eig["ari"],
            "mfari": m_eig["mfari"],
            "jaccard": m_eig["jaccard"],
            "cluster_sizes": m_eig["cluster_sizes"],
            "time_sec": round(dt, 2),
        }
        logger.info(f"    Eigengap: k={k_eig} (correct={k_eig==gt_k}), Jac={m_eig['jaccard']:.4f}")

        # Strategy (b): Oracle
        t0 = time.time()
        lbl_orc, _ = sponge_sym(CoI, k=gt_k)
        m_orc = evaluate_clustering(lbl_orc, gt_labels, gt_modules, meta)
        dt = time.time() - t0
        variant_k_results["oracle"] = {
            "k_selected": gt_k,
            "k_correct": True,
            "ari": m_orc["ari"],
            "mfari": m_orc["mfari"],
            "jaccard": m_orc["jaccard"],
            "cluster_sizes": m_orc["cluster_sizes"],
            "time_sec": round(dt, 2),
        }
        logger.info(f"    Oracle: k={gt_k}, Jac={m_orc['jaccard']:.4f}")

        # Strategy (b2): Oracle with best tau
        t0 = time.time()
        lbl_orc_bt, _ = sponge_sym(CoI, k=gt_k, tau_p=bt_p, tau_n=bt_n)
        m_orc_bt = evaluate_clustering(lbl_orc_bt, gt_labels, gt_modules, meta)
        dt = time.time() - t0
        variant_k_results["oracle_best_tau"] = {
            "k_selected": gt_k,
            "k_correct": True,
            "tau_p": bt_p,
            "tau_n": bt_n,
            "ari": m_orc_bt["ari"],
            "mfari": m_orc_bt["mfari"],
            "jaccard": m_orc_bt["jaccard"],
            "cluster_sizes": m_orc_bt["cluster_sizes"],
            "time_sec": round(dt, 2),
        }
        logger.info(f"    Oracle+BestTau: Jac={m_orc_bt['jaccard']:.4f}")

        # Strategy (c): Silhouette
        t0 = time.time()
        k_sil = _silhouette_k(CoI)
        lbl_sil, _ = sponge_sym(CoI, k=k_sil)
        m_sil = evaluate_clustering(lbl_sil, gt_labels, gt_modules, meta)
        dt = time.time() - t0
        variant_k_results["silhouette"] = {
            "k_selected": k_sil,
            "k_correct": k_sil == gt_k,
            "ari": m_sil["ari"],
            "mfari": m_sil["mfari"],
            "jaccard": m_sil["jaccard"],
            "cluster_sizes": m_sil["cluster_sizes"],
            "time_sec": round(dt, 2),
        }
        logger.info(f"    Silhouette: k={k_sil} (correct={k_sil==gt_k}), Jac={m_sil['jaccard']:.4f}")

        # Strategy (d): Stability-based
        # Skip for highdim if > 5 min likely
        timeout = 60.0 if vname == "highdim_8mod" else 120.0
        n_boot = 5 if vname == "highdim_8mod" else 10

        t0 = time.time()
        try:
            k_stab = _stability_k(CoI, n_bootstrap=n_boot, timeout_sec=timeout)
            lbl_stab, _ = sponge_sym(CoI, k=k_stab)
            m_stab = evaluate_clustering(lbl_stab, gt_labels, gt_modules, meta)
            dt = time.time() - t0
            variant_k_results["stability"] = {
                "k_selected": k_stab,
                "k_correct": k_stab == gt_k,
                "ari": m_stab["ari"],
                "mfari": m_stab["mfari"],
                "jaccard": m_stab["jaccard"],
                "cluster_sizes": m_stab["cluster_sizes"],
                "time_sec": round(dt, 2),
            }
            logger.info(f"    Stability: k={k_stab} (correct={k_stab==gt_k}), Jac={m_stab['jaccard']:.4f}")
        except Exception:
            logger.exception(f"    Stability failed for {vname}")
            variant_k_results["stability"] = {"skipped": True, "reason": "error"}

        # Unsigned spectral baselines (oracle k)
        us_labels, _ = unsigned_spectral(CoI, k=gt_k)
        m_us = evaluate_clustering(us_labels, gt_labels, gt_modules, meta)
        variant_k_results["unsigned_oracle"] = {
            "k_selected": gt_k,
            "k_correct": True,
            "ari": m_us["ari"],
            "mfari": m_us["mfari"],
            "jaccard": m_us["jaccard"],
            "cluster_sizes": m_us["cluster_sizes"],
        }
        logger.info(f"    Unsigned (oracle): Jac={m_us['jaccard']:.4f}")

        k_results[vname] = variant_k_results

    return k_results


# ═══════════════════════════════════════════════════════════════════════════════
# INVESTIGATION 3: SIGNED VS UNSIGNED DECOMPOSITION
# ═══════════════════════════════════════════════════════════════════════════════

def investigation_3_decomposition(
    coi_matrices: dict[str, np.ndarray],
    variant_metas: dict[str, dict],
    gt_labels_map: dict[str, np.ndarray],
) -> dict:
    """Analyze WHY signed ~ unsigned on these CoI graphs."""
    logger.info("\n" + "=" * 60)
    logger.info("INVESTIGATION 3: SIGNED VS UNSIGNED DECOMPOSITION")
    logger.info("=" * 60)

    decomp_results = {}

    for vname in STRUCTURED_VARIANTS:
        if vname not in coi_matrices:
            continue
        CoI = coi_matrices[vname]
        meta = variant_metas[vname]
        gt_labels = gt_labels_map[vname]
        gt_modules = meta.get("ground_truth_modules", [])
        gt_k = len(gt_modules)
        d = CoI.shape[0]

        if gt_k < 2:
            continue

        logger.info(f"\n  {vname}: d={d}, k={gt_k}")
        vresult = {}

        # (a) Sign distribution
        upper_tri = CoI[np.triu_indices(d, k=1)]
        n_total = len(upper_tri)
        frac_positive = float(np.sum(upper_tri > 0.01) / n_total)
        frac_negative = float(np.sum(upper_tri < -0.01) / n_total)
        frac_near_zero = float(np.sum(np.abs(upper_tri) < 0.01) / n_total)

        nonzero_mask = np.abs(upper_tri) > 0.01
        n_nonzero = nonzero_mask.sum()
        sign_ratio = float(np.sum(upper_tri[nonzero_mask] < 0) / n_nonzero) if n_nonzero > 0 else 0.5

        vresult["sign_distribution"] = {
            "frac_positive": round(frac_positive, 6),
            "frac_negative": round(frac_negative, 6),
            "frac_near_zero": round(frac_near_zero, 6),
            "sign_ratio_neg_over_nonzero": round(sign_ratio, 6),
            "n_nonzero_pairs": int(n_nonzero),
            "n_total_pairs": int(n_total),
        }
        logger.info(f"    Sign dist: pos={frac_positive:.3f}, neg={frac_negative:.3f}, "
                     f"~0={frac_near_zero:.3f}, ratio={sign_ratio:.3f}")

        # (b) Condition numbers
        A_mat, B_mat = _build_sponge_matrices(CoI, 1.0, 1.0)
        try:
            cond_A = float(np.linalg.cond(A_mat))
            cond_B = float(np.linalg.cond(B_mat))
        except Exception:
            cond_A = cond_B = float('inf')

        W_unsigned = np.abs(CoI)
        np.fill_diagonal(W_unsigned, 0)
        dv = W_unsigned.sum(axis=1)
        Di = np.diag(_safe_inv_sqrt(dv))
        L_us = np.diag(dv) - W_unsigned
        Ls_us = Di @ L_us @ Di
        try:
            cond_L = float(np.linalg.cond(Ls_us))
        except Exception:
            cond_L = float('inf')

        vresult["condition_numbers"] = {
            "cond_A_sponge": round(cond_A, 4) if np.isfinite(cond_A) else None,
            "cond_B_sponge": round(cond_B, 4) if np.isfinite(cond_B) else None,
            "cond_L_unsigned": round(cond_L, 4) if np.isfinite(cond_L) else None,
        }
        logger.info(f"    Condition: A={cond_A:.1f}, B={cond_B:.1f}, L_us={cond_L:.1f}")

        # (c) Eigenvector similarity (subspace angles)
        try:
            V_signed, _ = sponge_sym_embedding(CoI, k=gt_k)
            _, V_unsigned_raw = unsigned_spectral(CoI, k=gt_k)

            # Normalize rows
            V_unsigned = V_unsigned_raw.copy()
            norms_u = np.linalg.norm(V_unsigned, axis=1, keepdims=True)
            norms_u = np.maximum(norms_u, 1e-12)
            V_unsigned = V_unsigned / norms_u

            # Canonical angles via SVD
            M = V_signed.T @ V_unsigned
            S = np.linalg.svd(M, compute_uv=False)
            S = np.clip(S, 0, 1)
            cos_angles = S.tolist()
            mean_cos_angle = float(np.mean(S))

            # Grassmann distance
            angles = np.arccos(np.clip(S, -1, 1))
            grassmann_dist = float(np.sqrt(np.sum(angles**2)))

            vresult["eigenvector_similarity"] = {
                "cos_canonical_angles": [round(x, 6) for x in cos_angles],
                "mean_cos_angle": round(mean_cos_angle, 6),
                "grassmann_distance": round(grassmann_dist, 6),
            }
            logger.info(f"    Eigvec similarity: mean_cos={mean_cos_angle:.4f}, "
                         f"grassmann={grassmann_dist:.4f}")
        except Exception:
            logger.exception(f"    Eigenvector similarity failed for {vname}")
            vresult["eigenvector_similarity"] = {"error": "computation failed"}

        # (d) Differential clustering
        try:
            lbl_signed, _ = sponge_sym(CoI, k=gt_k)
            lbl_unsigned, _ = unsigned_spectral(CoI, k=gt_k)

            diff_features = []
            for fi in range(d):
                if lbl_signed[fi] != lbl_unsigned[fi]:
                    is_syn = any(fi in mod for mod in gt_modules)
                    is_red = any(fi == p[1] for p in meta.get("redundant_pairs", []))
                    is_noise = fi in meta.get("noise_features", [])

                    # Strongest CoI connections
                    coi_row = CoI[fi].copy()
                    coi_row[fi] = 0
                    top_idx = np.argsort(np.abs(coi_row))[-3:][::-1]
                    top_conns = [(int(idx), round(float(coi_row[idx]), 6)) for idx in top_idx]

                    diff_features.append({
                        "feature": fi,
                        "signed_cluster": int(lbl_signed[fi]),
                        "unsigned_cluster": int(lbl_unsigned[fi]),
                        "is_synergistic": is_syn,
                        "is_redundant": is_red,
                        "is_noise": is_noise,
                        "top_connections": top_conns,
                    })

            vresult["differential_features"] = {
                "n_different": len(diff_features),
                "n_total": d,
                "frac_different": round(len(diff_features) / d, 4),
                "details": diff_features[:20],  # Limit output size
            }
            logger.info(f"    Differential: {len(diff_features)}/{d} features differ")
        except Exception:
            logger.exception(f"    Differential clustering failed for {vname}")
            vresult["differential_features"] = {"error": "computation failed"}

        # (e) Spectral frustration
        frustration = compute_frustration_index(CoI)
        vresult["frustration_index"] = round(frustration, 6)
        logger.info(f"    Frustration: {frustration:.4f}")

        # (f) Effective sign structure
        abs_upper = np.abs(upper_tri)
        nonzero_vals = abs_upper[abs_upper > 0.001]
        if len(nonzero_vals) > 0:
            median_nz = float(np.median(nonzero_vals))
            strong_mask_flat = abs_upper > median_nz
            strong_vals = upper_tri[strong_mask_flat]
            strong_pos = int(np.sum(strong_vals > 0))
            strong_neg = int(np.sum(strong_vals < 0))
            strong_total = strong_pos + strong_neg
        else:
            median_nz = 0.0
            strong_pos = strong_neg = strong_total = 0

        vresult["effective_sign_structure"] = {
            "median_nonzero_coi": round(median_nz, 6),
            "strong_positive_edges": strong_pos,
            "strong_negative_edges": strong_neg,
            "strong_total": strong_total,
            "strong_neg_frac": round(strong_neg / strong_total, 4) if strong_total > 0 else 0.0,
        }
        logger.info(f"    Strong edges: pos={strong_pos}, neg={strong_neg}")

        decomp_results[vname] = vresult

    return decomp_results


# ═══════════════════════════════════════════════════════════════════════════════
# INVESTIGATION 4: HIGHDIM FIX
# ═══════════════════════════════════════════════════════════════════════════════

def investigation_4_highdim_fix(
    coi_matrices: dict[str, np.ndarray],
    variant_metas: dict[str, dict],
    gt_labels_map: dict[str, np.ndarray],
    best_taus: dict[str, tuple[float, float]],
) -> dict:
    """Fix catastrophic collapse of SPONGE on highdim_8mod."""
    logger.info("\n" + "=" * 60)
    logger.info("INVESTIGATION 4: HIGHDIM FIX")
    logger.info("=" * 60)

    vname = "highdim_8mod"
    if vname not in coi_matrices:
        logger.warning("  highdim_8mod not available, skipping")
        return {"skipped": True}

    CoI = coi_matrices[vname]
    meta = variant_metas[vname]
    gt_labels = gt_labels_map[vname]
    gt_modules = meta.get("ground_truth_modules", [])
    gt_k = len(gt_modules)
    d = CoI.shape[0]
    bt_p, bt_n = best_taus.get(vname, (1.0, 1.0))

    hd_results = {}

    # (a) Sparsification
    logger.info("\n  (a) Sparsification sweep")
    sparsification_results = []
    for keep_frac in [0.05, 0.10, 0.20, 0.50]:
        abs_CoI = np.abs(CoI)
        upper = abs_CoI[np.triu_indices(d, k=1)]
        threshold = float(np.percentile(upper, 100 * (1 - keep_frac)))

        CoI_sparse = CoI.copy()
        CoI_sparse[abs_CoI < threshold] = 0.0
        np.fill_diagonal(CoI_sparse, 0)

        n_nonzero = int(np.sum(np.abs(CoI_sparse) > 0)) // 2
        logger.info(f"    keep={keep_frac}: thr={threshold:.4f}, edges={n_nonzero}")

        # SPONGE with default tau
        try:
            lbl_s1, _ = sponge_sym(CoI_sparse, k=gt_k, tau_p=1.0, tau_n=1.0)
            m_s1 = evaluate_clustering(lbl_s1, gt_labels, gt_modules, meta)
        except Exception:
            m_s1 = {"ari": 0.0, "mfari": None, "jaccard": 0.0, "cluster_sizes": []}

        # SPONGE with best tau
        try:
            lbl_s2, _ = sponge_sym(CoI_sparse, k=gt_k, tau_p=bt_p, tau_n=bt_n)
            m_s2 = evaluate_clustering(lbl_s2, gt_labels, gt_modules, meta)
        except Exception:
            m_s2 = {"ari": 0.0, "mfari": None, "jaccard": 0.0, "cluster_sizes": []}

        # Unsigned
        lbl_u, _ = unsigned_spectral(CoI_sparse, k=gt_k)
        m_u = evaluate_clustering(lbl_u, gt_labels, gt_modules, meta)

        entry = {
            "keep_frac": keep_frac,
            "threshold": round(threshold, 6),
            "n_nonzero_edges": n_nonzero,
            "sponge_default_tau": m_s1,
            "sponge_best_tau": m_s2,
            "unsigned": m_u,
        }
        sparsification_results.append(entry)
        logger.info(f"      SPONGE(1,1)={m_s1['jaccard']:.4f}, "
                     f"SPONGE(best)={m_s2['jaccard']:.4f}, "
                     f"unsigned={m_u['jaccard']:.4f}")

    hd_results["sparsification"] = sparsification_results

    # (b) Large tau
    logger.info("\n  (b) Large tau sweep")
    large_tau_results = []
    for tau_val in [1.0, 5.0, 10.0, 50.0, 100.0]:
        try:
            lbl, _ = sponge_sym(CoI, k=gt_k, tau_p=tau_val, tau_n=tau_val)
            m = evaluate_clustering(lbl, gt_labels, gt_modules, meta)
        except Exception:
            m = {"ari": 0.0, "mfari": None, "jaccard": 0.0, "cluster_sizes": []}

        entry = {"tau": tau_val, "metrics": m}
        large_tau_results.append(entry)
        logger.info(f"    tau={tau_val}: Jac={m['jaccard']:.4f}, sizes={m['cluster_sizes'][:5]}")

    hd_results["large_tau"] = large_tau_results

    # (c) Absolute threshold zeroing
    logger.info("\n  (c) Threshold zeroing sweep")
    threshold_results = []
    for thresh in [0.001, 0.005, 0.01, 0.02, 0.05]:
        CoI_clean = CoI.copy()
        CoI_clean[np.abs(CoI) < thresh] = 0.0
        np.fill_diagonal(CoI_clean, 0)
        n_nz = int(np.sum(np.abs(CoI_clean) > 0)) // 2

        try:
            lbl_s, _ = sponge_sym(CoI_clean, k=gt_k)
            m_s = evaluate_clustering(lbl_s, gt_labels, gt_modules, meta)
        except Exception:
            m_s = {"ari": 0.0, "mfari": None, "jaccard": 0.0, "cluster_sizes": []}

        lbl_u, _ = unsigned_spectral(CoI_clean, k=gt_k)
        m_u = evaluate_clustering(lbl_u, gt_labels, gt_modules, meta)

        entry = {
            "threshold": thresh,
            "n_nonzero_edges": n_nz,
            "sponge": m_s,
            "unsigned": m_u,
        }
        threshold_results.append(entry)
        logger.info(f"    thresh={thresh}: SPONGE={m_s['jaccard']:.4f}, unsigned={m_u['jaccard']:.4f}")

    hd_results["threshold_zeroing"] = threshold_results

    # (d) Combined best: best sparsification + best tau
    logger.info("\n  (d) Combined best strategies")

    # Find best sparsification
    best_sparse_jac = -1
    best_sparse_frac = 0.10
    for sr in sparsification_results:
        j = max(sr["sponge_default_tau"]["jaccard"],
                sr["sponge_best_tau"]["jaccard"])
        if j > best_sparse_jac:
            best_sparse_jac = j
            best_sparse_frac = sr["keep_frac"]

    # Find best large tau
    best_tau_jac = -1
    best_tau_val = 1.0
    for lt in large_tau_results:
        if lt["metrics"]["jaccard"] > best_tau_jac:
            best_tau_jac = lt["metrics"]["jaccard"]
            best_tau_val = lt["tau"]

    # Find best threshold
    best_thresh_jac = -1
    best_thresh_val = 0.01
    for tr in threshold_results:
        if tr["sponge"]["jaccard"] > best_thresh_jac:
            best_thresh_jac = tr["sponge"]["jaccard"]
            best_thresh_val = tr["threshold"]

    # Combine: sparsify + tau
    abs_CoI = np.abs(CoI)
    upper = abs_CoI[np.triu_indices(d, k=1)]
    thr_val = float(np.percentile(upper, 100 * (1 - best_sparse_frac)))
    CoI_sparse_best = CoI.copy()
    CoI_sparse_best[abs_CoI < thr_val] = 0.0
    np.fill_diagonal(CoI_sparse_best, 0)

    try:
        lbl_comb1, _ = sponge_sym(CoI_sparse_best, k=gt_k,
                                   tau_p=best_tau_val, tau_n=best_tau_val)
        m_comb1 = evaluate_clustering(lbl_comb1, gt_labels, gt_modules, meta)
    except Exception:
        m_comb1 = {"ari": 0.0, "mfari": None, "jaccard": 0.0, "cluster_sizes": []}

    # Combine: threshold + tau
    CoI_thresh_best = CoI.copy()
    CoI_thresh_best[np.abs(CoI) < best_thresh_val] = 0.0
    np.fill_diagonal(CoI_thresh_best, 0)

    try:
        lbl_comb2, _ = sponge_sym(CoI_thresh_best, k=gt_k,
                                   tau_p=best_tau_val, tau_n=best_tau_val)
        m_comb2 = evaluate_clustering(lbl_comb2, gt_labels, gt_modules, meta)
    except Exception:
        m_comb2 = {"ari": 0.0, "mfari": None, "jaccard": 0.0, "cluster_sizes": []}

    # Unsigned baseline (full CoI)
    lbl_us_full, _ = unsigned_spectral(CoI, k=gt_k)
    m_us_full = evaluate_clustering(lbl_us_full, gt_labels, gt_modules, meta)

    hd_results["combined_best"] = {
        "sparsify_plus_tau": {
            "keep_frac": best_sparse_frac,
            "tau": best_tau_val,
            "metrics": m_comb1,
        },
        "threshold_plus_tau": {
            "threshold": best_thresh_val,
            "tau": best_tau_val,
            "metrics": m_comb2,
        },
        "unsigned_full_baseline": m_us_full,
    }
    logger.info(f"  Combined sparsify+tau: Jac={m_comb1['jaccard']:.4f}")
    logger.info(f"  Combined thresh+tau:   Jac={m_comb2['jaccard']:.4f}")
    logger.info(f"  Unsigned baseline:     Jac={m_us_full['jaccard']:.4f}")

    return hd_results


# ═══════════════════════════════════════════════════════════════════════════════
# INVESTIGATION 5: FINAL VERDICT
# ═══════════════════════════════════════════════════════════════════════════════

def investigation_5_verdict(
    tau_results: dict,
    k_results: dict,
    decomp_results: dict,
    hd_results: dict,
    coi_matrices: dict[str, np.ndarray],
    variant_metas: dict[str, dict],
    gt_labels_map: dict[str, np.ndarray],
) -> dict:
    """Compile all results into a verdict."""
    logger.info("\n" + "=" * 60)
    logger.info("INVESTIGATION 5: FINAL VERDICT")
    logger.info("=" * 60)

    per_variant = {}
    any_signed_beats = False
    any_signed_beats_tuned = False

    for vname in STRUCTURED_VARIANTS:
        if vname not in tau_results:
            continue

        tr = tau_results[vname]
        dr = decomp_results.get(vname, {})

        best_signed_jac = tr["best_jaccard"]
        best_unsigned_jac = tr["unsigned_jaccard"]
        signed_advantage = best_signed_jac - best_unsigned_jac

        if signed_advantage > 0:
            any_signed_beats_tuned = True
        if tr.get("default_tau1_jaccard", 0) > best_unsigned_jac:
            any_signed_beats = True

        # K-selection best
        kr = k_results.get(vname, {})
        best_k_strategy = "oracle"
        best_k_jac = 0
        for strat, data in kr.items():
            if isinstance(data, dict) and "jaccard" in data:
                if data["jaccard"] > best_k_jac:
                    best_k_jac = data["jaccard"]
                    best_k_strategy = strat

        per_variant[vname] = {
            "best_signed_jaccard": round(best_signed_jac, 6),
            "best_unsigned_jaccard": round(best_unsigned_jac, 6),
            "signed_advantage": round(signed_advantage, 6),
            "best_tau_p": tr["best_tau_p"],
            "best_tau_n": tr["best_tau_n"],
            "best_k_strategy": best_k_strategy,
            "frustration_index": dr.get("frustration_index", None),
            "sign_distribution": dr.get("sign_distribution", {}).get(
                "sign_ratio_neg_over_nonzero", None
            ),
            "eigenvector_similarity": dr.get("eigenvector_similarity", {}).get(
                "mean_cos_angle", None
            ),
        }

        logger.info(f"  {vname}: signed={best_signed_jac:.4f} vs "
                     f"unsigned={best_unsigned_jac:.4f} "
                     f"(advantage={signed_advantage:+.4f})")

    # Cross-variant analysis
    advantages = [pv["signed_advantage"] for pv in per_variant.values()]
    frustrations = [pv["frustration_index"] for pv in per_variant.values()
                    if pv["frustration_index"] is not None]
    sign_ratios = [pv["sign_distribution"] for pv in per_variant.values()
                   if pv["sign_distribution"] is not None]

    # Correlation between frustration and signed advantage
    corr_frust_adv = None
    if len(frustrations) >= 3 and len(advantages) >= 3:
        n_corr = min(len(frustrations), len(advantages))
        f_arr = np.array(frustrations[:n_corr])
        a_arr = np.array(advantages[:n_corr])
        if np.std(f_arr) > 1e-10 and np.std(a_arr) > 1e-10:
            try:
                corr_frust_adv = float(np.corrcoef(f_arr, a_arr)[0, 1])
            except Exception:
                pass

    # Correlation between sign_ratio and signed advantage
    corr_sign_adv = None
    if len(sign_ratios) >= 3:
        n_corr = min(len(sign_ratios), len(advantages))
        s_arr = np.array(sign_ratios[:n_corr])
        a_arr = np.array(advantages[:n_corr])
        if np.std(s_arr) > 1e-10 and np.std(a_arr) > 1e-10:
            try:
                corr_sign_adv = float(np.corrcoef(s_arr, a_arr)[0, 1])
            except Exception:
                pass

    # Root cause diagnosis
    mean_advantage = float(np.mean(advantages)) if advantages else 0.0
    mean_frust = float(np.mean(frustrations)) if frustrations else 0.0
    mean_sign_ratio = float(np.mean(sign_ratios)) if sign_ratios else 0.5

    if mean_advantage > 0.02:
        diagnosis = "signed_spectral_helps_with_tuning"
    elif mean_frust > 0.3:
        diagnosis = "high_frustration_limits_signed_benefit"
    elif mean_sign_ratio < 0.2 or mean_sign_ratio > 0.8:
        diagnosis = "skewed_sign_distribution_reduces_signed_benefit"
    else:
        diagnosis = "coi_estimation_noise_drowns_sign_info"

    # Best global tau
    best_global_tau_p = 1.0
    best_global_tau_n = 1.0
    best_global_jac = -1.0
    for vname, tr in tau_results.items():
        if tr["best_jaccard"] > best_global_jac:
            best_global_jac = tr["best_jaccard"]
            best_global_tau_p = tr["best_tau_p"]
            best_global_tau_n = tr["best_tau_n"]

    # Best k strategy overall (non-oracle strategies only)
    skip_strats = {"oracle", "oracle_best_tau", "unsigned_oracle"}
    k_correct_counts: dict[str, int] = {}
    k_total_counts: dict[str, int] = {}
    for vname, kr in k_results.items():
        for strat, data in kr.items():
            if strat in skip_strats:
                continue
            if isinstance(data, dict) and "k_correct" in data:
                k_correct_counts[strat] = k_correct_counts.get(strat, 0) + (1 if data["k_correct"] else 0)
                k_total_counts[strat] = k_total_counts.get(strat, 0) + 1

    best_k_strat = "eigengap"
    best_k_accuracy = 0.0
    for strat in k_correct_counts:
        accuracy = k_correct_counts[strat] / k_total_counts[strat] if k_total_counts[strat] > 0 else 0
        if accuracy > best_k_accuracy:
            best_k_accuracy = accuracy
            best_k_strat = strat

    # Recommendation
    if any_signed_beats_tuned:
        recommendation = (
            "Signed spectral clustering (SPONGE) can outperform unsigned spectral "
            "clustering when tau parameters are properly tuned. Use tau sweep to "
            "find optimal regularization per dataset."
        )
    else:
        recommendation = (
            "Unsigned spectral clustering on |CoI| is a simpler and equally effective "
            "alternative to SPONGE for these synthetic planted-synergy datasets. "
            "The signed information in the CoI matrix does not provide sufficient "
            "additional clustering signal beyond magnitude alone."
        )

    cross_variant = {
        "mean_signed_advantage": round(mean_advantage, 6),
        "mean_frustration": round(mean_frust, 6) if frustrations else None,
        "mean_sign_ratio": round(mean_sign_ratio, 6) if sign_ratios else None,
        "corr_frustration_advantage": round(corr_frust_adv, 6) if corr_frust_adv is not None else None,
        "corr_sign_ratio_advantage": round(corr_sign_adv, 6) if corr_sign_adv is not None else None,
        "best_k_strategy_non_oracle": best_k_strat,
        "best_k_strategy_accuracy": round(best_k_accuracy, 4),
    }

    verdict = {
        "per_variant": per_variant,
        "cross_variant": cross_variant,
    }

    summary = {
        "signed_beats_unsigned_any_variant": any_signed_beats,
        "signed_beats_unsigned_with_tuning": any_signed_beats_tuned,
        "best_global_tau_p": best_global_tau_p,
        "best_global_tau_n": best_global_tau_n,
        "best_k_selection_strategy": best_k_strat,
        "root_cause_diagnosis": diagnosis,
        "recommendation": recommendation,
    }

    logger.info(f"\n  Verdict: signed beats unsigned (default): {any_signed_beats}")
    logger.info(f"  Verdict: signed beats unsigned (tuned):   {any_signed_beats_tuned}")
    logger.info(f"  Diagnosis: {diagnosis}")

    return verdict, summary


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

@logger.catch
def main() -> None:
    overall_t0 = time.time()
    logger.info("=" * 60)
    logger.info("SPONGE Signed Spectral Diagnostic Experiment")
    logger.info("=" * 60)

    # ── Step 0: Load data ─────────────────────────────────────────────────
    logger.info("\nSTEP 0: LOADING DATA")
    results = load_data()
    variant_data = {r["name"]: r for r in results}

    # ── Step 1: Compute CoI matrices ──────────────────────────────────────
    logger.info("\nSTEP 1: COMPUTING COI MATRICES")
    coi_matrices: dict[str, np.ndarray] = {}
    mi_individuals: dict[str, np.ndarray] = {}
    variant_metas: dict[str, dict] = {}
    gt_labels_map: dict[str, np.ndarray] = {}

    for vname in VARIANT_ORDER:
        if vname not in variant_data:
            continue
        vd = variant_data[vname]
        X, y, meta = vd["X"], vd["y"], vd["meta"]
        d = X.shape[1]
        variant_metas[vname] = meta
        gt_labels_map[vname] = construct_ground_truth_labels(meta)

        logger.info(f"\n  {vname}: n={X.shape[0]}, d={d}")

        subsample_n = SUBSAMPLE_N
        if d >= 100:
            subsample_n = min(SUBSAMPLE_N, 8000)
            logger.info(f"    High-dim: subsample -> {subsample_n}")

        t0 = time.time()
        CoI, mi_ind = compute_coi_matrix(
            X, y, k=COI_K, n_subsample=subsample_n, n_jobs=NUM_WORKERS,
        )
        dt = time.time() - t0
        logger.info(f"    CoI computed in {dt:.1f}s")

        coi_matrices[vname] = CoI
        mi_individuals[vname] = mi_ind

        # Free raw data
        del vd["X"], vd["y"]
        gc.collect()

    total_coi_time = time.time() - overall_t0
    logger.info(f"\nTotal CoI computation: {total_coi_time:.1f}s")

    # ── Investigation 1: Tau sensitivity ──────────────────────────────────
    t_inv1 = time.time()
    tau_results = investigation_1_tau_sensitivity(
        coi_matrices, variant_metas, gt_labels_map,
    )
    logger.info(f"Investigation 1 done in {time.time() - t_inv1:.1f}s")

    # Extract best taus for later use
    best_taus: dict[str, tuple[float, float]] = {}
    for vname, tr in tau_results.items():
        best_taus[vname] = (tr["best_tau_p"], tr["best_tau_n"])

    # ── Investigation 2: K-selection ──────────────────────────────────────
    t_inv2 = time.time()
    k_results = investigation_2_k_selection(
        coi_matrices, variant_metas, gt_labels_map, best_taus,
    )
    logger.info(f"Investigation 2 done in {time.time() - t_inv2:.1f}s")

    # ── Investigation 3: Decomposition analysis ──────────────────────────
    t_inv3 = time.time()
    decomp_results = investigation_3_decomposition(
        coi_matrices, variant_metas, gt_labels_map,
    )
    logger.info(f"Investigation 3 done in {time.time() - t_inv3:.1f}s")

    # ── Investigation 4: Highdim fix ──────────────────────────────────────
    t_inv4 = time.time()
    hd_results = investigation_4_highdim_fix(
        coi_matrices, variant_metas, gt_labels_map, best_taus,
    )
    logger.info(f"Investigation 4 done in {time.time() - t_inv4:.1f}s")

    # ── Investigation 5: Final verdict ────────────────────────────────────
    t_inv5 = time.time()
    verdict, summary = investigation_5_verdict(
        tau_results, k_results, decomp_results, hd_results,
        coi_matrices, variant_metas, gt_labels_map,
    )
    logger.info(f"Investigation 5 done in {time.time() - t_inv5:.1f}s")

    total_wall = time.time() - overall_t0

    # ── Build output datasets (exp_gen_sol_out format) ────────────────────
    logger.info("\nBuilding output datasets...")
    output_datasets: list[dict] = []

    for vname in VARIANT_ORDER:
        if vname not in coi_matrices:
            continue
        CoI = coi_matrices[vname]
        meta = variant_metas[vname]
        gt_labels = gt_labels_map[vname]
        gt_modules = meta.get("ground_truth_modules", [])
        gt_k = len(gt_modules)
        d = CoI.shape[0]
        mi_ind = mi_individuals.get(vname, np.zeros(d))

        bt_p, bt_n = best_taus.get(vname, (1.0, 1.0))

        # Get labels for signed default, signed best-tau, unsigned
        if gt_k >= 2:
            lbl_signed_default, _ = sponge_sym(CoI, k=gt_k, tau_p=1.0, tau_n=1.0)
            lbl_signed_best, _ = sponge_sym(CoI, k=gt_k, tau_p=bt_p, tau_n=bt_n)
            lbl_unsigned, _ = unsigned_spectral(CoI, k=gt_k)
        else:
            lbl_signed_default = np.arange(d)
            lbl_signed_best = np.arange(d)
            lbl_unsigned = np.arange(d)

        examples: list[dict] = []
        for fi in range(d):
            is_syn = any(fi in mod for mod in gt_modules)
            is_red = any(fi == p[1] for p in meta.get("redundant_pairs", []))
            is_noise = fi in meta.get("noise_features", [])

            ex = {
                "input": json.dumps({
                    "variant": vname,
                    "feature_index": fi,
                    "tau_p": bt_p,
                    "tau_n": bt_n,
                    "k_strategy": "oracle",
                }),
                "output": str(int(gt_labels[fi])),
                "predict_signed_default": str(int(lbl_signed_default[fi])),
                "predict_signed_best_tau": str(int(lbl_signed_best[fi])),
                "predict_unsigned": str(int(lbl_unsigned[fi])),
                "metadata_variant": vname,
                "metadata_feature_index": fi,
                "metadata_individual_mi": round(float(mi_ind[fi]), 6),
                "metadata_is_synergistic": is_syn,
                "metadata_is_redundant": is_red,
                "metadata_is_noise": is_noise,
                "metadata_tau_p": bt_p,
                "metadata_tau_n": bt_n,
            }
            examples.append(ex)

        output_datasets.append({"dataset": vname, "examples": examples})

    # ── Build final output ────────────────────────────────────────────────
    output = {
        "metadata": {
            "experiment": "sponge_signed_diagnostic",
            "description": (
                "Systematic diagnostic experiment investigating why unsigned "
                "spectral clustering matches signed SPONGE on synthetic CoI data"
            ),
            "total_wallclock_sec": round(total_wall, 2),
            "summary": summary,
            "tau_sensitivity": tau_results,
            "k_selection": k_results,
            "decomposition_analysis": decomp_results,
            "highdim_fix": hd_results,
            "verdict": verdict,
        },
        "datasets": output_datasets,
    }

    out_path = WORKSPACE / "method_out.json"
    out_path.write_text(json.dumps(output, indent=2))
    logger.info(f"\nOutput -> {out_path}")
    logger.info(f"Total wall-clock: {total_wall:.1f}s")

    logger.info("\n=== SUMMARY ===")
    for k, v in summary.items():
        logger.info(f"  {k}: {v}")


# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    main()

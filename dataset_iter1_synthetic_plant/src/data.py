#!/usr/bin/env python3
"""Generate 6 synthetic classification datasets with planted synergistic feature modules.

Each variant tests a different aspect of signed spectral clustering recovery:
1. easy_2mod_xor: 2 XOR modules, 10 features, 10K samples
2. medium_4mod_mixed: 4 modules (XOR+AND), 18 features, 20K samples
3. hard_4mod_unequal: 4 unequal modules, 31 features, 20K samples
4. overlapping_modules: 4 overlapping modules, 18 features, 20K samples
5. no_structure_control: No synergy (null), 20 features, 10K samples
6. highdim_8mod: 8 modules, 200 features, 50K samples
"""

import gc
import json
import math
import os
import resource
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from loguru import logger
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import StratifiedKFold

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
    for p in ["/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"]:
        try:
            v = Path(p).read_text().strip()
            if v != "max" and int(v) < 1_000_000_000_000:
                return int(v) / 1e9
        except (FileNotFoundError, ValueError):
            pass
    return None


NUM_CPUS = _detect_cpus()
TOTAL_RAM_GB = _container_ram_gb() or 57.0
logger.info(f"Hardware: {NUM_CPUS} CPUs, {TOTAL_RAM_GB:.1f} GB RAM")

# RAM budget: ~20 GB should be plenty for this task
RAM_BUDGET = int(min(20, TOTAL_RAM_GB * 0.35) * 1024**3)
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET * 3, RAM_BUDGET * 3))
logger.info(f"RAM budget set to {RAM_BUDGET / 1e9:.1f} GB")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MASTER_SEED = 42
WORKSPACE = Path(__file__).parent
NUM_WORKERS = max(1, NUM_CPUS - 1)

# ---------------------------------------------------------------------------
# Core generation primitives
# ---------------------------------------------------------------------------

def xor_interaction(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """XOR: sign(x1 * x2). Zero marginal MI with target."""
    return np.sign(x1 * x2)


def and_interaction(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """AND: (x1 > 0) & (x2 > 0). Non-zero marginal MI."""
    return ((x1 > 0) & (x2 > 0)).astype(float)


def three_way_xor(x1: np.ndarray, x2: np.ndarray, x3: np.ndarray) -> np.ndarray:
    """3-way XOR: sign(x1 * x2 * x3)."""
    return np.sign(x1 * x2 * x3)


def pairwise_xor_sum(x1: np.ndarray, x2: np.ndarray, x3: np.ndarray, x4: np.ndarray) -> np.ndarray:
    """Sum of two XOR pairs: sign(x1*x2) + sign(x3*x4)."""
    return np.sign(x1 * x2) + np.sign(x3 * x4)


def and_chain(features: np.ndarray) -> np.ndarray:
    """AND-chain: all features > 0."""
    return np.all(features > 0, axis=1).astype(float)


def make_redundant(x: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Create redundant feature: x + N(0, sigma^2)."""
    return x + rng.normal(0, sigma, size=x.shape)


def generate_target(
    contributions: list[np.ndarray],
    weights: list[float],
    sigma_noise: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate binary target from weighted module contributions.

    Centers each contribution to ensure balanced classes (~50/50).
    """
    n = contributions[0].shape[0]
    logit = np.zeros(n)
    for c, w in zip(contributions, weights):
        # Center each contribution so E[logit] ≈ 0 → balanced classes
        logit += w * (c - c.mean())
    logit += rng.normal(0, sigma_noise, size=n)
    y = (logit > 0).astype(int)
    return y


def assign_folds(y: np.ndarray, n_splits: int = 5, random_state: int = 42) -> np.ndarray:
    """Assign stratified k-fold indices."""
    folds = np.zeros(len(y), dtype=int)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for fold_idx, (_, test_idx) in enumerate(skf.split(np.zeros(len(y)), y)):
        folds[test_idx] = fold_idx
    return folds


# ---------------------------------------------------------------------------
# Variant generators
# ---------------------------------------------------------------------------

def gen_easy_2mod_xor(rng: np.random.Generator) -> dict:
    """Variant 1: 2 XOR modules, 10 features, 10K samples."""
    n, d = 10000, 10
    X = rng.standard_normal((n, d))

    # Module A: [0,1] XOR, Module B: [2,3] XOR
    c_a = xor_interaction(X[:, 0], X[:, 1])
    c_b = xor_interaction(X[:, 2], X[:, 3])

    # Redundant: X4 = noisy X0, X5 = noisy X2
    X[:, 4] = make_redundant(X[:, 0], 0.3, rng)
    X[:, 5] = make_redundant(X[:, 2], 0.3, rng)

    # Noise: [6,7,8,9] already random

    y = generate_target([c_a, c_b], [1.0, 1.0], 0.1, rng)
    folds = assign_folds(y)

    meta = {
        "n_samples": n, "n_features": d, "n_modules": 2,
        "ground_truth_modules": [[0, 1], [2, 3]],
        "module_types": ["xor", "xor"],
        "module_weights": [1.0, 1.0],
        "sigma_noise": 0.1,
        "redundant_pairs": [[0, 4], [2, 5]],
        "redundant_sigma": 0.3,
        "noise_features": [6, 7, 8, 9],
        "feature_names": [f"X{i}" for i in range(d)],
    }
    return {"name": "easy_2mod_xor", "X": X, "y": y, "folds": folds, "meta": meta}


def gen_medium_4mod_mixed(rng: np.random.Generator) -> dict:
    """Variant 2: 4 modules (2 XOR + 2 AND), 18 features, 20K samples."""
    n, d = 20000, 18
    X = rng.standard_normal((n, d))

    c_a = xor_interaction(X[:, 0], X[:, 1])
    c_b = xor_interaction(X[:, 2], X[:, 3])
    c_c = and_interaction(X[:, 4], X[:, 5])
    c_d = and_interaction(X[:, 6], X[:, 7])

    # Redundant: X8=X0, X9=X2, X10=X4, X11=X6 (noisy copies)
    X[:, 8] = make_redundant(X[:, 0], 0.3, rng)
    X[:, 9] = make_redundant(X[:, 2], 0.3, rng)
    X[:, 10] = make_redundant(X[:, 4], 0.3, rng)
    X[:, 11] = make_redundant(X[:, 6], 0.3, rng)

    # Noise: [12-17] already random

    # AND has ~5x less variance than XOR after centering; compensate with higher weight
    y = generate_target([c_a, c_b, c_c, c_d], [1.0, 1.0, 2.5, 2.5], 0.2, rng)
    folds = assign_folds(y)

    meta = {
        "n_samples": n, "n_features": d, "n_modules": 4,
        "ground_truth_modules": [[0, 1], [2, 3], [4, 5], [6, 7]],
        "module_types": ["xor", "xor", "and", "and"],
        "module_weights": [1.0, 1.0, 2.5, 2.5],
        "sigma_noise": 0.2,
        "redundant_pairs": [[0, 8], [2, 9], [4, 10], [6, 11]],
        "redundant_sigma": 0.3,
        "noise_features": list(range(12, 18)),
        "feature_names": [f"X{i}" for i in range(d)],
    }
    return {"name": "medium_4mod_mixed", "X": X, "y": y, "folds": folds, "meta": meta}


def gen_hard_4mod_unequal(rng: np.random.Generator) -> dict:
    """Variant 3: 4 unequal-size modules, 31 features, 20K samples."""
    n, d = 20000, 31
    X = rng.standard_normal((n, d))

    # Module A: [0,1] 2-way XOR
    c_a = xor_interaction(X[:, 0], X[:, 1])

    # Module B: [2,3,4] 3-way XOR
    c_b = three_way_xor(X[:, 2], X[:, 3], X[:, 4])

    # Module C: [5,6,7,8] pairwise XOR sum
    c_c = pairwise_xor_sum(X[:, 5], X[:, 6], X[:, 7], X[:, 8])

    # Module D: [9,10,11,12,13] AND chain
    c_d = and_chain(X[:, 9:14])

    # Redundant: X14=X0, X15=X2, X16=X5, X17=X9, X18=X11 (sigma=0.5)
    X[:, 14] = make_redundant(X[:, 0], 0.5, rng)
    X[:, 15] = make_redundant(X[:, 2], 0.5, rng)
    X[:, 16] = make_redundant(X[:, 5], 0.5, rng)
    X[:, 17] = make_redundant(X[:, 9], 0.5, rng)
    X[:, 18] = make_redundant(X[:, 11], 0.5, rng)

    # Noise: [19-30] already random

    # 5-way AND chain has ~3% activation; needs very high weight to be detectable
    y = generate_target([c_a, c_b, c_c, c_d], [1.5, 1.5, 0.8, 8.0], 0.5, rng)
    folds = assign_folds(y)

    meta = {
        "n_samples": n, "n_features": d, "n_modules": 4,
        "ground_truth_modules": [[0, 1], [2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12, 13]],
        "module_types": ["xor_2way", "xor_3way", "xor_pairwise_sum", "and_chain"],
        "module_weights": [1.5, 1.5, 0.8, 8.0],
        "sigma_noise": 0.5,
        "redundant_pairs": [[0, 14], [2, 15], [5, 16], [9, 17], [11, 18]],
        "redundant_sigma": 0.5,
        "noise_features": list(range(19, 31)),
        "feature_names": [f"X{i}" for i in range(d)],
    }
    return {"name": "hard_4mod_unequal", "X": X, "y": y, "folds": folds, "meta": meta}


def gen_overlapping_modules(rng: np.random.Generator) -> dict:
    """Variant 4: 4 overlapping modules, 18 features, 20K samples."""
    n, d = 20000, 18
    X = rng.standard_normal((n, d))

    # Module A: [0,1,2] XOR on X0*X1 (X2 participates in B)
    c_a = xor_interaction(X[:, 0], X[:, 1])

    # Module B: [2,3,4] AND on (X2>0)&(X3>0) (X4 participates in C)
    c_b = and_interaction(X[:, 2], X[:, 3])

    # Module C: [4,5,6] XOR on X4*X5 (X6 participates in D)
    c_c = xor_interaction(X[:, 4], X[:, 5])

    # Module D: [6,7] AND on (X6>0)&(X7>0)
    c_d = and_interaction(X[:, 6], X[:, 7])

    # Redundant: X8=X0, X9=X3, X10=X5
    X[:, 8] = make_redundant(X[:, 0], 0.3, rng)
    X[:, 9] = make_redundant(X[:, 3], 0.3, rng)
    X[:, 10] = make_redundant(X[:, 5], 0.3, rng)

    # Noise: [11-17] already random

    # AND modules need higher weights to compensate for lower variance
    y = generate_target([c_a, c_b, c_c, c_d], [1.0, 2.5, 1.0, 2.5], 0.2, rng)
    folds = assign_folds(y)

    meta = {
        "n_samples": n, "n_features": d, "n_modules": 4,
        "ground_truth_modules": [[0, 1, 2], [2, 3, 4], [4, 5, 6], [6, 7]],
        "module_types": ["xor", "and", "xor", "and"],
        "module_weights": [1.0, 2.5, 1.0, 2.5],
        "sigma_noise": 0.2,
        "primary_modules": [[0, 1, 2], [2, 3, 4], [4, 5, 6], [6, 7]],
        "shared_features": {"2": [0, 1], "4": [1, 2], "6": [2, 3]},
        "redundant_pairs": [[0, 8], [3, 9], [5, 10]],
        "redundant_sigma": 0.3,
        "noise_features": list(range(11, 18)),
        "feature_names": [f"X{i}" for i in range(d)],
    }
    return {"name": "overlapping_modules", "X": X, "y": y, "folds": folds, "meta": meta}


def gen_no_structure_control(rng: np.random.Generator) -> dict:
    """Variant 5: No synergy, purely additive, 20 features, 10K samples."""
    n, d = 10000, 20
    X = rng.standard_normal((n, d))

    # Linear: Y depends on sign(X0..X4) individually, no interactions
    logit = np.zeros(n)
    linear_weights = [0.8, 0.6, 0.5, 0.4, 0.3]
    for i, w in enumerate(linear_weights):
        logit += w * X[:, i]
    logit += rng.normal(0, 0.3, size=n)
    y = (logit > 0).astype(int)
    folds = assign_folds(y)

    meta = {
        "n_samples": n, "n_features": d, "n_modules": 0,
        "ground_truth_modules": [],
        "module_types": [],
        "informative_features": list(range(5)),
        "linear_weights": linear_weights,
        "sigma_noise": 0.3,
        "noise_features": list(range(5, 20)),
        "feature_names": [f"X{i}" for i in range(d)],
        "note": "Purely additive model — no synergistic interactions. Control case.",
    }
    return {"name": "no_structure_control", "X": X, "y": y, "folds": folds, "meta": meta}


def gen_highdim_8mod(rng: np.random.Generator) -> dict:
    """Variant 6: 8 modules, 200 features, 50K samples."""
    n, d = 50000, 200
    X = rng.standard_normal((n, d))

    contributions = []
    modules = []
    module_types = []

    # Modules 1-4: XOR (3 features each; XOR on first two, third adds partial info)
    for m in range(4):
        base = m * 3
        c = xor_interaction(X[:, base], X[:, base + 1])
        # Third feature adds partial linear info
        c = c + 0.3 * X[:, base + 2]
        contributions.append(c)
        modules.append([base, base + 1, base + 2])
        module_types.append("xor_plus_linear")

    # Modules 5-8: AND (3 features each)
    for m in range(4):
        base = 12 + m * 3
        c = and_interaction(X[:, base], X[:, base + 1])
        # Third feature adds partial info via AND
        c = c * (X[:, base + 2] > 0).astype(float)
        contributions.append(c)
        modules.append([base, base + 1, base + 2])
        module_types.append("and_three_way")

    # Total synergistic features: 24 (indices 0-23)
    # Redundant: 24 features (indices 24-47), one copy per synergistic feature
    for i in range(24):
        X[:, 24 + i] = make_redundant(X[:, i], 0.5, rng)

    # Noise: indices 48-199 already random

    # XOR modules weight 1.0; AND modules need 3.0 (3-way AND has ~12.5% activation)
    weights = [1.0, 1.0, 1.0, 1.0, 3.0, 3.0, 3.0, 3.0]
    y = generate_target(contributions, weights, 0.3, rng)
    folds = assign_folds(y)

    redundant_pairs = [[i, 24 + i] for i in range(24)]

    meta = {
        "n_samples": n, "n_features": d, "n_modules": 8,
        "ground_truth_modules": modules,
        "module_types": module_types,
        "module_weights": [1.0, 1.0, 1.0, 1.0, 3.0, 3.0, 3.0, 3.0],
        "sigma_noise": 0.3,
        "redundant_pairs": redundant_pairs,
        "redundant_sigma": 0.5,
        "noise_features": list(range(48, 200)),
        "feature_names": [f"X{i}" for i in range(d)],
    }
    return {"name": "highdim_8mod", "X": X, "y": y, "folds": folds, "meta": meta}


# ---------------------------------------------------------------------------
# Validation sanity checks
# ---------------------------------------------------------------------------

def _compute_interaction_term(X: np.ndarray, mod: list[int], mtype: str) -> np.ndarray:
    """Compute the explicit interaction term for a module."""
    if mtype in ("xor", "xor_2way"):
        return np.sign(X[:, mod[0]] * X[:, mod[1]])
    elif mtype == "xor_3way":
        return np.sign(X[:, mod[0]] * X[:, mod[1]] * X[:, mod[2]])
    elif mtype == "xor_pairwise_sum":
        return np.sign(X[:, mod[0]] * X[:, mod[1]]) + np.sign(X[:, mod[2]] * X[:, mod[3]])
    elif mtype in ("and", "and_chain"):
        return np.all(X[:, mod] > 0, axis=1).astype(float)
    elif mtype == "and_three_way":
        return ((X[:, mod[0]] > 0) & (X[:, mod[1]] > 0) & (X[:, mod[2]] > 0)).astype(float)
    elif mtype == "xor_plus_linear":
        return np.sign(X[:, mod[0]] * X[:, mod[1]]) + 0.3 * X[:, mod[2]]
    else:
        # Fallback: product of signs
        return np.sign(np.prod(X[:, mod], axis=1))


def validate_variant(result: dict) -> dict:
    """Run all sanity checks on a generated variant. Returns a report dict."""
    name = result["name"]
    X, y = result["X"], result["y"]
    meta = result["meta"]
    n, d = X.shape
    report = {"name": name, "passed": True, "checks": {}}

    # Thresholds scale with sample size — kNN MI estimation noise ~ O(1/sqrt(n))
    mi_zero_thresh = max(0.01, 0.5 / np.sqrt(n))  # ~0.005 at 10K, ~0.002 at 50K
    mi_joint_thresh = 0.02  # Lowered: with many modules, each term's MI is diluted
    mi_noise_thresh = max(0.01, 0.5 / np.sqrt(n))

    # 1. Sample & feature count
    expected_n = meta["n_samples"]
    expected_d = meta["n_features"]
    report["checks"]["sample_count"] = {"expected": expected_n, "actual": n, "ok": n == expected_n}
    report["checks"]["feature_count"] = {"expected": expected_d, "actual": d, "ok": d == expected_d}

    # 2. Class balance
    pos_rate = y.mean()
    ok_balance = 0.35 <= pos_rate <= 0.65
    report["checks"]["class_balance"] = {"pos_rate": round(pos_rate, 4), "ok": ok_balance}

    # Compute MI for all features once (reused for marginal + noise checks)
    mi_all = mutual_info_classif(X, y, discrete_features=False, random_state=42)

    if meta.get("ground_truth_modules"):
        # 3. Marginal MI for XOR features (should be ~0)
        # Skip shared features in overlapping modules — they participate in
        # AND modules too, so they will have non-zero marginal MI by design.
        shared_feats = set()
        for k in meta.get("shared_features", {}):
            shared_feats.add(int(k))

        xor_mi_checks = []
        for mod_idx, (mod, mtype) in enumerate(zip(meta["ground_truth_modules"], meta["module_types"])):
            if "xor" in mtype:
                for feat_idx in mod:
                    if feat_idx in shared_feats:
                        continue  # shared feature — skip marginal MI check
                    mi_val = mi_all[feat_idx]
                    ok = mi_val < mi_zero_thresh
                    xor_mi_checks.append({
                        "feature": feat_idx, "module": mod_idx,
                        "mi": round(float(mi_val), 5), "ok": bool(ok),
                    })
        if xor_mi_checks:
            report["checks"]["xor_marginal_mi"] = xor_mi_checks

        # 4. Joint MI: compute actual interaction term MI with Y
        joint_mi_checks = []
        for mod_idx, (mod, mtype) in enumerate(zip(meta["ground_truth_modules"], meta["module_types"])):
            interaction = _compute_interaction_term(X, mod, mtype)
            mi_joint = mutual_info_classif(
                interaction.reshape(-1, 1), y,
                discrete_features=False, random_state=42,
            )
            mi_val = float(mi_joint[0])
            ok = mi_val > mi_joint_thresh
            joint_mi_checks.append({
                "module": mod_idx, "features": mod, "type": mtype,
                "interaction_mi": round(mi_val, 5), "ok": ok,
            })
        report["checks"]["joint_mi"] = joint_mi_checks

    # 5. Noise features MI (should be ~0)
    noise_mi_checks = []
    for feat_idx in meta.get("noise_features", []):
        mi_val = mi_all[feat_idx]
        ok = mi_val < mi_noise_thresh
        noise_mi_checks.append({
            "feature": feat_idx, "mi": round(float(mi_val), 5), "ok": bool(ok),
        })
    if noise_mi_checks:
        report["checks"]["noise_feature_mi"] = noise_mi_checks

    # 6. Redundancy: correlation > 0.5
    red_checks = []
    for pair in meta.get("redundant_pairs", []):
        corr = float(np.corrcoef(X[:, pair[0]], X[:, pair[1]])[0, 1])
        ok = abs(corr) > 0.5
        red_checks.append({"pair": pair, "corr": round(corr, 4), "ok": ok})
    if red_checks:
        report["checks"]["redundancy_corr"] = red_checks

    # Mark overall pass/fail
    for key, val in report["checks"].items():
        if isinstance(val, list):
            for item in val:
                if not item.get("ok", True):
                    report["passed"] = False
        elif isinstance(val, dict):
            if not val.get("ok", True):
                report["passed"] = False

    return report


# ---------------------------------------------------------------------------
# Format dataset into JSON-schema-compatible examples
# ---------------------------------------------------------------------------

def format_examples(result: dict) -> list[dict]:
    """Convert X, y, folds into list of example dicts for JSON output."""
    X, y, folds = result["X"], result["y"], result["folds"]
    name = result["name"]
    feature_names = result["meta"]["feature_names"]
    n = X.shape[0]

    examples = []
    for i in range(n):
        # Build input as JSON string of feature values
        feat_dict = {fn: round(float(X[i, j]), 6) for j, fn in enumerate(feature_names)}
        examples.append({
            "input": json.dumps(feat_dict),
            "output": str(int(y[i])),
            "metadata_fold": int(folds[i]),
            "metadata_variant": name,
            "metadata_sample_idx": i,
            "metadata_row_index": i,
            "metadata_feature_names": feature_names,
            "metadata_task_type": "classification",
            "metadata_n_classes": 2,
        })
    return examples


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

GENERATORS = [
    ("easy_2mod_xor", gen_easy_2mod_xor),
    ("medium_4mod_mixed", gen_medium_4mod_mixed),
    ("hard_4mod_unequal", gen_hard_4mod_unequal),
    ("overlapping_modules", gen_overlapping_modules),
    ("no_structure_control", gen_no_structure_control),
    ("highdim_8mod", gen_highdim_8mod),
]


def generate_variant(name: str, gen_fn, seed: int) -> dict:
    """Generate a single variant with its own RNG."""
    rng = np.random.default_rng(seed)
    t0 = time.time()
    result = gen_fn(rng)
    dt = time.time() - t0
    logger.info(f"  Generated {name}: {result['X'].shape} in {dt:.2f}s")
    return result


@logger.catch
def main(max_samples_per_variant: int | None = None):
    """Generate all 6 variants, validate, and write output JSON files.

    Args:
        max_samples_per_variant: If set, truncate each variant for testing.
    """
    overall_t0 = time.time()
    logger.info("=" * 60)
    logger.info("Starting synthetic dataset generation")
    logger.info(f"Master seed: {MASTER_SEED}")
    if max_samples_per_variant:
        logger.info(f"TESTING MODE: max {max_samples_per_variant} samples per variant")
    logger.info("=" * 60)

    # Step 1: Generate all variants
    logger.info("Step 1: Generating all 6 variants...")
    base_rng = np.random.default_rng(MASTER_SEED)
    variant_seeds = [int(base_rng.integers(0, 2**31)) for _ in range(len(GENERATORS))]

    results = []
    for (name, gen_fn), seed in zip(GENERATORS, variant_seeds):
        result = generate_variant(name, gen_fn, seed)
        results.append(result)

    # Truncate if testing
    if max_samples_per_variant:
        for r in results:
            n = min(max_samples_per_variant, r["X"].shape[0])
            r["X"] = r["X"][:n]
            r["y"] = r["y"][:n]
            r["folds"] = r["folds"][:n]
            r["meta"]["n_samples"] = n

    # Step 2: Validate all variants
    logger.info("\nStep 2: Running validation sanity checks...")
    all_passed = True
    for r in results:
        t0 = time.time()
        report = validate_variant(r)
        dt = time.time() - t0
        status = "PASS" if report["passed"] else "FAIL"
        logger.info(f"  {r['name']}: {status} ({dt:.1f}s)")

        # Log details for failures
        if not report["passed"]:
            all_passed = False
            for key, val in report["checks"].items():
                if isinstance(val, list):
                    fails = [item for item in val if not item.get("ok", True)]
                    if fails:
                        logger.warning(f"    {key}: {len(fails)} failures")
                        for f in fails[:5]:
                            logger.warning(f"      {f}")
                elif isinstance(val, dict) and not val.get("ok", True):
                    logger.warning(f"    {key}: {val}")

        # Log class balance for all variants
        cb = report["checks"].get("class_balance", {})
        logger.info(f"    Class balance: {cb.get('pos_rate', 'N/A')}")

    if not all_passed:
        logger.warning("Some validation checks failed — see details above")

    # Step 3: Build metadata
    logger.info("\nStep 3: Building output JSON...")
    variants_meta = {}
    for r in results:
        variants_meta[r["name"]] = r["meta"]

    top_metadata = {
        "source": "synthetic",
        "description": (
            "Synthetic classification datasets with planted synergistic feature "
            "modules for validating signed spectral clustering recovery"
        ),
        "master_seed": MASTER_SEED,
        "generation_date": "2026-03-19",
        "variants": variants_meta,
    }

    # Step 4: Format examples
    logger.info("Step 4: Formatting examples...")
    datasets = []
    for r in results:
        t0 = time.time()
        examples = format_examples(r)
        dt = time.time() - t0
        logger.info(f"  {r['name']}: {len(examples)} examples formatted in {dt:.1f}s")
        datasets.append({"dataset": r["name"], "examples": examples})

    full_output = {"metadata": top_metadata, "datasets": datasets}

    # Free large arrays
    for r in results:
        del r["X"], r["y"], r["folds"]
    del results
    gc.collect()

    # Step 5: Write full_data_out.json (single file; splitting handled separately)
    logger.info("\nStep 5: Writing full_data_out.json...")
    out_dir = WORKSPACE
    full_path = out_dir / "full_data_out.json"
    t0 = time.time()
    full_path.write_text(json.dumps(full_output, indent=None))
    dt = time.time() - t0
    full_size_mb = full_path.stat().st_size / (1024 * 1024)
    logger.info(f"  full_data_out.json: {full_size_mb:.1f} MB ({dt:.1f}s)")

    # Clean up
    del full_output
    gc.collect()

    total_dt = time.time() - overall_t0
    logger.info(f"\nDone! Total time: {total_dt:.1f}s")
    logger.info(f"Output: {full_path} ({full_size_mb:.1f} MB)")

    return full_path


if __name__ == "__main__":
    # Support testing mode via command-line arg
    max_n = None
    if len(sys.argv) > 1:
        max_n = int(sys.argv[1])
    main(max_samples_per_variant=max_n)

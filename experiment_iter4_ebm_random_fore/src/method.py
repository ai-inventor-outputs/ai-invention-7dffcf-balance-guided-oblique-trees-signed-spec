#!/usr/bin/env python3
"""EBM + Random Forest + Logistic/Ridge Baselines on 8 Real Datasets.

Runs three baseline methods (EBM, Random Forest, Logistic/Ridge Regression) on
all 8 Grinsztajn benchmark datasets using pre-assigned 5-fold CV splits.
Records balanced_accuracy, AUC, R2 (regression), fit_time, and EBM-specific
interpretability metrics. Outputs method_out.json with per-fold results,
aggregated stats, and per-example predictions for downstream comparison.
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
from loguru import logger

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

# Set memory limits — use 50% of container RAM
RAM_BUDGET_GB = min(14, TOTAL_RAM_GB * 0.5)
RAM_BUDGET = int(RAM_BUDGET_GB * 1024**3)
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET * 3, RAM_BUDGET * 3))
resource.setrlimit(resource.RLIMIT_CPU, (3600, 3600))
logger.info(f"RAM budget: {RAM_BUDGET_GB:.1f} GB, CPU limit: 3600s")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
WORKSPACE = Path(__file__).parent
NUM_WORKERS = max(1, NUM_CPUS)

DATA_ID4_DIR = Path("/ai-inventor/aii_pipeline/runs/jamnik-sgfigs-pid-v2/"
                     "3_invention_loop/iter_1/gen_art/data_id4_it1__opus")
DATA_ID5_DIR = Path("/ai-inventor/aii_pipeline/runs/jamnik-sgfigs-pid-v2/"
                     "3_invention_loop/iter_2/gen_art/data_id5_it2__opus")

EXPECTED_DATASETS = ["electricity", "adult", "california_housing", "jannis",
                     "higgs_small", "eye_movements", "credit", "miniboone"]

METHOD_NAMES = ["ebm", "random_forest", "linear"]

# Max examples per dataset in final output to control file size
MAX_OUTPUT_EXAMPLES = 2000

# ---------------------------------------------------------------------------
# SECTION 1: DATA LOADING
# ---------------------------------------------------------------------------

def load_split_json(directory: Path, pattern: str) -> dict:
    """Load multi-part JSON files and merge by dataset name."""
    all_datasets: dict[str, list] = {}
    for part_file in sorted(directory.glob(pattern)):
        logger.info(f"  Loading {part_file.name}...")
        raw = part_file.read_text()
        data = json.loads(raw)
        del raw
        gc.collect()
        for ds_block in data["datasets"]:
            name = ds_block["dataset"]
            all_datasets.setdefault(name, []).extend(ds_block["examples"])
        del data
        gc.collect()
    return all_datasets


# ---------------------------------------------------------------------------
# SECTION 2: DATA PARSING
# ---------------------------------------------------------------------------

def parse_dataset(name: str, examples: list) -> dict:
    """Convert list of examples to X (numpy), y (numpy), folds (numpy), metadata."""
    first = examples[0]

    # Extract full feature names from input JSON (metadata_feature_names is truncated)
    first_input = json.loads(first["input"])
    feature_names_full = list(first_input.keys())

    task_type = first["metadata_task_type"]
    n_classes = first.get("metadata_n_classes", None)
    is_regression = ("regression" in task_type)

    n = len(examples)
    d = len(feature_names_full)
    X = np.zeros((n, d), dtype=np.float64)
    y_raw = []
    folds = np.zeros(n, dtype=np.int32)
    row_indices = np.zeros(n, dtype=np.int64)

    for i, ex in enumerate(examples):
        feats = json.loads(ex["input"])
        for j, fname in enumerate(feature_names_full):
            X[i, j] = float(feats[fname])
        y_raw.append(ex["output"])
        folds[i] = int(ex["metadata_fold"])
        row_indices[i] = int(ex.get("metadata_row_index", i))

    if is_regression:
        y = np.array([float(v) for v in y_raw], dtype=np.float64)
    else:
        unique_labels = sorted(set(y_raw))
        label_map = {lbl: idx for idx, lbl in enumerate(unique_labels)}
        y = np.array([label_map[v] for v in y_raw], dtype=np.int64)
        n_classes = len(unique_labels)

    return {
        "name": name,
        "X": X, "y": y, "folds": folds,
        "row_indices": row_indices,
        "feature_names": feature_names_full,
        "task_type": task_type,
        "is_regression": is_regression,
        "n_classes": n_classes,
        "n_samples": n, "n_features": d,
        "examples_raw": examples,  # keep for output
    }


# ---------------------------------------------------------------------------
# SECTION 3: MODEL DEFINITIONS
# ---------------------------------------------------------------------------

def make_model(method_name: str, is_regression: bool, n_classes: int | None = None):
    """Instantiate a fresh model for the given method and task type."""
    if method_name == "ebm":
        if is_regression:
            from interpret.glassbox import ExplainableBoostingRegressor
            return ExplainableBoostingRegressor(
                outer_bags=8,
                interactions=10,
                max_rounds=5000,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
        else:
            from interpret.glassbox import ExplainableBoostingClassifier
            return ExplainableBoostingClassifier(
                outer_bags=8,
                interactions=10,
                max_rounds=5000,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
    elif method_name == "random_forest":
        if is_regression:
            from sklearn.ensemble import RandomForestRegressor
            return RandomForestRegressor(
                n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
        else:
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(
                n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
    elif method_name == "linear":
        if is_regression:
            from sklearn.linear_model import Ridge
            return Ridge(alpha=1.0)
        else:
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression(
                max_iter=1000, C=1.0, random_state=RANDOM_STATE)
    else:
        raise ValueError(f"Unknown method: {method_name}")


# ---------------------------------------------------------------------------
# SECTION 4: METRICS COMPUTATION
# ---------------------------------------------------------------------------

from sklearn.metrics import balanced_accuracy_score, roc_auc_score, r2_score
from sklearn.preprocessing import StandardScaler


def compute_metrics(model, method_name: str, X_test: np.ndarray,
                    y_test: np.ndarray, is_regression: bool,
                    n_classes: int | None) -> dict:
    """Compute all relevant metrics for one fold."""
    metrics: dict = {}

    if is_regression:
        y_pred = model.predict(X_test)
        metrics["r2"] = float(r2_score(y_test, y_pred))
        metrics["balanced_accuracy"] = None
        metrics["auc"] = None
    else:
        y_pred = model.predict(X_test)
        metrics["balanced_accuracy"] = float(
            balanced_accuracy_score(y_test, y_pred))
        metrics["r2"] = None

        # AUC computation
        try:
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test)
            elif hasattr(model, "decision_function"):
                y_prob = model.decision_function(X_test)
            else:
                y_prob = None

            if y_prob is not None:
                if n_classes == 2:
                    if y_prob.ndim == 2:
                        metrics["auc"] = float(
                            roc_auc_score(y_test, y_prob[:, 1]))
                    else:
                        metrics["auc"] = float(
                            roc_auc_score(y_test, y_prob))
                else:
                    metrics["auc"] = float(roc_auc_score(
                        y_test, y_prob,
                        multi_class="ovr", average="weighted"))
            else:
                metrics["auc"] = None
        except Exception as e:
            logger.warning(f"AUC computation failed: {e}")
            metrics["auc"] = None

    # EBM-specific interpretability metrics
    if method_name == "ebm":
        try:
            metrics["n_terms"] = len(model.term_names_)
            metrics["n_interaction_terms"] = sum(
                1 for t in model.term_features_ if len(t) >= 2)
            metrics["n_main_effects"] = sum(
                1 for t in model.term_features_ if len(t) == 1)
        except AttributeError as e:
            logger.warning(f"EBM interpretability metrics failed: {e}")

    return metrics


# ---------------------------------------------------------------------------
# SECTION 5: MAIN EXPERIMENT LOOP
# ---------------------------------------------------------------------------

@logger.catch
def main():
    total_start = time.time()
    logger.info("=" * 60)
    logger.info("Starting EBM + RF + Linear baselines experiment")
    logger.info("=" * 60)

    # -----------------------------------------------------------------------
    # Load data
    # -----------------------------------------------------------------------
    logger.info("Loading data from data_id4...")
    ds4 = load_split_json(DATA_ID4_DIR / "full_data_out", "full_data_out_*.json")
    logger.info(f"  data_id4 datasets: {list(ds4.keys())}")

    logger.info("Loading data from data_id5...")
    ds5 = load_split_json(DATA_ID5_DIR / "full_data_out", "full_data_out_*.json")
    logger.info(f"  data_id5 datasets: {list(ds5.keys())}")

    all_datasets_raw = {**ds4, **ds5}
    del ds4, ds5
    gc.collect()

    # Verify all 8 datasets present
    found = set(all_datasets_raw.keys())
    expected = set(EXPECTED_DATASETS)
    if found != expected:
        logger.error(f"Dataset mismatch! Found={found}, Expected={expected}")
        logger.error(f"Missing: {expected - found}")
        logger.error(f"Extra: {found - expected}")
        # Continue with whatever we have
    else:
        logger.info("All 8 expected datasets found")

    # -----------------------------------------------------------------------
    # Parse datasets
    # -----------------------------------------------------------------------
    datasets = []
    for name in EXPECTED_DATASETS:
        if name not in all_datasets_raw:
            logger.warning(f"Skipping missing dataset: {name}")
            continue
        logger.info(f"Parsing {name}: {len(all_datasets_raw[name])} examples")
        ds = parse_dataset(name, all_datasets_raw[name])
        datasets.append(ds)
        logger.info(f"  -> {ds['n_samples']} x {ds['n_features']}, "
                     f"task={ds['task_type']}, n_classes={ds['n_classes']}")
        # Free raw examples after parsing
        del all_datasets_raw[name]
        gc.collect()

    del all_datasets_raw
    gc.collect()

    # Sort by size ascending for quick early results
    datasets_sorted = sorted(datasets, key=lambda d: d["n_samples"])

    # -----------------------------------------------------------------------
    # Run experiment
    # -----------------------------------------------------------------------
    all_results: dict = {}  # dataset_name -> {method -> {fold_results, aggregate}}
    all_predictions: dict = {}  # dataset_name -> {method -> {fold -> predictions}}

    for ds_idx, ds in enumerate(datasets_sorted):
        ds_name = ds["name"]
        X, y, folds_arr = ds["X"], ds["y"], ds["folds"]
        is_reg = ds["is_regression"]
        n_cls = ds["n_classes"]
        n_folds = len(np.unique(folds_arr))

        logger.info(f"\n{'=' * 60}")
        logger.info(f"Dataset {ds_idx + 1}/{len(datasets_sorted)}: {ds_name} "
                     f"({ds['n_samples']} x {ds['n_features']}, "
                     f"{'regression' if is_reg else f'{n_cls}-class classification'})")

        ds_results: dict = {}
        ds_predictions: dict = {}

        # Process methods: ebm (slowest) -> rf -> linear
        for method_name in METHOD_NAMES:
            logger.info(f"  Method: {method_name}")
            fold_metrics = []
            fold_predictions: dict = {}
            method_total_time = 0.0

            for fold_id in range(n_folds):
                test_mask = (folds_arr == fold_id)
                train_mask = ~test_mask
                X_train, X_test = X[train_mask], X[test_mask]
                y_train, y_test = y[train_mask], y[test_mask]

                # Scale for linear models
                if method_name == "linear":
                    scaler = StandardScaler()
                    X_train_use = scaler.fit_transform(X_train)
                    X_test_use = scaler.transform(X_test)
                else:
                    X_train_use = X_train
                    X_test_use = X_test

                # Train
                t0 = time.time()
                model = make_model(method_name, is_reg, n_cls)
                try:
                    model.fit(X_train_use, y_train)
                    fit_time = time.time() - t0

                    # Metrics
                    metrics = compute_metrics(
                        model, method_name, X_test_use, y_test,
                        is_reg, n_cls)
                    metrics["fit_time"] = round(fit_time, 3)
                    metrics["fold"] = fold_id
                    metrics["status"] = "success"

                    # Store predictions on test set
                    preds = model.predict(X_test_use)
                    fold_predictions[fold_id] = {
                        "predictions": preds.tolist(),
                        "test_indices": np.where(test_mask)[0].tolist(),
                    }

                except Exception as e:
                    logger.exception(f"    Fold {fold_id} FAILED: {e}")
                    fit_time = time.time() - t0
                    metrics = {
                        "fold": fold_id, "status": "error",
                        "error": str(e),
                        "fit_time": round(fit_time, 3),
                        "balanced_accuracy": None,
                        "auc": None, "r2": None,
                    }

                fold_metrics.append(metrics)
                method_total_time += metrics.get("fit_time", 0)
                logger.info(
                    f"    Fold {fold_id}: "
                    f"bal_acc={metrics.get('balanced_accuracy')}, "
                    f"auc={metrics.get('auc')}, "
                    f"r2={metrics.get('r2')}, "
                    f"time={metrics.get('fit_time', 0):.1f}s")

                del model
                gc.collect()

            # Aggregate across folds
            successful = [m for m in fold_metrics if m["status"] == "success"]
            aggregate: dict = {}
            for metric_key in ["balanced_accuracy", "auc", "r2", "fit_time"]:
                vals = [m[metric_key] for m in successful
                        if m.get(metric_key) is not None]
                if vals:
                    aggregate[f"{metric_key}_mean"] = round(
                        float(np.mean(vals)), 6)
                    aggregate[f"{metric_key}_std"] = round(
                        float(np.std(vals)), 6)
                else:
                    aggregate[f"{metric_key}_mean"] = None
                    aggregate[f"{metric_key}_std"] = None
            aggregate["total_fit_time"] = round(method_total_time, 2)
            aggregate["n_successful_folds"] = len(successful)

            # EBM-specific aggregate
            if method_name == "ebm" and successful:
                for ekey in ["n_terms", "n_interaction_terms", "n_main_effects"]:
                    evals = [m[ekey] for m in successful if ekey in m]
                    if evals:
                        aggregate[f"{ekey}_mean"] = round(
                            float(np.mean(evals)), 1)

            ds_results[method_name] = {
                "fold_results": fold_metrics,
                "aggregate": aggregate,
            }
            ds_predictions[method_name] = fold_predictions

            logger.info(f"  {method_name} aggregate: {aggregate}")

        all_results[ds_name] = ds_results
        all_predictions[ds_name] = ds_predictions

        # CHECKPOINT: write intermediate results after each dataset
        elapsed = round(time.time() - total_start, 1)
        intermediate = {
            "metadata": {
                "status": "in_progress",
                "datasets_completed": ds_idx + 1,
                "elapsed_sec": elapsed,
            },
            "results": all_results,
        }
        (WORKSPACE / "method_out_intermediate.json").write_text(
            json.dumps(intermediate, indent=2, default=str))
        logger.info(f"Checkpoint saved after {ds_name} "
                     f"({ds_idx + 1}/{len(datasets_sorted)}, {elapsed:.0f}s)")

        # Free dataset arrays
        ds["X"] = None
        ds["y"] = None
        gc.collect()

    # -----------------------------------------------------------------------
    # SECTION 6: BUILD FINAL OUTPUT — method_out.json
    # -----------------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("Building final output...")

    # Build cross-dataset summary table
    summary_table: dict = {}
    for method_name in METHOD_NAMES:
        summary_table[method_name] = {}
        for metric in ["balanced_accuracy_mean", "auc_mean", "r2_mean",
                       "total_fit_time"]:
            vals = []
            for ds_name in EXPECTED_DATASETS:
                if (ds_name in all_results
                        and method_name in all_results[ds_name]):
                    v = all_results[ds_name][method_name]["aggregate"].get(
                        metric)
                    vals.append({"dataset": ds_name, "value": v})
            summary_table[method_name][metric] = vals

    # Build datasets block conforming to exp_gen_sol_out schema
    output_datasets = []
    for ds in datasets_sorted:
        ds_name = ds["name"]
        examples_raw = ds.get("examples_raw", [])
        n_total = len(examples_raw)

        # Subsample to control output file size
        if n_total > MAX_OUTPUT_EXAMPLES:
            step = n_total / MAX_OUTPUT_EXAMPLES
            indices = [int(i * step) for i in range(MAX_OUTPUT_EXAMPLES)]
        else:
            indices = list(range(n_total))

        # Build index -> prediction mapping per method per fold
        pred_maps: dict = {}
        for method_name in METHOD_NAMES:
            pred_maps[method_name] = {}
            if (ds_name in all_predictions
                    and method_name in all_predictions[ds_name]):
                for fold_id_str, fold_data in all_predictions[ds_name][method_name].items():
                    fold_id = int(fold_id_str)
                    test_indices = fold_data["test_indices"]
                    preds = fold_data["predictions"]
                    for ti, pred_val in zip(test_indices, preds):
                        pred_maps[method_name][ti] = pred_val

        output_examples = []
        for idx in indices:
            ex = examples_raw[idx]
            out_ex: dict = {
                "input": ex["input"],
                "output": str(ex["output"]),
            }
            # Add metadata fields
            for k, v in ex.items():
                if k.startswith("metadata_"):
                    out_ex[k] = v
            # Add predictions
            for method_name in METHOD_NAMES:
                pred_key = f"predict_{method_name}"
                if idx in pred_maps[method_name]:
                    out_ex[pred_key] = str(pred_maps[method_name][idx])
                else:
                    out_ex[pred_key] = ""
            output_examples.append(out_ex)

        output_datasets.append({
            "dataset": ds_name,
            "examples": output_examples,
        })

    total_runtime = round(time.time() - total_start, 1)

    output = {
        "metadata": {
            "experiment": "baselines_ebm_rf_linear",
            "description": ("EBM, Random Forest, and Logistic/Ridge Regression "
                            "baselines on 8 Grinsztajn benchmark datasets"),
            "date": "2026-03-19",
            "methods": METHOD_NAMES,
            "n_folds": 5,
            "random_state": RANDOM_STATE,
            "ebm_config": {
                "outer_bags": 8, "interactions": 10, "max_rounds": 5000},
            "rf_config": {"n_estimators": 100},
            "linear_config": {
                "logistic_C": 1.0, "logistic_max_iter": 1000,
                "ridge_alpha": 1.0},
            "total_runtime_s": total_runtime,
            "per_dataset_results": all_results,
            "cross_dataset_summary": summary_table,
        },
        "datasets": output_datasets,
    }

    out_path = WORKSPACE / "method_out.json"
    out_text = json.dumps(output, indent=None, default=str)
    out_path.write_text(out_text)
    out_size_mb = len(out_text.encode()) / 1e6
    logger.info(f"Output written to {out_path} ({out_size_mb:.1f} MB)")

    # Warn if too large
    if out_size_mb > 100:
        logger.warning(f"Output file is {out_size_mb:.1f} MB — exceeds 100 MB!")
        logger.info("Reducing to metadata-only output...")
        output_slim = {
            "metadata": output["metadata"],
            "datasets": [{"dataset": d["dataset"],
                          "examples": d["examples"][:100]}
                         for d in output_datasets],
        }
        out_path.write_text(
            json.dumps(output_slim, indent=None, default=str))
        logger.info("Reduced output written")

    logger.info(f"\nTotal runtime: {total_runtime:.1f}s "
                f"({total_runtime / 60:.1f} min)")
    logger.info("DONE")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Process and standardize 5 tabular benchmark datasets for Balance-Guided Oblique Trees evaluation.

Datasets:
  1. electricity  - binary classification, 7 numeric features (from Grinsztajn HF)
  2. adult        - binary classification, 6 numeric features (from HF + OpenML fallback)
  3. higgs_small  - binary classification, 24 numeric features (subsampled from Grinsztajn HF)
  4. jannis       - multiclass classification, 54 numeric features (from Grinsztajn HF)
  5. california_housing - regression, 8 numeric features (from sklearn)
"""

import json
import gc
import math
import os
import resource
import sys
import warnings
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import StratifiedKFold, KFold

warnings.filterwarnings("ignore")

# ── Logging ──────────────────────────────────────────────────────────────────
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add("logs/run.log", rotation="30 MB", level="DEBUG")

# ── Hardware detection ───────────────────────────────────────────────────────
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
RAM_BUDGET = int(TOTAL_RAM_GB * 0.7 * 1e9)  # 70% of container RAM
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET * 3, RAM_BUDGET * 3))

logger.info(f"Hardware: {NUM_CPUS} CPUs, {TOTAL_RAM_GB:.1f} GB RAM, budget={RAM_BUDGET/1e9:.1f} GB")

# ── Paths ────────────────────────────────────────────────────────────────────
WORKSPACE = Path("/ai-inventor/aii_pipeline/runs/jamnik-sgfigs-pid-v2/3_invention_loop/iter_1/gen_art/data_id4_it1__opus")
RAW_DIR = WORKSPACE / "temp" / "datasets"
OUT_DIR = WORKSPACE
RANDOM_STATE = 42

# ── Dataset configs ──────────────────────────────────────────────────────────
DATASET_CONFIGS = {
    "electricity": {
        "raw_file": "full_inria-soda_tabular-benchmark_clf_num_electricity_train.json",
        "target_col": "class",
        "task": "binary_classification",
        "source": "hf_inria-soda/tabular-benchmark_clf_num_electricity",
        "target_encoding": {"DOWN": 0, "UP": 1},
        "numeric_only": False,  # all features are already numeric
        "subsample": None,
    },
    "adult": {
        "raw_file": "full_scikit-learn_adult-census-income_train.json",
        "target_col": "income",
        "task": "binary_classification",
        "source": "hf_scikit-learn/adult-census-income",
        "target_encoding": {"<=50K": 0, "<=50K.": 0, ">50K": 1, ">50K.": 1},
        "numeric_only": True,  # keep only numeric features per plan Option A
        "numeric_features": ["age", "fnlwgt", "education.num", "capital.gain", "capital.loss", "hours.per.week"],
        "subsample": None,
    },
    "higgs_small": {
        "raw_file": "full_inria-soda_tabular-benchmark_clf_num_Higgs_train.json",
        "target_col": "target",
        "task": "binary_classification",
        "source": "hf_inria-soda/tabular-benchmark_clf_num_Higgs",
        "target_encoding": None,  # already 0/1 integers
        "numeric_only": False,
        "subsample": 98050,  # subsample to match plan spec
    },
    "jannis": {
        "raw_file": "full_inria-soda_tabular-benchmark_clf_num_jannis_train.json",
        "target_col": "class",
        "task": "auto_detect",  # auto-detect: binary if 2 classes, multiclass if >2
        "source": "hf_inria-soda/tabular-benchmark_clf_num_jannis",
        "target_encoding": None,  # will be mapped to 0-indexed
        "numeric_only": False,
        "subsample": None,
    },
    "california_housing": {
        "raw_file": None,  # loaded from sklearn
        "target_col": "MedHouseVal",
        "task": "regression",
        "source": "sklearn_california_housing",
        "target_encoding": None,
        "numeric_only": False,
        "subsample": None,
    },
}


def load_raw_dataset(name: str, config: dict) -> pd.DataFrame:
    """Load raw dataset from file or sklearn."""
    if name == "california_housing":
        logger.info("Loading california_housing from sklearn")
        data = fetch_california_housing(as_frame=True)
        df = data.data.copy()
        df["MedHouseVal"] = data.target
        logger.info(f"  california_housing: {df.shape}")
        return df

    raw_path = RAW_DIR / config["raw_file"]
    logger.info(f"Loading {name} from {raw_path.name}")

    # Stream-read large JSON files
    file_size_mb = raw_path.stat().st_size / 1e6
    if file_size_mb > 100:
        logger.info(f"  Large file ({file_size_mb:.0f} MB), reading in chunks...")
        # For the higgs file (717MB), read as JSON array
        with open(raw_path, "r") as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        del data
        gc.collect()
    else:
        with open(raw_path, "r") as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        del data
        gc.collect()

    logger.info(f"  {name}: {df.shape}")
    return df


def validate_raw(name: str, df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Validate and clean raw dataset."""
    logger.info(f"Validating {name}...")

    # Check shape
    logger.info(f"  Shape: {df.shape}")

    # Check for NaN/missing
    nan_counts = df.isnull().sum()
    total_nan = nan_counts.sum()
    if total_nan > 0:
        logger.warning(f"  Found {total_nan} NaN values, dropping rows...")
        before = len(df)
        df = df.dropna().reset_index(drop=True)
        logger.info(f"  Dropped {before - len(df)} rows with NaN")

    # Handle '?' values in adult
    if name == "adult":
        for col in df.columns:
            if df[col].dtype == object:
                mask = df[col].astype(str).str.strip() == "?"
                if mask.any():
                    logger.info(f"  Found {mask.sum()} '?' values in {col}")

    # Apply numeric-only filter for adult
    target_col = config["target_col"]
    if config.get("numeric_only") and "numeric_features" in config:
        keep_cols = config["numeric_features"] + [target_col]
        available = [c for c in keep_cols if c in df.columns]
        logger.info(f"  Keeping numeric features only: {[c for c in available if c != target_col]}")
        df = df[available].copy()

    # Verify numeric features
    feature_cols = [c for c in df.columns if c != target_col]
    numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    non_numeric = [c for c in feature_cols if c not in numeric_cols]
    if non_numeric:
        logger.warning(f"  Non-numeric feature columns: {non_numeric}")
        # Convert to numeric where possible
        for col in non_numeric:
            try:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                logger.info(f"    Converted {col} to numeric")
            except Exception:
                logger.warning(f"    Could not convert {col}, dropping")
                df = df.drop(columns=[col])
        # Drop any new NaN rows from conversion
        if df.isnull().any().any():
            before = len(df)
            df = df.dropna().reset_index(drop=True)
            logger.info(f"  Dropped {before - len(df)} rows after numeric conversion")

    # Subsample if needed
    if config.get("subsample"):
        n = config["subsample"]
        if len(df) > n:
            logger.info(f"  Subsampling from {len(df)} to {n} rows (stratified)")
            if config["task"] != "regression":
                # Stratified subsample
                from sklearn.model_selection import train_test_split
                _, df_sub = train_test_split(
                    df, test_size=n, stratify=df[target_col],
                    random_state=RANDOM_STATE
                )
                df = df_sub.reset_index(drop=True)
            else:
                df = df.sample(n=n, random_state=RANDOM_STATE).reset_index(drop=True)
            logger.info(f"  After subsample: {df.shape}")

    # Log target distribution
    if config["task"] == "regression":
        logger.info(f"  Target stats: {df[target_col].describe().to_dict()}")
    else:
        logger.info(f"  Target distribution ({df[target_col].nunique()} classes): {df[target_col].value_counts().to_dict()}")

    # Log feature info
    feature_cols = [c for c in df.columns if c != target_col]
    logger.info(f"  Features ({len(feature_cols)}): {feature_cols[:10]}{'...' if len(feature_cols) > 10 else ''}")
    logger.info(f"  Dtypes: {df[feature_cols].dtypes.value_counts().to_dict()}")

    return df


def encode_target(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Encode target variable."""
    target_col = config["target_col"]
    encoding = config.get("target_encoding")

    if encoding:
        # Map string labels to integers
        original_values = df[target_col].unique()
        logger.info(f"  Encoding target: {dict(zip(original_values[:5], [encoding.get(str(v).strip(), v) for v in original_values[:5]]))}")
        df[target_col] = df[target_col].astype(str).str.strip().map(encoding)
        # Check for unmapped values
        if df[target_col].isnull().any():
            unmapped = df[df[target_col].isnull()].index
            logger.warning(f"  {len(unmapped)} rows with unmapped target values, dropping")
            df = df.dropna(subset=[target_col]).reset_index(drop=True)
        df[target_col] = df[target_col].astype(int)
    elif config["task"] == "multiclass_classification":
        # Map to 0-indexed integers
        unique_classes = sorted(df[target_col].unique())
        class_map = {c: i for i, c in enumerate(unique_classes)}
        logger.info(f"  Multiclass mapping: {len(class_map)} classes, first 5: {dict(list(class_map.items())[:5])}")
        df[target_col] = df[target_col].map(class_map)
        df[target_col] = df[target_col].astype(int)
    elif config["task"] == "binary_classification":
        df[target_col] = df[target_col].astype(int)
    else:
        # regression - keep as float
        df[target_col] = df[target_col].astype(float)

    return df


def assign_folds(df: pd.DataFrame, config: dict) -> np.ndarray:
    """Assign 5-fold cross-validation fold indices."""
    target_col = config["target_col"]
    y = df[target_col].values

    if config["task"] in ("binary_classification", "multiclass_classification"):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        fold_assignments = np.zeros(len(df), dtype=int)
        for fold_idx, (_, val_idx) in enumerate(skf.split(df, y)):
            fold_assignments[val_idx] = fold_idx
    else:
        kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        fold_assignments = np.zeros(len(df), dtype=int)
        for fold_idx, (_, val_idx) in enumerate(kf.split(df)):
            fold_assignments[val_idx] = fold_idx

    # Verify balance
    unique, counts = np.unique(fold_assignments, return_counts=True)
    logger.info(f"  Fold distribution: {dict(zip(unique.tolist(), counts.tolist()))}")
    return fold_assignments


def df_to_records(df: pd.DataFrame, folds: np.ndarray, name: str, config: dict) -> list[dict]:
    """Convert DataFrame to list of standardized JSON records."""
    target_col = config["target_col"]
    feature_cols = [c for c in df.columns if c != target_col]
    task = config["task"]
    source = config["source"]

    # Determine n_classes
    if task == "binary_classification":
        n_classes = 2
    elif task == "multiclass_classification":
        n_classes = int(df[target_col].nunique())
    else:
        n_classes = None

    records = []
    for i in range(len(df)):
        row = df.iloc[i]
        record = {
            "input": {col: float(row[col]) for col in feature_cols},
            "output": int(row[target_col]) if task != "regression" else float(row[target_col]),
            "metadata_fold": int(folds[i]),
            "metadata_dataset": name,
            "metadata_task": task,
            "metadata_source": source,
        }
        if n_classes is not None:
            record["metadata_n_classes"] = n_classes
        records.append(record)

    return records


def create_mini(records: list[dict], config: dict, n: int = 2000) -> list[dict]:
    """Create stratified mini subset of <=2000 rows."""
    if len(records) <= n:
        return records

    rng = np.random.RandomState(RANDOM_STATE)

    if config["task"] != "regression":
        # Stratified sample by target class
        by_class = {}
        for r in records:
            c = r["output"]
            by_class.setdefault(c, []).append(r)

        mini = []
        for cls, cls_records in by_class.items():
            cls_n = max(1, int(n * len(cls_records) / len(records)))
            indices = rng.choice(len(cls_records), min(cls_n, len(cls_records)), replace=False)
            mini.extend([cls_records[i] for i in indices])
    else:
        indices = rng.choice(len(records), n, replace=False)
        mini = [records[i] for i in indices]

    return mini[:n]


def write_jsonl(records: list[dict], path: Path) -> None:
    """Write records as JSONL (one JSON object per line)."""
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r, separators=(",", ":")) + "\n")
    size_mb = path.stat().st_size / 1e6
    logger.info(f"  Wrote {path.name}: {len(records)} rows, {size_mb:.1f} MB")


def process_single_dataset(name: str) -> dict:
    """Process a single dataset end-to-end. Returns metadata dict."""
    config = DATASET_CONFIGS[name]
    logger.info(f"{'='*60}")
    logger.info(f"Processing: {name}")
    logger.info(f"{'='*60}")

    # Step 1: Load
    df = load_raw_dataset(name, config)

    # Step 2: Validate & clean
    df = validate_raw(name, df, config)

    # Step 2.5: Auto-detect task type if needed
    if config["task"] == "auto_detect":
        target_col = config["target_col"]
        n_unique = df[target_col].nunique()
        if n_unique == 2:
            config["task"] = "binary_classification"
            logger.info(f"  Auto-detected task: binary_classification ({n_unique} classes)")
        elif n_unique <= 100:
            config["task"] = "multiclass_classification"
            logger.info(f"  Auto-detected task: multiclass_classification ({n_unique} classes)")
        else:
            config["task"] = "regression"
            logger.info(f"  Auto-detected task: regression ({n_unique} unique values)")

    # Step 3: Encode target
    df = encode_target(df, config)

    # Step 4: Assign folds
    folds = assign_folds(df, config)

    # Step 5: Convert to records
    target_col = config["target_col"]
    feature_cols = [c for c in df.columns if c != target_col]
    logger.info(f"  Converting {len(df)} rows to JSON records...")
    records = df_to_records(df, folds, name, config)
    logger.info(f"  Created {len(records)} records")

    # Step 6: Create mini and preview
    mini_records = create_mini(records, config, n=2000)
    preview_records = mini_records[:100]
    logger.info(f"  Mini: {len(mini_records)} rows, Preview: {len(preview_records)} rows")

    # Step 7: Write files
    full_path = OUT_DIR / f"data_out_{name}.json"
    mini_path = OUT_DIR / f"data_out_{name}_mini.json"
    preview_path = OUT_DIR / f"data_out_{name}_preview.json"

    write_jsonl(records, full_path)
    write_jsonl(mini_records, mini_path)
    write_jsonl(preview_records, preview_path)

    # Build metadata
    n_classes = None
    if config["task"] == "binary_classification":
        n_classes = 2
    elif config["task"] == "multiclass_classification":
        n_classes = int(df[target_col].nunique())

    metadata = {
        "name": name,
        "task": config["task"],
        "n_samples": len(records),
        "n_features": len(feature_cols),
        "n_classes": n_classes,
        "source": config["source"],
        "feature_names": feature_cols,
        "target_name": target_col,
        "files": {
            "full": f"data_out_{name}.json",
            "mini": f"data_out_{name}_mini.json",
            "preview": f"data_out_{name}_preview.json",
        },
    }

    # Free memory
    del df, records, mini_records, preview_records, folds
    gc.collect()

    logger.info(f"  Done: {name} ({metadata['n_samples']} samples, {metadata['n_features']} features)")
    return metadata


def validate_output_file(path: Path, expected_keys: set) -> tuple[bool, str]:
    """Validate a JSONL output file."""
    errors = []
    try:
        rows = []
        with open(path) as f:
            for i, line in enumerate(f):
                try:
                    row = json.loads(line.strip())
                    rows.append(row)
                except json.JSONDecodeError:
                    errors.append(f"Line {i}: invalid JSON")
                    if len(errors) > 5:
                        break

        if not rows:
            return False, "Empty file"

        # Check keys
        for i, row in enumerate(rows[:10]):
            missing = expected_keys - set(row.keys())
            if missing:
                errors.append(f"Row {i}: missing keys {missing}")

        # Check fold distribution
        folds = [r.get("metadata_fold") for r in rows]
        fold_set = set(folds)
        if not fold_set.issubset({0, 1, 2, 3, 4}):
            errors.append(f"Invalid fold values: {fold_set - {0,1,2,3,4}}")

        # Check input dict
        first_input = rows[0].get("input", {})
        n_features = len(first_input)
        for i, row in enumerate(rows[:10]):
            if len(row.get("input", {})) != n_features:
                errors.append(f"Row {i}: expected {n_features} features, got {len(row.get('input', {}))}")

        if errors:
            return False, "; ".join(errors[:5])
        return True, f"OK ({len(rows)} rows, {n_features} features)"

    except Exception as e:
        return False, str(e)


@logger.catch
def main():
    logger.info("Starting dataset processing pipeline")
    logger.info(f"Workspace: {WORKSPACE}")
    logger.info(f"Raw data dir: {RAW_DIR}")
    logger.info(f"Output dir: {OUT_DIR}")

    # Process datasets sequentially to manage memory (higgs is large)
    dataset_names = ["electricity", "adult", "california_housing", "jannis", "higgs_small"]
    all_metadata = []

    for name in dataset_names:
        try:
            meta = process_single_dataset(name)
            all_metadata.append(meta)
        except Exception:
            logger.exception(f"Failed to process {name}")
            raise

    # Write manifest
    manifest = {"datasets": all_metadata}
    manifest_path = OUT_DIR / "data_out.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    logger.info(f"Wrote manifest: {manifest_path}")

    # ── Validation ──────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Validating all output files...")
    logger.info("=" * 60)

    expected_keys = {"input", "output", "metadata_fold", "metadata_dataset", "metadata_task", "metadata_source"}
    all_ok = True

    for meta in all_metadata:
        for version, fname in meta["files"].items():
            fpath = OUT_DIR / fname
            ok, msg = validate_output_file(fpath, expected_keys)
            status = "PASS" if ok else "FAIL"
            logger.info(f"  [{status}] {fname}: {msg}")
            if not ok:
                all_ok = False

    if all_ok:
        logger.info("All validations PASSED")
    else:
        logger.error("Some validations FAILED")

    # ── Summary ─────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Summary:")
    for meta in all_metadata:
        logger.info(f"  {meta['name']}: {meta['n_samples']} samples, {meta['n_features']} features, task={meta['task']}")

    # Check file sizes
    logger.info("File sizes:")
    for meta in all_metadata:
        for version, fname in meta["files"].items():
            fpath = OUT_DIR / fname
            size_mb = fpath.stat().st_size / 1e6
            logger.info(f"  {fname}: {size_mb:.1f} MB")

    logger.info("Pipeline complete!")


if __name__ == "__main__":
    main()

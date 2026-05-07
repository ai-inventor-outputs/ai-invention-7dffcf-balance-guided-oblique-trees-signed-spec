#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["loguru"]
# ///
"""Convert per-dataset JSONL files to grouped exp_sel_data_out.json schema.

Reads the manifest (data_out.json) and per-dataset JSONL files produced by
process_datasets.py, then outputs full_data_out.json in the grouped schema:
{
  "datasets": [
    {"dataset": "electricity", "examples": [{"input": "...", "output": "...", ...}, ...]},
    ...
  ]
}

Schema requirements (exp_sel_data_out.json):
  - input: STRING (JSON-serialized feature dict)
  - output: STRING (target value as string)
  - metadata_* fields only (no split, dataset, context at example level)
"""

import json
import sys
import gc
from pathlib import Path

from loguru import logger

# ── Logging ──────────────────────────────────────────────────────────────────
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add("logs/data.log", rotation="30 MB", level="DEBUG")

# ── Paths ────────────────────────────────────────────────────────────────────
WORKSPACE = Path("/ai-inventor/aii_pipeline/runs/jamnik-sgfigs-pid-v2/"
                 "3_invention_loop/iter_1/gen_art/data_id4_it1__opus")
MANIFEST = WORKSPACE / "data_out.json"
OUTPUT_DIR = WORKSPACE / "full_data_out"
MAX_FILE_SIZE_BYTES = 100 * 1024 * 1024  # 100 MB limit


def load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file (one JSON object per line)."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def task_type_label(task: str) -> str:
    """Map detailed task names to simple classification/regression."""
    if "classification" in task:
        return "classification"
    return "regression"


@logger.catch
def main():
    logger.info("Loading manifest from data_out.json")
    manifest = json.loads(MANIFEST.read_text())
    dataset_metas = manifest["datasets"]
    logger.info(f"Found {len(dataset_metas)} datasets in manifest")

    all_datasets = []

    for ds_meta in dataset_metas:
        name = ds_meta["name"]
        full_file = ds_meta["files"]["full"]
        full_path = WORKSPACE / full_file
        feature_names = ds_meta["feature_names"]
        task = ds_meta["task"]
        n_classes = ds_meta.get("n_classes")
        source = ds_meta["source"]
        n_features = ds_meta["n_features"]

        logger.info(f"Processing {name}: {full_path.name} ({ds_meta['n_samples']} samples)")

        # Load JSONL records
        records = load_jsonl(full_path)
        logger.info(f"  Loaded {len(records)} records")

        # Convert each record to schema format
        examples = []
        for row_idx, rec in enumerate(records):
            # input: JSON-serialize the feature dict to a string
            input_str = json.dumps(rec["input"], separators=(",", ":"))

            # output: convert to string
            output_val = rec["output"]
            output_str = str(output_val)

            example = {
                "input": input_str,
                "output": output_str,
                "metadata_fold": rec["metadata_fold"],
                "metadata_feature_names": feature_names,
                "metadata_task_type": task_type_label(task),
                "metadata_source": source,
                "metadata_row_index": row_idx,
            }

            # Add n_classes only for classification
            if n_classes is not None:
                example["metadata_n_classes"] = n_classes

            examples.append(example)

        logger.info(f"  Converted {len(examples)} examples for {name}")

        all_datasets.append({
            "dataset": name,
            "examples": examples,
        })

        # Free memory from records
        del records
        gc.collect()

    # Build split parts that each stay under 100MB
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Group datasets into parts under the size limit
    # Strategy: write one dataset per part; if a single dataset exceeds
    # the limit, split its examples across multiple parts.
    parts: list[dict] = []
    current_datasets: list[dict] = []
    current_size_est = 50  # overhead for {"datasets":[...]}

    def estimate_size(ds_dict: dict) -> int:
        """Rough byte estimate of a dataset dict serialized as compact JSON."""
        return len(json.dumps(ds_dict, separators=(",", ":")))

    for ds in all_datasets:
        ds_size = estimate_size(ds)
        if current_size_est + ds_size < MAX_FILE_SIZE_BYTES:
            current_datasets.append(ds)
            current_size_est += ds_size
        else:
            # Flush current batch if non-empty
            if current_datasets:
                parts.append({"datasets": current_datasets})
                current_datasets = []
                current_size_est = 50

            # Check if single dataset exceeds limit
            if ds_size < MAX_FILE_SIZE_BYTES:
                current_datasets.append(ds)
                current_size_est += ds_size
            else:
                # Split large dataset's examples into chunks
                examples = ds["examples"]
                chunk_size = len(examples) // ((ds_size // MAX_FILE_SIZE_BYTES) + 1)
                for start in range(0, len(examples), chunk_size):
                    chunk_ds = {"dataset": ds["dataset"],
                                "examples": examples[start:start + chunk_size]}
                    parts.append({"datasets": [chunk_ds]})
                    logger.info(f"  Split {ds['dataset']} chunk: "
                                f"examples {start}-{min(start+chunk_size, len(examples))}")

    if current_datasets:
        parts.append({"datasets": current_datasets})

    # Write parts
    total_examples = sum(len(ds["examples"]) for ds in all_datasets)
    logger.info(f"Writing {len(parts)} parts to {OUTPUT_DIR.name}/")
    for i, part in enumerate(parts, 1):
        path = OUTPUT_DIR / f"full_data_out_{i}.json"
        with open(path, "w") as f:
            json.dump(part, f, separators=(",", ":"))
        size_mb = path.stat().st_size / 1e6
        n_ex = sum(len(d["examples"]) for d in part["datasets"])
        ds_names = [d["dataset"] for d in part["datasets"]]
        logger.info(f"  Part {i}: {size_mb:.1f} MB, {n_ex} examples, datasets={ds_names}")

    logger.info(f"Total: {len(parts)} parts, {len(all_datasets)} datasets, "
                f"{total_examples} examples")

    # Summary
    logger.info("=" * 60)
    logger.info("Summary:")
    for ds in all_datasets:
        n = len(ds["examples"])
        ex = ds["examples"][0]
        logger.info(f"  {ds['dataset']}: {n} examples, "
                    f"task={ex['metadata_task_type']}, "
                    f"features={len(json.loads(ex['input']))}")
    logger.info("Done!")


if __name__ == "__main__":
    main()

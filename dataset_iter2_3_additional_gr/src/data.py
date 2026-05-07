#!/usr/bin/env python3
"""Convert processed JSONL dataset files to exp_sel_data_out.json schema.

Reads the 3 JSONL dataset files (eye_movements, credit, miniboone) produced by
process_datasets.py and converts them to the standardized exp_sel_data_out.json
format where:
- Each row becomes one example with input (JSON string) and output (string label)
- Examples are grouped by dataset
- Metadata stored in metadata_* fields per example
"""

import json
import gc
import resource
import sys
from pathlib import Path

from loguru import logger

# ── Logging ──────────────────────────────────────────────────────────────────
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add("logs/data.log", rotation="30 MB", level="DEBUG")

# ── Hardware detection ───────────────────────────────────────────────────────
def _container_ram_gb() -> float | None:
    for p in ["/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"]:
        try:
            v = Path(p).read_text().strip()
            if v != "max" and int(v) < 1_000_000_000_000:
                return int(v) / 1e9
        except (FileNotFoundError, ValueError):
            pass
    return None

TOTAL_RAM_GB = _container_ram_gb() or 57.0
RAM_BUDGET = int(TOTAL_RAM_GB * 0.5 * 1e9)
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET * 3, RAM_BUDGET * 3))
logger.info(f"RAM: {TOTAL_RAM_GB:.1f} GB total, budget={RAM_BUDGET/1e9:.1f} GB")

# ── Paths ────────────────────────────────────────────────────────────────────
WORKSPACE = Path("/ai-inventor/aii_pipeline/runs/jamnik-sgfigs-pid-v2/3_invention_loop/iter_2/gen_art/data_id5_it2__opus")

# ── Dataset configs (hardcoded from process_datasets.py output) ──────────────
DATASETS = [
    {
        "name": "eye_movements",
        "jsonl_file": "data_out_eye_movements.json",
        "task": "binary_classification",
        "n_classes": 2,
        "source": "hf_inria-soda/tabular-benchmark_clf_num_eye_movements",
        "feature_names": [
            "lineNo", "assgNo", "prevFixDur", "firstfixDur", "firstPassFixDur",
            "nextFixDur", "firstSaccLen", "lastSaccLen", "prevFixPos", "landingPos",
            "leavingPos", "totalFixDur", "meanFixDur", "regressLen", "regressDur",
            "pupilDiamMax", "pupilDiamLag", "timePrtctg", "titleNo", "wordNo",
        ],
    },
    {
        "name": "credit",
        "jsonl_file": "data_out_credit.json",
        "task": "binary_classification",
        "n_classes": 2,
        "source": "hf_inria-soda/tabular-benchmark_clf_num_credit",
        "feature_names": [
            "RevolvingUtilizationOfUnsecuredLines", "age",
            "NumberOfTime30-59DaysPastDueNotWorse", "DebtRatio", "MonthlyIncome",
            "NumberOfOpenCreditLinesAndLoans", "NumberOfTimes90DaysLate",
            "NumberRealEstateLoansOrLines", "NumberOfTime60-89DaysPastDueNotWorse",
            "NumberOfDependents",
        ],
    },
    {
        "name": "miniboone",
        "jsonl_file": ["data_out_miniboone/data_out_miniboone_1.json", "data_out_miniboone/data_out_miniboone_2.json"],
        "task": "binary_classification",
        "n_classes": 2,
        "source": "hf_inria-soda/tabular-benchmark_clf_num_MiniBooNE",
        "feature_names": [f"ParticleID_{i}" for i in range(50)],
    },
]


def load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file (one JSON object per line)."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def convert_dataset(ds_cfg: dict) -> dict:
    """Convert one dataset's JSONL rows to exp_sel_data_out schema format."""
    name = ds_cfg["name"]
    jsonl_files = ds_cfg["jsonl_file"]
    if isinstance(jsonl_files, str):
        jsonl_files = [jsonl_files]
    feature_names = ds_cfg["feature_names"]
    task = ds_cfg["task"]
    n_classes = ds_cfg["n_classes"]
    source = ds_cfg["source"]

    rows = []
    for jf in jsonl_files:
        full_file = WORKSPACE / jf
        logger.info(f"Loading {name} from {full_file.name}...")
        rows.extend(load_jsonl(full_file))
    logger.info(f"  Loaded {len(rows)} rows total")

    examples = []
    for idx, row in enumerate(rows):
        example = {
            "input": json.dumps(row["input"], separators=(",", ":")),
            "output": str(row["output"]),
            "metadata_fold": row["metadata_fold"],
            "metadata_feature_names": feature_names,
            "metadata_task_type": task,
            "metadata_n_classes": n_classes,
            "metadata_source": source,
            "metadata_row_index": idx,
        }
        examples.append(example)

    logger.info(f"  Converted {len(examples)} examples for {name}")
    del rows
    gc.collect()

    return {"dataset": name, "examples": examples}


@logger.catch
def main():
    logger.info("Converting datasets to exp_sel_data_out.json schema")

    datasets = []
    for ds_cfg in DATASETS:
        ds = convert_dataset(ds_cfg)
        datasets.append(ds)
        logger.info(f"  {ds['dataset']}: {len(ds['examples'])} examples")

    output = {
        "metadata": {
            "description": "3 additional Grinsztajn clf_num benchmark datasets (eye_movements, credit, miniboone)",
            "source": "inria-soda/tabular-benchmark (HuggingFace)",
            "benchmark": "Grinsztajn et al. 2022 - Why do tree-based models still outperform deep learning on tabular data?",
            "n_datasets": len(datasets),
            "total_examples": sum(len(d["examples"]) for d in datasets),
        },
        "datasets": datasets,
    }

    # Write full output — split into parts under 100MB
    out_dir = WORKSPACE / "full_data_out"
    out_dir.mkdir(exist_ok=True)

    # Part 1: eye_movements + credit (~21MB)
    part1 = {"metadata": output["metadata"], "datasets": [datasets[0], datasets[1]]}
    # Parts 2-3: miniboone split in half
    mb_examples = datasets[2]["examples"]
    half = len(mb_examples) // 2
    part2 = {"metadata": output["metadata"], "datasets": [{"dataset": "miniboone", "examples": mb_examples[:half]}]}
    part3 = {"metadata": output["metadata"], "datasets": [{"dataset": "miniboone", "examples": mb_examples[half:]}]}

    for i, part in enumerate([part1, part2, part3], 1):
        p = out_dir / f"full_data_out_{i}.json"
        with open(p, "w") as f:
            json.dump(part, f, separators=(",", ":"))
        n_ex = sum(len(d["examples"]) for d in part["datasets"])
        logger.info(f"  Wrote {p.name}: {n_ex} examples, {p.stat().st_size/1e6:.1f} MB")

    # Summary
    logger.info("=" * 60)
    logger.info("Summary:")
    for ds in datasets:
        ex = ds["examples"][0]
        logger.info(f"  {ds['dataset']}: {len(ds['examples'])} examples, "
                     f"n_classes={ex['metadata_n_classes']}, task={ex['metadata_task_type']}")
    logger.info(f"Total: {output['metadata']['total_examples']} examples across {output['metadata']['n_datasets']} datasets")
    logger.info("Done!")


if __name__ == "__main__":
    main()

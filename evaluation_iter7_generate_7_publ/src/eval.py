#!/usr/bin/env python3
"""Generate 7 publication-quality figures from experiment outputs.

Produces:
  fig1_pipeline.png       - Method pipeline diagram
  fig2_cd_diagram.png     - Critical difference diagram (Friedman + Nemenyi)
  fig3_accuracy_bars.png  - Main accuracy comparison (grouped bar chart)
  fig4_pareto_scatter.png - Arity-accuracy Pareto scatter
  fig5_forest_plot.png    - Signed-vs-unsigned forest plot
  fig6_module_recovery.png- Synthetic module recovery bars
  fig7_scaling.png        - Scaling analysis (log-log scatter + power law)
  eval_out.json           - Figure metadata and evaluation output
"""

import json
import sys
import os
import math
import resource
import gc
from pathlib import Path
from collections import defaultdict

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from scipy import stats
import scikit_posthocs as sp
from loguru import logger

# =============================================================================
# LOGGING
# =============================================================================
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
logger.add(str(LOG_DIR / "run.log"), rotation="30 MB", level="DEBUG")

# =============================================================================
# HARDWARE DETECTION & MEMORY LIMITS
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
    return os.cpu_count() or 1


def _container_ram_gb():
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
TOTAL_RAM_GB = _container_ram_gb() or 57.0
RAM_BUDGET = int(TOTAL_RAM_GB * 0.5 * 1e9)
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET * 3, RAM_BUDGET * 3))
logger.info(f"Hardware: {NUM_CPUS} CPUs, {TOTAL_RAM_GB:.1f}GB RAM, budget={RAM_BUDGET/1e9:.1f}GB")

# =============================================================================
# PATHS
# =============================================================================
WORKSPACE = Path(__file__).parent
BASE = Path("/ai-inventor/aii_pipeline/runs/jamnik-sgfigs-pid-v2/3_invention_loop")

EXP_PATHS = {
    'exp1_real':  (BASE / "iter_5/gen_art/exp_id1_it5__opus/full_method_out.json",
                   BASE / "iter_5/gen_art/exp_id1_it5__opus/mini_method_out.json"),
    'exp2_base':  (BASE / "iter_4/gen_art/exp_id2_it4__opus/full_method_out.json",
                   BASE / "iter_4/gen_art/exp_id2_it4__opus/mini_method_out.json"),
    'exp3_synth': (BASE / "iter_3/gen_art/exp_id3_it3__opus/full_method_out.json",
                   BASE / "iter_3/gen_art/exp_id3_it3__opus/mini_method_out.json"),
    'exp1_recov': (BASE / "iter_2/gen_art/exp_id1_it2__opus/full_method_out.json",
                   BASE / "iter_2/gen_art/exp_id1_it2__opus/mini_method_out.json"),
    'exp2_frust': (BASE / "iter_5/gen_art/exp_id2_it5__opus/full_method_out.json",
                   BASE / "iter_5/gen_art/exp_id2_it5__opus/mini_method_out.json"),
}

# =============================================================================
# CONSTANTS
# =============================================================================
CLASSIFICATION_DATASETS = [
    'adult', 'electricity', 'credit', 'eye_movements',
    'higgs_small', 'miniboone', 'jannis',
]
DATASET_NFEATURES = {
    'adult': 6, 'electricity': 7, 'credit': 10,
    'eye_movements': 20, 'higgs_small': 24,
    'miniboone': 50, 'jannis': 54,
}

FIGS_METHODS = [
    'axis_aligned', 'random_oblique', 'hard_threshold',
    'unsigned_spectral', 'signed_spectral',
]
BASELINE_METHODS = ['ebm', 'random_forest', 'linear']
ALL_METHODS = FIGS_METHODS + BASELINE_METHODS

METHOD_COLORS = {
    'axis_aligned':      '#BBBBBB',
    'random_oblique':    '#4477AA',
    'hard_threshold':    '#66CCEE',
    'unsigned_spectral': '#228833',
    'signed_spectral':   '#EE6677',
    'ebm':               '#AA3377',
    'random_forest':     '#CCBB44',
    'linear':            '#DDDDDD',
}

METHOD_LABELS = {
    'axis_aligned':      'FIGS (Axis)',
    'random_oblique':    'FIGS (Rand-Obl.)',
    'hard_threshold':    'FIGS (Hard-Thr.)',
    'unsigned_spectral': 'FIGS (Uns-Spec.)',
    'signed_spectral':   'FIGS (Sgn-Spec.)',
    'ebm':               'EBM',
    'random_forest':     'Random Forest',
    'linear':            'Logistic/Ridge',
}

METHOD_MARKERS = {
    'axis_aligned': 'o', 'random_oblique': 's', 'hard_threshold': 'D',
    'unsigned_spectral': '^', 'signed_spectral': 'p',
}

# Nemenyi q-values at alpha=0.05, from Demsar (2006)
NEMENYI_Q = {
    2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728,
    6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164,
}

SYNTH_VARIANTS_RECOVERY = [
    'easy_2mod_xor', 'medium_4mod_mixed', 'hard_4mod_unequal',
    'overlapping_modules', 'highdim_8mod',
]
SYNTH_VARIANT_LABELS = {
    'easy_2mod_xor':      'Easy (2-mod)',
    'medium_4mod_mixed':  'Medium (4-mod)',
    'hard_4mod_unequal':  'Hard (unequal)',
    'overlapping_modules':'Overlapping',
    'highdim_8mod':       'High-dim (8-mod)',
    'no_structure_control':'No Structure',
}
ALL_SYNTH_VARIANTS = [
    'easy_2mod_xor', 'medium_4mod_mixed', 'hard_4mod_unequal',
    'overlapping_modules', 'no_structure_control', 'highdim_8mod',
]

# =============================================================================
# ACADEMIC STYLE
# =============================================================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif', 'Times New Roman', 'Times'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def sanitize_number(v) -> float:
    """Ensure value is a valid JSON-safe float."""
    if v is None:
        return 0.0
    v = float(v)
    if np.isnan(v) or np.isinf(v):
        return 0.0
    return v


def load_json_safe(primary: Path, fallback: Path, name: str) -> dict:
    """Load JSON with fallback to mini version."""
    for path in [primary, fallback]:
        try:
            logger.info(f"Loading {name} from {path.name}")
            data = json.loads(path.read_text())
            logger.info(f"Loaded {name} successfully ({path.name})")
            return data
        except FileNotFoundError:
            logger.warning(f"{path.name} not found, trying fallback")
        except json.JSONDecodeError:
            logger.warning(f"{path.name} invalid JSON, trying fallback")
        except MemoryError:
            logger.warning(f"{path.name} too large, trying fallback")
    raise RuntimeError(f"Could not load {name}")


def get_best_figs_data(exp1_meta: dict) -> dict:
    """Select best max_splits per (dataset, method) from exp1.

    Returns: {(dataset, method): {
        'ba_mean', 'ba_std', 'per_fold_ba', 'arity_mean', 'arity_std',
        'fit_time_mean', 'max_splits'
    }}
    """
    results_summary = exp1_meta['results_summary']
    results_per_fold = exp1_meta['results_per_fold']

    # Best max_splits per (dataset, method) by balanced_accuracy_mean
    best_summary = {}
    for entry in results_summary:
        ds = entry['dataset']
        m = entry['method']
        if ds not in CLASSIFICATION_DATASETS:
            continue
        key = (ds, m)
        if key not in best_summary or entry['balanced_accuracy_mean'] > best_summary[key]['balanced_accuracy_mean']:
            best_summary[key] = entry

    # Collect per-fold results at best max_splits
    result = {}
    for key, summary in best_summary.items():
        ds, m = key
        ms = summary['max_splits']
        per_fold = [r['balanced_accuracy'] for r in results_per_fold
                    if r['dataset'] == ds and r['method'] == m and r['max_splits'] == ms]
        result[key] = {
            'ba_mean': summary['balanced_accuracy_mean'],
            'ba_std': summary['balanced_accuracy_std'],
            'per_fold_ba': per_fold,
            'arity_mean': summary.get('avg_split_arity_mean', 1.0),
            'arity_std': summary.get('avg_split_arity_std', 0.0),
            'fit_time_mean': summary.get('fit_time_s_mean', 0.0),
            'max_splits': ms,
        }
    return result


def get_baseline_data(exp2_meta: dict) -> dict:
    """Extract baseline data from exp2.

    Returns: {(dataset, method): {'ba_mean', 'ba_std', 'per_fold_ba'}}
    """
    pdr = exp2_meta['per_dataset_results']
    result = {}
    for ds in CLASSIFICATION_DATASETS:
        if ds not in pdr:
            logger.warning(f"Dataset {ds} not in baselines")
            continue
        for method in BASELINE_METHODS:
            if method not in pdr[ds]:
                logger.warning(f"Method {method} not in baselines for {ds}")
                continue
            entry = pdr[ds][method]
            fold_ba = [f['balanced_accuracy'] for f in entry['fold_results']
                       if f.get('balanced_accuracy') is not None]
            agg = entry['aggregate']
            result[(ds, method)] = {
                'ba_mean': agg['balanced_accuracy_mean'],
                'ba_std': agg['balanced_accuracy_std'],
                'per_fold_ba': fold_ba,
            }
    return result


def hedges_g(group1: list, group2: list) -> tuple:
    """Compute Hedges' g = (mean1 - mean2)/pooled_sd * J and bootstrap 95% CI.

    Positive g means group1 > group2.
    """
    a1 = np.array(group1, dtype=float)
    a2 = np.array(group2, dtype=float)
    n1, n2 = len(a1), len(a2)

    if n1 < 2 or n2 < 2:
        return 0.0, (-1.0, 1.0)

    m1, m2 = a1.mean(), a2.mean()
    s1, s2 = a1.std(ddof=1), a2.std(ddof=1)
    pooled_sd = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))

    if pooled_sd < 1e-12:
        return 0.0, (-0.5, 0.5)

    J = 1 - 3 / (4 * (n1 + n2 - 2) - 1)
    g = (m1 - m2) / pooled_sd * J

    # Bootstrap CI
    rng = np.random.default_rng(42)
    boot_gs = np.empty(10000)
    for b in range(10000):
        idx1 = rng.integers(0, n1, size=n1)
        idx2 = rng.integers(0, n2, size=n2)
        b1, b2 = a1[idx1], a2[idx2]
        bs1 = b1.std(ddof=1)
        bs2 = b2.std(ddof=1)
        bp = np.sqrt(((n1 - 1) * bs1**2 + (n2 - 1) * bs2**2) / (n1 + n2 - 2))
        if bp > 1e-12:
            boot_gs[b] = (b1.mean() - b2.mean()) / bp * J
        else:
            boot_gs[b] = 0.0

    ci_low = float(np.percentile(boot_gs, 2.5))
    ci_high = float(np.percentile(boot_gs, 97.5))
    return float(g), (ci_low, ci_high)


def find_cd_cliques(sorted_ranks: list, cd: float) -> list:
    """Find maximal non-significant groups for CD diagram.

    Returns list of (start_idx, end_idx) pairs where all methods
    within each group have rank difference <= cd.
    """
    n = len(sorted_ranks)
    cliques = []
    for i in range(n):
        for j in range(n - 1, i, -1):
            if sorted_ranks[j] - sorted_ranks[i] <= cd:
                cliques.append((i, j))
                break

    # Remove subset cliques
    final = []
    for c in cliques:
        is_subset = any(
            o[0] <= c[0] and o[1] >= c[1] and o != c for o in cliques
        )
        if not is_subset and c[0] != c[1]:
            final.append(c)
    return final


# =============================================================================
# FIGURE 1: METHOD PIPELINE DIAGRAM
# =============================================================================

def fig1_pipeline(save_path: Path) -> dict:
    """Draw a left-to-right pipeline schematic."""
    logger.info("Generating Figure 1: Pipeline diagram")

    fig, ax = plt.subplots(1, 1, figsize=(10, 3))
    ax.set_xlim(-0.2, 11.0)
    ax.set_ylim(-1.2, 2.2)
    ax.axis('off')

    # Stage definitions: (x, width, title, subtitle, facecolor)
    stages = [
        (0.0, 1.8, 'Raw Features', '$X_1 \\ldots X_d$', '#F0F0F0'),
        (2.5, 1.8, 'Pairwise\nCo-Information', 'CoI matrix', '#E8D5E0'),
        (5.0, 1.8, 'Signed\nGraph', '+/\u2212 edges', '#D5E0E8'),
        (7.5, 1.8, 'Spectral\nClustering', 'SPONGE', '#D5E8D5'),
        (10.0, 1.4, 'FIGS\nEnsemble', 'oblique splits', '#F5E6D0'),
    ]

    for x, w, title, sub, color in stages:
        h = 1.2
        y = 0.2
        rect = FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.12",
            facecolor=color, edgecolor='#333333', linewidth=1.3,
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2 + 0.1, title,
                ha='center', va='center', fontsize=9, fontweight='bold')
        ax.text(x + w / 2, y + 0.15, sub,
                ha='center', va='center', fontsize=7, fontstyle='italic',
                color='#555555')

    # Arrows
    arrow_kw = dict(arrowstyle="Simple,tail_width=3,head_width=10,head_length=6",
                    color='#555555')
    for x_end_prev, x_start_next in [(1.8, 2.5), (4.3, 5.0), (6.8, 7.5), (9.3, 10.0)]:
        arrow = FancyArrowPatch(
            (x_end_prev + 0.05, 0.8), (x_start_next - 0.05, 0.8), **arrow_kw,
        )
        ax.add_patch(arrow)

    # Annotation below
    annotations = [
        (1.35, 'Input data'),
        (3.4, 'Positive \u2192 redundancy\nNegative \u2192 synergy'),
        (5.9, 'Weighted\nadjacency'),
        (8.4, 'Feature\nmodules'),
        (10.7, 'Final\nmodel'),
    ]
    for x, txt in annotations:
        ax.text(x, -0.35, txt, ha='center', va='top', fontsize=7, color='#666666')

    ax.set_title('Signed Spectral FIGS Pipeline',
                 fontsize=13, fontweight='bold', pad=15)

    fig.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    logger.info(f"Saved {save_path.name}")

    return {
        "filename": "fig1_pipeline.png",
        "description": "Method pipeline: Raw Features -> CoI matrix -> Signed Graph -> Spectral Clustering -> FIGS Ensemble",
        "data_sources": [],
        "width_inches": 10,
        "height_inches": 3,
    }


# =============================================================================
# FIGURE 2: CRITICAL DIFFERENCE DIAGRAM
# =============================================================================

def fig2_cd_diagram(figs_data: dict, base_data: dict, save_path: Path) -> dict:
    """Draw a Demsar-style critical difference diagram."""
    logger.info("Generating Figure 2: Critical difference diagram")

    n_ds = len(CLASSIFICATION_DATASETS)
    n_m = len(ALL_METHODS)

    # Build accuracy matrix: rows=datasets, cols=methods
    matrix = np.full((n_ds, n_m), 0.5)
    for i, ds in enumerate(CLASSIFICATION_DATASETS):
        for j, method in enumerate(ALL_METHODS):
            key = (ds, method)
            if key in figs_data:
                matrix[i, j] = figs_data[key]['ba_mean']
            elif key in base_data:
                matrix[i, j] = base_data[key]['ba_mean']
            else:
                logger.warning(f"Missing {ds}/{method}, defaulting to 0.5")

    # Friedman test
    try:
        fstat, fpval = stats.friedmanchisquare(
            *[matrix[:, j] for j in range(n_m)]
        )
    except Exception:
        logger.exception("Friedman test failed")
        fstat, fpval = 0.0, 1.0
    logger.info(f"Friedman chi2={fstat:.3f}, p={fpval:.6f}")

    # Ranks: 1 = best (highest accuracy)
    ranks = np.zeros_like(matrix)
    for i in range(n_ds):
        ranks[i, :] = stats.rankdata(-matrix[i, :])
    avg_ranks = ranks.mean(axis=0)

    # Sort by rank
    order = np.argsort(avg_ranks)
    s_methods = [ALL_METHODS[i] for i in order]
    s_ranks = avg_ranks[order].tolist()

    logger.info("Average ranks (sorted):")
    for m, r in zip(s_methods, s_ranks):
        logger.info(f"  {METHOD_LABELS[m]}: {r:.3f}")

    # Critical difference
    q_alpha = NEMENYI_Q.get(n_m, 3.031)
    cd = q_alpha * np.sqrt(n_m * (n_m + 1) / (6 * n_ds))
    logger.info(f"CD = {cd:.3f}  (k={n_m}, N={n_ds}, q_alpha={q_alpha})")

    # Cliques
    cliques = find_cd_cliques(s_ranks, cd)
    logger.info(f"Non-significant cliques: {len(cliques)}")

    # Nemenyi p-values for reporting
    sig_pairs = []
    try:
        nem_p = sp.posthoc_nemenyi_friedman(matrix)
        for ii in range(n_m):
            for jj in range(ii + 1, n_m):
                if nem_p.iloc[ii, jj] < 0.05:
                    sig_pairs.append((ALL_METHODS[ii], ALL_METHODS[jj],
                                     float(nem_p.iloc[ii, jj])))
    except Exception:
        logger.exception("Nemenyi post-hoc failed")

    # ---- Draw ----
    fig, ax = plt.subplots(figsize=(8, 4))

    y_axis = 0.0
    ax.set_xlim(0.5, n_m + 0.5)
    half = n_m // 2

    # Axis line + ticks
    ax.hlines(y_axis, 1, n_m, color='black', linewidth=1)
    for t in range(1, n_m + 1):
        ax.vlines(t, y_axis - 0.08, y_axis + 0.08, color='black', linewidth=1)
        ax.text(t, y_axis + 0.15, str(t), ha='center', va='bottom', fontsize=8)

    # CD bracket at top
    y_cd = y_axis + 0.55
    ax.hlines(y_cd, 1, 1 + cd, color='#333333', linewidth=2)
    ax.vlines(1, y_cd - 0.06, y_cd + 0.06, color='#333333', linewidth=2)
    ax.vlines(1 + cd, y_cd - 0.06, y_cd + 0.06, color='#333333', linewidth=2)
    ax.text(1 + cd / 2, y_cd + 0.1, f'CD = {cd:.2f}',
            ha='center', va='bottom', fontsize=8, fontweight='bold')

    # Left-side labels (better-ranked methods, going upward)
    for idx in range(half):
        m = s_methods[idx]
        r = s_ranks[idx]
        y_label = y_axis + 0.6 + idx * 0.5
        color = METHOD_COLORS[m]
        weight = 'bold' if m == 'signed_spectral' else 'normal'

        ax.plot(r, y_axis, 'o', color=color, markersize=7,
                markeredgecolor='black', markeredgewidth=0.5, zorder=5)
        ax.plot([r, r], [y_axis + 0.08, y_label - 0.08],
                color='#888888', linewidth=0.6)
        ax.plot([r, 0.8], [y_label, y_label],
                color='#888888', linewidth=0.6)
        ax.text(0.75, y_label, f'{METHOD_LABELS[m]} ({r:.2f})',
                ha='right', va='center', fontsize=8, color=color,
                fontweight=weight)

    # Right-side labels (worse-ranked methods, going upward)
    for idx_offset, idx in enumerate(range(half, n_m)):
        m = s_methods[idx]
        r = s_ranks[idx]
        y_label = y_axis + 0.6 + idx_offset * 0.5
        color = METHOD_COLORS[m]
        weight = 'bold' if m == 'signed_spectral' else 'normal'

        ax.plot(r, y_axis, 'o', color=color, markersize=7,
                markeredgecolor='black', markeredgewidth=0.5, zorder=5)
        ax.plot([r, r], [y_axis + 0.08, y_label - 0.08],
                color='#888888', linewidth=0.6)
        ax.plot([r, n_m + 0.2], [y_label, y_label],
                color='#888888', linewidth=0.6)
        ax.text(n_m + 0.25, y_label, f'{METHOD_LABELS[m]} ({r:.2f})',
                ha='left', va='center', fontsize=8, color=color,
                fontweight=weight)

    # Clique bars below the axis
    for ci, (start, end) in enumerate(cliques):
        y_bar = y_axis - 0.3 - ci * 0.22
        ax.hlines(y_bar, s_ranks[start], s_ranks[end],
                  color='#333333', linewidth=3.0)

    ax.set_ylim(y_axis - 0.3 - len(cliques) * 0.22 - 0.3,
                y_axis + 0.6 + max(half, n_m - half) * 0.5 + 0.8)
    ax.axis('off')

    sig_str = "significant" if fpval < 0.05 else "not significant"
    ax.set_title(
        f'Critical Difference Diagram\n'
        f'(Friedman \u03c7\u00b2={fstat:.1f}, p={fpval:.4f}, {sig_str})',
        fontsize=11, fontweight='bold', pad=10,
    )

    fig.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    logger.info(f"Saved {save_path.name}")

    avg_ranks_dict = {ALL_METHODS[i]: float(avg_ranks[i]) for i in range(n_m)}
    return {
        "filename": "fig2_cd_diagram.png",
        "description": (f"Critical difference diagram (Friedman + Nemenyi) "
                        f"for {n_m} methods on {n_ds} classification datasets"),
        "data_sources": ["exp_id1_it5__opus", "exp_id2_it4__opus"],
        "friedman_chi2": float(fstat),
        "friedman_p": float(fpval),
        "critical_difference": float(cd),
        "average_ranks": avg_ranks_dict,
        "significant_pairs": len(sig_pairs),
    }


# =============================================================================
# FIGURE 3: MAIN ACCURACY COMPARISON (GROUPED BAR CHART)
# =============================================================================

def fig3_accuracy_bars(figs_data: dict, base_data: dict,
                       save_path: Path) -> dict:
    """Draw grouped bar chart of balanced accuracy."""
    logger.info("Generating Figure 3: Accuracy comparison bars")

    fig, ax = plt.subplots(figsize=(12, 5))

    n_ds = len(CLASSIFICATION_DATASETS)
    n_m = len(ALL_METHODS)
    bar_w = 0.1
    x_base = np.arange(n_ds)

    for j, method in enumerate(ALL_METHODS):
        means, stds = [], []
        for ds in CLASSIFICATION_DATASETS:
            key = (ds, method)
            if key in figs_data:
                means.append(figs_data[key]['ba_mean'])
                stds.append(figs_data[key]['ba_std'])
            elif key in base_data:
                means.append(base_data[key]['ba_mean'])
                stds.append(base_data[key]['ba_std'])
            else:
                means.append(0.5)
                stds.append(0.0)

        offset = (j - n_m / 2 + 0.5) * bar_w
        edge_kw = (dict(edgecolor='black', linewidth=1.2)
                   if method == 'signed_spectral'
                   else dict(edgecolor='none'))

        ax.bar(x_base + offset, means, bar_w, yerr=stds,
               color=METHOD_COLORS[method], label=METHOD_LABELS[method],
               capsize=2, error_kw={'linewidth': 0.7}, **edge_kw)

    xlabels = [f"{ds}\n(d={DATASET_NFEATURES[ds]})" for ds in CLASSIFICATION_DATASETS]
    ax.set_xticks(x_base)
    ax.set_xticklabels(xlabels, fontsize=8)
    ax.set_ylabel('Balanced Accuracy')
    ax.set_xlabel('Dataset (ordered by dimensionality)')
    ax.set_title('Method Comparison Across Classification Datasets',
                 fontweight='bold')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=7,
              frameon=True)
    ax.set_ylim(0.4, 1.0)
    ax.axhline(y=0.5, color='grey', linestyle=':', linewidth=0.5, alpha=0.5)

    fig.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    logger.info(f"Saved {save_path.name}")

    method_avg = {}
    for method in ALL_METHODS:
        vals = []
        for ds in CLASSIFICATION_DATASETS:
            key = (ds, method)
            if key in figs_data:
                vals.append(figs_data[key]['ba_mean'])
            elif key in base_data:
                vals.append(base_data[key]['ba_mean'])
        method_avg[method] = float(np.mean(vals)) if vals else 0.0

    return {
        "filename": "fig3_accuracy_bars.png",
        "description": (f"Grouped bar chart of balanced accuracy for "
                        f"{n_m} methods across {n_ds} datasets"),
        "data_sources": ["exp_id1_it5__opus", "exp_id2_it4__opus"],
        "method_avg_accuracy": method_avg,
    }


# =============================================================================
# FIGURE 4: ARITY-ACCURACY PARETO SCATTER
# =============================================================================

def fig4_pareto_scatter(figs_data: dict, save_path: Path) -> dict:
    """Draw arity vs accuracy scatter with Pareto frontier."""
    logger.info("Generating Figure 4: Pareto scatter")

    fig, ax = plt.subplots(figsize=(7, 5))

    all_pts = []
    for method in FIGS_METHODS:
        for ds in CLASSIFICATION_DATASETS:
            key = (ds, method)
            if key in figs_data:
                d = figs_data[key]
                all_pts.append({
                    'arity': d['arity_mean'],
                    'accuracy': d['ba_mean'],
                    'method': method,
                    'dataset': ds,
                    'n_features': DATASET_NFEATURES[ds],
                })

    # Size scaling
    min_nf, max_nf = 6, 54
    min_sz, max_sz = 30, 150

    for method in FIGS_METHODS:
        pts = [p for p in all_pts if p['method'] == method]
        if not pts:
            continue
        x = [p['arity'] for p in pts]
        y = [p['accuracy'] for p in pts]
        sizes = [min_sz + (p['n_features'] - min_nf)
                 / max(1, max_nf - min_nf) * (max_sz - min_sz) for p in pts]
        ax.scatter(x, y, s=sizes, c=METHOD_COLORS[method],
                   marker=METHOD_MARKERS[method],
                   label=METHOD_LABELS[method], alpha=0.8,
                   edgecolors='black', linewidths=0.5, zorder=3)

    # Pareto frontier: minimize arity, maximize accuracy
    frontier = []
    if all_pts:
        sorted_by_arity = sorted(all_pts, key=lambda p: (p['arity'], -p['accuracy']))
        best_acc = -1
        for p in sorted_by_arity:
            if p['accuracy'] > best_acc:
                frontier.append((p['arity'], p['accuracy']))
                best_acc = p['accuracy']

        if len(frontier) > 1:
            fx, fy = zip(*frontier)
            ax.step(list(fx), list(fy), where='post',
                    color='#333333', linestyle='--', linewidth=1.5, alpha=0.5,
                    label='Pareto frontier', zorder=2)

    ax.set_xlabel('Mean Split Arity')
    ax.set_ylabel('Mean Balanced Accuracy')
    ax.set_title('Accuracy\u2013Interpretability Tradeoff', fontweight='bold')

    # Build combined legend with size examples
    handles, labels = ax.get_legend_handles_labels()
    for nf_val, lbl in [(6, 'd=6'), (30, 'd=30'), (54, 'd=54')]:
        sz = min_sz + (nf_val - min_nf) / max(1, max_nf - min_nf) * (max_sz - min_sz)
        h = ax.scatter([], [], s=sz, c='grey', alpha=0.5,
                       edgecolors='black', linewidths=0.5)
        handles.append(h)
        labels.append(lbl)
    ax.legend(handles, labels, fontsize=7, frameon=True,
              loc='lower right', ncol=2)

    fig.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    logger.info(f"Saved {save_path.name}")

    return {
        "filename": "fig4_pareto_scatter.png",
        "description": "Arity vs accuracy scatter with Pareto frontier for 5 FIGS methods",
        "data_sources": ["exp_id1_it5__opus"],
        "n_points": len(all_pts),
        "pareto_frontier_size": len(frontier),
    }


# =============================================================================
# FIGURE 5: SIGNED-VS-UNSIGNED FOREST PLOT
# =============================================================================

def fig5_forest_plot(figs_data: dict, exp3_meta: dict,
                     save_path: Path) -> dict:
    """Draw forest plot of Hedges' g (unsigned - signed spectral)."""
    logger.info("Generating Figure 5: Forest plot")

    rows = []  # (name, g, ci_lo, ci_hi, group)
    hedges_g_results = {}

    # --- Real datasets ---
    for ds in CLASSIFICATION_DATASETS:
        key_u = (ds, 'unsigned_spectral')
        key_s = (ds, 'signed_spectral')
        if key_u in figs_data and key_s in figs_data:
            uf = figs_data[key_u]['per_fold_ba']
            sf = figs_data[key_s]['per_fold_ba']
            if len(uf) >= 2 and len(sf) >= 2:
                g, (ci_lo, ci_hi) = hedges_g(uf, sf)
                rows.append((ds, g, ci_lo, ci_hi, 'Real'))
                hedges_g_results[ds] = float(g)

    # --- Synthetic variants from exp3 ---
    pvr = exp3_meta.get('per_variant_results', {})
    for var in ALL_SYNTH_VARIANTS:
        if var not in pvr:
            logger.warning(f"Variant {var} not in exp3")
            continue
        methods = pvr[var].get('methods', {})
        unsigned = methods.get('unsigned_spectral', {})
        signed = methods.get('signed_spectral', {})

        u_folds = [f['balanced_accuracy'] for f in unsigned.get('best_folds', [])]
        s_folds = [f['balanced_accuracy'] for f in signed.get('best_folds', [])]

        if len(u_folds) >= 2 and len(s_folds) >= 2:
            g, (ci_lo, ci_hi) = hedges_g(u_folds, s_folds)
            label = SYNTH_VARIANT_LABELS.get(var, var)
            rows.append((label, g, ci_lo, ci_hi, 'Synthetic'))
            hedges_g_results[var] = float(g)

    if not rows:
        logger.warning("No data for forest plot")
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.text(0.5, 0.5, 'No data available', transform=ax.transAxes,
                ha='center', va='center', fontsize=14)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        return {"filename": "fig5_forest_plot.png",
                "description": "No data available", "data_sources": []}

    # --- Draw (real at top, synthetic at bottom) ---
    real_rows = [r for r in rows if r[4] == 'Real']
    synth_rows = [r for r in rows if r[4] == 'Synthetic']
    n_rows = len(rows)
    fig_h = max(4, 0.55 * n_rows + 2.5)
    fig, ax = plt.subplots(figsize=(8, fig_h))

    # Assign y from top (high) to bottom (low)
    y_positions = []
    y_labels = []
    total_height = len(real_rows) + len(synth_rows) + 2  # +2 for headers/gap
    y = total_height

    def _draw_row(ax, y_val, name, g_val, ci_lo_val, ci_hi_val):
        color = '#EE6677' if g_val < 0 else '#4477AA'
        ci_w = max(abs(ci_hi_val - ci_lo_val), 0.01)
        sq_sz = max(4, min(12, 20 / ci_w))
        ax.plot([ci_lo_val, ci_hi_val], [y_val, y_val],
                color=color, linewidth=1.5, zorder=2)
        ax.plot(g_val, y_val, 's', color=color, markersize=sq_sz,
                markeredgecolor='black', markeredgewidth=0.5, zorder=3)

    # Real section header
    if real_rows:
        ax.text(0.01, y + 0.2, 'Real Datasets', fontweight='bold',
                fontsize=9, ha='left', va='bottom',
                transform=ax.get_yaxis_transform())

    for name, g, ci_lo, ci_hi, _ in real_rows:
        y -= 1
        y_positions.append(y)
        y_labels.append(name)
        _draw_row(ax, y, name, g, ci_lo, ci_hi)

    # Separator
    if real_rows and synth_rows:
        y -= 0.5
        ax.axhline(y=y, color='grey', linestyle='--',
                   linewidth=0.5, alpha=0.5, xmin=0.05, xmax=0.95)
        y -= 0.3
        ax.text(0.01, y + 0.2, 'Synthetic Variants', fontweight='bold',
                fontsize=9, ha='left', va='bottom',
                transform=ax.get_yaxis_transform())

    for name, g, ci_lo, ci_hi, _ in synth_rows:
        y -= 1
        y_positions.append(y)
        y_labels.append(name)
        _draw_row(ax, y, name, g, ci_lo, ci_hi)

    # Zero line
    ax.axvline(x=0, color='black', linestyle='--', linewidth=0.8,
               alpha=0.7, zorder=1)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=9)
    ax.set_xlabel("Hedges' g  (positive \u2192 unsigned better, "
                  "negative \u2192 signed better)", fontsize=10)
    ax.set_title('Signed vs Unsigned Spectral: Effect Sizes',
                 fontweight='bold')

    # Direction annotations
    ax.annotate('\u2190 Signed better', xy=(0.02, 0.02),
                xycoords='axes fraction', fontsize=7, color='#EE6677',
                fontstyle='italic')
    ax.annotate('Unsigned better \u2192', xy=(0.98, 0.02),
                xycoords='axes fraction', fontsize=7, color='#4477AA',
                ha='right', fontstyle='italic')

    ax.set_ylim(y - 0.5, total_height + 0.5)
    fig.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    logger.info(f"Saved {save_path.name}")

    return {
        "filename": "fig5_forest_plot.png",
        "description": ("Forest plot of Hedges' g for signed vs unsigned "
                        "spectral on real and synthetic datasets"),
        "data_sources": ["exp_id1_it5__opus", "exp_id3_it3__opus"],
        "hedges_g_per_dataset": hedges_g_results,
        "n_real_datasets": len(real_rows),
        "n_synthetic_variants": len(synth_rows),
    }


# =============================================================================
# FIGURE 6: SYNTHETIC MODULE RECOVERY BARS
# =============================================================================

def fig6_module_recovery(exp1_recov_meta: dict, save_path: Path) -> dict:
    """Draw grouped bars of synergistic pair Jaccard."""
    logger.info("Generating Figure 6: Module recovery bars")

    per_variant = exp1_recov_meta.get('per_variant', {})

    recovery_methods = ['sponge_oracle_k', 'sponge_auto_k',
                        'unsigned_spectral', 'hard_threshold']
    recovery_colors = {
        'sponge_oracle_k':   '#EE6677',
        'sponge_auto_k':     '#CC3355',
        'unsigned_spectral':  '#228833',
        'hard_threshold':     '#66CCEE',
    }
    recovery_labels = {
        'sponge_oracle_k':   'SPONGE (Oracle k)',
        'sponge_auto_k':     'SPONGE (Auto k)',
        'unsigned_spectral':  'Unsigned Spectral',
        'hard_threshold':     'Hard Threshold',
    }

    fig, ax = plt.subplots(figsize=(9, 4.5))

    n_var = len(SYNTH_VARIANTS_RECOVERY)
    n_meth = len(recovery_methods)
    bar_w = 0.18
    x_base = np.arange(n_var)

    jaccard_data = {}

    for j, method in enumerate(recovery_methods):
        jaccards = []
        for var in SYNTH_VARIANTS_RECOVERY:
            if var in per_variant:
                mdict = per_variant[var].get('methods', {})
                if method in mdict:
                    jac = mdict[method].get('synergistic_pair_jaccard')
                    jac_val = float(jac) if jac is not None else 0.0
                    jaccards.append(jac_val)
                    jaccard_data[(var, method)] = jac_val
                else:
                    jaccards.append(0.0)
            else:
                jaccards.append(0.0)

        offset = (j - n_meth / 2 + 0.5) * bar_w
        ax.bar(x_base + offset, jaccards, bar_w,
               color=recovery_colors[method],
               label=recovery_labels[method],
               edgecolor='white', linewidth=0.5)

    # Random partition reference dots
    first_rp = True
    for vi, var in enumerate(SYNTH_VARIANTS_RECOVERY):
        if var in per_variant:
            mdict = per_variant[var].get('methods', {})
            if 'random_partition' in mdict:
                rp_jac = mdict['random_partition'].get('synergistic_pair_jaccard')
                if rp_jac is not None:
                    ax.plot(vi, float(rp_jac), 'x', color='grey',
                            markersize=8, markeredgewidth=2, zorder=5,
                            label='Random Partition' if first_rp else None)
                    first_rp = False

    # Success threshold
    ax.axhline(y=0.8, color='#333333', linestyle='--', linewidth=1,
               alpha=0.5, label='Success threshold (0.8)')

    xlabels = [SYNTH_VARIANT_LABELS.get(v, v) for v in SYNTH_VARIANTS_RECOVERY]
    ax.set_xticks(x_base)
    ax.set_xticklabels(xlabels, fontsize=9)
    ax.set_ylabel('Synergistic Pair Jaccard')
    ax.set_xlabel('Synthetic Variant')
    ax.set_title('Module Recovery: Synergistic Pair Jaccard',
                 fontweight='bold')
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=7, frameon=True, loc='upper right', ncol=2)

    fig.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    logger.info(f"Saved {save_path.name}")

    return {
        "filename": "fig6_module_recovery.png",
        "description": ("Module recovery Jaccard for 4 clustering methods "
                        "across 5 synthetic variants"),
        "data_sources": ["exp_id1_it2__opus"],
        "jaccard_scores": {
            f"{var}_{method}": float(v)
            for (var, method), v in jaccard_data.items()
        },
    }


# =============================================================================
# FIGURE 7: SCALING ANALYSIS
# =============================================================================

def fig7_scaling(exp2_frust_meta: dict, save_path: Path) -> dict:
    """Draw log-log scatter of computation time vs n_features."""
    logger.info("Generating Figure 7: Scaling analysis")

    pdr = exp2_frust_meta.get('per_dataset_results', {})

    data_pts = []
    for ds, info in pdr.items():
        nf = info.get('n_features')
        coi_time = info.get('coi_computation', {}).get('time_s', 0)
        sponge_time = info.get('signed_spectral_sponge', {}).get('time_s', 0)

        if nf and nf > 0 and coi_time > 0:
            data_pts.append({
                'dataset': ds,
                'n_features': nf,
                'coi_time': coi_time,
                'sponge_time': sponge_time,
                'total_time': coi_time + sponge_time,
            })

    if not data_pts:
        logger.warning("No scaling data")
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.text(0.5, 0.5, 'No data available', transform=ax.transAxes,
                ha='center', va='center', fontsize=14)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        return {"filename": "fig7_scaling.png",
                "description": "No data available", "data_sources": []}

    fig, ax = plt.subplots(figsize=(7, 5))

    nf_arr = np.array([p['n_features'] for p in data_pts])
    coi_arr = np.array([p['coi_time'] for p in data_pts])
    sponge_arr = np.array([p['sponge_time'] for p in data_pts])
    total_arr = np.array([p['total_time'] for p in data_pts])

    coi_mask = coi_arr > 1e-6
    sponge_mask = sponge_arr > 1e-6
    total_mask = total_arr > 1e-6

    # Scatter
    ax.scatter(nf_arr[coi_mask], coi_arr[coi_mask], c='#4477AA',
               marker='o', s=50, label='CoI computation', alpha=0.8,
               edgecolors='black', linewidths=0.5, zorder=3)
    ax.scatter(nf_arr[sponge_mask], sponge_arr[sponge_mask], c='#EE6677',
               marker='^', s=50, label='SPONGE clustering', alpha=0.8,
               edgecolors='black', linewidths=0.5, zorder=3)
    ax.scatter(nf_arr[total_mask], total_arr[total_mask], c='#228833',
               marker='s', s=50, label='Total pipeline', alpha=0.8,
               edgecolors='black', linewidths=0.5, zorder=3)

    # Power-law fits
    exponents = {}
    series = [
        (coi_arr, coi_mask, 'coi', '#4477AA'),
        (sponge_arr, sponge_mask, 'sponge', '#EE6677'),
        (total_arr, total_mask, 'total', '#228833'),
    ]
    for arr, mask, name, color in series:
        v_nf = nf_arr[mask]
        v_t = arr[mask]
        if len(v_nf) >= 3 and len(np.unique(v_nf)) >= 2:
            log_nf = np.log10(v_nf.astype(float))
            log_t = np.log10(v_t.astype(float))
            coeffs = np.polyfit(log_nf, log_t, 1)
            exponent = coeffs[0]
            exponents[name] = float(exponent)

            nf_range = np.linspace(v_nf.min(), v_nf.max(), 100)
            fit_line = 10 ** np.polyval(coeffs, np.log10(nf_range))
            ax.plot(nf_range, fit_line, color=color, linestyle='--',
                    linewidth=1.5, alpha=0.6)

            mid = len(nf_range) // 2
            ax.annotate(
                f'O(d$^{{{exponent:.1f}}}$)',
                xy=(nf_range[mid], fit_line[mid]),
                xytext=(10, 10), textcoords='offset points',
                fontsize=8, color=color, fontweight='bold',
            )

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Number of Features (d)')
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Computational Scaling Analysis', fontweight='bold')
    ax.legend(fontsize=8, frameon=True)

    # Label a few datasets
    for p in data_pts:
        if p['n_features'] >= 50 or p['n_features'] <= 8:
            ax.annotate(p['dataset'], xy=(p['n_features'], p['total_time']),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=6, alpha=0.7)

    fig.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    logger.info(f"Saved {save_path.name}")

    return {
        "filename": "fig7_scaling.png",
        "description": ("Log-log scatter of computation time vs features "
                        "with power-law fits"),
        "data_sources": ["exp_id2_it5__opus"],
        "power_law_exponents": exponents,
        "n_datasets": len(data_pts),
    }


# =============================================================================
# BUILD EVAL OUTPUT
# =============================================================================

def build_eval_output(figures_meta: list, figs_data: dict, base_data: dict,
                      exp1_recov_meta: dict, exp2_frust_meta: dict) -> dict:
    """Build eval_out.json conforming to exp_eval_sol_out schema."""
    logger.info("Building eval_out.json")

    # --- metrics_agg ---
    metrics_agg = {"n_figures_generated": len(figures_meta)}

    for fm in figures_meta:
        fname = fm.get('filename', '')

        if fname == 'fig2_cd_diagram.png':
            metrics_agg['friedman_chi2'] = sanitize_number(
                fm.get('friedman_chi2'))
            metrics_agg['friedman_p_value'] = sanitize_number(
                fm.get('friedman_p'))
            metrics_agg['critical_difference'] = sanitize_number(
                fm.get('critical_difference'))
            for m, r in fm.get('average_ranks', {}).items():
                metrics_agg[f'avg_rank_{m}'] = sanitize_number(r)

        elif fname == 'fig3_accuracy_bars.png':
            for m, avg in fm.get('method_avg_accuracy', {}).items():
                metrics_agg[f'mean_accuracy_{m}'] = sanitize_number(avg)

        elif fname == 'fig5_forest_plot.png':
            hg = fm.get('hedges_g_per_dataset', {})
            real_gs = [v for k, v in hg.items()
                       if k in CLASSIFICATION_DATASETS]
            synth_gs = [v for k, v in hg.items()
                        if k not in CLASSIFICATION_DATASETS]
            if real_gs:
                metrics_agg['mean_hedges_g_real'] = sanitize_number(
                    float(np.mean(real_gs)))
            if synth_gs:
                metrics_agg['mean_hedges_g_synthetic'] = sanitize_number(
                    float(np.mean(synth_gs)))

        elif fname == 'fig7_scaling.png':
            for nm, exp_val in fm.get('power_law_exponents', {}).items():
                metrics_agg[f'scaling_exponent_{nm}'] = sanitize_number(
                    exp_val)

    # --- datasets ---
    datasets = []

    # 1. Accuracy comparison
    acc_ex = []
    for ds in CLASSIFICATION_DATASETS:
        for method in ALL_METHODS:
            key = (ds, method)
            ba = 0.5
            if key in figs_data:
                ba = figs_data[key]['ba_mean']
            elif key in base_data:
                ba = base_data[key]['ba_mean']
            acc_ex.append({
                "input": json.dumps({"dataset": ds, "method": method,
                                     "n_features": DATASET_NFEATURES.get(ds, 0)}),
                "output": f"{ba:.6f}",
                "eval_balanced_accuracy_mean": sanitize_number(ba),
                "metadata_dataset": ds,
                "metadata_method": method,
            })
    if acc_ex:
        datasets.append({"dataset": "accuracy_comparison",
                         "examples": acc_ex})

    # 2. Effect sizes
    eff_ex = []
    for fm in figures_meta:
        if fm.get('filename') == 'fig5_forest_plot.png':
            for ds_name, g_val in fm.get('hedges_g_per_dataset', {}).items():
                eff_ex.append({
                    "input": json.dumps({
                        "comparison": "unsigned_vs_signed_spectral",
                        "dataset": ds_name}),
                    "output": f"{g_val:.6f}",
                    "eval_hedges_g": sanitize_number(g_val),
                    "metadata_dataset": ds_name,
                })
    if eff_ex:
        datasets.append({"dataset": "effect_sizes", "examples": eff_ex})

    # 3. Module recovery
    recov_ex = []
    per_variant = exp1_recov_meta.get('per_variant', {})
    for var in SYNTH_VARIANTS_RECOVERY:
        if var not in per_variant:
            continue
        mdict = per_variant[var].get('methods', {})
        for method in ['sponge_oracle_k', 'sponge_auto_k',
                       'unsigned_spectral', 'hard_threshold']:
            if method in mdict:
                jac = mdict[method].get('synergistic_pair_jaccard')
                if jac is not None:
                    recov_ex.append({
                        "input": json.dumps({"variant": var,
                                             "method": method}),
                        "output": f"{jac:.6f}",
                        "eval_jaccard": sanitize_number(jac),
                        "metadata_variant": var,
                        "metadata_method": method,
                    })
    if recov_ex:
        datasets.append({"dataset": "module_recovery",
                         "examples": recov_ex})

    # 4. Scaling
    scl_ex = []
    for ds, info in exp2_frust_meta.get('per_dataset_results', {}).items():
        nf = info.get('n_features')
        coi_t = info.get('coi_computation', {}).get('time_s')
        if nf and coi_t and coi_t > 0:
            scl_ex.append({
                "input": json.dumps({"dataset": ds, "n_features": nf}),
                "output": f"{coi_t:.4f}",
                "eval_coi_time_s": sanitize_number(coi_t),
                "eval_n_features": sanitize_number(nf),
                "metadata_dataset": ds,
            })
    if scl_ex:
        datasets.append({"dataset": "scaling_analysis",
                         "examples": scl_ex})

    # Ensure at least one dataset (schema requires minItems: 1)
    if not datasets:
        datasets.append({
            "dataset": "summary",
            "examples": [{
                "input": "figure_generation_summary",
                "output": f"{len(figures_meta)} figures generated",
                "eval_n_figures": float(len(figures_meta)),
            }],
        })

    return {
        "metadata": {
            "evaluation": "figure_generation",
            "description": ("7 publication-quality figures from "
                            "5 experiment outputs"),
            "figures": figures_meta,
        },
        "metrics_agg": metrics_agg,
        "datasets": datasets,
    }


# =============================================================================
# MAIN
# =============================================================================

@logger.catch
def main():
    logger.info("=" * 60)
    logger.info("Starting figure generation evaluation")
    logger.info("=" * 60)

    # --- Load all experiment data ---
    experiments = {}
    for name, (primary, fallback) in EXP_PATHS.items():
        experiments[name] = load_json_safe(primary, fallback, name)

    exp1_meta = experiments['exp1_real']['metadata']
    exp2_meta = experiments['exp2_base']['metadata']
    exp3_meta = experiments['exp3_synth']['metadata']
    exp1_recov_meta = experiments['exp1_recov']['metadata']
    exp2_frust_meta = experiments['exp2_frust']['metadata']

    # Free datasets sections (only need metadata)
    for exp in experiments.values():
        if 'datasets' in exp:
            del exp['datasets']
    gc.collect()

    # --- Pre-process ---
    logger.info("Pre-processing FIGS results (best max_splits)")
    figs_data = get_best_figs_data(exp1_meta)
    logger.info(f"Best results for {len(figs_data)} (dataset, method) pairs")

    logger.info("Pre-processing baseline results")
    base_data = get_baseline_data(exp2_meta)
    logger.info(f"Baseline results for {len(base_data)} (dataset, method) pairs")

    # --- Generate all 7 figures ---
    figures_meta = []

    figure_funcs = [
        ("Figure 1", lambda: fig1_pipeline(WORKSPACE / "fig1_pipeline.png")),
        ("Figure 2", lambda: fig2_cd_diagram(
            figs_data, base_data, WORKSPACE / "fig2_cd_diagram.png")),
        ("Figure 3", lambda: fig3_accuracy_bars(
            figs_data, base_data, WORKSPACE / "fig3_accuracy_bars.png")),
        ("Figure 4", lambda: fig4_pareto_scatter(
            figs_data, WORKSPACE / "fig4_pareto_scatter.png")),
        ("Figure 5", lambda: fig5_forest_plot(
            figs_data, exp3_meta, WORKSPACE / "fig5_forest_plot.png")),
        ("Figure 6", lambda: fig6_module_recovery(
            exp1_recov_meta, WORKSPACE / "fig6_module_recovery.png")),
        ("Figure 7", lambda: fig7_scaling(
            exp2_frust_meta, WORKSPACE / "fig7_scaling.png")),
    ]

    for name, func in figure_funcs:
        try:
            meta = func()
            figures_meta.append(meta)
        except Exception:
            logger.exception(f"Failed to generate {name}")

    logger.info(f"Generated {len(figures_meta)}/7 figures")

    # --- Build and save eval_out.json ---
    eval_output = build_eval_output(
        figures_meta, figs_data, base_data,
        exp1_recov_meta, exp2_frust_meta,
    )

    out_path = WORKSPACE / "eval_out.json"
    out_path.write_text(json.dumps(eval_output, indent=2))
    size_kb = out_path.stat().st_size / 1024
    logger.info(f"Saved eval_out.json ({size_kb:.1f} KB)")

    # --- Summary ---
    logger.info("=" * 60)
    logger.info("EVALUATION COMPLETE")
    logger.info(f"Figures generated: {len(figures_meta)}/7")
    logger.info(f"Output datasets: {len(eval_output['datasets'])}")
    logger.info(f"Aggregate metrics: {len(eval_output['metrics_agg'])}")
    for k, v in eval_output['metrics_agg'].items():
        if isinstance(v, float):
            logger.info(f"  {k}: {v:.4f}")
        else:
            logger.info(f"  {k}: {v}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

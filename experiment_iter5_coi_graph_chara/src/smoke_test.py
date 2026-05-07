#!/usr/bin/env python3
"""Smoke test: run pipeline on 1 tiny synthetic dataset to verify all functions work."""
import json
import sys
import time
import numpy as np
from pathlib import Path

WORKSPACE = Path(__file__).parent
sys.path.insert(0, str(WORKSPACE))

from method import (
    _bin_feature,
    _discretize_y,
    compute_coi_matrix,
    characterize_coi_graph,
    unsigned_spectral_clustering,
    sponge_sym_clustering,
    compute_frustration_index,
    compute_eigenspectrum,
    QuickFIGS,
    quick_figs_comparison,
    assign_ground_truth_labels,
)

def test_on_synthetic():
    """Test all functions on easy_2mod_xor synthetic data."""
    print("=== SMOKE TEST: easy_2mod_xor ===")
    t_start = time.time()

    # Generate data
    from synth_data import gen_easy_2mod_xor
    rng = np.random.default_rng(42)
    # Use seed 0 from same sequence as main pipeline
    base_rng = np.random.default_rng(42)
    variant_seeds = [int(base_rng.integers(0, 2**31)) for _ in range(6)]
    result = gen_easy_2mod_xor(np.random.default_rng(variant_seeds[0]))
    X, y, meta = result["X"], result["y"], result["meta"]
    folds = result["folds"]
    n, d = X.shape
    print(f"  Data: X={X.shape}, y unique={np.unique(y)}, d={d}")

    # Subsample
    idx = np.random.default_rng(42).choice(n, min(5000, n), replace=False)
    X_sub, y_sub = X[idx], y[idx]

    # Test binning
    binned = _bin_feature(X_sub[:, 0], 10)
    assert len(binned) == len(X_sub), "binning length mismatch"
    print(f"  _bin_feature: OK, unique bins={len(np.unique(binned))}")

    # Test discretize_y
    y_disc = _discretize_y(y_sub, 10)
    assert len(y_disc) == len(y_sub)
    print(f"  _discretize_y: OK, unique={len(np.unique(y_disc))}")

    # Test CoI matrix
    t0 = time.time()
    coi_matrix, mi_ind = compute_coi_matrix(X_sub, y_sub, n_bins=10)
    coi_time = time.time() - t0
    print(f"  compute_coi_matrix: ({coi_matrix.shape}) in {coi_time:.2f}s")
    assert coi_matrix.shape == (d, d), f"Expected ({d},{d}), got {coi_matrix.shape}"
    assert np.allclose(coi_matrix, coi_matrix.T), "CoI not symmetric"
    assert np.allclose(np.diag(coi_matrix), 0), "CoI diagonal not zero"
    print(f"    CoI[0,1] (XOR pair) = {coi_matrix[0,1]:.6f} (should be negative)")
    print(f"    CoI[0,4] (redundant pair) = {coi_matrix[0,4]:.6f} (should be positive/near-zero)")
    print(f"    MI_ind[0] = {mi_ind[0]:.6f} (XOR feature, should be near 0)")
    print(f"    MI_ind[6] = {mi_ind[6]:.6f} (noise feature, should be near 0)")

    # Test graph characterization
    stats = characterize_coi_graph(coi_matrix, "easy_2mod_xor", meta=meta)
    print(f"  characterize_coi_graph: n_features={stats['n_features']}")
    print(f"    frac_negative={stats['sign_distribution']['frac_negative']}")
    print(f"    frac_positive={stats['sign_distribution']['frac_positive']}")
    if "ground_truth_analysis" in stats:
        syn_pairs = stats["ground_truth_analysis"]["synergistic_pairs"]
        red_pairs = stats["ground_truth_analysis"]["redundant_pairs"]
        print(f"    synergistic_pairs: {len(syn_pairs)}, redundant_pairs: {len(red_pairs)}")

    # Test unsigned spectral
    t0 = time.time()
    us_modules, us_k, us_labels, us_evals, us_sil = unsigned_spectral_clustering(coi_matrix)
    print(f"  unsigned_spectral: k={us_k}, sil={us_sil:.3f} in {time.time()-t0:.2f}s")
    assert us_k >= 2, f"Expected k>=2, got {us_k}"
    assert len(us_labels) == d

    # Test SPONGE
    t0 = time.time()
    ss_modules, ss_k, ss_labels, ss_evals, ss_sil = sponge_sym_clustering(coi_matrix)
    print(f"  sponge_sym: k={ss_k}, sil={ss_sil:.3f} in {time.time()-t0:.2f}s")

    # Test frustration index
    frust = compute_frustration_index(coi_matrix)
    print(f"  frustration: raw={frust['frustration_raw']:.6f}, norm={frust['normalized_by_max']:.6f}")
    assert np.isfinite(frust["frustration_raw"]), "frustration not finite"

    # Test eigenspectrum
    eigenspec = compute_eigenspectrum(coi_matrix, top_k=5)
    print(f"  eigenspectrum: unsigned={len(eigenspec['unsigned_laplacian_eigenvalues'])} vals")

    # Test GT recovery
    gt_labels = assign_ground_truth_labels(meta, d)
    mask = gt_labels >= 0
    print(f"  GT labels: {gt_labels}, mask sum={mask.sum()}")
    from sklearn.metrics import adjusted_rand_score
    ari_us = adjusted_rand_score(gt_labels[mask], us_labels[mask])
    ari_ss = adjusted_rand_score(gt_labels[mask], ss_labels[mask])
    print(f"  ARI: unsigned={ari_us:.4f}, sponge={ari_ss:.4f}")

    # Test FIGS comparison
    t0 = time.time()
    figs_result = quick_figs_comparison(X, y, folds, "classification", n_classes=2)
    print(f"  FIGS: aa={figs_result['metric_axis_aligned']:.4f}, "
          f"ro={figs_result['metric_random_oblique']:.4f}, "
          f"benefit={figs_result['oblique_benefit']:.4f} in {time.time()-t0:.1f}s")

    total_time = time.time() - t_start
    print(f"\n=== SMOKE TEST PASSED in {total_time:.1f}s ===")
    return True


if __name__ == "__main__":
    test_on_synthetic()

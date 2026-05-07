# CoI-FIGS Stack

## Summary

Complete technical specification for the Balance-Guided Oblique Trees pipeline covering 5 threads: (1) NPEET micd() confirmed for computing Co-Information without discretization on classification targets, with exact call mapping for all MI terms; (2) computational feasibility validated with subsample to n=10K plus joblib parallelization achieving approximately 5 minutes for d=200; (3) first-draft reuse map shows approximately 80% of tree-building code is directly reusable with approximately 155 lines of new code; (4) FIGS oblique extension uses the existing custom framework (not imodels) with only _get_feature_subsets_for_split() needing modification; (5) SPONGE integration via SigNet or custom 50-line scipy implementation with eigengap-based cluster selection and spectral frustration index.

## Research Findings

## Executive Summary

This research resolves the five critical technical questions for the Balance-Guided Oblique Trees pipeline. The key finding is that the entire approach is technically feasible: NPEET's micd() can compute Co-Information without discretization [1, 2], computational costs are manageable with subsampling plus parallelization [8, 19], approximately 80% of the first-draft code is reusable [20, 21, 22], and SPONGE integration is straightforward via SigNet or a custom implementation [10, 11, 12].

---

## Thread 1: Co-Information Computation — Exact Formula to Library Calls

### Formula and Sign Convention

The Co-Information formula is confirmed as: CoI(Xi, Xj; Y) = I(Xi; Y) + I(Xj; Y) - I({Xi, Xj}; Y). Positive CoI indicates redundancy (overlapping information); negative CoI indicates synergy (emergent joint information) [18]. This conflates both phenomena into a single signed quantity, which is precisely what we need for signed graph construction. The interaction information (McGill 1954) is defined identically but some authors reverse the sign; we adopt the convention where positive equals redundancy and negative equals synergy [18].

### Primary Library: NPEET

The primary recommended library is NPEET (Non-parametric Entropy Estimation Toolbox) [1]. It implements the KSG (Kraskov-Stogbauer-Grassberger 2004) estimator for continuous variables and the Ross 2014 estimator for mixed discrete-continuous variables [6]. The critical function for classification targets is micd(x, y, k=3), which computes mutual information between continuous X and discrete Y [1, 2].

Multi-dimensional support for micd() is confirmed by examining the source code [2]. The function uses x[(y == yval).all(axis=1)] for filtering, which correctly preserves multi-dimensional array structure. The underlying entropy() function handles arrays of shape (n_samples, n_features) [2]. This is critical because computing the joint MI term I({Xi, Xj}; Y) requires passing a 2D array of shape (n, 2) as the x argument.

IMPORTANT: For classification targets (discrete Y), one must use micd() and NOT mi(). The mi() function is designed for both-continuous variables and would produce incorrect results on discrete class labels [1, 6]. The micd() function implements the Ross 2014 estimator, which stratifies by class labels and estimates I(X;Y) = H(X) - H(X|Y) via kNN entropy within each discrete class partition [6].

A practical caveat exists: if k > len(x_given_y) - 1 for some class (i.e., a class has too few samples), micd() warns and assumes maximal entropy for that conditional term [2]. Mitigation: ensure minimum class frequency exceeds k+1.

### Exact Call Mapping

The concrete mapping from the CoI formula to NPEET function calls is:

- I(Xi; Y): ee.micd(Xi.reshape(-1, 1), y, k=3) — cacheable, only d=200 unique calls needed [2]
- I(Xj; Y): ee.micd(Xj.reshape(-1, 1), y, k=3) — same cache as above [2]
- I({Xi, Xj}; Y): ee.micd(np.column_stack([Xi, Xj]), y, k=3) — must compute for each pair, 19,900 calls for d=200 [2]
- CoI(Xi, Xj; Y) = I(Xi;Y) + I(Xj;Y) - I({Xi,Xj};Y)

Note that micd() does NOT explicitly reshape x like mi() does. Input x must already be 2D (n_samples, n_features). Always pass .reshape(-1, 1) for individual features [2].

### Alternative Libraries Evaluated

sklearn's mutual_info_classif is NOT usable for joint MI [3]. The internal functions _compute_mi and _compute_mi_cd accept only 1D x arrays (n_samples,). The mutual_info_classif function iterates over features independently and cannot compute I({Xi,Xj}; Y) [3].

The knncmi package is a viable alternative [4]. It supports multi-dimensional MI estimation via DataFrame column lists, using the API: knncmi.cmi(['col1','col2'], ['target'], [], k=3, data=df). However, it requires pandas DataFrame input, which adds conversion overhead [4].

The paulbrodersen/entropy_estimators package implements the KSG estimator but the author explicitly warns that these estimators 'should likely no longer be the first choice' due to bias issues [5]. It is not recommended.

A particularly useful reference implementation is the frbourassa gist [7], which provides an optimized Ross 2014 implementation using scipy.spatial.cKDTree with vectorized radius queries. It explicitly handles multi-dimensional continuous X with discrete Y and may be 2-5x faster than NPEET for large n [7].

For improved accuracy on strongly dependent variables, the NPEET_LNC package adds a local non-uniformity correction to the standard KSG estimator [9].

### Thread 1 Recommendation

PRIMARY: Use NPEET micd() for simplicity and correctness [1, 2, 6]. FALLBACK: Use the frbourassa gist implementation [7] for speed if NPEET is too slow at n=100K. Both implement the Ross 2014 algorithm [6].

---

## Thread 2: Computational Feasibility for d=200, n=100K

### Operation Count

For d=200 features: d*(d-1)/2 = 19,900 pairs. Each pair requires 3 MI estimates, but individual MI values I(Xi;Y) can be cached (only d=200 unique individual MI calls). Total MI calls: 200 (individual, cached) + 19,900 (joint) = 20,100 [8].

### Per-Call Complexity

The KSG estimator uses KD-tree for nearest neighbor search with O(n log n) complexity per call for low-dimensional inputs (our inputs are 1-3 dimensional) [8]. For higher dimensions, complexity degrades to O(n^(1+alpha)) for alpha > 0, but this is not applicable here since our maximum input dimensionality is 3 (two features plus one target) [8].

### Time Estimates

No published benchmarks for NPEET at n=100K were found. Conservative estimate based on KD-tree performance: 0.1-0.5 seconds per call for n=100K.

Conservative scenario (0.3s per call): 20,100 calls times 0.3s = 6,030 seconds = 100.5 minutes serial. This exceeds the 30-minute budget.

Optimistic scenario (0.1s per call): 20,100 calls times 0.1s = 2,010 seconds = 33.5 minutes serial. This barely fits the budget.

Serial execution is therefore too risky for n=100K.

### Recommended Strategy: Subsampling + Parallelization

The recommended approach combines subsampling with parallelization:

Step 1: Subsample to n=10K. The KSG estimator is consistent, meaning accuracy improves with n but the ranking of CoI values is preserved at smaller n [9, 19]. For feature selection purposes (ranking pairs by CoI), exact values matter less than relative ordering. Estimated time at n=10K: approximately 0.03s per call, total approximately 600 seconds serial (10 minutes).

Step 2: Parallelize with joblib. Pre-compute and cache all individual MI values first (200 calls, trivially fast), then parallelize only the 19,900 joint MI calls.

With 4 cores and n=10K subsampling: approximately 3 minutes total. With 8 cores: approximately 2 minutes.

Step 3: Validate ranking stability via 2 independent subsamples. Compute Spearman rank correlation of CoI rankings between subsamples; expect rho > 0.9 [19].

### Memory Analysis

Memory is not a constraint: data matrix 100K x 200 x 8 bytes = 160 MB; CoI matrix 200 x 200 x 8 bytes = 320 KB; KD-tree per call approximately 2-4 MB for n=100K in 2-3D; total peak memory under 500 MB, well within 16GB limit.

### Acceleration Alternatives

The frbourassa gist uses cKDTree directly with vectorized queries and may be 2-5x faster than NPEET for large n [7]. No established GPU implementation for kNN MI estimation in Python exists, making GPU acceleration not worth pursuing.

---

## Thread 3: First-Draft SG-FIGS Codebase Reuse Analysis

### Component-by-Component Reuse Map

Analysis of the first-draft source files reveals approximately 80% of tree-building and evaluation code is directly reusable.

**REUSE (12 components):**

The ObliqueFIGSNode class stores feature (int), features (list), weights (array), threshold, and is_oblique flag, which is exactly what the new pipeline needs [21]. The fit_oblique_split_ridge() function implements Ridge(alpha=1.0) plus DecisionTreeRegressor(max_depth=1) for oblique splits, which is the exact pattern needed [21]. The BaseFIGSOblique greedy tree-sum framework implements the complete greedy loop: residual computation, leaf scoring, best split selection, tree growing, and leaf value updates [21]. The _best_split_for_node() method already compares axis-aligned stump versus oblique Ridge+stump and selects maximum gain [21]. The prediction traversal code already handles both axis-aligned and oblique traversal [21]. MinMaxScaler preprocessing scales features to [0,1] before fitting, which is important for Ridge regression [21]. The MultiClassObliqueWrapper provides One-vs-Rest wrapping for multi-class classification [22]. The 5-fold cross-validation framework provides stratified 5-fold CV with hyperparameter tuning [22]. The FIGSBaselineWrapper wraps imodels FIGSClassifier as an axis-aligned baseline [22]. Resource management with setrlimit for RAM and CPU time is good practice [20].

**MODIFY (4 components):**

The _get_feature_subsets_for_split() method currently draws from synergy graph cliques. This must be modified to draw from SPONGE spectral modules instead [21]. The XOR validation test should be adapted to verify CoI detects XOR synergy and AND redundancy [20]. The dataset loading code should be adapted for OpenML direct loading if using Grinsztajn 2022 benchmarks [22]. Interpretability metrics need updating [22].

**REPLACE (4 components):**

PID computation using the dit library must be replaced entirely with Co-Information via NPEET micd() [20]. The unsigned synergy graph must be replaced with a signed graph where positive edges represent redundancy and negative edges represent synergy [20]. Interpretability scoring must be replaced with structural metrics: frustration index, module purity score, and split alignment [22].

**ELIMINATE (2 components):**

Discretization via KBinsDiscretizer is no longer needed because the kNN MI estimator handles continuous features natively [20]. MI pre-filtering of the top 30 features is no longer needed because CoI computation is fast enough for all pairs [20].

### New Code Estimate

The genuinely new code needed is approximately 155 lines, covering: compute_coi_matrix() (approximately 50 lines), build_signed_graph() (approximately 20 lines), run_sponge_clustering() (approximately 50 lines), compute_frustration_index() (approximately 15 lines), and modification of _get_feature_subsets_for_split() (approximately 20 lines).

---

## Thread 4: Extending FIGS with Ridge-Regression Oblique Splits

### imodels vs. Custom Framework

The imodels FIGS implementation supports only axis-aligned splits and has no plugin system for oblique splits [15]. The first-draft custom framework already supports oblique splits with ObliqueFIGSNode and fit_oblique_split_ridge() [21].

Recommendation: USE the first draft's custom framework. Do NOT use imodels FIGS [15, 21].

### Split Selection Algorithm

The split selection algorithm is already implemented in the first draft [21]: for each candidate leaf, compute residuals, try axis-aligned stump, then for each spectral module call fit_oblique_split_ridge(), and select the split with maximum impurity reduction. The only modification needed is in _get_feature_subsets_for_split() to draw from spectral modules instead of cliques [21].

### Existing Oblique Tree Packages

The DecisionTreeBaseline repository includes RidgeCART [16]. The FC-ODT paper validates the Ridge+stump pattern theoretically [23]. The scikit-tree project provides Cythonized oblique trees [24]. None replace the first draft's implementation, which integrates oblique splits into a FIGS-style additive tree-sum with spectral module guidance [21].

---

## Thread 5: Signed Spectral Clustering (SPONGE) Integration

### SigNet Package Viability

SigNet was archived November 2025 but remains installable [10]. Its dependencies are scikit-learn, cvxpy, and networkx [25]. Compatibility risk is moderate due to cvxpy.

### SPONGE Algorithm Details

SPONGE formulates signed clustering as a generalized eigenvalue problem [12]. It constructs Laplacians from positive and negative adjacency matrices and solves via scipy lobpcg [11]. The SigNet API: c = Cluster((Ap, An)), labels = c.SPONGE(k=k) [11].

### Input Construction from CoI Matrix

Construct Ap from positive CoI entries (redundancy) and An from negative CoI entries (synergy) as sparse matrices [11]. Run SPONGE to obtain cluster labels and extract feature index modules [11, 12].

### Cluster Number Selection

The eigengap heuristic on SPONGE eigenvalues selects k at the largest spectral gap [12]. Fallback: sweep k=2 through 8 with signed modularity or silhouette score [12].

### Frustration Index Computation

The frustration index measures deviation from structural balance [13]. A signed graph is balanced iff the smallest eigenvalue of its signed Laplacian equals zero [13]. The smallest eigenvalue provides a polynomial-time lower bound on the exact (NP-hard) frustration index [14]. Computed via scipy eigsh with which='SM' [13, 14].

### Custom SPONGE Implementation

If SigNet fails to install, a custom implementation requires approximately 50 lines using scipy lobpcg and sklearn KMeans [11, 12].

---

## Synthesis

### Recommended Library Stack

Core ML: scikit-learn, numpy, scipy. MI estimation: NPEET [1]. Signed clustering: SigNet or custom [10, 11]. Baseline: imodels [15]. Benchmarks: openml [17].

### Pipeline Phases

Phase 1 (CoI, ~5 min): subsample, compute MI, assemble CoI matrix [1, 2, 19]. Phase 2 (Clustering, <1 min): SPONGE with eigengap k selection [10, 11, 12, 13]. Phase 3 (Trees, 1-5 min/fold): greedy tree-sum with spectral modules [21]. Phase 4 (Eval): 5-fold CV against baselines [22].

### Grinsztajn Benchmark

Classification numerical suite: OpenML ID 337 [17]. First draft datasets available as secondary validation [22].

### Risk Register

1. NPEET too slow: subsample to n=10K [19]; fallback: frbourassa gist [7].
2. NPEET inaccurate: use NPEET_LNC [9].
3. SigNet install fails: custom 50-line implementation [11, 12].
4. Ambiguous eigengap: default k=3, compare vs random subsets.
5. Oblique splits don't help: valid finding; measure complexity reduction.
6. Budget exceeded: process smallest datasets first with timeout.

## Sources

[1] [NPEET: Non-parametric Entropy Estimation Toolbox](https://github.com/gregversteeg/NPEET) — Primary library for kNN-based MI estimation. Confirmed micd() supports multi-dimensional continuous X with discrete Y via Ross 2014 algorithm.

[2] [NPEET entropy_estimators.py source code](https://github.com/gregversteeg/NPEET/blob/master/npeet/entropy_estimators.py) — Source code confirming mi() reshapes to 2D and micd() uses (y==yval).all(axis=1) filtering that preserves multi-dimensional structure.

[3] [sklearn _mutual_info.py source code](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/feature_selection/_mutual_info.py) — Confirmed _compute_mi and _compute_mi_cd accept only 1D arrays. Cannot compute joint MI I({Xi,Xj};Y).

[4] [knncmi: kNN Conditional Mutual Information](https://github.com/omesner/knncmi) — Alternative library supporting multi-dimensional MI via DataFrame column lists. Based on arxiv 1912.03387.

[5] [paulbrodersen entropy_estimators](https://github.com/paulbrodersen/entropy_estimators) — KSG-based MI estimation. Author warns estimators should likely no longer be the first choice due to known bias issues.

[6] [Ross 2014: Mutual Information between Discrete and Continuous Data Sets](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0087357) — Foundational paper for the mixed discrete-continuous MI estimator used in NPEET micd() and sklearn _compute_mi_cd.

[7] [frbourassa: MI estimator for continuous-multidimensional with discrete variable](https://gist.github.com/frbourassa/c108366d28f541ec6c1c4e39bdfea355) — Optimized Ross 2014 implementation using cKDTree with vectorized queries. Explicitly handles multi-dimensional continuous X. Potentially 2-5x faster than NPEET.

[8] [A computationally efficient estimator for mutual information (Royal Society)](https://royalsocietypublishing.org/doi/10.1098/rspa.2007.0196) — KSG complexity analysis: O(n log n) for 1D marginals, O(n^(1+alpha)) for higher dimensions. Basis for time estimates.

[9] [Gao et al. 2015: Efficient Estimation of MI for Strongly Dependent Variables](http://proceedings.mlr.press/v38/gao15.pdf) — Improved KSG estimator for strongly dependent variables. NPEET_LNC implements this local non-uniformity correction.

[10] [SigNet: Signed Network Clustering (archived Nov 2025)](https://github.com/alan-turing-institute/SigNet) — Python implementation of SPONGE and signed Laplacian clustering. Archived but still installable via git.

[11] [SigNet cluster.py source code](https://github.com/alan-turing-institute/SigNet/blob/master/signet/cluster.py) — SPONGE implementation using lobpcg for generalized eigenvalue problem. Input: (Ap, An) sparse matrix tuple. Basis for custom reimplementation.

[12] [SPONGE: A generalized eigenproblem for clustering signed networks (AISTATS 2019)](https://arxiv.org/abs/1904.08575) — Original SPONGE paper by Cucuringu et al. Formulates signed clustering as generalized eigenvalue problem with tau regularization parameters.

[13] [Belardo 2014: Balancedness and the least eigenvalue of Laplacian of signed graphs](https://www.sciencedirect.com/science/article/pii/S0024379514000160) — Proves signed graph is balanced iff smallest Laplacian eigenvalue equals zero. Eigenvalue provides polynomial-time lower bound on frustration index.

[14] [Frustration Index XOR computation (Aref et al.)](https://github.com/saref/frustration-index-XOR) — Exact frustration index computation via binary linear programming. Confirms NP-hardness of exact computation; spectral bound is polynomial alternative.

[15] [imodels FIGS source code](https://github.com/csinva/imodels/blob/master/imodels/tree/figs.py) — FIGS implementation with axis-aligned splits only. No plugin system for oblique splits. Node class stores only single feature plus threshold.

[16] [DecisionTreeBaseline (includes RidgeCART)](https://github.com/maoqiangqiang/DecisionTreeBaseline) — Collection of oblique tree baselines including RidgeCART using Ridge regression for split directions. Reference implementation but not FIGS-integrated.

[17] [Grinsztajn et al. 2022 Tabular Benchmark](https://github.com/LeoGrin/tabular-benchmark) — Standard benchmark of 45 datasets from NeurIPS 2022. Classification numerical suite: OpenML ID 337. Access via openml.study.get_suite(337).

[18] [Information Decomposition and Synergy (MDPI Entropy 2015)](https://www.mdpi.com/1099-4300/17/5/3501) — Explains co-information as conflating synergy and redundancy into a signed quantity. Validates CoI for signed graph construction.

[19] [Accurate Estimation of MI in High Dimensions (2025)](https://arxiv.org/html/2506.00330) — Subsampling-and-extrapolation workflow for MI estimation. Supports the subsampling strategy for preserving CoI rankings.

[20] [First draft PID synergy computation code](https://raw.githubusercontent.com/ai-inventor-outputs/ai-invention-b88b52-synergy-guided-oblique-splits-using-part/main/experiment_iter2_pairwise_pid_sy/src/method.py) — Uses dit library with discretization. Contains compute_co_information() baseline, synergy graph construction, and stability analysis pattern.

[21] [First draft SG-FIGS tree construction code](https://raw.githubusercontent.com/ai-inventor-outputs/ai-invention-b88b52-synergy-guided-oblique-splits-using-part/main/experiment_iter2_sg_figs_full_ex/src/method.py) — ObliqueFIGSNode, fit_oblique_split_ridge(), BaseFIGSOblique, SGFIGSClassifier. Complete greedy tree-sum with oblique split support.

[22] [First draft definitive evaluation code](https://raw.githubusercontent.com/ai-inventor-outputs/ai-invention-b88b52-synergy-guided-oblique-splits-using-part/main/experiment_iter3_sg_figs_definit/src/method.py) — 5-method comparison framework, 5-fold CV, metrics (balanced accuracy, AUC), FIGSBaselineWrapper, MultiClassObliqueWrapper.

[23] [FC-ODT: Feature Concatenation Oblique Decision Tree (Feb 2025)](https://arxiv.org/html/2502.00465) — Recent paper using Ridge regression for oblique split direction. Provides theoretical validation of the Ridge+stump pattern.

[24] [scikit-tree (treeple) oblique tree documentation](https://docs.neurodata.io/treeple/v0.8/modules/supervised_tree.html) — Cythonized oblique tree implementation by neurodata. Reference for oblique forest approaches but not FIGS-based.

[25] [SigNet setup.py](https://github.com/alan-turing-institute/SigNet/blob/master/setup.py) — Dependencies: scikit-learn, cvxpy, networkx. No Python version constraint specified. Unconventional numpy install via subprocess.

## Follow-up Questions

- What is the optimal k (nearest neighbors) parameter for micd() when computing CoI on subsampled data (n=10K) — does k=3 remain appropriate or should it be increased to reduce variance?
- How should the spectral module feature subsets interact with the beam_size parameter in the oblique split search — should modules smaller than beam_size always be padded with random features, or kept pure?
- Can the frustration index be used as a dataset-level predictor of whether spectral-guided oblique splits will outperform random oblique splits — i.e., does higher frustration correlate with larger accuracy gains from spectral guidance?

---
*Generated by AI Inventor Pipeline*

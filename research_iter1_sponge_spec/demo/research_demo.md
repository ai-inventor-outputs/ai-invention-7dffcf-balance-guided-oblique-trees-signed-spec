# SPONGE Spec

## Summary

Comprehensive implementation recipe for SPONGE signed spectral clustering on Co-Information feature graphs. Covers exact mathematical formulations for both SPONGE and SPONGE_sym generalized eigenvalue problems, SigNet library API (archived Nov 2025, recommend vendoring ~250 lines), eigensolver selection for dense 200-1000 node graphs (scipy.linalg.eigh for d<=500, eigsh shift-invert for d<=1000), spectral frustration index via lambda_min of signed Laplacian, eigengap+silhouette k-selection, Co-Information computation via sklearn KSG with custom joint MI implementation, complete 9-step pseudocode pipeline, scalability estimates, and seven documented pitfalls with mitigations.

## Research Findings

## 1. Algorithm Overview

SPONGE (Signed Positive Over Negative Generalized Eigenproblem) is a spectral clustering algorithm for signed graphs introduced by Cucuringu et al. [1]. Unlike standard spectral clustering which only handles positive (similarity) edge weights, SPONGE explicitly models both positive edges (agreement/redundancy) and negative edges (disagreement/synergy) by decomposing the signed adjacency matrix into separate positive and negative components and solving a generalized eigenvalue problem [1]. The algorithm simultaneously maximizes within-cluster positive edges while maximizing between-cluster negative edges [1, 2]. Two variants exist: unnormalized SPONGE and normalized SPONGE_sym, with SPONGE_sym generally preferred for robustness across diverse settings including varying cluster sizes and sparse graphs [1, 2].

## 2. Mathematical Formulation

### 2.1 Signed Adjacency Decomposition

Given a signed weighted adjacency matrix A, decompose it into positive and negative components [1]: A_plus_ij = max(A_ij, 0) for positive entries only (redundancy edges in the Co-Information context), A_minus_ij = max(-A_ij, 0) for the absolute value of negative entries (synergy edges), with the identity A = A_plus - A_minus [1].

### 2.2 Degree Matrices and Graph Laplacians

D_plus_ii = sum_j A_plus_ij is the positive degree; D_minus_ii = sum_j A_minus_ij is the negative degree; D_bar_ii = D_plus_ii + D_minus_ii is the total absolute degree [1, 14]. The Laplacians are: L_plus = D_plus - A_plus; L_minus = D_minus - A_minus (unnormalized); L_sym_plus = (D_plus)^{-1/2} L_plus (D_plus)^{-1/2}; L_sym_minus = (D_minus)^{-1/2} L_minus (D_minus)^{-1/2} (normalized) [1, 6]. The signed Laplacian L_signed = D_bar - A is used for the frustration index [14, 24].

### 2.3 Generalized Eigenvalue Problems

The SPONGE generalized eigenvalue problem is: (L_plus + tau_neg * D_minus) w = lambda (L_minus + tau_pos * D_plus) w [1]. The SPONGE_sym variant is: (L_sym_plus + tau_neg * I) w = lambda (L_sym_minus + tau_pos * I) w [1, 2]. The solution takes k eigenvectors corresponding to the k smallest eigenvalues, forms an n x k matrix V, and applies k-means++ to the rows [1]. SigNet's implementation divides eigenvectors by eigenvalues before k-means [6].

### 2.4 Regularization and Guarantees

Regularization parameters tau_plus and tau_minus default to 1.0 and must be strictly positive to prevent singular matrices [1, 3]. The theoretical constraint for correct ordering under SSBM is: tau_minus < tau_plus * (n/2 - 1 + eta) / (n/2 - eta) [1]. For sparse graphs, adaptive scaling gamma = (np)^{6/7} is recommended [2]. Under SSBM, SPONGE recovers the true partition with high probability when p >= c1' * log(n)/n in the dense regime [1, 2].

## 3. SigNet Library Reference

The SigNet library was archived on November 4, 2025 and is now read-only [5]. Its last significant commit dates to September 2018 [5]. Dependencies include scikit-learn, cvxpy (only for SDP_cluster, not SPONGE), and networkx [7]. No Python version is specified; core APIs are compatible with Python 3.10/3.11 [7].

The Cluster class accepts a tuple of (A_pos, A_neg) as scipy.sparse.csc_matrix objects [3, 4]. Key methods: SPONGE(k=4, tau_p=1, tau_n=1, eigens=None, mi=None) and SPONGE_sym with the same signature [3, 6]. SPONGE constructs matrix1 = (D_p - A_pos) + tau_n * D_n and matrix2 = (D_n - A_neg) + tau_p * D_p, solves via lobpcg [6]. Zero-degree handling uses sqrtinvdiag with safeguard dd = 1/max(sqrt(x), 1/999999999) [8].

**Recommendation:** VENDOR cluster.py (~200 lines) and utils.py (~50 lines) directly to avoid the cvxpy dependency and replace LOBPCG with dense eigensolvers [5, 6, 7, 8].

## 4. Eigensolver Selection for Dense Graphs (d=200-1000)

This is a CRITICAL decision. Co-Information feature graphs are fully dense, and SigNet's default LOBPCG solver is inappropriate [10].

LOBPCG breaks when 5k > n [10]. For k=20 clusters with n=200 features, this constraint is violated. Known convergence failures exist (scipy issue #10258) [20].

**Solver recommendations:** For d <= 500, use scipy.linalg.eigh(A, b=B, subset_by_index) — exact, under 1 second [9]. B must be positive definite [9]. For 500 < d <= 1000, use scipy.linalg.eigh (5-30 sec) or scipy.sparse.linalg.eigsh with shift-invert sigma=0.0 [9, 11]. For d > 1000, use lobpcg with sparse matrices and random initial vectors [10].

The SPONGE B matrix (L_minus + tau_pos * D_plus) is positive definite as long as every node has at least one positive edge [1, 9]. For safety, add epsilon * I with epsilon=1e-10 [9].

## 5. Spectral Frustration Index

The signed Laplacian is L(Sigma) = D_bar - A_signed [14]. A signed graph is balanced if and only if lambda_min(L(Sigma)) = 0 [14, 24]. When lambda_min > 0, it measures how far from balanced the graph is [14, 24].

The signed Cheeger inequality bounds lambda_1/2 <= h_1_sigma <= sqrt(2 * lambda_1), connecting the spectral measure to the combinatorial frustration index [14]. The combinatorial frustration index (minimum edges to flip for balance) is NP-hard, making the spectral bound invaluable [15].

Normalized measure: f = lambda_min / lambda_max gives a scale-invariant ratio [14, 15]. The spectrum is switching-invariant, capturing intrinsic structure [14].

**Computation:** Construct L = D_bar - W where W is the signed CoI matrix, compute eigenvalues via eigvalsh, take ratio of smallest to largest [14].

**Interpretation:** LOW frustration (ratio < 0.1) means clean synergistic/redundant modules exist — oblique splits should help [14, 24]. HIGH frustration (ratio > 0.3) means no clean partition exists [14, 15].

## 6. Choosing k (Number of Clusters)

**Eigengap heuristic:** Sort eigenvalues and compute gaps; choose k = argmax(gaps) [18]. Justified by perturbation theory and Davis-Kahan theorem [18]. Works well for well-separated clusters but may be ambiguous for gradual community structure [18].

**Silhouette on embedding:** Sweep k from 2 to k_max and compute silhouette scores (b-a)/max(a,b), choosing k with best score [19].

**Combined heuristic:** Identify top-3 eigengap candidates, evaluate silhouette for each, choose best [18, 19]. Fallback when ambiguous: k = ceil(sqrt(d/2)) [18]. Practical range: k in {2, ..., min(20, d/3)} [18, 19].

## 7. Co-Information Computation

Co-Information is defined as CoI(Xi, Xj; Y) = I(Xi; Y) + I(Xj; Y) - I(Xi, Xj; Y) [21, 22]. Positive CoI = redundancy (positive edge); negative CoI = synergy (negative edge) [21, 22].

**Individual MI:** Use sklearn.feature_selection.mutual_info_classif with the KSG estimator [12]. The formula is I(X;Y) = psi(k) + psi(N) - mean(psi(n_x+1)) - mean(psi(n_y+1)) using Chebyshev distance [13, 17]. Recommend n_neighbors=5 for reduced variance [12].

**CRITICAL: Joint MI.** sklearn's mutual_info_classif processes each column INDEPENDENTLY — passing X[:, {i,j}] returns individual MIs, NOT the joint [12, 13]. A custom implementation is required. The recommended approach implements the Ross (2014) estimator for multivariate continuous X with discrete Y, using KDTree in 2D Chebyshev space (~20 lines) [13]. Alternatively, NPEET supports multivariate inputs via list-of-lists format but has known bias [16]. The KSG estimator has known negative bias for strongly dependent variables [23], which could slightly overestimate synergy.

**Recipe:** Compute individual MI via sklearn [12], joint MI via custom KSG [13], CoI = mi_i + mi_j - jmi_ij [21, 22]. Parallelize d*(d-1)/2 pairs via joblib.

## 8. Implementation Pipeline

9-step pipeline: (1) Individual MI via sklearn [12]; (2) Pairwise joint MI via custom KSG with joblib parallelization [13]; (3) CoI matrix construction [21, 22]; (4) Signed decomposition into A_pos and A_neg [1]; (5) Frustration index via signed Laplacian eigenvalues [14, 24]; (6) Degeneracy check for sign distribution [14]; (7) k selection via eigengap + silhouette [18, 19]; (8) SPONGE clustering via scipy.linalg.eigh [1, 9]; (9) Module extraction [1].

## 9. Scalability Estimates

CoI computation dominates at O(d^2 * n * log n) [12]. On 8 cores: d=200/n=50K takes ~4 min; d=500/n=100K takes ~26 min; d=1000/n=100K takes ~106 min [12]. Eigendecomposition is negligible (under 30 sec for d=1000) [9]. Memory: 8 MB for d=1000.

## 10. Pitfalls and Mitigations

Seven key pitfalls: (1) Zero-degree nodes making Laplacians singular — mitigated by tau regularization plus epsilon*I [1, 8, 9]; (2) All-same-sign graphs causing degeneracy — check distribution, fall back to unsigned clustering [14, 24]; (3) KSG bias underestimating joint MI — use n_neighbors=5, consider NPEET_LNC [16, 23]; (4) LOBPCG failure for small dense graphs — use scipy.linalg.eigh instead [10, 20]; (5) Dense/sparse storage mismatch — keep as dense numpy arrays [9]; (6) Non-determinism — set random_state everywhere [12]; (7) Eigenvector normalization — SigNet divides by eigenvalues which can amplify noise [6].

## Sources

[1] [Cucuringu et al. (2019) SPONGE: A generalized eigenproblem for clustering signed networks](https://ar5iv.labs.arxiv.org/html/1904.08575) — Original SPONGE paper with complete mathematical formulations for the generalized eigenvalue problem, SPONGE_sym variant, regularization parameters, tau constraints, and theoretical guarantees under the Signed Stochastic Block Model.

[2] [Cucuringu et al. (2022) Regularized spectral methods for clustering signed networks (JMLR)](https://ar5iv.labs.arxiv.org/html/2011.01737) — Extended analysis with adaptive regularization parameters scaling as (np)^{6/7}, sparse regime guarantees, SSBM recovery conditions for general k >= 2, and evidence that SPONGEsym outperforms unnormalized SPONGE.

[3] [SigNet Cluster API Documentation](https://signet.readthedocs.io/en/latest/cluster.html) — Complete API reference for Cluster class including SPONGE, SPONGE_sym, and other methods with parameter signatures, defaults, and input format (tuple of sparse CSC matrices).

[4] [SigNet Usage Documentation](https://signet.readthedocs.io/en/latest/usage.html) — Usage examples showing complete workflow: initialize Cluster from (A_pos, A_neg) tuple, call clustering methods, evaluate with adjusted_rand_score.

[5] [SigNet GitHub Repository (archived Nov 4, 2025)](https://github.com/alan-turing-institute/SigNet) — Repository archived Nov 4, 2025. Last significant commit Sept 2018. ~40 stars, ~10 forks. Install via pip from GitHub. Read-only status confirmed.

[6] [SigNet cluster.py Source Code](https://github.com/alan-turing-institute/SigNet/blob/master/signet/cluster.py) — Complete SPONGE implementation: constructs L_plus + tau_n*D_neg and L_neg + tau_p*D_plus, solves via lobpcg, divides eigenvectors by eigenvalues, applies KMeans. SPONGE_sym uses normalized Laplacians with identity regularization.

[7] [SigNet setup.py](https://github.com/alan-turing-institute/SigNet/blob/master/setup.py) — Version 0.1.0. Dependencies: scikit-learn, cvxpy (only for SDP_cluster), networkx (no version constraints). No Python version specified. Pre-installs numpy for ecos.

[8] [SigNet utils.py Source Code](https://github.com/alan-turing-institute/SigNet/blob/master/signet/utils.py) — sqrtinvdiag uses safeguard dd=1/max(sqrt(x), 1/999999999) to prevent division by zero for zero-degree nodes. Same pattern in invdiag utility.

[9] [scipy.linalg.eigh Documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.eigh.html) — Dense symmetric eigenvalue solver supporting generalized problem Aw=lambda*Bw. B must be positive DEFINITE. subset_by_index enables computing only k smallest. Returns (eigenvalues, eigenvectors) in ascending order.

[10] [scipy.sparse.linalg.lobpcg Documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.lobpcg.html) — Iterative sparse eigensolver that BREAKS when 5k > n. Designed for extremely large n/k ratios. Solves 3k x 3k dense eigenproblems per iteration. Supports generalized problem via B parameter.

[11] [scipy.sparse.linalg.eigsh Documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigsh.html) — ARPACK-based sparse symmetric eigensolver. Shift-invert mode (sigma=0) efficiently finds smallest eigenvalues. Allows M (B matrix) to be positive semidefinite when sigma is specified.

[12] [sklearn mutual_info_classif Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_classif.html) — KSG-based MI estimation between features and discrete target. Processes each feature column independently. n_neighbors=3 default. Returns MI in nat units, clips negatives to 0. Supports n_jobs for parallelization.

[13] [sklearn _mutual_info.py Source Code](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/feature_selection/_mutual_info.py) — Internal KSG implementation: _compute_mi_cd uses NearestNeighbors with Chebyshev metric and KDTree for radius queries. Digamma formula: psi(n)+psi(k)-mean(psi(n_x+1))-mean(psi(n_y+1)). Adds 1e-10 noise.

[14] [Atay & Liu (2019) Cheeger constants, structural balance, and spectral clustering for signed graphs](https://ar5iv.labs.arxiv.org/html/1411.3530) — Signed Cheeger inequality: lambda_1/2 <= h_1_sigma <= sqrt(2*lambda_1). Fundamental result: balance iff lambda_1=0. Switching invariance of spectrum. Multi-way extensions for k balanced components.

[15] [Belardo et al. (2021) Inequalities for Laplacian Eigenvalues of Signed Graphs with Given Frustration Number](https://www.mdpi.com/2073-8994/13/10/1902) — Bounds relating lambda_min to combinatorial frustration index. Computing frustration is NP-hard, making spectral lower bound invaluable as practical proxy.

[16] [NPEET: Non-Parametric Entropy Estimation Toolbox](https://github.com/gregversteeg/NPEET) — KSG-based entropy and MI estimation supporting multivariate inputs via list-of-lists format. Known bias for strong dependencies. NPEET_LNC variant provides Local Non-uniformity Correction.

[17] [Kraskov et al. (2004) Estimating Mutual Information](https://arxiv.org/abs/cond-mat/0305641) — Original KSG estimator paper. Two classes of k-NN based MI estimators that are data-efficient, adaptive, and have minimal bias. Uses Chebyshev distance for neighbor queries.

[18] [Von Luxburg (2007) A Tutorial on Spectral Clustering](https://people.csail.mit.edu/dsontag/courses/ml14/notes/Luxburg07_tutorial_spectral_clustering.pdf) — Comprehensive tutorial covering eigengap heuristic definition, Davis-Kahan perturbation theory justification, normalized vs unnormalized spectral clustering algorithms, and practical guidance.

[19] [sklearn silhouette_score Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html) — Mean silhouette coefficient: (b-a)/max(a,b). Range -1 to 1. Higher = better-defined clusters. Model-based evaluation method for choosing k.

[20] [LOBPCG convergence failure (scipy #10258)](https://github.com/scipy/scipy/issues/10258) — Known bug: LOBPCG fails with provided initial guesses, producing 'leading minor not positive definite' error. Workaround: use random initial vectors.

[21] [dit library: Co-Information Documentation](https://dit.readthedocs.io/en/latest/measures/multivariate/coinformation.html) — Python library for discrete information theory with coinformation function. Co-Information defined via inclusion-exclusion over entropies. Positive = redundancy, negative = synergy.

[22] [Wikipedia: Interaction Information (Co-Information)](https://en.wikipedia.org/wiki/Multivariate_mutual_information) — Co-information can be positive (redundancy) or negative (synergy). For three variables: I(X;Y;Z) = I(X;Y) - I(X;Y|Z). Equivalent formulations via entropies and conditional MI.

[23] [Gao et al. (2016) Demystifying Fixed k-NN Information Estimators](https://arxiv.org/pdf/1604.03006) — Bias-improved KSG (BI-KSG) estimator. KSG shown to have severe negative bias for log-normal distributions. BI-KSG proven consistent with better sample complexity for moderate N.

[24] [Belardo & Simic (2014) Balancedness and the least eigenvalue of Laplacian of signed graphs](https://www.sciencedirect.com/science/article/pii/S0024379514000160) — Fundamental result: signed graph is balanced iff lambda_min(L)=0. lambda_min serves as spectral proxy for balance deviation, with larger values indicating greater frustration.

## Follow-up Questions

- Should we vendor SigNet's cluster.py (~200 lines) and utils.py (~50 lines) or reimplement the SPONGE algorithm from scratch using scipy.linalg.eigh? Vendoring preserves the exact SigNet convention (eigenvector/eigenvalue weighting) while reimplementation avoids the cvxpy dependency and LOBPCG limitation entirely.
- Is the eigengap heuristic reliable for Co-Information feature graphs specifically? The Co-Information matrix may have a gradually varying spectrum rather than sharp gaps, especially when features have mixed synergy/redundancy patterns. Should we default to silhouette-based k selection instead?
- What is the empirical accuracy of KSG-based Co-Information estimation vs binning-based methods for continuous features? The KSG estimator's known negative bias for strongly dependent variables could systematically underestimate joint MI, leading to overestimation of synergy (more negative CoI values). How significant is this effect for typical tabular datasets?

---
*Generated by AI Inventor Pipeline*

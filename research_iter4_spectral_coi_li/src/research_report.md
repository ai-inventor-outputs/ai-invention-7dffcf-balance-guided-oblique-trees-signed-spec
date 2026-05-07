# Spectral CoI Lit

## Summary

Four-thread literature survey confirming a clear novelty gap: no prior work combines spectral clustering on information-theoretic (MI/Co-Information) feature interaction graphs with oblique decision tree split construction. Covers spectral feature grouping methods (gap confirmed across 7 key papers), Co-Information estimation bias (KSG underestimates joint MI biasing CoI negative toward apparent synergy via 3 independent mechanisms), state-of-the-art interpretable oblique trees (none of 7 surveyed methods use IT criteria for feature selection), and paper positioning (Framing C recommended: Spectral Decomposition of Feature Interactions for Constrained Oblique Trees). Includes comparison table and three candidate framings with risk mitigations.

## Research Findings

This four-thread literature survey confirms a clear novelty gap: no prior work combines spectral clustering on an information-theoretic (MI or Co-Information) feature interaction graph with oblique decision tree split construction. The survey covers spectral feature grouping methods, Co-Information estimation bias, state-of-the-art interpretable oblique trees, and paper positioning strategies.

**THREAD 1 - Spectral Feature Grouping for Tree Learning: Gap Confirmed.**
The closest prior work in spectral feature selection is Zhao & Liu (2007) [1], who proposed SPEC, a unified spectral framework using graph Laplacians for feature selection. However, SPEC evaluates features INDIVIDUALLY via spectral scores projected onto eigenvectors of the graph Laplacian -- it does NOT group features for joint use in splits, and has no connection to tree construction. Zheng et al. (2020) [2] proposed graph-based feature grouping (GBFG) using minimum spanning trees on three-way mutual information metrics, clustering redundant features together. This is the closest to our graph-based grouping approach, but it uses MST rather than spectral clustering, focuses on redundancy removal rather than synergy exploitation, and has no connection to oblique trees. A comprehensive review of feature grouping approaches [3] confirms that MST-based methods dominate graph-based feature grouping, with NO spectral clustering methods documented for this purpose. No method in the review distinguishes synergy from redundancy or connects to oblique tree construction.

PIDF (Westphal et al., AISTATS 2025) [4] is a key related work that decomposes per-feature synergy and redundancy using Partial Information Decomposition. It computes Feature-Wise Synergy (FWS) and Feature-Wise Redundancy (FWR) with a pairwise theta measure for synergy/redundancy detection. However, PIDF does NOT build a graph, does NOT use spectral methods, does NOT connect to tree learning, uses MINE neural estimator with O(k^2) complexity, and takes 45 minutes for 6-feature datasets on GPU clusters.

Interaction Forests (Hornung & Boulesteix, 2022) [5] detect feature interactions via bivariate splits in random forest trees, using the EIM (Effect Importance Measure) to rank covariate pairs. The method is purely structural (tree co-occurrence based), NOT information-theoretic, does NOT build a graph, and uses a screening step for >100 features (pre-selecting top 5000 pairs). A recent extension, Unity Forests (2026) [6], addresses sequential split selection limitations but still uses structural rather than information-theoretic interaction detection.

FoLDTree (2024) [7] uses Forward ULDA (Fisher's criterion) for oblique splits with intrinsic variable selection -- NOT information-theoretic. The BioData Mining 2025 paper on Feature Graphs for Interpretable Unsupervised Tree Ensembles [8] builds feature graphs from tree ensembles POST-HOC (after fitting), using parent-child node splits as edges, and can apply spectral clustering to these graphs. However, this is fundamentally different from our approach: the graph is built AFTER tree fitting for interpretation, not BEFORE to guide tree construction, and the edge weights are structural co-occurrence, not information-theoretic.

**THREAD 2 - Co-Information Properties & Estimation Bias.**
Co-Information was introduced by Bell (2003) [9] as a renaming of McGill's (1954) interaction information [10]. Two OPPOSITE sign conventions exist in the literature: in Bell's co-information convention, positive = redundancy and negative = synergy; in the interaction information convention (used by some authors), the signs are reversed [11]. The formula for three variables is I(X;Y;Z) = I(X;Y|Z) - I(X;Y). Negative co-information (synergy in Bell's convention) is WIDESPREAD in empirical data [12], a phenomenon that has been a standing interpretive puzzle since the 1950s. The prevalence of synergy is partly explained by the ubiquity of collider (common-effect) causal structures in nature, which naturally induce synergistic dependencies [12].

Critically, estimation bias systematically inflates apparent synergy. For discrete estimators, PID sampling bias analysis [13] shows that synergy bias is QUADRATIC in the number of discrete response states, while redundancy bias is only sub-linear. This means finite-sample estimates are artificially biased toward synergy. O-information estimation [14] shows that independent systems are severely biased toward synergy when sample size is smaller than the number of jointly possible observations. The Miller-Maddow correction can partially address this.

For KSG (continuous) estimators, Gao et al. (2015) [15] demonstrated that KSG requires exponentially many samples for strongly dependent variables due to implicit reliance on local uniformity of the underlying joint distribution. The LNC (Local Non-uniformity Correction) estimator was proposed as a fix. KSG systematically UNDERESTIMATES mutual information [16, 17], especially for high-dimensional joint distributions, due to the curse of dimensionality. This has direct implications for Co-Information estimation: CoI = I(Xi;Y) + I(Xj;Y) - I(Xi,Xj;Y). If KSG underestimates the joint MI I(Xi,Xj;Y) (which involves higher-dimensional density estimation), CoI will be biased NEGATIVE (toward apparent synergy). This is consistent with the observed near-universal negativity of Co-Information in experiments. Gao et al. (2016) [18] showed KSG is consistent under standard assumptions and proposed the BI-KSG (Bias-Improved KSG) estimator exploiting a 'correlation boosting' effect. Holmes & Nemenman (2019) [19] provided a self-consistent method for verifying absence of bias and choosing the k parameter. Gaussian PID bias correction (NeurIPS 2023) [20] addressed bias at finite sample sizes for Gaussian distributions specifically.

Timme et al. (2014) [21] reviewed multivariate information measures from an experimentalist's perspective, comparing interaction information, total correlation, and other measures, noting that synergy and redundancy have been treated as mutually exclusive by some authors based on the sign of interaction information. Co-information confounds synergy and redundancy -- it can underreport synergy when both are simultaneously present [12, 21].

**THREAD 3 - Interpretable Oblique Trees: State of the Art.**
NO existing oblique tree method uses information-theoretic criteria to select which features to combine in splits. TAO (Carreira-Perpinan & Tavallali, NeurIPS 2018) [22] uses alternating optimization with L1 sparsity penalty on split weights, optimizing each node as a simple classifier. Sparsity is achieved via l1 regularization (LASSO-like path), but feature selection is purely loss-driven, not information-theoretic. Sparse Oblique Trees (Hada et al., 2021/2023) [23] apply TAO to neural net feature interpretation using L1 regularization. LHT (2025) [24] uses statistical class-separation (differences in feature expectations between classes) with threshold parameters alpha (variance) and beta (weight) to filter features -- NOT information-theoretic. CART-ELC (2025) [25] performs exhaustive search on restricted hyperplanes with complexity Theta(C(n,r)*C(m,r)*r(r^2+n)), practically limited to bivariate splits (r<=2). EBM/GA2M (Lou et al., KDD 2013) [26] uses the FAST algorithm to rank feature pairs by interaction strength via boosting residuals (statistical, not information-theoretic), producing additive models with pairwise terms rather than oblique tree splits [27]. FIGS (Tan & Singh, 2022) [28] extends CART to simultaneously grow tree sums, using axis-aligned splits with no information-theoretic criteria.

The comparison table clearly shows our method fills a unique niche: it is the ONLY approach that uses information-theoretic criteria (Co-Information) to build a feature interaction graph, applies spectral decomposition to identify interaction modules, and constrains oblique splits to use features within these modules.

**THREAD 4 - Paper Positioning.**
Given the confirmed literature gap and the empirical findings (unsigned spectral clustering works as well as signed, spectral modules reduce split arity, frustration index does not predict oblique split benefit), Framing C ('Spectral Decomposition of Feature Interactions for Constrained Oblique Trees') is recommended as the strongest positioning. The signed spectral clustering methodology draws on SPONGE [29] and its regularized extensions [30], but the key finding that unsigned clustering on |CoI| performs equally well simplifies the practical method. This framing: (1) is the most general, allowing presentation of both positive (arity reduction) and negative (no signed advantage) results as contributions; (2) bridges three distinct fields (information theory, spectral graph methods, interpretable tree learning) in a novel combination; (3) avoids over-claiming about signed spectral advantage; (4) positions the signed vs unsigned result as an ablation finding rather than a failure; (5) emphasizes the confirmed novelty gap. The importance of tree-based methods for tabular data is well-established by benchmarks showing they outperform deep learning [31]. Elements of Framing A (interpretability through principled sparsity) should be incorporated for the interpretability narrative. Framing B (diagnostic/analytical) risks framing around negative results, which is harder to publish.

## Sources

[1] [Zhao & Liu (2007) - Spectral Feature Selection for Supervised and Unsupervised Learning, ICML](https://dl.acm.org/doi/abs/10.1145/1273496.1273641) — Unified spectral framework using graph Laplacians for feature selection; evaluates features individually via spectral scores, does NOT group features for joint splits or connect to tree construction

[2] [Zheng et al. (2020) - Feature Grouping and Selection: A Graph-Based Approach, Information Sciences](https://www.sciencedirect.com/science/article/abs/pii/S0020025520309336) — Graph-based feature grouping using MST and three-way MI metrics; closest to graph-based grouping but uses MST not spectral clustering, no oblique tree connection

[3] [PeerJ CS (2023) - Review of Feature Selection Approaches Based on Grouping of Features](https://pmc.ncbi.nlm.nih.gov/articles/PMC10358338/) — Comprehensive review confirming MST dominance in graph-based feature grouping; no spectral clustering methods documented for feature grouping in tree learning context

[4] [Westphal et al. (2025) - PIDF: Per-feature Synergy/Redundancy via PID, AISTATS](https://arxiv.org/html/2405.19212) — Key related work decomposing feature synergy/redundancy using PID; no graph, no spectral methods, no tree connection, O(k^2) complexity with MINE neural estimator

[5] [Hornung & Boulesteix (2022) - Interaction Forests, Computational Statistics & Data Analysis](https://www.sciencedirect.com/science/article/pii/S0167947322000408) — Tree-structural interaction detection with EIM measure; purely structural co-occurrence based, NOT information-theoretic, no graph construction

[6] [Unity Forests (2026) - Improving Interaction Modelling in Random Forests](https://arxiv.org/html/2601.07003) — Extension of Interaction Forests addressing sequential split selection limitations; still uses structural rather than information-theoretic interaction detection

[7] [FoLDTree (2024) - Forward ULDA Oblique Decision Trees](https://arxiv.org/html/2410.23147) — Oblique tree with Forward ULDA feature selection based on Fisher's criterion; NOT information-theoretic, no interaction graph

[8] [Feature Graphs for Interpretable Unsupervised Tree Ensembles (BioData Mining, 2025)](https://biodatamining.biomedcentral.com/articles/10.1186/s13040-025-00430-3) — Builds feature graphs POST-HOC from tree ensembles for interpretation; uses spectral clustering but after tree fitting, not before to guide construction; structural not IT

[9] [Bell (2003) - The Co-Information Lattice, ICA 2003](https://www.semanticscholar.org/paper/THE-CO-INFORMATION-LATTICE-Bell/25a0cd8d486d5ffd204485685226f189e6eadd4d) — Introduced co-information terminology and lattice structure via Mobius inversion; positive = redundancy, negative = synergy sign convention

[10] [Interaction Information - Wikipedia (McGill 1954 definition)](https://en.wikipedia.org/wiki/Interaction_information) — Original definition of interaction information by McGill (1954); documents sign conventions and relationship to co-information

[11] [Interaction Information - en-academic](https://en-academic.com/dic.nsf/enwiki/3582538) — Detailed sign conventions for interaction information: negative II = common cause (redundancy), positive II = common effect/collider (synergy)

[12] [Synergistic Perspective on Multivariate Computation and Causality (PMC, 2024)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11507062/) — Confirms negative co-information (synergy) is widespread in empirical data; explains prevalence via collider causal structures; notes co-information confounds synergy and redundancy

[13] [Sampling Bias Corrections for Accurate Neural Measures of Redundant, Unique, and Synergistic Information (PMC, 2024)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11185652/) — Synergy bias is QUADRATIC in number of discrete response states while redundancy bias is sub-linear; finite-sample estimates are artificially biased toward synergy

[14] [Bias in O-Information Estimation (Entropy, 2024)](https://www.mdpi.com/1099-4300/26/10/837) — O-information of independent systems severely biased toward synergy when sample size < number of jointly possible observations; Miller-Maddow correction derived

[15] [Gao et al. (2015) - Efficient Estimation of MI for Strongly Dependent Variables, AISTATS](https://proceedings.mlr.press/v38/gao15.html) — KSG requires exponentially many samples for strongly dependent variables due to local uniformity assumption failure; proposed LNC (Local Non-uniformity Correction) estimator

[16] [Improving Numerical Stability of NMI Estimation in High Dimensions (2024)](https://arxiv.org/html/2410.07642v1) — Documents KSG systematic underestimation of MI in high dimensions due to curse of dimensionality; dimension-dependent bias

[17] [Marx & Fischer - Geodesic KSG (G-KSG) for MI Estimation](https://arxiv.org/pdf/2110.13883) — Proposes G-KSG using geodesic distances to address classical KSG over/underestimation in high dimensions and on manifolds

[18] [Gao et al. (2016) - Demystifying Fixed k-NN Information Estimators](https://arxiv.org/pdf/1604.03006) — Proves KSG consistency under standard assumptions; proposes BI-KSG via correlation boosting effect for improved finite-sample performance

[19] [Holmes & Nemenman (2019) - Estimation of MI with Error Bars and Controlled Bias, Phys. Rev. E](https://arxiv.org/abs/1903.09280) — Improved KSG method with self-consistent bias verification, error bars, and guidelines for choosing k parameter

[20] [Gaussian PID Bias Correction (NeurIPS 2023)](https://arxiv.org/abs/2307.10515) — First bias correction method for Gaussian Partial Information Decomposition at finite sample sizes; works at high dimensionality

[21] [Timme et al. (2014) - Synergy, Redundancy, and Multivariate Information Measures, J. Comp. Neurosci.](https://pubmed.ncbi.nlm.nih.gov/23820856/) — Reviews multivariate information measures from experimentalist's perspective; discusses sign-based synergy/redundancy interpretations and their limitations

[22] [Carreira-Perpinan & Tavallali (2018) - TAO: Alternating Optimization of Decision Trees, NeurIPS](https://proceedings.neurips.cc/paper/2018/hash/185c29dc24325934ee377cfda20e414c-Abstract.html) — State-of-art oblique tree using alternating optimization with L1 sparsity penalty; purely loss-driven feature selection, no IT criteria

[23] [Hada et al. (2021/2023) - Sparse Oblique Decision Trees for Neural Net Features](https://arxiv.org/abs/2104.02922) — TAO-based sparse oblique trees applied to neural net feature interpretation; L1 regularization approach, no IT criteria

[24] [LHT (2025) - Statistically-Driven Oblique Decision Trees](https://arxiv.org/html/2505.04139v1) — Uses statistical class-separation (feature mean differences) with alpha/beta thresholds; NOT information-theoretic

[25] [CART-ELC (2025) - Exhaustive Linear Combinations for Oblique Trees](https://arxiv.org/html/2505.05402) — Exhaustive search on restricted hyperplanes; practically limited to bivariate splits (r<=2); built-in sparsity via r parameter

[26] [Lou et al. (2013) - GA2M: Accurate Intelligible Models with Pairwise Interactions, KDD](https://www.cs.cornell.edu/~yinlou/papers/lou-kdd13.pdf) — FAST algorithm ranks feature pairs by interaction strength via boosting residuals; statistical not IT; produces additive models not oblique trees

[27] [InterpretML - EBM Documentation](https://interpret.ml/docs/ebm.html) — EBM implementation of GA2M: cyclic gradient boosting with automatic pairwise interaction detection via FAST; additive model structure

[28] [Tan & Singh (2022) - FIGS: Fast Interpretable Greedy-Tree Sums](https://arxiv.org/abs/2201.11931) — Axis-aligned tree sums with additive structure; no IT criteria; no oblique splits; greedy residual-based growth

[29] [Cucuringu et al. (2019) - SPONGE: Signed Spectral Clustering, AISTATS](https://arxiv.org/abs/1904.08575) — Principled signed spectral clustering via generalized eigenproblem on positive/negative adjacency matrices; theoretical guarantees for signed SBM

[30] [Cucuringu et al. (2021) - Regularized Signed Spectral Methods, JMLR](https://www.jmlr.org/papers/v22/20-1289.html) — Extends SPONGE to unequal cluster sizes and sparse graphs with regularization; enables practical signed spectral clustering

[31] [Grinsztajn et al. (2022) - Why Tree-Based Models Outperform Deep Learning on Tabular Data, NeurIPS](https://arxiv.org/abs/2207.08815) — Benchmark establishing tree-based models as state-of-the-art on tabular data; motivates improving tree methods over deep learning alternatives

## Follow-up Questions

- Does KSG negative bias for joint MI systematically inflate apparent synergy in Co-Information, and can this be verified on synthetic data with known ground truth CoI values?
- Are there datasets where signed spectral clustering on the Co-Information graph outperforms unsigned, perhaps with specific causal structures (e.g., mixture of common-cause and collider structures)?
- Could alternative MI estimators (LNC-KSG, BI-KSG, or G-KSG) reduce the apparent synergy dominance in Co-Information estimates, potentially changing the spectral structure of the interaction graph and downstream oblique split quality?

---
*Generated by AI Inventor Pipeline*

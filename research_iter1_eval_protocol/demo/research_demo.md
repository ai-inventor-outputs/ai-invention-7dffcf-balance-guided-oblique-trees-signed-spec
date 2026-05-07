# Eval Protocol

## Summary

Comprehensive survey of InterpretML EBM/GA2M API (constructor parameters, FAST interaction detection algorithm, fitted attributes), Grinsztajn 2022 tabular benchmark protocol (45 datasets across 4 OpenML suites 334-337, random search tuning, dataset selection criteria), interpretability metrics for tree-based and additive models (split arity, path length, total splits, number of terms), Friedman/Nemenyi statistical testing with autorank and scikit-posthocs implementations, and Bayesian signed-rank test with ROPE. Synthesized into complete evaluation protocol comparing 6 methods (axis-aligned FIGS, random-oblique FIGS, hard-threshold SG-FIGS, unsigned spectral FIGS, signed spectral FIGS, EBM) across 5-8 classification-numerical benchmark datasets with CD diagram visualization.

## Research Findings

This survey establishes a complete, actionable evaluation protocol for comparing Balance-Guided Oblique Trees against state-of-the-art interpretable models on tabular data, covering five tightly integrated components.

**EBM Baseline Configuration.** The ExplainableBoostingClassifier from InterpretML [1, 3] provides the gold-standard interpretable baseline. Key defaults include interactions='3x' (3 times feature count), max_bins=1024 for mains and 64 for interactions, outer_bags=14 for bagging, learning_rate=0.015, and max_leaves=2 (depth-1 stumps) [1]. After fitting, interpretability is measured via len(model.term_names_) for total terms and counting 2-tuples in model.term_features_ for interaction count [1, 24]. The FAST algorithm underlying EBM's interaction detection [4, 5] works by training main effects first via cyclic gradient boosting, computing residuals, then scoring all O(d^2) feature pairs by fitting shallow trees to residuals on discretized features (O(b^2+n) per pair) [4, 23]. This is fundamentally different from the Co-Information spectral approach: FAST uses greedy residual-based pair ranking with no structural analysis, while the spectral method builds a signed interaction graph enabling spectral clustering of synergistic feature groups [4, 5].

**Benchmark Protocol.** Following Grinsztajn et al. 2022 [7, 8], 5-8 classification-numerical datasets from OpenML Suite 337 are recommended [8, 9]: electricity (~38K samples, 8 features), california (~21K, 8), Higgs (~98K, 28), jannis (~58K, 54), eye_movements (~7.6K, 23), credit (~17K, 10), MiniBooNE (~73K, 50), and heloc (~10K, 22). Dataset selection criteria require d/n less than 1/10, d less than 500, no missing data, heterogeneous columns, and real-world origin [7]. Training sets are truncated to 10,000 samples per the benchmark protocol. Evaluation uses 5-fold stratified cross-validation with random search hyperparameter tuning. Metrics include balanced accuracy (primary) and AUC-ROC (secondary for binary classification) [7]. Datasets can be loaded via OpenML API (openml.study.get_suite(337)) or HuggingFace (load_dataset('inria-soda/tabular-benchmark', config_name)) [8, 9].

**Interpretability Metrics.** Beyond accuracy, the protocol measures: total number of splits (FIGS: model.complexity_) [16, 18], average split arity (features per split: 1 for axis-aligned, greater than 1 for oblique) [20, 21], average path length, number of trees (len(model.trees_) for FIGS) [18], and for EBM: number of terms and interaction terms [1]. Recent evidence challenges the strict performance-interpretability tradeoff, showing EBM matches or exceeds black-box models while remaining fully interpretable [19]. The split arity metric is particularly important for oblique trees: bivariate splits (arity=2) retain reasonable interpretability while being competitive with full multivariate splits [21]. FIGS uses max_rules (default=12) to cap total splits, and complexity_ tracks the actual count [16, 18].

**Statistical Testing.** Following Demsar 2006 [10] and Garcia and Herrera 2008 [11], k=6 methods across N=5-8 datasets are compared using the Friedman test followed by Nemenyi post-hoc test. The autorank library [12] automates test selection: Shapiro-Wilk normality check, then Friedman + Nemenyi for non-normal multi-population comparisons. Critical difference diagrams are generated via scikit-posthocs [13, 14] or autorank's plot_stats() [12]. Kendall's W effect size (W = chi2/(N*(K-1)), thresholds: 0.1 small, 0.3 moderate, 0.5 large) is reported alongside p-values [25]. For focused pairwise comparisons, paired Wilcoxon tests with Holm correction are used [11]. The Bayesian signed-rank test with ROPE provides a complementary analysis capable of declaring practical equivalence [15].

**Complete Protocol.** Six methods are compared: (i) Axis-aligned FIGS (max_rules grid [4,8,12,16,20]), (ii) Random-oblique FIGS, (iii) Hard-threshold SG-FIGS, (iv) Unsigned spectral FIGS, (v) Signed spectral FIGS (ours), (vi) EBM (default settings) [1, 16]. The evaluation checklist covers: balanced accuracy, AUC-ROC, total splits, split arity, path length, training time, EBM term counts, frustration index, and frustration-vs-accuracy-gain correlation. Reporting includes accuracy tables (mean +/- std), interpretability metric tables, CD diagrams, frustration index scatter plots, and full statistical test results with Friedman p-values, Kendall's W, and pairwise Nemenyi p-values [10, 12, 14].

## Sources

[1] [ExplainableBoostingClassifier API - InterpretML Documentation](https://interpret.ml/docs/python/api/ExplainableBoostingClassifier.html) — Complete constructor parameters (interactions='3x', max_bins=1024, outer_bags=14, learning_rate=0.015, max_leaves=2), methods (fit, predict, explain_global, explain_local, term_importances, eval_terms), and fitted attributes (term_features_, term_names_, term_scores_, intercept_) for EBM classifier.

[2] [ExplainableBoostingRegressor API - InterpretML Documentation](https://interpret.ml/docs/python/api/ExplainableBoostingRegressor.html) — Regressor variant parameters showing key differences from classifier: interactions='5x', learning_rate=0.04, smoothing_rounds=500, objective='rmse'.

[3] [Explainable Boosting Machine - InterpretML Documentation](https://interpret.ml/docs/ebm.html) — Conceptual overview of EBM training: cyclic gradient boosting, round-robin feature training, automatic interaction detection, and fast C++/Python implementation.

[4] [Accurate Intelligible Models with Pairwise Interactions (Lou et al., KDD 2013)](https://dl.acm.org/doi/10.1145/2487575.2487579) — Original GA2M/FAST algorithm paper. FAST ranks O(d^2) pairs via residual-based shallow tree scoring on discretized features. 3-4 orders of magnitude faster than ANOVA/Grove methods.

[5] [Microsoft Research - Accurate Intelligible Models with Pairwise Interactions](https://www.microsoft.com/en-us/research/publication/accurate-intelligible-models-pairwise-interactions/) — FAST algorithm overview: GA2M models with pairwise interactions achieve near full-complexity model accuracy while remaining intelligible.

[6] [InterpretML GitHub Repository](https://github.com/interpretml/interpret) — EBM implementation with FAST integration, greedy/smoothing defaults added in v0.5.1 (Feb 2024). FAST runs on each outer_bag and aggregates rankings.

[7] [Why do tree-based models still outperform deep learning on tabular data? (Grinsztajn et al., NeurIPS 2022)](https://arxiv.org/abs/2207.08815) — Benchmark paper: 45 datasets, 4 OpenML suites, ~20K compute hours random search per learner, training truncated to 10K samples, Gaussianization preprocessing.

[8] [Tabular Benchmark GitHub Repository](https://github.com/LeoGrin/tabular-benchmark) — Dataset loading via OpenML suites 334-337, scikit-learn API integration for custom benchmarks, WandB sweep infrastructure.

[9] [inria-soda/tabular-benchmark - Hugging Face Datasets](https://huggingface.co/datasets/inria-soda/tabular-benchmark) — 60+ datasets in 4 categories (clf_num, clf_cat, reg_num, reg_cat). Parquet format with HuggingFace datasets library loading.

[10] [Statistical Comparisons of Classifiers over Multiple Data Sets (Demsar, JMLR 2006)](https://jmlr.org/papers/volume7/demsar06a/demsar06a.pdf) — Foundational paper: Wilcoxon signed-rank for 2 classifiers, Friedman + Nemenyi post-hoc for 3+ classifiers across multiple datasets. CD diagrams.

[11] [An Extension on Statistical Comparisons of Classifiers (Garcia & Herrera, JMLR 2008)](https://www.jmlr.org/papers/volume9/garcia08a/garcia08a.pdf) — More powerful post-hoc corrections: Holm, Hochberg, Hommel, Shaffer, Bergmann-Hommel. Holm-corrected Wilcoxon for control comparisons.

[12] [Autorank - Automated Statistical Testing](https://sherbold.github.io/autorank/) — Python library auto-selecting tests via Shapiro-Wilk normality check. Functions: autorank(), plot_stats(), create_report(), latex_table(). Supports frequentist and Bayesian approaches.

[13] [scikit-posthocs posthoc_nemenyi_friedman Documentation](https://scikit-posthocs.readthedocs.io/en/latest/generated/scikit_posthocs.posthoc_nemenyi_friedman.html) — Function signature: posthoc_nemenyi_friedman(a, y_col, group_col, block_col, melted). Returns p-value DataFrame for pairwise comparisons.

[14] [scikit-posthocs Tutorial](https://scikit-posthocs.readthedocs.io/en/latest/tutorial.html) — End-to-end workflow: Friedman test via scipy, Nemenyi post-hoc, critical_difference_diagram() visualization with customizable styling.

[15] [Time for a Change: Comparing Multiple Classifiers Through Bayesian Analysis (Benavoli et al., JMLR 2017)](https://www.jmlr.org/papers/volume18/16-305/16-305.pdf) — Bayesian signed-rank test with ROPE. Outputs three probabilities (A better, equivalent, B better). More conservative than NHST; can declare practical equivalence.

[16] [FIGS Documentation - imodels](https://csinva.io/imodels/figs.html) — FIGS algorithm: extends CART to grow sum of trees simultaneously. max_rules caps total splits for interpretability. Limits of 2-16 splits recommended.

[17] [FIGS: Attaining XGBoost-level performance with interpretability (BAIR Blog)](https://bair.berkeley.edu/blog/2022/06/30/figs/) — FIGS grows trees in competition, number/shape emerge from data. Predicts well with very few splits. Bagging-FIGS competitive with XGBoost/RF.

[18] [FIGS Source Code - imodels](https://github.com/csinva/imodels/blob/master/imodels/tree/figs.py) — Constructor: max_rules=12, max_trees, min_impurity_decrease=0.0, max_features, max_depth. Fitted: trees_ (root nodes), complexity_ (total splits count), feature_importances_.

[19] [Challenging the Performance-Interpretability Trade-Off (Springer BISE, 2024)](https://link.springer.com/article/10.1007/s12599-024-00922-2) — EBM, IGANN, GAMI-Net achieve high performance while fully interpretable. EBM surpasses most black-box models on average for tabular data.

[20] [LHT: Statistically-Driven Oblique Decision Trees for Interpretable Classification](https://arxiv.org/html/2505.04139v1) — Oblique tree with explicit feature contributions at each split node. Interpretability from statistical framework.

[21] [Optimal Bivariate Splits for Oblique Decision Trees (Applied Intelligence)](https://link.springer.com/article/10.1007/s10489-021-02281-x) — Bivariate oblique splits (arity=2) as interpretability middle ground. Competitive with full multivariate while remaining fairly interpretable.

[22] [CD-Diagram: Critical Difference Diagrams with Wilcoxon-Holm](https://github.com/hfawaz/cd-diagram) — Python implementation for CD diagrams. Runs Friedman test then Wilcoxon-Holm post-hoc analysis.

[23] [Explainable Boosting Machines - Emergent Mind](https://www.emergentmind.com/topics/explainable-boosting-machines) — EBM overview: cyclic boosting on individual features, FAST interaction scoring via pseudo-residuals, greedy/smoothing enabled by default since v0.5.1.

[24] [How are feature importances calculated in EBM? - GitHub Issue](https://github.com/interpretml/interpret/issues/263) — term_importances computes average of absolute predicted values per feature across training set.

[25] [Friedman Test Effect Size (Kendall's W) - rstatix](https://rpkgs.datanovia.com/rstatix/reference/friedman_effsize.html) — Kendall's W formula: W = chi2/(N*(K-1)). Cohen's thresholds: 0.1 small, 0.3 moderate, 0.5 large. Report alongside Friedman p-value.

## Follow-up Questions

- What are the exact dataset sizes (n_samples, n_features, n_classes) for each of the 16 classification-numerical datasets in OpenML Suite 337, and which 5-8 best match the 10K-100K sample range hypothesis?
- Does FIGS support sample_weight for handling imbalanced datasets, and what is the recommended strategy for class imbalance across the benchmark datasets?
- What is the recommended random search budget (number of iterations) for EBM hyperparameter tuning, given that the default configuration already performs well — should EBM be tuned or use defaults only?
- How should the frustration index computation be handled for datasets with many features (d>20) where the pairwise Co-Information matrix becomes expensive to compute?

---
*Generated by AI Inventor Pipeline*

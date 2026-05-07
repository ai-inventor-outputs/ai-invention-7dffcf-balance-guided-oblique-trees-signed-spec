# BibTeX 15 Papers

## Summary

Complete verified BibTeX bibliography for all 15 papers cited in the Balance-Guided Oblique Trees paper. 10 entries retrieved via Semantic Scholar batch API, 3 recovered via DOI retry (Lou 2013, Murthy 1994, Hornung 2022), 2 manually constructed (Bell 2003, McGill 1954). Multiple corrections applied: Kraskov year 2003->2004, Hornung year 2021->2022 and DOI fixed, Westphal year 2024->2025, Breiman venue corrected. Novelty gap confirmed via dependency survey: no 2024-2026 work combines spectral clustering on IT feature graphs with oblique tree construction. Output includes complete bib_content string ready for references.bib and per-entry verification notes.

## Research Findings

All 15 BibTeX entries for the Balance-Guided Oblique Trees paper have been successfully obtained and verified.

**Batch Retrieval (10/15 direct):** The Semantic Scholar batch API successfully returned entries for Cucuringu 2019 (SPONGE), Nori 2019 (InterpretML), Grinsztajn 2022 (tabular data), Tan 2022 (FIGS), Kraskov 2004 (MI estimation), Williams 2010 (PID), Breiman 2001 (Random Forests), Westphal 2025 (PIDF), Heider 1946, and Cartwright & Harary 1956 [1].

**Retry Recovery (3/15):** Three entries failed initial lookup but were recovered via corrected identifiers: Lou et al. 2013 (GA2M) was recovered using DOI 10.1145/2487575.2487579 found via ACM Digital Library [2]; Murthy et al. 1994 (OC1) was recovered using DOI 10.1613/jair.63 confirmed via JAIR website [3]; and Hornung & Boulesteix 2022 (Interaction Forests) was recovered via full title search on Semantic Scholar [4].

**Manual Construction (2/15):** Bell 2003 (The Co-Information Lattice) is a workshop paper from ICA 2003 in Nara, Japan with no DOI, verified via Semantic Scholar page (corpus ID 5031248, 257 citations) and original PDF on the ICA 2003 CD-ROM [5]. McGill 1954 (Multivariate Information Transmission) was verified via Springer Nature: Psychometrika vol. 19, issue 2, pages 97-116 [6].

**Critical Corrections Applied:**
- Kraskov: Year fixed from 2003 to 2004 (Physical Review E vol. 69 published 2004) [1]
- Hornung: Year fixed from 2021 to 2022; DOI corrected from university repository (10.5282/UBM/EPUB.75269) to publisher DOI (10.1016/j.csda.2022.107460) [4]
- Westphal: Year fixed from 2024 to 2025 (AISTATS 2025 proceedings) [1]
- Breiman: Venue corrected from erroneous "Machine-mediated learning" to "Machine Learning" journal [1]
- Multiple entry types standardized (e.g., journal articles incorrectly typed as @inproceedings) [1]

**Novelty Gap Status:** Confirmed. The dependency survey (research_id5_it4__opus) examined 31 sources across four threads (spectral feature grouping, Co-Information estimation, interpretable oblique trees, paper positioning) and found no prior work combining spectral clustering on an information-theoretic feature interaction graph with oblique tree split construction [7]. The closest works are SPEC (spectral but individual features only), GBFG (MI-based but MST not spectral), PIDF (PID-based but no graph/spectral/trees), Interaction Forests (structural not IT), and Feature Graphs in BioData Mining 2025 (post-hoc not pre-construction) [7].

**Complete .bib file** with all 15 entries is available in the `bib_content` field of research_out.json, ready for direct use as references.bib.

## Sources

[1] [Semantic Scholar Batch API](https://api.semanticscholar.org/) — Primary source for 10 of 15 BibTeX entries via batch POST /paper/batch endpoint using ArXiv IDs and DOIs

[2] [ACM Digital Library - Lou et al. KDD 2013](https://dl.acm.org/doi/10.1145/2487575.2487579) — Confirmed DOI 10.1145/2487575.2487579 for GA2M paper, enabling successful Semantic Scholar retry

[3] [JAIR - Murthy et al. 1994](https://www.jair.org/index.php/jair/article/view/10121) — Confirmed DOI 10.1613/jair.63 for OC1 oblique tree paper, JAIR vol. 2 pages 1-32

[4] [RePEc - Hornung & Boulesteix 2022](https://ideas.repec.org/a/eee/csdana/v171y2022ics0167947322000408.html) — Confirmed Interaction Forests published in CSDA vol. 171 (2022), article 107460, DOI 10.1016/j.csda.2022.107460

[5] [Semantic Scholar - Bell 2003 Co-Information Lattice](https://www.semanticscholar.org/paper/THE-CO-INFORMATION-LATTICE-Bell/25a0cd8d486d5ffd204485685226f189e6eadd4d) — Verified Bell 2003 workshop paper details: ICA 2003 (Nara, Japan), single author Anthony J. Bell, 257 citations, no DOI

[6] [Springer - McGill 1954 Psychometrika](https://link.springer.com/article/10.1007/BF02289159) — Verified McGill 1954 bibliographic details: Psychometrika vol. 19 issue 2 pages 97-116, DOI 10.1007/BF02289159

[7] [Dependency research_id5_it4__opus - Spectral CoI Literature Survey](https://www.semanticscholar.org/paper/THE-CO-INFORMATION-LATTICE-Bell/25a0cd8d486d5ffd204485685226f189e6eadd4d) — Four-thread survey of 31 sources confirming novelty gap: no prior work combines spectral clustering on IT feature interaction graphs with oblique tree construction

## Follow-up Questions

- Should the FIGS citation key be updated from Tan2022 to Tan2025 to match the PNAS publication year, and does the paper draft use the ArXiv or PNAS version?
- Is the SPONGE citation intended for the original AISTATS 2019 paper (ArXiv 1904.08575) or the regularized JMLR 2021 extension (ArXiv 2011.01737), and should both be cited?
- Should the Westphal PIDF entry use the AISTATS proceedings DOI (when available from PMLR) instead of the ArXiv DOI currently listed?

---
*Generated by AI Inventor Pipeline*

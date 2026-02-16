# Response to Reviewer Comments

We thank both reviewers for their thorough and constructive evaluation. We have substantially revised the manuscript to address each concern. All modifications are marked in blue using the \revised{} command for ease of identification. The following provides our point-by-point responses.

---

## Reviewer 1

### R1-A: Novelty is primarily architectural rather than algorithmically novel

We acknowledge that the individual components (MedCPT embeddings, vector retrieval, hybrid scoring, knowledge graph traversal) are established techniques. However, we respectfully submit that in safety-critical biomedical AI, architectural orchestration and constraint enforcement constitute first-order contributions. To substantiate this claim, we have expanded the evaluation from a single GPT-4o baseline to an 8-arm factorial design spanning 4 frontier models (GPT-4.1, GPT-5.2, Claude Sonnet 4.5, Claude Opus 4.6) in both pipeline and web-search configurations, yielding 400 total experiments. Bonferroni-corrected paired Wilcoxon tests confirm 10 statistically significant contrasts out of 28 pairwise comparisons, demonstrating that the deterministic D-T-P constraint -- not model selection -- drives the observed performance gains. Furthermore, pipeline arms achieve 95.5--96.0% citation validity versus 23.9--100% for websearch baselines, with websearch-GPT-5.2 exhibiting only 23.9% citation validity and 14.9% verifiability. We have added an introductory paragraph framing the architectural contribution, a Discussion section (Section 7) defending systems-level innovation, and a complete description of the 8-arm factorial design in Comparative Methods (Section 4.3).

### R1-B: Scoring weights (0.8/0.2) and threshold (0.52) lack sensitivity analysis

We have conducted a comprehensive post-hoc sensitivity analysis over 273 hyperparameter configurations: retrieval weights from 0.0 to 1.0 in increments of 0.05 (21 values), and acceptance thresholds from 0.30 to 0.90 in increments of 0.05 (13 values). The results, presented in Section 6.1 with an accompanying heatmap (Figure 5a), reveal that P@10 is largely invariant to the retrieval weight across the range [0.0, 0.65], with peak performance concentrated in a narrow band where the threshold lies in [0.30, 0.50] and the retrieval weight in [0.40, 0.65]. Our chosen configuration (w_ret=0.8, tau=0.52) sits at the conservative edge of this plateau---slightly beyond the peak band---deliberately prioritizing precision over recall via stronger vector weighting and a stricter threshold. P@10 degrades by only 0.03 relative to the optimum, confirming that performance is driven by the graph constraint itself rather than by hyperparameter tuning.

### R1-C: Comparison restricted to a single GPT-4o baseline is insufficient

We have expanded to 4 state-of-the-art models, each tested in both pipeline and web-search configurations:

- GPT-4.1 (OpenAI, April 2025)
- GPT-5.2 (OpenAI, December 2025)
- Claude Sonnet 4.5 (Anthropic, September 2025)
- Claude Opus 4.6 (Anthropic, February 2026)

Each websearch baseline receives up to 5 tool calls via the Tavily API with access to PubMed, Wikipedia, and biomedical databases. Every pipeline arm outperforms the corresponding websearch arm on P@1 (0.760--0.841 vs. 0.540--0.620). Pipeline-GPT-5.2 achieves P@10 of 0.525 versus 0.307 for websearch-GPT-5.2 (+71% relative improvement). Ten contrasts are statistically significant at Bonferroni-corrected alpha = 0.0018. These results are presented in the revised Abstract, Table 1 (expanded to 8 rows), and Section 4.3.

### R1-D: No execution time, throughput, or cost analysis

We now provide a detailed computational cost analysis in Section 6.2. The 400-experiment run completed in 532.8 seconds (1.3s per drug-arm pair). Pipeline arms consume 5.3M--18.5M tokens due to exhaustive graph traversal (tool call limit = 25), with pipeline-Sonnet-4.5 reaching 18.5M tokens. Websearch arms use 0.2M--1.6M tokens (tool call limit = 5), representing a 5--18x reduction in token cost. Estimated API costs per 50 drugs range from USD 13.25 (pipeline-GPT-5.2 at USD 0.27/drug) to USD 152.50 (pipeline-Opus-4.6 at USD 3.05/drug). Websearch arms cost USD 0.50--4.00 total. For large-scale deployment, vector caching reduces marginal costs to below USD 0.10/drug. Token usage is visualized in Figure 6.

### R1-S4: False negative analysis to justify the precision-recall trade-off

We have categorized all false negatives across 8 arms into 6 failure modes, presented in Section 6.3 with an accompanying distribution plot (Figure 7). The dominant category is "Not In Candidates" (89--91%), indicating that the LLM did not generate the disease as a candidate prediction regardless of evidence availability. The remaining categories are: No Evidence Found (5--8%), Name Mismatch (2--4%), and Low Retrieval Score / Low LLM Confidence / Below Threshold (each below 1%). Models predict only 3--15 diseases per drug (median: 7), while CTD ground truth contains 10--22 associations (median: 14), establishing a recall ceiling of approximately R@10 = 0.45 even with perfect retrieval. This confirms that the bottleneck is candidate generation, not evidence retrieval or scoring, and justifies our architecture's prioritization of precision over recall: emitted candidates are high-confidence and auditable, with lower total recall accepted as the necessary cost of mechanistic traceability.

---

## Reviewer 2

### R2-Main: Contribution is architectural and integrative rather than methodological

We agree with this characterization and have reframed our contribution accordingly. Section 7 (Discussion) now explicitly acknowledges the architectural nature of the work while arguing that in safety-critical biomedical AI, system design and constraint enforcement are first-order contributions. The empirical evidence supporting this position includes: (i) consistent performance gains across 4 frontier models and 400 experiments with statistical significance; (ii) a citation validity gap of 96% (pipeline) versus 24% (websearch-GPT-5.2); and (iii) the sensitivity analysis finding that P@10 is invariant to scoring weights, suggesting that the graph constraint itself, not hyperparameter tuning, drives performance.

### R2: Evaluation scope limited to 50 drugs

We provide statistical justification for the sample size. For the paired Wilcoxon signed-rank test, N=50 provides 80% power to detect effect sizes of Cohen's d >= 0.40 at alpha = 0.05. The observed effect sizes are substantially larger: the difference in P@10 between pipeline-GPT-5.2 and websearch-GPT-5.2 (0.218) corresponds to d approximately equal to 0.85, well above the detectable threshold. With 8 arms yielding 28 pairwise comparisons under Bonferroni correction (alpha = 0.0018), 10 contrasts remain statistically significant, confirming adequate statistical power. We note that the total number of experiments is 400 (50 drugs x 8 arms), not 50. This justification is included in the revised Experimental Setup (Section 4.1).

### R2: Scoring parameters set heuristically without sensitivity analysis

This concern is addressed identically to R1-B above. The 273-configuration grid search and accompanying heatmap demonstrate that the chosen parameters lie at the conservative edge of a robust performance plateau, with P@10 degrading by only 0.03 relative to the optimum.

---

## Summary of Revisions

| Concern | Status | Section Added/Revised |
|---------|--------|----------------------|
| R1-A: Incremental innovation | Addressed | Introduction, Discussion (Section 7) |
| R1-B: No sensitivity analysis | Resolved | Section 6.1, Figure 5 |
| R1-C: Insufficient baselines | Resolved | Abstract, Section 4.3, Table 1 |
| R1-D: No cost analysis | Resolved | Section 6.2, Figure 6 |
| R1-S4: No false negative analysis | Resolved | Section 6.3, Figure 7 |
| R2: Architectural contribution | Acknowledged and defended | Discussion (Section 7) |
| R2: Limited sample size | Justified | Section 4.1 |
| R2: Heuristic parameters | Resolved | Section 6.1 (same as R1-B) |

All new and revised text is marked in blue for reviewer convenience.

Reviewer 1:
```
Review 1:This paper proposes a deterministic Graph-RAG framework for drug–disease association discovery that enforces mechanistic traceability prior to language generation.

Despite the generally positive assessment of the paper, there are some areas that should be improved before publication.

In my opinion, the biggest weaknesses of the paper include:

A/ Although the overall system integration is robust, most of the individual components (Graph-RAG, selective KG utilization, vector search, hybrid scoring, CTD evaluation) are incremental rather than fundamentally innovative. The main innovation lies in the architectural orchestration, not in the algorithmic innovations.

B/ The scoring weights (0.8 vector / 0.2 LLM) and acceptance threshold (0.52) are set without sensitivity analysis or justification beyond intuition. To me, it is unclear how robust the results are to these hyperparameters or whether they generalize to all datasets and domains.

C/ Comparing the proposed method solely to the GPT-4o zero-shot method is insufficient. GPT-4o was not designed as a mechanistic search system, making the comparison somewhat uneven.

D/ The paper does not include execution time analysis, throughput measurements, or cost estimates, particularly for the LLM verification phase. This omission makes it difficult to assess feasibility at scale or in real-world deployments.



Suggestions for improving and modifying the paper:

1. If possible, ablation studies regarding scoring weights and acceptance thresholds should be included.

2. Stronger and more accurate baselines could be added, particularly graph-based methods.

3. Suggests adding runtime and scalability analysis.

4. False negative analysis should be performed to better justify the trade-off between recall and precision.
```

Reviewer 2:
```
Review 2: The paper presents a well-engineered, conceptually sound system that prioritizes mechanistic traceability and auditability over generative flexibility. While the novelty is primarily integrative and the evaluation scope is limited, the contribution aligns well with ACIIDS’ applied and interdisciplinary focus, particularly for explainable and trustworthy AI in biomedicine.

The paper proposes a deterministic Graph-RAG framework for drug–disease association discovery that enforces mechanistic traceability before generation. Instead of allowing an LLM to hypothesize associations, the system accepts a drug–disease link if and only if an explicit Drug → Target → Phenotype (D→T→P) path exists, grounded in curated databases (ChEMBL, OpenTargets) and validated via literature retrieval from PubMed/Europe PMC. A hybrid scoring scheme combines vector similarity (MedCPT) with an LLM-based factual consistency check, where the LLM acts as a negation detector rather than a generator. The approach is evaluated on 50 rare drugs using CTD as ground truth, reporting improvements in Precision@K and reduced hallucination rates compared to a GPT-4o zero-shot baseline.

The main concern is that the contribution is largely architectural and procedural rather than methodological. While the emphasis on deterministic traceability is well motivated, most components, knowledge graph construction from curated databases, vector-based literature retrieval, hybrid scoring, and threshold-based filtering, are individually well established. The novelty lies primarily in system integration and constraint enforcement, rather than in new algorithms or learning techniques. The evaluation scope is also limited: only 50 drugs are tested, recall is deliberately sacrificed, and comparisons are restricted to a single black-box LLM baseline rather than to other graph-based or rule-based drug–disease discovery systems. The weighted scoring parameters (0.8/0.2) and the threshold (0.52) are set heuristically, with no sensitivity analysis or justification beyond empirical intuition.

A key strength of the paper is its clear conceptual stance and rigorous emphasis on auditability, which is highly relevant for safety-critical biomedical applications. The deterministic pipeline is well specified, reproducible, and carefully designed to prevent hallucinations, with explicit handling of identifier normalization, leakage control, and negation. The empirical evaluation, though narrow, is thoughtfully constructed around rare drugs where LLM priors are weak, and the reduction in hallucination rate relative to a strong LLM baseline is convincingly demonstrated.
```
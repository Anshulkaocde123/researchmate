# QASPER Gold Evidence to Chunk Mapping Report

> Generated: 2026-10-03 16:55:40 UTC
> Corpus: 34,280 chunks generated across 1,585 papers.

## 1. Evidence Mapping Performance

| Metric | Count / Value |
|---|---:|
| **Evaluation Questions** | 200 |
| **Total Evidence Strings** | 598 |
| **Successfully Grounded Evidence** | 498 |
| **Overall Match Rate** | **83.3%** |
| **Unmatched Evidence Strings** | 100 |

## 2. Root Cause Analysis of Unmatched Evidence

In QASPER, annotators had the option to highlight float objects (tables and figures) as evidence using the annotation tool. These appear in raw annotations prefixed with `FLOAT SELECTED: Table ...` or `FLOAT SELECTED: Figure ...`. Because ResearchMate's paragraph chunker specifically extracts textual body paragraphs, ungrounded table and figure captions remain unmatched as expected.

## 3. Sample Unmatched Evidence Examples

### Example 1
- **Paper ID:** `1906.10225`
- **Question:** *"what were the evaluation metrics?"*
- **Reason:** `figure_or_table_evidence`
- **Evidence Text:**
  > FLOAT SELECTED: Table 3: Results from training RNNGs on induced trees from various models (Induced RNNG) on the PTB. Induced URNNG indicates fine-tuning with the URNNG objective. We show perplexity (PPL), grammaticality judgment performance (Syntactic Eval.), and unlabeled F1. PPL/F1 are calculated on the PTB test set and Syntactic Eval. is from Marvin and Linzen (2018)’s dataset. Results on top do not make any use of annotated trees, while the bottom two results are trained on binarized gold trees. The perplexity numbers here are not comparable to standard results on the PTB since our models are generative model of sentences and hence we do not carry information across sentence boundaries. Also note that all the RNN-based models above (i.e. LSTM/PRPN/ON/RNNG/URNNG) have roughly the same model capacity (see appendix A.3).

### Example 2
- **Paper ID:** `1906.10225`
- **Question:** *"what were the evaluation metrics?"*
- **Reason:** `figure_or_table_evidence`
- **Evidence Text:**
  > FLOAT SELECTED: Table 1: Unlabeled sentence-level F1 scores on PTB and CTB test sets. Top shows results from previous work while the rest of the results are from this paper. Mean/Max scores are obtained from 4 runs of each model with different random seeds. Oracle is the maximum score obtainable with binarized trees, since we compare against the non-binarized gold trees per convention. Results with † are trained on a version of PTB with punctuation, and hence not strictly comparable to the present work. For URNNG/DIORA, we take the parsed test set provided by the authors from their best runs and evaluate F1 with our evaluation setup, which ignores punctuation. ough hyperparameter search.13 See appendix A.2 for the full results (including corpus-level F1) broken down by sentence length.

### Example 3
- **Paper ID:** `1906.10225`
- **Question:** *"what were the evaluation metrics?"*
- **Reason:** `figure_or_table_evidence`
- **Evidence Text:**
  > FLOAT SELECTED: Table 2: (Top) Mean F1 similarity against Gold, Left, Right, and Self trees. Self F1 score is calculated by averaging over all 6 pairs obtained from 4 different runs. (Bottom) Fraction of ground truth constituents that were predicted as a constituent by the models broken down by label (i.e. label recall).

### Example 4
- **Paper ID:** `1802.00396`
- **Question:** *"Do they use number of votes as an indicator of preference?"*
- **Reason:** `figure_or_table_evidence`
- **Evidence Text:**
  > FLOAT SELECTED: Figure 1: NATO-Warsaw intra-bloc differences in votes and speeches.

### Example 5
- **Paper ID:** `1805.07133`
- **Question:** *"what methods were used to reduce data sparsity effects?"*
- **Reason:** `no_paragraph_match (best_score=0.58)`
- **Evidence Text:**
  > Mix-Source Approach

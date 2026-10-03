# QASPER Dataset Profile & Evaluation Set Report

> Generated: 2026-10-03 16:43:26 UTC
> Scope: Full empirical inspection across train, validation, and test splits.

## 1. High-Level Split Statistics

| Split | Papers | Paragraphs | Questions | Annotations | Empty Papers |
|---|---:|---:|---:|---:|---:|
| **train** | 888 | 46,882 | 2,593 | 2,675 | 1 (1801.07804) |
| **validation** | 281 | 13,266 | 1,005 | 1,764 | 0 |
| **test** | 416 | 19,817 | 1,451 | 3,554 | 0 |
| **Total** | **1,585** | **79,965** | **5,049** | **7,993** | **1** |

## 2. Paragraph Distribution per Paper

| Split | Mean | Median | Max | Min |
|---|---:|---:|---:|---:|
| train | 52.8 | 45.0 | 526 | 0 |
| validation | 47.2 | 43 | 289 | 7 |
| test | 47.6 | 43.0 | 290 | 6 |

## 3. Paragraph Length in Tokens (whitespace words)

| Split | Mean Tokens | Median Tokens | Max Tokens | Min Tokens |
|---|---:|---:|---:|---:|
| train | 70.3 | 59.0 | 2501 | 1 |
| validation | 73.1 | 63.0 | 1132 | 1 |
| test | 70.5 | 61 | 3624 | 1 |

## 4. Questions per Paper

| Split | Mean Qs | Median Qs | Max Qs | Min Qs |
|---|---:|---:|---:|---:|
| train | 2.9 | 3.0 | 12 | 1 |
| validation | 3.6 | 3 | 12 | 1 |
| test | 3.5 | 3.0 | 10 | 1 |

## 5. Question Type & Answer Breakdown

### Question-Level Categories (Prioritized: Unanswerable > Yes/No > Extractive > Free-form)

| Split | Extractive | Free-form | Unanswerable | Yes/No | Total |
|---|---:|---:|---:|---:|---:|
| train | 1316 (50.8%) | 602 (23.2%) | 278 (10.7%) | 397 (15.3%) | 2,593 |
| validation | 622 (61.9%) | 146 (14.5%) | 135 (13.4%) | 102 (10.1%) | 1,005 |
| test | 877 (60.4%) | 137 (9.4%) | 243 (16.7%) | 194 (13.4%) | 1,451 |

### Annotator-Level Answer Distribution

| Split | Extractive Spans % | Free-form % | Unanswerable % | Yes/No % |
|---|---:|---:|---:|---:|
| train | 51.0% | 23.3% | 10.5% | 15.3% |
| validation | 54.5% | 24.4% | 9.2% | 11.8% |
| test | 51.1% | 24.7% | 10.3% | 13.9% |

## 6. Dataset Integrity & Assertions

- **Duplicate Paper IDs:** 0 across all splits (asserted).
- **Duplicate Question IDs:** 0 across all splits (asserted).
- **Zero-Paragraph Papers in Validation / Test:** 0 (asserted).
- **Zero-Paragraph Papers in Train:** 1 (`1801.07804` — Vietnamese Open Information Extraction, raw source paper had empty text extraction; excluded from chunking).

## 7. Fixed Evaluation Set (Stratified Sample)

- **Destination:** `data/eval/eval_set.jsonl`
- **Sample Size:** 200 questions
- **Random Seed:** 42 (deterministic)
- **SHA-256 Checksum:** `4eb8700b90124644710d85de04f6cc9bfe9a86d69d8f529b1ea70970e0df846a`

| Stratum | Validation Population | Val Share | Sample Quota | Sample Share |
|---|---:|---:|---:|---:|
| **extractive** | 622 | 61.9% | 124 | 62.0% |
| **free_form** | 146 | 14.5% | 29 | 14.5% |
| **unanswerable** | 135 | 13.4% | 27 | 13.5% |
| **yes_no** | 102 | 10.1% | 20 | 10.0% |

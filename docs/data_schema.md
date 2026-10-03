# QASPER Data Schema

> Observed from the real v0.3 release.
> Source: `qasper-train-dev-v0.3.tgz` + `qasper-test-and-evaluator-v0.3.tgz`
> (AllenAI S3 bucket, SHA-256 checksums in `data/raw/MANIFEST.json`)

## Dataset statistics

| Split      | Papers | QA pairs (total annotations) |
|------------|--------|-------------------------------|
| train      | 888    | ~4,900                        |
| validation | 281    | ~1,500                        |
| test       | 416    | ~1,600                        |
| **Total**  | **1,585** | **~7,993 annotations**     |

Each question has **2 independent annotator answers** on average.

## Answer type distribution (all splits, all annotators)

| Type              | Count | Fraction |
|-------------------|-------|----------|
| Extractive spans  | 4,142 | 51 %     |
| Free-form         | 1,931 | 24 %     |
| Yes / No          | 1,110 | 13 %     |
| Unanswerable      |   810 | 10 %     |
| **Total**         | **7,993** | 100 %  |

## Raw file format

The original JSON files are **dict-of-papers**: `{paper_id: {paper fields}}`.
The loader converts these to **JSONL** (`data/raw/<split>.jsonl`) where each
line has `paper_id` injected as the first key.

## Paper object

```
{
  "paper_id": "2002.06424",              # str  — arXiv ID
  "title":    "Multi-task ...",          # str
  "abstract": "Multi-task learning ...", # str
  "full_text": [                         # list of section dicts, one per section
    {
      "section_name": "Introduction",    # str — may be "" for unnamed sections
      "paragraphs":   ["Para 1 ...",     # list[str] — raw paragraphs
                       "Para 2 ..."]
    },
    ...
  ],
  "qas": [                               # list — one entry per question
    {
      "question":       "What is X?",    # str
      "question_id":    "abc123",        # str — unique across dataset
      "nlp_background": "yes",           # str — annotator metadata, safe to ignore
      "topic_background": "yes",
      "paper_read":     "skimmed",
      "search_query":   "...",
      "question_writer": "worker_id_hash",
      "answers": [                       # list — one entry per annotator (usually 2)
        {
          "annotation_id": "hash",       # str
          "worker_id":     "hash",       # str — anonymised
          "answer": {                    # dict — the actual answer
            "unanswerable":     false,   # bool  — true iff no answer in paper
            "yes_no":           null,    # bool | null — "yes"/"no" or null
            "extractive_spans": [],      # list[str] — verbatim quoted spans
            "free_form_answer": "...",   # str — paraphrase answer (may be "")
            "evidence":         ["..."],  # list[str] — supporting paragraphs
            "highlighted_evidence": ["..."]  # list[str] — shorter evidence excerpts
          }
        }
      ]
    }
  ],
  "figures_and_tables": [               # list — usually irrelevant for RAG
    {
      "file":    "figure1.png",          # str
      "caption": "Figure 1: ..."         # str
    }
  ]
}
```

## Answer semantics

An answer is **unanswerable** when `unanswerable == true` (all other fields empty/null).

An answer is **yes/no** when `yes_no` is `true` or `false` (not `null`).

An answer is **extractive** when `extractive_spans` is non-empty.

An answer is **free-form** when `free_form_answer` is non-empty (and none of the above apply).

These are mutually exclusive categories in practice but the field schema allows overlap.

## Evidence field notes

- `evidence` contains full paragraphs from the paper verbatim (used for RAG grounding).
- `highlighted_evidence` contains shorter excerpts — may differ from `evidence`.
- Evidence strings often contain citation tokens like `BIBREF0`, `BIBREF1` and
  figure/table placeholders like `FLOAT SELECTED` (handled in Step 1.3 cleaning).
- Evidence for figure/table questions often matches a `figures_and_tables` caption,
  not a text paragraph — these will show up as unmatched during evidence mapping.

## Key observations for the pipeline

1. **`full_text` is a list**, not a dict — iterate with `for section in paper["full_text"]`.
2. **Section names can be empty strings** (`""`) — treat them as `"(no section)"`.
3. **Questions are paper-scoped** — every question is about exactly one paper,
   so scoped retrieval (search within the paper) is the primary evaluation mode.
4. **Multiple annotators** — take the union of evidence across annotators as gold.
5. **`evidence` contains the RAG gold standard** — map these strings to chunks in Step 2.2.

## Files produced

| File | Description |
|------|-------------|
| `data/raw/train.jsonl` | 888 papers as JSONL |
| `data/raw/validation.jsonl` | 281 papers as JSONL |
| `data/raw/test.jsonl` | 416 papers as JSONL |
| `data/raw/MANIFEST.json` | Source, date, row counts, sizes, SHA-256 |

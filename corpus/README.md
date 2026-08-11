# Corpus provenance

Everything here is **fiction** — invented parties, courts, statutes and facts. No real
client data, and nothing from Legixo production systems.

| Files | Origin |
|---|---|
| `01_matter_memo_arvind_v_northfield.md` … `06_property_lease_clause.md` | Supplied in `gen_ai_takehome_sample_corpus.zip`, **unmodified** |
| `07_employment_agreement_vantage.md` … `30_costs_schedule.md` | **Written for this project** (24 files), same fictional style |

The brief permits this: *"You can use this as your whole corpus, or mix in more files in
the same style."*

## Why

Six files chunk to 15 vectors. At `TOP_K=5` that returns a third of the corpus per query,
so retrieval recall was trivially perfect and neither reranking nor hybrid search had
anything to improve. Thirty files chunk to 93, a query sees ~5%, and retrieval becomes a
real problem worth measuring.

## The additions are adversarial on purpose

Three employment agreements carry **different** notice periods (60 / 30 / 90 days), three
leases carry different units and deposits, and several duration facts compete across
documents. Answering correctly therefore requires selecting the right *document*, not
merely the right topic.

Cross-references are kept internally consistent — `CV-2024-8812` appears in six files with
the same parties and dates throughout.

Full rationale: [`../README.md#corpus-provenance`](../README.md#corpus-provenance).

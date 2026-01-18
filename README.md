# RAG Research

Metadata-Filtered RAG pipeline for internal documentation Q&A.

## Quick Start

```bash
# Install dependencies
poetry install

# Run the RAG pipeline
poetry run python -m rag_research.rag_pipeline

# Run evaluation
poetry run python -m rag_research.eval
```

## Architecture

```
Query → Vector Search (with category filter) → LLM Generation → Answer
```

| Component  | File              | Description                                     |
| ---------- | ----------------- | ----------------------------------------------- |
| Ingestion  | `chunker.py`      | Sliding window chunking (400 chars, 80 overlap) |
| Retrieval  | `vector_store.py` | Word2Vec embeddings + cosine similarity         |
| Generation | `rag_pipeline.py` | Azure OpenAI GPT-4o-mini                        |
| Evaluation | `eval.py`         | Precision@K metric                              |

## RAG Pattern

**Metadata-Filtered RAG** — Filter by category before vector search.

Categories: `root`, `docs`, `tools`, `apps`, `api`

## Sample Output

```
>>> Query: What is Global Billing?
>>> Category: apps

[RETRIEVAL] Retrieved 5 chunks:
    1. [apps] Billing.md (sim=0.796)
    2. [apps] integration_plan_global.md (sim=0.784)
    3. [apps] billing_backup_BILLING_STRATEGY.md (sim=0.757)

ANSWER:
Global Billing is a comprehensive system designed to streamline
third-party vendor billing by automatically synchronizing customer
data and usage information from multiple security vendors. It enables
to efficiently track, manage, and bill for third-party services
across their customer base.

SOURCES: 5 chunks from apps/
```

## Evaluation Results

```
Precision@5 Comparison
──────────────────────────────────────
Unfiltered:  49%
Filtered:   100%
Improvement: +51%
```

| Category | Unfiltered | Filtered |
| -------- | ---------- | -------- |
| api      | 20%        | 100%     |
| apps     | 80%        | 100%     |
| docs     | 80%        | 100%     |
| root     | 47%        | 100%     |
| tools    | 10%        | 100%     |

## Environment

Requires `.env` with:

```
AZURE_OPENAI_ENDPOINT=https://...
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_DEPLOYMENT=gpt-4o-mini
```

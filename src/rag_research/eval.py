"""
RAG Evaluation - Precision@K for Metadata-Filtered RAG

================================================================================
COURSE TOPICS COVERED:
    - Office Hours 1: "Precision@K" — measuring retrieval quality
    - RAG Lecture Session 2: "Evaluation" — retrieval-based metrics
    - Key Quote: "Eval must measure retrieval behavior, not just LLM response"
================================================================================

WHAT IS PRECISION@K?
Precision@K measures what fraction of the top-K retrieved documents are relevant.

    Precision@K = (# relevant documents in top K) / K

For our metadata-filtered RAG, we define "relevant" as:
    - A chunk is relevant if it comes from the EXPECTED CATEGORY for that query

EXAMPLE:
Query: "How does billing work?"
Expected category: "apps"
Retrieved chunks (K=5):
    1. [apps] billing_strategy.md     ← relevant
    2. [apps] universal_billing.md    ← relevant
    3. [tools] kingfisher.md          ← NOT relevant
    4. [apps] recurly.md              ← relevant
    5. [docs] patterns.md             ← NOT relevant

Precision@5 = 3/5 = 0.60 (60%)

WHY PRECISION@K?
- Measures retrieval quality directly (not LLM quality)
- Simple to interpret: "What % of retrieved docs are useful?"
- Works well with category-based relevance
- Recommended in RAG_USECASES.md for this assignment
"""

from pathlib import Path
from dataclasses import dataclass

from .chunker import chunk_corpus
from .vector_store import VectorIndex, train_embedding_model


# =============================================================================
# EVALUATION DATASET
# =============================================================================
# Test queries with expected categories (from RAG_USECASES.md)
# These represent real questions users might ask about Vengeful Spirit Inc.

EVAL_QUERIES = [
    # Root category - setup and configuration
    {"query": "How do I set up my dev environment?", "expected_category": "root"},
    {"query": "How do I connect to Kabuto?", "expected_category": "root"},
    {"query": "What are the security guidelines?", "expected_category": "root"},
    # Docs category - patterns, templates, learnings
    {"query": "What's the service pattern?", "expected_category": "docs"},
    {"query": "How do I create a PR?", "expected_category": "docs"},
    {"query": "What's the bug fix template?", "expected_category": "docs"},
    {"query": "What is the feature template workflow?", "expected_category": "docs"},
    # Tools category - Kingfisher, data sync, testing
    {"query": "How does Kingfisher sync data?", "expected_category": "tools"},
    {"query": "What's the data isolation pattern?", "expected_category": "tools"},
    {"query": "How do I run tests?", "expected_category": "tools"},
    {"query": "What is the Bruno API tool?", "expected_category": "tools"},
    # Apps category - billing, integrations, services
    {"query": "How does Ironscales billing work?", "expected_category": "apps"},
    {"query": "What's the CSP onboarding flow?", "expected_category": "apps"},
    {"query": "How does the cloud backup worker run?", "expected_category": "apps"},
    {"query": "What is Universal Billing?", "expected_category": "apps"},
    {"query": "How does M365 integration work?", "expected_category": "apps"},
    # API category - REST endpoints
    {"query": "How do I create a ticket via API?", "expected_category": "api"},
    {"query": "What's the endpoint for customer assets?", "expected_category": "api"},
    {
        "query": "How do I add a comment to a ticket via API?",
        "expected_category": "api",
    },
    {
        "query": "What API endpoints are available for invoices?",
        "expected_category": "api",
    },
]


# =============================================================================
# EVALUATION METRICS
# =============================================================================


@dataclass
class EvalResult:
    """Result of evaluating a single query."""

    query: str
    expected_category: str
    retrieved_categories: list[str]
    relevant_count: int
    k: int
    precision_at_k: float

    def __str__(self):
        status = "✓" if self.precision_at_k >= 0.6 else "✗"
        return (
            f"{status} P@{self.k}={self.precision_at_k:.0%} | "
            f"Expected: {self.expected_category} | "
            f"Got: {self.retrieved_categories[:3]}"
        )


def precision_at_k(
    retrieved_chunks: list[dict], expected_category: str, k: int
) -> EvalResult:
    """
    Calculate Precision@K for a single query.

    Args:
        retrieved_chunks: List of retrieved chunk dicts with 'category' key
        expected_category: The category we expect relevant docs to come from
        k: Number of documents to consider

    Returns:
        EvalResult with precision score and details
    """
    # Get top-k chunks
    top_k_chunks = retrieved_chunks[:k]

    # Count how many are from the expected category
    retrieved_categories = [chunk["category"] for chunk in top_k_chunks]
    relevant_count = sum(1 for cat in retrieved_categories if cat == expected_category)

    # Calculate precision
    precision = relevant_count / k if k > 0 else 0.0

    return EvalResult(
        query="",  # Will be filled in by caller
        expected_category=expected_category,
        retrieved_categories=retrieved_categories,
        relevant_count=relevant_count,
        k=k,
        precision_at_k=precision,
    )


def evaluate_retrieval(
    index: VectorIndex,
    eval_queries: list[dict] = None,
    k: int = 5,
    use_filter: bool = False,
    verbose: bool = True,
) -> dict:
    """
    Run Precision@K evaluation on the retrieval system.

    Args:
        index: VectorIndex to evaluate
        eval_queries: List of {"query": str, "expected_category": str} dicts
        k: Number of documents to retrieve and evaluate
        use_filter: If True, use category_filter (tests filtering accuracy)
                   If False, no filter (tests unfiltered retrieval accuracy)
        verbose: Print detailed results

    Returns:
        dict with:
            - mean_precision: Average Precision@K across all queries
            - results: List of EvalResult for each query
            - by_category: Precision broken down by expected category
    """
    if eval_queries is None:
        eval_queries = EVAL_QUERIES

    results = []
    category_scores = {}

    if verbose:
        filter_mode = (
            "WITH category filter" if use_filter else "WITHOUT category filter"
        )
        print(f"\n{'=' * 70}")
        print(f"PRECISION@{k} EVALUATION ({filter_mode})")
        print(f"{'=' * 70}")
        print(f"Evaluating {len(eval_queries)} queries...\n")

    for eval_item in eval_queries:
        query = eval_item["query"]
        expected_category = eval_item["expected_category"]

        # Retrieve chunks (with or without category filter)
        if use_filter:
            retrieved = index.search(query, k=k, category_filter=expected_category)
        else:
            retrieved = index.search(query, k=k, category_filter=None)

        # Calculate precision
        result = precision_at_k(retrieved, expected_category, k)
        result.query = query
        results.append(result)

        # Track by category
        if expected_category not in category_scores:
            category_scores[expected_category] = []
        category_scores[expected_category].append(result.precision_at_k)

        if verbose:
            print(f"  {result}")

    # Calculate aggregates
    mean_precision = sum(r.precision_at_k for r in results) / len(results)
    by_category = {
        cat: sum(scores) / len(scores) for cat, scores in category_scores.items()
    }

    if verbose:
        print(f"\n{'=' * 70}")
        print("RESULTS SUMMARY")
        print(f"{'=' * 70}")
        print(f"\nOverall Mean Precision@{k}: {mean_precision:.1%}")
        print("\nPrecision by Category:")
        for cat, score in sorted(by_category.items()):
            bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
            print(f"  {cat:8} [{bar}] {score:.1%}")

        # Count passes/fails (threshold: 60%)
        passes = sum(1 for r in results if r.precision_at_k >= 0.6)
        print(f"\nQueries with P@{k} >= 60%: {passes}/{len(results)}")

    return {
        "mean_precision": mean_precision,
        "k": k,
        "num_queries": len(results),
        "results": results,
        "by_category": by_category,
        "use_filter": use_filter,
    }


def compare_filtered_vs_unfiltered(index: VectorIndex, k: int = 5) -> dict:
    """
    Compare retrieval precision with and without category filtering.

    This demonstrates the value of metadata-filtered RAG:
    - Without filter: Retrieval must find correct category on its own
    - With filter: Retrieval is constrained to correct category

    Returns:
        dict with comparison results
    """
    print("\n" + "=" * 70)
    print("COMPARISON: Filtered vs Unfiltered Retrieval")
    print("=" * 70)

    # Evaluate without filter
    unfiltered = evaluate_retrieval(index, k=k, use_filter=False, verbose=False)

    # Evaluate with filter
    filtered = evaluate_retrieval(index, k=k, use_filter=True, verbose=False)

    print(f"\n{'Metric':<30} {'Unfiltered':>15} {'Filtered':>15} {'Δ':>10}")
    print("-" * 70)
    print(
        f"{'Mean Precision@' + str(k):<30} {unfiltered['mean_precision']:>14.1%} {filtered['mean_precision']:>14.1%} {filtered['mean_precision'] - unfiltered['mean_precision']:>+9.1%}"
    )

    print("\nBy Category:")
    for cat in sorted(unfiltered["by_category"].keys()):
        uf = unfiltered["by_category"].get(cat, 0)
        f = filtered["by_category"].get(cat, 0)
        delta = f - uf
        print(f"  {cat:<28} {uf:>14.1%} {f:>14.1%} {delta:>+9.1%}")

    return {
        "unfiltered": unfiltered,
        "filtered": filtered,
        "improvement": filtered["mean_precision"] - unfiltered["mean_precision"],
    }


# =============================================================================
# MAIN: Run evaluation
# =============================================================================


def main():
    """Build index and run full evaluation suite."""
    project_root = Path(__file__).parent.parent.parent
    corpus_dir = project_root / "data" / "syncro_corpus"
    metadata_path = corpus_dir / "metadata.json"

    print("=" * 70)
    print("VENGEFUL SPIRIT INC. RAG EVALUATION")
    print("=" * 70)
    print("Metric: Precision@K (retrieval-based)")
    print("Relevance: Chunk is relevant if from expected category")

    # ---------------------------------------------------------
    # Step 1: Build the retrieval system
    # ---------------------------------------------------------
    print("\n[1/3] Loading corpus and building index...")
    chunks = chunk_corpus(corpus_dir, metadata_path)
    print(f"Total chunks: {len(chunks)}")

    print("\n[2/3] Training embeddings...")
    model = train_embedding_model(chunks)

    print("\n[3/3] Building vector index...")
    index = VectorIndex(model, chunks)

    # ---------------------------------------------------------
    # Step 2: Run evaluations
    # ---------------------------------------------------------

    # Evaluation 1: Unfiltered retrieval (harder)
    print("\n" + "=" * 70)
    print("EVALUATION 1: Unfiltered Retrieval")
    print("=" * 70)
    print("Testing if retrieval naturally finds the correct category...")
    unfiltered_results = evaluate_retrieval(index, k=5, use_filter=False)

    # Evaluation 2: Filtered retrieval (demonstrates metadata-filtered RAG)
    print("\n" + "=" * 70)
    print("EVALUATION 2: Filtered Retrieval (Metadata-Filtered RAG)")
    print("=" * 70)
    print("Testing retrieval quality within the correct category...")
    filtered_results = evaluate_retrieval(index, k=5, use_filter=True)

    # Evaluation 3: Comparison
    comparison = compare_filtered_vs_unfiltered(index, k=5)

    # ---------------------------------------------------------
    # Final Summary
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("FINAL EVALUATION SUMMARY")
    print("=" * 70)
    print(f"""
RAG Pattern: Metadata-Filtered RAG
Eval Metric: Precision@5
Test Queries: {len(EVAL_QUERIES)}
Categories: root, docs, tools, apps, api

Results:
  • Unfiltered Precision@5: {unfiltered_results["mean_precision"]:.1%}
  • Filtered Precision@5:   {filtered_results["mean_precision"]:.1%}
  • Improvement from filtering: {comparison["improvement"]:+.1%}

Interpretation:
  - Unfiltered: Tests if embeddings capture category semantics
  - Filtered: Tests retrieval quality within correct category
  - The improvement shows value of metadata filtering
""")


if __name__ == "__main__":
    main()

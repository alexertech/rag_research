import os
from functools import wraps
from typing import Any, Callable

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Global flag to track if tracing is enabled
_tracing_enabled = False
_langsmith_client = None


def init_tracing() -> bool:
    """
    Initialize LangSmith tracing if configured.

    Returns:
        True if tracing was successfully initialized, False otherwise.

    WHAT THIS DOES:
        1. Checks if LANGCHAIN_API_KEY is set
        2. Verifies LANGCHAIN_TRACING_V2 is "true"
        3. Initializes the LangSmith client
        4. Returns status so caller knows if tracing is active
    """
    global _tracing_enabled, _langsmith_client

    api_key = os.getenv("LANGCHAIN_API_KEY")
    tracing_enabled = os.getenv("LANGCHAIN_TRACING_V2", "").lower() == "true"
    project = os.getenv("LANGCHAIN_PROJECT", "warp_analysis_data")

    if not api_key:
        print("[Observability] LANGCHAIN_API_KEY not set - tracing disabled")
        return False

    if not tracing_enabled:
        print("[Observability] LANGCHAIN_TRACING_V2 not 'true' - tracing disabled")
        return False

    try:
        from langsmith import Client

        _langsmith_client = Client(api_key=api_key)
        _tracing_enabled = True
        print(f"[Observability] LangSmith tracing enabled for project: {project}")
        return True
    except ImportError:
        print("[Observability] langsmith package not installed - tracing disabled")
        return False
    except Exception as e:
        print(f"[Observability] Failed to initialize LangSmith: {e}")
        return False


def is_tracing_enabled() -> bool:
    """Check if tracing is currently enabled."""
    return _tracing_enabled


def trace_retrieval(func: Callable) -> Callable:
    """
    Decorator to trace retrieval operations (LoreIndex.search).

    WHAT IT TRACES:
        - Input query
        - Number of results requested (k)
        - Retrieved chunks with similarity scores
        - Faction distribution of results

    This helps answer: "What did the system retrieve for this query?"
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        if not _tracing_enabled:
            return func(*args, **kwargs)

        try:
            from langsmith import traceable

            @traceable(run_type="retriever", name="lore_search")
            def traced_func(*args, **kwargs):
                return func(*args, **kwargs)

            return traced_func(*args, **kwargs)
        except Exception:
            # Fallback to untraced execution if tracing fails
            return func(*args, **kwargs)

    return wrapper


def trace_prediction(func: Callable) -> Callable:
    """
    Decorator to trace prediction operations (RAGClassifier.predict).

    WHAT IT TRACES:
        - Input text
        - MLP confidence and prediction
        - Whether RAG was activated
        - Lore votes (if RAG used)
        - Combined scores
        - Final prediction

    This helps answer: "Why did the system make this prediction?"
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        if not _tracing_enabled:
            return func(*args, **kwargs)

        try:
            from langsmith import traceable

            @traceable(run_type="chain", name="rag_prediction")
            def traced_func(*args, **kwargs):
                return func(*args, **kwargs)

            return traced_func(*args, **kwargs)
        except Exception:
            return func(*args, **kwargs)

    return wrapper


def log_retrieval(query: str, results: list[dict[str, Any]], k: int, category_filter: str = None) -> None:
    """
    Log a retrieval operation to LangSmith.

    USE THIS FOR:
        - Tracking what chunks are being retrieved
        - Debugging retrieval quality issues
        - Building retrieval evaluation datasets

    Args:
        query: The query text used for retrieval
        results: List of retrieved chunk dicts with similarity scores
        k: Number of results requested
        category_filter: Optional category filter applied
    """
    if not _tracing_enabled or not _langsmith_client:
        return

    try:
        from langsmith.run_helpers import trace

        # Summarize results for logging
        category_counts = {}
        for r in results:
            category = r.get("category", "unknown")
            category_counts[category] = category_counts.get(category, 0) + 1

        with trace(
            name="rag_retrieval",
            run_type="retriever",
            inputs={"query": query, "k": k, "category_filter": category_filter},
            project_name=os.getenv("LANGCHAIN_PROJECT", "rag_research"),
        ) as run:
            run.end(
                outputs={
                    "num_results": len(results),
                    "category_distribution": category_counts,
                    "top_similarity": results[0].get("similarity") if results else None,
                    "results": [
                        {
                            "category": r.get("category"),
                            "source": r.get("source"),
                            "similarity": r.get("similarity"),
                            "text_preview": r.get("text", "")[:100] + "...",
                        }
                        for r in results
                    ],
                }
            )
    except Exception as e:
        print(f"[Observability] Warning: Failed to log retrieval: {e}")


def log_generation(query: str, context: str, answer: str, sources: list[dict], category_filter: str = None) -> None:
    """
    Log a RAG generation operation to LangSmith.

    USE THIS FOR:
        - Tracking full RAG pipeline executions
        - Debugging answer quality issues
        - Monitoring LLM usage

    Args:
        query: The user's question
        context: The formatted context sent to LLM
        answer: The generated answer
        sources: List of source chunks used
        category_filter: Optional category filter applied
    """
    if not _tracing_enabled or not _langsmith_client:
        return

    try:
        from langsmith.run_helpers import trace

        with trace(
            name="rag_generation",
            run_type="llm",
            inputs={
                "query": query,
                "category_filter": category_filter,
                "context_length": len(context),
                "num_sources": len(sources),
            },
            project_name=os.getenv("LANGCHAIN_PROJECT", "rag_research"),
        ) as run:
            run.end(
                outputs={
                    "answer": answer,
                    "answer_length": len(answer),
                    "sources": [
                        {"category": s.get("category"), "source": s.get("source")}
                        for s in sources
                    ],
                }
            )
    except Exception as e:
        print(f"[Observability] Warning: Failed to log generation: {e}")


def create_evaluation_dataset(
    name: str, predictions: list[dict], ground_truth: list[str]
) -> str | None:
    """
    Create a LangSmith dataset from prediction results for evaluation.

    USE THIS FOR:
        - Building test datasets from production traffic
        - Creating golden datasets for regression testing
        - Human annotation workflows

    Args:
        name: Dataset name in LangSmith
        predictions: List of prediction result dicts
        ground_truth: List of correct labels (same order as predictions)

    Returns:
        Dataset ID if successful, None otherwise
    """
    if not _tracing_enabled or not _langsmith_client:
        print("[Observability] Cannot create dataset - tracing not enabled")
        return None

    try:
        # Create dataset
        dataset = _langsmith_client.create_dataset(
            dataset_name=name,
            description="RAG classifier evaluation dataset",
        )

        # Add examples
        for pred, truth in zip(predictions, ground_truth):
            _langsmith_client.create_example(
                inputs={"text": pred.get("text", "")},
                outputs={"expected": truth, "predicted": pred.get("final_prediction")},
                dataset_id=dataset.id,
                metadata={
                    "mlp_confidence": pred.get("mlp_confidence"),
                    "used_rag": pred.get("used_rag"),
                },
            )

        print(
            f"[Observability] Created dataset '{name}' with {len(predictions)} examples"
        )
        return dataset.id

    except Exception as e:
        print(f"[Observability] Failed to create dataset: {e}")
        return None

"""
RAG Pipeline - Retrieval-Augmented Generation for Vengeful Spirit Inc. Documentation

================================================================================
COURSE TOPICS COVERED:
    - RAG Lecture Session 1: Full RAG pipeline (chunking, embedding, retrieval)
    - RAG Lecture Session 2: "Metadata-Filtered RAG" (category filtering)
    - Key Quote: "RAG is just giving the model access to information" — Ash
================================================================================

This module combines:
1. Vector retrieval (from vector_store.py)
2. LLM generation (Azure OpenAI)

The RAG Pattern:
    Query → Retrieve relevant chunks → Feed to LLM → Generate answer

This is different from classification (faction voting). Here we:
- Retrieve context
- Let the LLM synthesize a natural language answer
- Return the answer with source citations
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from openai import AzureOpenAI

from .chunker import chunk_corpus
from .vector_store import VectorIndex, train_embedding_model
from .observability import init_tracing, log_retrieval, log_generation

# Load environment variables
load_dotenv()

# Initialize LangSmith tracing (optional - fails gracefully if not configured)
init_tracing()


class RAGPipeline:
    """
    RAG Pipeline combining vector retrieval with LLM generation.

    ARCHITECTURE:
    ┌─────────────────────────────────────────────────────────────────┐
    │                        RAG PIPELINE                             │
    ├─────────────────────────────────────────────────────────────────┤
    │                                                                 │
    │   Query: "How does billing work?"                               │
    │       │                                                         │
    │       ▼                                                         │
    │   ┌─────────────────┐                                           │
    │   │ VectorIndex     │  ← Retrieval (your existing code)         │
    │   │ .search()       │                                           │
    │   └────────┬────────┘                                           │
    │            │                                                    │
    │            ▼                                                    │
    │   Retrieved chunks: [chunk1, chunk2, chunk3]                    │
    │            │                                                    │
    │            ▼                                                    │
    │   ┌─────────────────┐                                           │
    │   │ Format Context  │  ← Prepare for LLM                        │
    │   └────────┬────────┘                                           │
    │            │                                                    │
    │            ▼                                                    │
    │   ┌─────────────────┐                                           │
    │   │ Azure OpenAI    │  ← Generation (GPT-4o-mini)               │
    │   │ (LLM)           │                                           │
    │   └────────┬────────┘                                           │
    │            │                                                    │
    │            ▼                                                    │
    │   Answer: "Vengeful Spirit billing uses subscription model..."           │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘
    """

    def __init__(self, vector_index: VectorIndex):
        """
        Initialize RAG pipeline with vector index and Azure OpenAI client.

        Args:
            vector_index: VectorIndex instance for retrieval
        """
        self.index = vector_index

        # Initialize Azure OpenAI client
        self.client = AzureOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
        )
        self.deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")

    def generate_answer(
        self,
        query: str,
        category_filter: str = None,
        top_k: int = 5,
        verbose: bool = False,
    ) -> dict:
        """
        Retrieve relevant chunks and generate a natural language answer.

        COURSE TOPIC: RAG Session 1 — "The RAG Pipeline"
        This is the complete retrieve → augment → generate flow.

        Args:
            query: User's question
            category_filter: Optional category to filter retrieval
                            (root, docs, tools, apps, api)
            top_k: Number of chunks to retrieve
            verbose: Print debug information

        Returns:
            dict with:
                - answer: Generated natural language answer
                - sources: List of retrieved chunks used
                - query: Original query
        """
        # =====================================================================
        # STEP 1: RETRIEVE relevant chunks
        # =====================================================================
        retrieved_chunks = self.index.search(
            query=query, k=top_k, category_filter=category_filter
        )

        # Log retrieval to LangSmith (if enabled)
        log_retrieval(query, retrieved_chunks, top_k, category_filter)

        if verbose:
            print(f"\n[RETRIEVAL] Query: {query}")
            if category_filter:
                print(f"[RETRIEVAL] Category filter: {category_filter}")
            print(f"[RETRIEVAL] Retrieved {len(retrieved_chunks)} chunks:")
            for i, chunk in enumerate(retrieved_chunks[:3], 1):
                preview = (
                    chunk["text"][:60] + "..."
                    if len(chunk["text"]) > 60
                    else chunk["text"]
                )
                print(
                    f"    {i}. [{chunk['category']}] {chunk['source']} "
                    f"(sim={chunk['similarity']:.3f})"
                )
                print(f"       {preview}")

        # =====================================================================
        # STEP 2: FORMAT context from retrieved chunks
        # =====================================================================
        context = self._format_context(retrieved_chunks)

        # =====================================================================
        # STEP 3: BUILD prompt for LLM
        # =====================================================================
        prompt = self._build_prompt(query, context)

        if verbose:
            print(f"\n[GENERATION] Sending to {self.deployment}...")

        # =====================================================================
        # STEP 4: GENERATE answer using Azure OpenAI
        # =====================================================================
        answer = self._generate_with_llm(prompt)

        # Log generation to LangSmith (if enabled)
        log_generation(query, context, answer, retrieved_chunks, category_filter)

        if verbose:
            print(f"[GENERATION] Answer generated ({len(answer)} chars)")

        # =====================================================================
        # STEP 5: RETURN structured result
        # =====================================================================
        return {
            "answer": answer,
            "sources": [
                {
                    "text": chunk["text"],
                    "source": chunk["source"],
                    "category": chunk["category"],
                    "similarity": chunk["similarity"],
                }
                for chunk in retrieved_chunks
            ],
            "query": query,
            "category_filter": category_filter,
        }

    def _format_context(self, chunks: list[dict]) -> str:
        """
        Format retrieved chunks into context string for LLM.

        Each chunk is labeled with its source and category for attribution.
        """
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            context_parts.append(
                f"[Document {i} - {chunk['category']}/{chunk['source']}]\n"
                f"{chunk['text']}"
            )
        return "\n\n".join(context_parts)

    def _build_prompt(self, query: str, context: str) -> str:
        """
        Build the prompt sent to the LLM.

        COURSE TOPIC: Prompt Engineering
        - Clear instructions on using ONLY provided context
        - Explicit about saying "I don't know" when context is insufficient
        - Source attribution encouragement
        """
        return f"""You are a helpful assistant answering questions about the Vengeful Spirit Inc. platform based on internal documentation.

INSTRUCTIONS:
1. Use ONLY the information provided in the documentation below to answer the question.
2. If the documentation doesn't contain enough information to fully answer, say so.
3. When possible, mention which document(s) your answer comes from.
4. Be concise and direct.

DOCUMENTATION:
{context}

QUESTION: {query}

ANSWER:"""

    def _generate_with_llm(self, prompt: str) -> str:
        """
        Call Azure OpenAI to generate answer.

        Uses low temperature (0.3) for factual accuracy over creativity.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.deployment,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a helpful assistant that answers questions "
                            "about Vengeful Spirit Inc. based on provided documentation. "
                            "Be accurate and cite your sources when possible."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,  # Low temperature for factual accuracy
                max_tokens=500,  # Reasonable limit for documentation answers
                top_p=0.9,
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            return f"Error generating answer: {str(e)}"


# =============================================================================
# MAIN: Build and test RAG pipeline
# =============================================================================


def main():
    """Build RAG pipeline and test with sample queries."""
    project_root = Path(__file__).parent.parent.parent
    corpus_dir = project_root / "data" / "syncro_corpus"
    metadata_path = corpus_dir / "metadata.json"

    print("=" * 70)
    print("VENGEFUL SPIRIT INC. RAG PIPELINE")
    print("=" * 70)
    print("Pattern: Metadata-Filtered RAG with Azure OpenAI Generation")

    # ---------------------------------------------------------
    # Step 1: Load and chunk corpus
    # ---------------------------------------------------------
    print("\n[1/4] Loading and chunking Vengeful Spirit corpus...")
    chunks = chunk_corpus(corpus_dir, metadata_path)
    print(f"Total chunks: {len(chunks)}")

    # ---------------------------------------------------------
    # Step 2: Train embedding model
    # ---------------------------------------------------------
    print("\n[2/4] Training Word2Vec embedding model...")
    model = train_embedding_model(chunks)

    # ---------------------------------------------------------
    # Step 3: Build vector index
    # ---------------------------------------------------------
    print("\n[3/4] Building vector index...")
    index = VectorIndex(model, chunks)

    # ---------------------------------------------------------
    # Step 4: Create RAG pipeline
    # ---------------------------------------------------------
    print("\n[4/4] Initializing RAG pipeline with Azure OpenAI...")
    rag = RAGPipeline(index)

    # ---------------------------------------------------------
    # Test queries
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("TESTING RAG PIPELINE")
    print("=" * 70)

    test_queries = [
        ("How do I set up my dev environment?", "tools"),
        ("How are background workers structured?", "apps"),
        ("How do I run the test suite?", "tools"),
        ("What's the PR review process?", "docs"),
        ("What are the coding style guidelines?", "root"),
    ]

    for query, category in test_queries:
        print(f"\n{'=' * 70}")
        print(f"QUERY: {query}")
        print(f"CATEGORY FILTER: {category}")
        print("=" * 70)

        result = rag.generate_answer(query, category_filter=category, verbose=True)

        print(f"\nANSWER:\n{result['answer']}")
        print(f"\nSOURCES USED: {len(result['sources'])} chunks")
        for src in result["sources"][:3]:
            print(
                f"  - [{src['category']}] {src['source']} (sim={src['similarity']:.3f})"
            )


if __name__ == "__main__":
    main()

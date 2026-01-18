"""
Vector Store for RAG Pipeline

================================================================================
COURSE TOPICS COVERED:
    - RAG Lecture Session 1: "Embedding Models" — Word2Vec as local alternative
    - RAG Lecture Session 1: "Vector Similarity Search" — Cosine similarity
    - Office Hours 2: "Determinism" — Pure Python nodes for guaranteed computation
    - Key Quote: "LLMs solve language, not math — use tools for everything else" — Aaron
================================================================================

ADAPTATION FROM train_nn.py:
================================================================================
ORIGINAL (train_nn.py)                  THIS FILE (vector_store.py)
--------------------------------------------------------------------------------
Goal: Classification                    Goal: Similarity Search
Input: Short messages (CSV)             Input: Document chunks (lore corpus)
Output: Label (loyalist/heretic)        Output: Similar chunks to query
Training: Word2Vec + MLP Classifier     Training: Word2Vec only (no classifier)
Usage: predict_vox_intercept()          Usage: search()

WHAT WE KEEP:
- Text cleaning (clean_intercept_text)
- Tokenization (tokenize_sentence)
- Word2Vec embedding training
- Average vector calculation

WHAT WE CHANGE:
- No train/test split (we index ALL chunks)
- No MLP classifier (we do similarity search instead)
- Add cosine similarity search
- Add vector index storage

WHAT WE'RE BUILDING:
A searchable index where:
1. Each chunk gets embedded as a vector
2. Queries get embedded the same way
3. We find chunks with vectors closest to query vector
================================================================================
"""

import json
import re
from pathlib import Path

import nltk
import numpy as np
from gensim.models import Word2Vec
from tqdm import tqdm

from .chunker import chunk_corpus

# Download tokenizer data
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)


# =============================================================================
# 1. TEXT PREPROCESSING (IDENTICAL TO train_nn.py)
# =============================================================================
# We reuse the EXACT same cleaning logic.
# Why? The embedding model learns word patterns. If we clean text differently
# during indexing vs querying, the vectors won't match properly.


def clean_text(text: str) -> str:
    """
    Standardizes text for embedding.
    COPIED from train_nn.py - must match exactly for consistent embeddings.
    """
    text = str(text).lower()
    text = re.sub(r"<.*?>", "", text)  # Remove HTML
    text = text.replace("-", " ")  # Split compound words
    text = re.sub(r"[^a-z0-9\s]", "", text)  # Keep alphanumeric only
    text = re.sub(r"\s+", " ", text).strip()  # Collapse whitespace
    return text


def tokenize(text: str) -> list:
    """
    Tokenize text into list of word lists.
    COPIED from train_nn.py.
    """
    return [nltk.word_tokenize(sent) for sent in nltk.sent_tokenize(text)]


# =============================================================================
# 2. EMBEDDING MODEL TRAINING
# =============================================================================
# DIFFERENCE FROM train_nn.py:
# - train_nn.py trained on short messages from message_log.csv
# - Here we train on ALL chunk text from lore corpus
# - Larger corpus = richer vocabulary = better embeddings for lore queries


def train_embedding_model(chunks: list[dict], vector_size: int = 50) -> Word2Vec:
    """
    Train Word2Vec on chunk texts.

    CHANGES FROM train_nn.py:
    - vector_size: 50 instead of 10 (larger corpus can support more dimensions)
    - No artificial expansion (corpus is already large enough)
    - window=5 (slightly larger for prose vs short messages)
    """
    print("Tokenizing chunks for embedding training...")
    all_sentences = []
    for chunk in tqdm(chunks):
        clean = clean_text(chunk["text"])
        tokens = tokenize(clean)
        all_sentences.extend(tokens)

    print(f"Training Word2Vec on {len(all_sentences)} sentences...")
    model = Word2Vec(
        sentences=all_sentences,
        vector_size=vector_size,  # Embedding dimensions
        window=5,  # Context window
        min_count=2,  # Ignore very rare words (lore has weird names)
        epochs=30,  # Training iterations
        sg=1,  # Skip-gram
        seed=42,
    )

    # Sanity check
    try:
        similar = model.wv.most_similar("emperor", topn=3)
        print(f"Embedding check - similar to 'emperor': {[w for w, _ in similar]}")
    except KeyError:
        print("Warning: 'emperor' not in vocabulary")

    return model


# =============================================================================
# 3. VECTORIZATION (IDENTICAL LOGIC TO train_nn.py)
# =============================================================================
# Same averaging approach: chunk → words → word vectors → average = chunk vector


def embed_text(text: str, model: Word2Vec) -> np.ndarray:
    """
    Convert text to a single vector by averaging word embeddings.
    SAME LOGIC as calculate_avg_review_embedding in train_nn.py.
    """
    clean = clean_text(text)
    tokens = tokenize(clean)
    vocab = set(model.wv.index_to_key)
    vector_size = model.vector_size

    word_vectors = []
    for sentence in tokens:
        for word in sentence:
            if word in vocab:
                word_vectors.append(model.wv[word])

    if not word_vectors:
        return np.zeros(vector_size)

    return np.mean(word_vectors, axis=0)


# =============================================================================
# 4. VECTOR INDEX (NEW - NOT IN train_nn.py)
# =============================================================================
# This is the core RAG addition: store all chunk vectors for similarity search


class VectorIndex:
    """
    In-memory vector store for chunk retrieval.

    COURSE TOPIC: RAG Session 1 — "Vector Index Architecture"
    - This is a simplified version of what FAISS/Pinecone/Chroma do
    - Pure numpy implementation = no external dependencies
    - Trade-off: O(n) search vs O(log n) with ANN algorithms

    WHAT THIS REPLACES:
    - train_nn.py used MLP classifier to map vectors → labels
    - We use cosine similarity to map query vector → similar chunk vectors

    STRUCTURE:
        vectors: numpy array shape (num_chunks, vector_size)
        chunks: list of chunk metadata (text, source, faction)
    """

    def __init__(self, model: Word2Vec, chunks: list[dict]):
        """
        Build index by embedding all chunks.

        Args:
            model: Trained Word2Vec model
            chunks: List of chunk dicts from chunker
        """
        self.model = model
        self.chunks = chunks
        self.vectors = self._build_index()

    def _build_index(self) -> np.ndarray:
        """Embed all chunks and stack into matrix."""
        print("Building vector index...")
        vectors = []
        for chunk in tqdm(self.chunks):
            vec = embed_text(chunk["text"], self.model)
            vectors.append(vec)
        return np.stack(vectors)

    def search(self, query: str, k: int = 5, faction_filter: str = None) -> list[dict]:
        """
        Find k most similar chunks to query.

        COURSE TOPIC: RAG Session 1 — "Retrieval" (the R in RAG)
        COURSE TOPIC: RAG Session 2 — "Metadata-Filtered RAG" (faction_filter param)

        THIS IS THE RAG RETRIEVAL STEP:
        1. Embed query using same model
        2. Compute cosine similarity to all chunk vectors
        3. Return top-k matches

        Args:
            query: Search query text
            k: Number of results to return
            faction_filter: Optional - only return chunks from this faction

        Returns:
            List of (chunk, similarity_score) tuples
        """
        # Embed query
        query_vec = embed_text(query, self.model)

        # Cosine similarity: dot(A,B) / (norm(A) * norm(B))
        # We compute against ALL chunk vectors at once (vectorized)
        query_norm = np.linalg.norm(query_vec)
        if query_norm == 0:
            return []

        chunk_norms = np.linalg.norm(self.vectors, axis=1)
        # Avoid division by zero
        chunk_norms = np.where(chunk_norms == 0, 1e-10, chunk_norms)

        similarities = np.dot(self.vectors, query_vec) / (chunk_norms * query_norm)

        # Get indices sorted by similarity (descending)
        sorted_indices = np.argsort(similarities)[::-1]

        # Collect results, applying faction filter if specified
        results = []
        for idx in sorted_indices:
            if len(results) >= k:
                break

            chunk = self.chunks[idx]

            # Apply faction filter
            if faction_filter and chunk["faction"] != faction_filter:
                continue

            results.append({
                "text": chunk["text"],
                "source": chunk["source"],
                "faction": chunk["faction"],
                "chunk_index": chunk["chunk_index"],
                "similarity": float(similarities[idx]),
            })

        return results


# =============================================================================
# 5. MAIN: BUILD AND TEST INDEX
# =============================================================================
# Equivalent to the training + evaluation flow in train_nn.py,
# but for retrieval instead of classification


def main():
    # Paths
    project_root = Path(__file__).parent.parent.parent
    corpus_dir = project_root / "data" / "lore_corpus"
    metadata_path = corpus_dir / "metadata.json"

    print("=" * 60)
    print("BUILDING RAG VECTOR INDEX")
    print("=" * 60)

    # ---------------------------------------------------------
    # Step 1: Load chunks (equivalent to loading CSV in train_nn.py)
    # ---------------------------------------------------------
    print("\n[1/3] Loading and chunking corpus...")
    chunks = chunk_corpus(corpus_dir, metadata_path)
    print(f"Total chunks: {len(chunks)}")

    # ---------------------------------------------------------
    # Step 2: Train embeddings (same as train_nn.py section 5)
    # ---------------------------------------------------------
    print("\n[2/3] Training embedding model...")
    model = train_embedding_model(chunks)

    # ---------------------------------------------------------
    # Step 3: Build index (REPLACES classifier training)
    # ---------------------------------------------------------
    print("\n[3/3] Building vector index...")
    index = VectorIndex(model, chunks)

    print("\n" + "=" * 60)
    print("INDEX READY - TESTING RETRIEVAL")
    print("=" * 60)

    # ---------------------------------------------------------
    # Test queries (equivalent to LIVE FIRE TESTING in train_nn.py)
    # ---------------------------------------------------------
    test_queries = [
        "What is the Golden Throne?",
        "Who are the Chaos Gods?",
        "Tell me about Space Marines",
        "What is the Warp?",
    ]

    for query in test_queries:
        print(f"\n>>> Query: {query}")
        results = index.search(query, k=3)
        for i, r in enumerate(results, 1):
            preview = r["text"][:80] + "..." if len(r["text"]) > 80 else r["text"]
            print(f"    {i}. [{r['faction']}] {r['source']} (sim={r['similarity']:.3f})")
            print(f"       {preview}")


if __name__ == "__main__":
    main()

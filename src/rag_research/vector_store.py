from pathlib import Path

import nltk
import re
import numpy as np
from gensim.models import Word2Vec
from tqdm import tqdm

from .chunker import chunk_corpus

# Download tokenizer data
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

#######
# 1. TEXT PREPROCESSING
#######


def clean_text(text: str) -> str:
    """
    Standardizes text for embedding.
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
    """
    return [nltk.word_tokenize(sent) for sent in nltk.sent_tokenize(text)]


#######
# 2. EMBEDDING MODEL TRAINING
#######


def train_embedding_model(chunks: list[dict], vector_size: int = 50) -> Word2Vec:
    """
    Train Word2Vec on chunk texts.
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
        similar = model.wv.most_similar("ticket", topn=3)
        print(f"Embedding check - similar to 'ticket': {[w for w, _ in similar]}")
    except KeyError:
        print("Warning: 'ticket' not in vocabulary")

    return model


########
# 3. VECTORIZATION (IDENTICAL LOGIC TO train_nn.py)
#######


def embed_text(text: str, model: Word2Vec) -> np.ndarray:
    """
    Convert text to a single vector by averaging word embeddings.
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


#######
# 4. VECTOR INDEX
#######

# Mental Note "core RAG": store all chunk vectors for similarity search


class VectorIndex:
    """
    In-memory vector store for chunk retrieval.

    COURSE TOPIC: RAG Session 1 — "Vector Index Architecture"
    - This is a simplified version of what FAISS/Pinecone/Chroma do
    - Pure numpy implementation = no external dependencies
    - Trade-off: O(n) search vs O(log n) with ANN algorithms

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

    def search(self, query: str, k: int = 5, category_filter: str = None) -> list[dict]:
        """
        Find k most similar chunks to query.

        COURSE TOPIC: RAG Session 1 — "Retrieval" (the R in RAG)
        COURSE TOPIC: RAG Session 2 — "Metadata-Filtered RAG" (category_filter param)

        THIS IS THE RAG RETRIEVAL STEP:
        1. Embed query using same model
        2. Compute cosine similarity to all chunk vectors
        3. Return top-k matches

        Args:
            query: Search query text
            k: Number of results to return
            category_filter: Optional - only return chunks from this category (root, docs, tools, apps, api)

        Returns:
            List of chunk dicts with similarity scores
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

        # Collect results, applying category filter if specified
        results = []
        for idx in sorted_indices:
            if len(results) >= k:
                break

            chunk = self.chunks[idx]

            # Apply category filter (metadata-filtered RAG)
            if category_filter and chunk["category"] != category_filter:
                continue

            results.append(
                {
                    "text": chunk["text"],
                    "source": chunk["source"],
                    "category": chunk["category"],
                    "chunk_index": chunk["chunk_index"],
                    "similarity": float(similarities[idx]),
                }
            )

        return results


#######
# 5. MAIN: BUILD AND TEST INDEX
#######


def main():
    # Paths
    project_root = Path(__file__).parent.parent.parent
    corpus_dir = project_root / "data" / "syncro_corpus"
    metadata_path = corpus_dir / "metadata.json"

    print("=" * 60)
    print("BUILDING RAG VECTOR INDEX")
    print("=" * 60)

    # ---------------------------------------------------------
    # Step 1: Load chunks (equivalent to loading CSV in train_nn.py)

    print("\n[1/3] Loading and chunking corpus...")
    chunks = chunk_corpus(corpus_dir, metadata_path)
    print(f"Total chunks: {len(chunks)}")

    # ---------------------------------------------------------
    # Step 2: Train embeddings (same as train_nn.py section 5)

    print("\n[2/3] Training embedding model...")
    model = train_embedding_model(chunks)

    # ---------------------------------------------------------
    # Step 3: Build index (REPLACES classifier training)

    print("\n[3/3] Building vector index...")
    index = VectorIndex(model, chunks)

    print("\n" + "=" * 60)
    print("INDEX READY - TESTING RETRIEVAL")
    print("=" * 60)

    # ---------------------------------------------------------
    # Test queries (Vengeful Spirit Inc. documentation)

    test_queries = [
        ("How do I set up my dev environment?", None),  # Should find root/tools
        ("What is Universal Billing?", "apps"),  # Filter to apps category
        ("How does Kingfisher sync data?", "tools"),  # Filter to tools
        ("What's the PR process?", "docs"),  # Filter to docs
    ]

    for query, category in test_queries:
        print(f"\n>>> Query: {query}")
        if category:
            print(f"    (filtered to category: {category})")
        results = index.search(query, k=3, category_filter=category)
        for i, r in enumerate(results, 1):
            preview = r["text"][:80] + "..." if len(r["text"]) > 80 else r["text"]
            print(
                f"    {i}. [{r['category']}] {r['source']} (sim={r['similarity']:.3f})"
            )
            print(f"       {preview}")


if __name__ == "__main__":
    main()

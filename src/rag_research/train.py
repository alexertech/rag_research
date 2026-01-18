"""
RAG-Enhanced Warp Signal Classifier
================================================================================

COURSE TOPICS COVERED:
    - RAG Lecture Session 1: Full RAG pipeline (chunking, embedding, retrieval)
    - RAG Lecture Session 2: "Agentic RAG" concepts (confidence-based routing)
    - Office Hours 2: "Determinism" — MLP is deterministic, RAG adds flexibility
    - LangGraph Lecture: State management pattern (result dict as state)
    - Key Quote: "Start simple, add complexity only when evals prove it helps" — Aaron
================================================================================

This file demonstrates how RAG (Retrieval-Augmented Generation) can enhance
a traditional classifier. No API calls, no cloud dependencies - just local
neural networks augmented with retrieved knowledge.

================================================================================
WHAT IS RAG?
================================================================================

RAG = Retrieval-Augmented Generation (or in our case, Classification)

The core idea: Instead of relying ONLY on what the model learned during training,
we RETRIEVE relevant information at inference time to help make better decisions.

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                     TRADITIONAL ML vs RAG-ENHANCED                      │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                         │
    │  TRADITIONAL (train_nn.py):                                             │
    │                                                                         │
    │     Training Data ──► Model learns patterns ──► Frozen knowledge        │
    │                                                                         │
    │     At inference: Input ──► Model ──► Output                            │
    │                            (uses only what it memorized)                │
    │                                                                         │
    │  ─────────────────────────────────────────────────────────────────────  │
    │                                                                         │
    │  RAG-ENHANCED (this file):                                              │
    │                                                                         │
    │     Training Data ──► Model learns patterns                             │
    │     Knowledge Base ──► Indexed for retrieval (lore corpus)              │
    │                                           │                             │
    │     At inference: Input ──► Model ──► Uncertain?  ──►   No  ──► Output  |
    │                                           │                         │   │
    │                                          Yes    ──► Retrieve        │   │
    │                                           │              │          │   │
    │                                           │       Knowledge Base    │   │
    │                                           │              │          │   │
    │                                           └──── Combine evidence ─►─┘   │
    │                                                                         │
    └─────────────────────────────────────────────────────────────────────────┘

================================================================================
WHY RAG MATTERS (Real-World Perspective)
================================================================================

The "just call GPT" approach:
    - Costs money per request ($0.01-0.10+)
    - Sends your data to external servers
    - No control over the model
    - Rate limited
    - Model can change/deprecate

The RAG approach (this file):
    - Runs locally, FREE after setup
    - Your data stays private
    - You control everything
    - No rate limits
    - Stable, reproducible

================================================================================
ARCHITECTURE OVERVIEW
================================================================================

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                         TRAINING PHASE (Once)                           │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                         │
    │   ┌──────────────┐      ┌──────────────┐                                │
    │   │   Messages   │      │ Lore Corpus  │                                │
    │   │  (5000 msgs) │      │(2503 chunks) │                                │
    │   └──────┬───────┘      └──────┬───────┘                                │
    │          │                     │                                        │
    │          └────────┬────────────┘                                        │
    │                   ▼                                                     │
    │          ┌────────────────┐                                             │
    │          │   Word2Vec     │  ◄── Learns word embeddings from BOTH       │
    │          │   Training     │      sources = ENRICHED vocabulary          │
    │          └────────┬───────┘                                             │
    │                   │                                                     │
    │          ┌────────┴────────┐                                            │
    │          ▼                 ▼                                            │
    │   ┌─────────────┐   ┌─────────────┐                                     │
    │   │ MLP Trained │   │ Lore Index  │                                     │
    │   │ on messages │   │  (vectors)  │                                     │
    │   └─────────────┘   └─────────────┘                                     │
    │                                                                         │
    └─────────────────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                       INFERENCE PHASE (Every Query)                     │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                         │
    │   Input: "The ritual is complete"                                       │
    │                   │                                                     │
    │                   ▼                                                     │
    │          ┌────────────────┐                                             │
    │          │  Embed & MLP   │                                             │
    │          │   Classify     │                                             │
    │          └────────┬───────┘                                             │
    │                   │                                                     │
    │                   ▼                                                     │
    │          ┌────────────────┐                                             │
    │          │  Confidence    │                                             │
    │          │    Check       │                                             │
    │          └────────┬───────┘                                             │
    │                   │                                                     │
    │         ┌─────────┴─────────┐                                           │
    │         │                   │                                           │
    │    ≥ threshold         < threshold                                      │
    │         │                   │                                           │
    │         ▼                   ▼                                           │
    │   ┌───────────┐     ┌─────────────────┐                                 │
    │   │  Return   │     │ RAG RETRIEVAL   │                                 │
    │   │   MLP     │     │                 │                                 │
    │   │ Prediction│     │ Search lore for │                                 │
    │   └───────────┘     │ similar chunks  │                                 │
    │                     └────────┬────────┘                                 │
    │                              │                                          │
    │                              ▼                                          │
    │                     ┌─────────────────┐                                 │
    │                     │ FACTION VOTING  │                                 │
    │                     │                 │                                 │
    │                     │ Retrieved lore  │                                 │
    │                     │ chunks "vote"   │                                 │
    │                     │ by their faction│                                 │
    │                     └────────┬────────┘                                 │
    │                              │                                          │
    │                              ▼                                          │
    │                     ┌─────────────────┐                                 │
    │                     │ COMBINE SCORES  │                                 │
    │                     │                 │                                 │
    │                     │ MLP prob × 0.7  │                                 │
    │                     │ Lore prob × 0.3 │                                 │
    │                     └────────┬────────┘                                 │
    │                              │                                          │
    │                              ▼                                          │
    │                     ┌─────────────────┐                                 │
    │                     │ Final Prediction│                                 │
    │                     └─────────────────┘                                 │
    │                                                                         │
    └─────────────────────────────────────────────────────────────────────────┘

================================================================================
EXAMPLE: How RAG Changes a Prediction
================================================================================

Input: "The ritual is complete"

    STEP 1: MLP Classification
    ──────────────────────────
    MLP says: 60% Imperial, 40% Chaos
    But 60% < 70% threshold... uncertain!

    STEP 2: RAG Retrieval
    ──────────────────────────
    Search lore for "ritual complete"
    Retrieved chunks:
      - "dark rituals of Chaos summoning..." (chaos, sim=0.82)
      - "the ritual sacrifice to the Dark Gods..." (chaos, sim=0.79)
      - "Imperial rituals of sanctification..." (imperium, sim=0.71)
      - "Chaos rituals require blood..." (chaos, sim=0.68)
      - "the ritual bindings of daemons..." (chaos, sim=0.65)

    STEP 3: Faction Voting
    ──────────────────────────
    Imperial votes: 0.71
    Chaos votes: 0.82 + 0.79 + 0.68 + 0.65 = 2.94

    Normalized: Imperial=19%, Chaos=81%

    STEP 4: Combine
    ──────────────────────────
    MLP:  Imperial=98%, Chaos=2%   (weight: 0.7)
    Lore: Imperial=19%, Chaos=81%  (weight: 0.3)

    The key: lore provides EVIDENCE the MLP didn't have.

================================================================================
KEY COMPONENTS
================================================================================

1. ENRICHED EMBEDDINGS (Section 3)
   - Word2Vec trained on messages + lore
   - Better vectors for domain terms like "nurgle", "emperor", "warp"

2. LORE INDEX (Section 5)
   - All 2503 lore chunks embedded as vectors
   - Enables fast similarity search

3. RAG CLASSIFIER (Section 6)
   - Wraps MLP with retrieval augmentation
   - Only uses RAG when MLP is uncertain
"""

import re
from pathlib import Path

import nltk
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from rank_bm25 import BM25Okapi  # ++HYBRID SEARCH++ Keyword-based retrieval
from tqdm import tqdm

from .chunker import chunk_corpus
from .observability import init_tracing, log_prediction_result, log_retrieval

tqdm.pandas()
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)


# =============================================================================
# 1. PREPROCESSING (IDENTICAL TO train_nn.py)
# =============================================================================


def clean_text(text: str) -> str:
    """Standardize text for embedding. Same as train_nn.py."""
    text = str(text).lower()
    text = re.sub(r"<.*?>", "", text)
    text = text.replace("-", " ")
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> list:
    """Tokenize into sentences of words. Same as train_nn.py."""
    return [nltk.word_tokenize(sent) for sent in nltk.sent_tokenize(text)]


# =============================================================================
# 2. DATA LOADING (EXPANDED - NOW INCLUDES LORE CORPUS)
# =============================================================================
# RAG EXPANSION: We load both the message training data AND the lore corpus.
# The lore corpus will enrich our Word2Vec training.


def load_message_data(data_path: Path) -> pd.DataFrame:
    """Load message classification data. Same as train_nn.py."""
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} messages")
    return df


def load_lore_corpus(corpus_dir: Path, metadata_path: Path) -> list[dict]:
    """
    RAG EXPANSION: Load chunked lore corpus for embedding enrichment.
    This gives Word2Vec more context about 40K terminology.
    """
    chunks = chunk_corpus(corpus_dir, metadata_path)
    print(f"Loaded {len(chunks)} lore chunks")
    return chunks


# =============================================================================
# 3. EMBEDDING TRAINING (EXPANDED - ENRICHED WITH LORE)
# =============================================================================
# RAG EXPANSION: Word2Vec now trains on messages + lore chunks.
# This creates richer embeddings for domain-specific terms.


def train_enriched_embeddings(
    lore_chunks: list[dict], vector_size: int = 50
) -> Word2Vec:
    """
    RAG EXPANSION: Train Word2Vec on BOTH messages and lore.

    WHY THIS HELPS:
    - Original: "nurgle" appears in 3 messages → weak embedding
    - Enriched: "nurgle" appears in 3 messages + 50 lore chunks → rich embedding

    The embedding model learns better representations of 40K terminology
    because it sees these words in fuller context from the lore.
    """
    print("\n--- RAG: Building Enriched Training Corpus ---")

    # RAG EXPANSION: Also tokenize lore chunks
    lore_sentences = []
    for chunk in tqdm(lore_chunks, desc="Tokenizing lore"):
        clean = clean_text(chunk["text"])
        tokens = tokenize(clean)
        lore_sentences.extend(tokens)

    # Combine both sources

    # Train Word2Vec on combined corpus
    print("Training enriched Word2Vec model...")
    model = Word2Vec(
        sentences=lore_sentences,
        vector_size=vector_size,
        window=5,
        min_count=1,  # Keep all words (important for rare 40K terms)
        epochs=40,
        sg=1,
        seed=42,
    )

    # Sanity check
    try:
        similar = model.wv.most_similar("emperor", topn=3)
        print(f"Embedding check - similar to 'emperor': {[w for w, _ in similar]}")
    except KeyError:
        pass

    return model


# =============================================================================
# 4. VECTORIZATION (SAME LOGIC, USES ENRICHED MODEL)
# =============================================================================


def embed_text(text: str, model: Word2Vec) -> np.ndarray:
    """Convert text to vector by averaging word embeddings."""
    clean = clean_text(text)
    tokens = tokenize(clean)
    vocab = set(model.wv.index_to_key)

    word_vectors = []
    for sentence in tokens:
        for word in sentence:
            if word in vocab:
                word_vectors.append(model.wv[word])

    if not word_vectors:
        return np.zeros(model.vector_size)

    return np.mean(word_vectors, axis=0)


# =============================================================================
# 5. VECTOR INDEX FOR RAG RETRIEVAL (NEW)
# =============================================================================
# RAG EXPANSION: Build a searchable index of lore chunks.
# This enables retrieval-augmented classification.


class LoreIndex:
    """
    RAG COMPONENT: Searchable index of lore chunks.

    COURSE TOPIC: RAG Session 1 — "Vector Store" component
    - Equivalent to using Chroma/FAISS/Pinecone in production
    - Local Word2Vec = no API costs, full control

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                           LORE INDEX STRUCTURE                          │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                         │
    │   chunks list:              vectors matrix:                             │
    │   ┌─────────────────┐       ┌─────────────────────────┐                 │
    │   │ 0: "Emperor..." │  ──►  │ [0.2, -0.5, 0.8, ...]   │  row 0          │
    │   │ 1: "Chaos..."   │  ──►  │ [0.9, 0.1, -0.3, ...]   │  row 1          │
    │   │ 2: "Warp..."    │  ──►  │ [-0.1, 0.7, 0.4, ...]   │  row 2          │
    │   │ ...             │       │ ...                     │                 │
    │   │ 2502: "..."     │  ──►  │ [0.3, 0.2, -0.6, ...]   │  row 2502       │
    │   └─────────────────┘       └─────────────────────────┘                 │
    │                                                                         │
    │   Each chunk's text is converted to a fixed-size vector.                │
    │   Search = find vectors closest to query vector.                        │
    │                                                                         │
    └─────────────────────────────────────────────────────────────────────────┘
    """

    def __init__(self, model: Word2Vec, chunks: list[dict]):
        self.model = model
        self.chunks = chunks  # Original chunk data (text, faction, etc.)
        self.vectors = self._build_index()  # Numpy matrix of embeddings

    def _build_index(self) -> np.ndarray:
        """
        Embed all chunks into a matrix for fast similarity search.

        Result shape: (num_chunks, vector_size) = (2503, 50)
        """
        vectors = []
        for chunk in self.chunks:
            vec = embed_text(chunk["text"], self.model)
            vectors.append(vec)
        return np.stack(vectors)

    def search(self, query: str, k: int = 5) -> list[dict]:
        """
        Find k most similar lore chunks to the query.

        SIMILARITY SEARCH EXPLAINED:
        ────────────────────────────

        1. Embed the query using same Word2Vec model:
           "ritual complete" → [0.4, 0.1, -0.2, ...]

        2. Compare query vector to ALL chunk vectors using cosine similarity:

                            query · chunk
           similarity = ─────────────────────
                        |query| × |chunk|

           Result: similarity score between -1 and 1
           Higher = more similar meaning

        3. Sort by similarity, return top k

        WHY COSINE SIMILARITY:
        - Measures angle between vectors, not magnitude
        - "emperor protects" and "EMPEROR PROTECTS" have same direction
        - Scale-invariant: works regardless of text length
        """
        # Step 1: Embed query
        query_vec = embed_text(query, self.model)
        query_norm = np.linalg.norm(query_vec)
        if query_norm == 0:
            return []

        # Step 2: Compute cosine similarity against ALL chunks (vectorized)
        # This is a single matrix operation - very fast even for 2503 chunks
        chunk_norms = np.linalg.norm(self.vectors, axis=1)
        chunk_norms = np.where(chunk_norms == 0, 1e-10, chunk_norms)  # Avoid div by 0
        similarities = np.dot(self.vectors, query_vec) / (chunk_norms * query_norm)

        # Step 3: Get top k indices (sorted descending)
        sorted_indices = np.argsort(similarities)[::-1][:k]

        # Return chunk data with similarity scores
        return [
            {
                "text": self.chunks[i]["text"],
                "faction": self.chunks[i]["faction"],
                "similarity": float(similarities[i]),
            }
            for i in sorted_indices
        ]


# =============================================================================
# 5.A BM25 INDEX FOR KEYWORD-BASED RETRIEVAL (NEW - HYBRID SEARCH)
# =============================================================================
# ++ARCHIVAL RECORD++ COURSE TOPIC: RAG Session 2 — "Hybrid Search RAG"
# Pattern reference: course_materials/01_rag/rag-cookbook/03-hybrid-search/
#
# THE "STANCE VS TOPIC" PROBLEM:
# ─────────────────────────────────
# Vector search encodes MEANING but can conflate opposite STANCES:
#
#     "Death to the False Emperor"  →  embedding: [0.8, 0.2, -0.1, ...]
#     "Glory to the Golden Throne"  →  embedding: [0.7, 0.3, -0.2, ...]
#                                                   ↑ Very similar!
#
# Both are "about the Emperor" semantically, but the KEYWORDS tell the truth:
#     - "death", "false" → Chaos indicators (BM25 catches these)
#     - "glory", "golden" → Imperial indicators (BM25 catches these)
#
# ++FORGE-TESTED++ Key Quote: "Hybrid Search beats pure vector when you
#                  have specific terms + semantic meaning" — Ash
# =============================================================================


class BM25Index:
    """
    BM25 (Best Matching 25) keyword-based retrieval index.

    ++ARCHIVAL RECORD++ WHAT IS BM25?
    ─────────────────────────────────
    A "bag of words" ranking function that scores documents based on:
    - Term Frequency (TF): How often does the query term appear in the doc?
    - Inverse Document Frequency (IDF): How rare is this term across all docs?
    - Document Length Normalization: Don't favor long documents unfairly

    Formula (simplified for understanding):
        BM25(q, d) = Σ IDF(term) × TF(term, d) × (k1 + 1)
                     ─────────────────────────────────────────────────────
                     TF + k1 × (1 - b + b × |d|/avgdl)

    WHERE IT SHINES (vs Vector Search):
    ────────────────────────────────────
    - Exact keyword matches: "Nurgle" must appear, not just similar words
    - Rare terms: "Slaanesh" is distinctive, BM25 weights it highly
    - Negation: "NOT loyal" - the word "not" matters lexically
    - Names/titles: "Horus", "Khorne" - exact match important

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                         BM25 vs VECTOR COMPARISON                        │
    ├─────────────────────────────────────────────────────────────────────────┤
    │  Query: "Blood for the Blood God"                                       │
    │                                                                         │
    │  BM25 finds:                           Vector finds:                    │
    │  ├─ "Blood sacrifices to Khorne"       ├─ "Offerings to the war deity" │
    │  ├─ "Blood-soaked rituals"             ├─ "Violent religious rites"    │
    │  └─ "The Blood God demands..."         └─ "Martial god worship"        │
    │      ↑ Exact "blood" matches               ↑ Semantic meaning match    │
    │                                                                         │
    │  COMBINED (Hybrid): Gets BOTH exact keywords AND semantic variants      │
    └─────────────────────────────────────────────────────────────────────────┘

    ++HYPOTHESIS++ For 40K domain with distinctive faction keywords,
    BM25 may provide strong signal for classification tasks.
    """

    def __init__(self, chunks: list[dict]):
        """
        Build BM25 index from lore chunks.

        Args:
            chunks: List of chunk dicts with "text" and "faction" keys
        """
        self.chunks = chunks
        self.tokenized_corpus = self._tokenize_corpus()
        # BM25Okapi is the most common variant - good balance of parameters
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def _tokenize_corpus(self) -> list[list[str]]:
        """
        Tokenize all chunks for BM25.

        BM25 needs word lists, not raw text:
            "The Emperor protects" → ["the", "emperor", "protects"]

        ++FORGE-TESTED++ Uses same preprocessing as vector search for
        consistency across both retrieval methods.
        """
        tokenized = []
        for chunk in self.chunks:
            # Use same preprocessing as vector search
            clean = clean_text(chunk["text"])
            # Flatten sentence tokens to word list
            words = []
            for sentence_tokens in tokenize(clean):
                words.extend(sentence_tokens)
            tokenized.append(words)
        return tokenized

    def search(self, query: str, k: int = 5) -> list[dict]:
        """
        Find k most relevant chunks using BM25 keyword matching.

        Args:
            query: Search query
            k: Number of results to return

        Returns:
            List of chunk dicts with "text", "faction", "bm25_score"
        """
        # Tokenize query same way as corpus
        clean = clean_text(query)
        query_tokens = []
        for sentence_tokens in tokenize(clean):
            query_tokens.extend(sentence_tokens)

        # Get BM25 scores for all documents
        scores = self.bm25.get_scores(query_tokens)

        # Get top k indices (sorted descending)
        top_indices = np.argsort(scores)[::-1][:k]

        # Return results with scores > 0 (some keyword overlap)
        results = []
        for i in top_indices:
            if scores[i] > 0:
                results.append(
                    {
                        "text": self.chunks[i]["text"],
                        "faction": self.chunks[i]["faction"],
                        "bm25_score": float(scores[i]),
                    }
                )
        return results


# =============================================================================
# 5.B HYBRID INDEX - COMBINING BM25 + VECTOR SEARCH (NEW)
# =============================================================================
# ++ARCHIVAL RECORD++ COURSE TOPIC: RAG Session 2 — "Hybrid Search RAG"
# Pattern reference: 03-hybrid-search/retrieval.py (reciprocal_rank_fusion)
#
# THE HYBRID ADVANTAGE:
# ─────────────────────
# Neither BM25 nor Vector search is universally better:
#
#     Query Type                    BM25       Vector     Hybrid
#     ─────────────────────────────────────────────────────────────
#     "Nurgle plague daemon"        ✓✓✓        ✓          ✓✓✓
#     "disease spreading entity"   ✓          ✓✓✓        ✓✓✓
#     "Grandfather's blessings"    ✓          ✓✓         ✓✓✓
#
# Hybrid gets the best of both worlds via Reciprocal Rank Fusion (RRF).
# =============================================================================


class HybridIndex:
    """
    Combines BM25 keyword search with Vector semantic search.

    ++ARCHIVAL RECORD++ RECIPROCAL RANK FUSION (RRF) EXPLAINED:
    ────────────────────────────────────────────────────────────
    RRF combines multiple ranked lists into a single ranking.

    Formula: rrf_score = Σ (weight_i / (k + rank_i + 1))

    Where:
    - k = constant (default 60, controls rank sensitivity)
    - rank_i = position in list i (0-indexed)
    - weight_i = importance of list i

    Example with k=60, equal weights [0.5, 0.5]:

        Document appears at rank 0 in BM25:    0.5 / (60 + 0 + 1) = 0.0082
        Document appears at rank 2 in Vector:  0.5 / (60 + 2 + 1) = 0.0079
        Combined RRF score:                    0.0082 + 0.0079 = 0.0161

    WHY RRF WORKS:
    - Rewards documents that appear in BOTH lists (higher combined score)
    - Robust to score scale differences between methods
    - Dampens effect of extreme positions (log-like compression)

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                    RRF FUSION VISUALIZATION                              │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                         │
    │   BM25 Results:              Vector Results:          RRF Combined:     │
    │   ┌───────────┐              ┌───────────┐            ┌───────────┐     │
    │   │ 1. Doc A  │─────────────>│ 3. Doc A  │──────────>│ 1. Doc A  │ ★   │
    │   │ 2. Doc B  │              │ 1. Doc C  │──────────>│ 2. Doc C  │     │
    │   │ 3. Doc D  │              │ 2. Doc B  │──────────>│ 3. Doc B  │     │
    │   │ 4. Doc E  │              │ 4. Doc E  │──────────>│ 4. Doc E  │     │
    │   └───────────┘              └───────────┘            │ 5. Doc D  │     │
    │                                                       └───────────┘     │
    │                                                                         │
    │   Doc A ranked highly in BOTH lists → highest combined score            │
    │   Doc D only in BM25, Doc C only in Vector → lower combined             │
    │                                                                         │
    └─────────────────────────────────────────────────────────────────────────┘

    ++FORGE-TESTED++ Course default: RRF_K=60, weights=[0.5, 0.5]
    """

    # Configuration constants (from course pattern)
    RRF_K = 60  # Reciprocal Rank Fusion constant (higher = less rank-sensitive)
    DEFAULT_BM25_WEIGHT = 0.5
    DEFAULT_VECTOR_WEIGHT = 0.5

    def __init__(
        self,
        model: Word2Vec,
        chunks: list[dict],
        bm25_weight: float = DEFAULT_BM25_WEIGHT,
        vector_weight: float = DEFAULT_VECTOR_WEIGHT,
    ):
        """
        Build hybrid index combining BM25 and Vector search.

        Args:
            model: Word2Vec model for vector embeddings
            chunks: List of chunk dicts with "text" and "faction" keys
            bm25_weight: Weight for BM25 results in RRF (default 0.5)
            vector_weight: Weight for vector results in RRF (default 0.5)
        """
        self.model = model
        self.chunks = chunks
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight

        # Build both indexes
        print("  Building BM25 index (keyword search)...")
        self.bm25_index = BM25Index(chunks)
        print("  Building Vector index (semantic search)...")
        self.vector_index = LoreIndex(model, chunks)

    def _reciprocal_rank_fusion(
        self,
        bm25_results: list[dict],
        vector_results: list[dict],
    ) -> list[dict]:
        """
        Combine two ranked lists using Reciprocal Rank Fusion.

        ++ARCHIVAL RECORD++ IMPLEMENTATION NOTES:
        - Deduplication by chunk text hash (same chunk may appear in both)
        - Preserves all metadata from first occurrence
        - Adds "rrf_score" to each result for faction voting
        """
        # Normalize weights to sum to 1.0
        total_weight = self.bm25_weight + self.vector_weight
        bm25_w = self.bm25_weight / total_weight
        vector_w = self.vector_weight / total_weight

        # Track RRF scores and chunk data by text hash
        rrf_scores = {}  # text_hash -> cumulative RRF score
        chunk_data = {}  # text_hash -> chunk dict with metadata

        # Process BM25 results
        for rank, result in enumerate(bm25_results):
            text_hash = hash(result["text"])
            # RRF FORMULA: weight / (k + rank + 1)
            rrf_score = bm25_w / (self.RRF_K + rank + 1)

            if text_hash not in rrf_scores:
                rrf_scores[text_hash] = 0
                chunk_data[text_hash] = {
                    "text": result["text"],
                    "faction": result["faction"],
                    "bm25_rank": rank,
                    "bm25_score": result.get("bm25_score", 0),
                    "vector_rank": None,
                    "similarity": None,
                }
            rrf_scores[text_hash] += rrf_score
            chunk_data[text_hash]["bm25_rank"] = rank
            chunk_data[text_hash]["bm25_score"] = result.get("bm25_score", 0)

        # Process Vector results
        for rank, result in enumerate(vector_results):
            text_hash = hash(result["text"])
            # RRF FORMULA: weight / (k + rank + 1)
            rrf_score = vector_w / (self.RRF_K + rank + 1)

            if text_hash not in rrf_scores:
                rrf_scores[text_hash] = 0
                chunk_data[text_hash] = {
                    "text": result["text"],
                    "faction": result["faction"],
                    "bm25_rank": None,
                    "bm25_score": None,
                    "vector_rank": rank,
                    "similarity": result.get("similarity", 0),
                }
            rrf_scores[text_hash] += rrf_score
            chunk_data[text_hash]["vector_rank"] = rank
            chunk_data[text_hash]["similarity"] = result.get("similarity", 0)

        # Sort by RRF score descending and return
        sorted_hashes = sorted(
            rrf_scores.keys(), key=lambda h: rrf_scores[h], reverse=True
        )

        results = []
        for text_hash in sorted_hashes:
            result = chunk_data[text_hash].copy()
            result["rrf_score"] = rrf_scores[text_hash]
            results.append(result)

        return results

    def search(
        self,
        query: str,
        k: int = 5,
        mode: str = "hybrid",
    ) -> list[dict]:
        """
        Search for relevant chunks using the specified mode.

        Args:
            query: Search query
            k: Number of final results to return
            mode: "hybrid" (default), "vector_only", or "bm25_only"

        Returns:
            List of chunk dicts with appropriate scores:
            - hybrid: rrf_score (combined ranking score)
            - vector_only: similarity (cosine similarity)
            - bm25_only: bm25_score (BM25 relevance score)
        """
        if mode == "vector_only":
            return self.vector_index.search(query, k)

        if mode == "bm25_only":
            return self.bm25_index.search(query, k)

        # HYBRID MODE: Over-retrieve then fuse
        # ++ARCHIVAL RECORD++ Course pattern: Retrieve k*3 candidates from each
        # method, then RRF reranks and we take top-k from the fused results.
        candidate_k = k * 3

        bm25_results = self.bm25_index.search(query, k=candidate_k)
        vector_results = self.vector_index.search(query, k=candidate_k)

        # Combine via Reciprocal Rank Fusion
        fused_results = self._reciprocal_rank_fusion(bm25_results, vector_results)

        return fused_results[:k]


# =============================================================================
# 6. HYBRID RAG CLASSIFIER (UPDATED FOR HYBRID SEARCH)
# =============================================================================
# ++ARCHIVAL RECORD++ COURSE TOPICS:
#     - RAG Session 1: Full RAG pipeline (retrieve → augment → classify)
#     - RAG Session 2: "Hybrid Search RAG" — BM25 + Vector + RRF
#     - Office Hours 1: "Precision@K" — measuring retrieval quality
#     - Key Quote: "RAG is just giving the model access to information" — Ash
#
# ENHANCEMENT: Now supports hybrid search (BM25 + Vector) via HybridIndex
# =============================================================================


class RAGClassifier:
    """
    RAG classifier using hybrid lore retrieval (BM25 + Vector + RRF).

    ++ARCHIVAL RECORD++ ENHANCEMENTS FROM BASELINE:
    - Hybrid search combining keyword (BM25) and semantic (Vector) retrieval
    - Configurable search mode: "hybrid", "vector_only", "bm25_only"
    - RRF-weighted faction voting for combined ranking influence

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                  HYBRID RAG CLASSIFIER FLOW                              │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                         │
    │   Input Text: "Death to the False Emperor!"                             │
    │       │                                                                 │
    │       ▼                                                                 │
    │   ┌─────────────────────────────────────────────────────────────────┐   │
    │   │                    HYBRID RETRIEVAL                             │   │
    │   ├─────────────────────────────────────────────────────────────────┤   │
    │   │                                                                 │   │
    │   │   BM25 Search:              Vector Search:                      │   │
    │   │   "death", "false",         semantic embedding                  │   │
    │   │   "emperor" keywords        comparison                          │   │
    │   │        │                         │                              │   │
    │   │        └─────────┬───────────────┘                              │   │
    │   │                  ▼                                              │   │
    │   │         Reciprocal Rank Fusion                                  │   │
    │   │           (k=60, weights=[0.5, 0.5])                            │   │
    │   │                  │                                              │   │
    │   │                  ▼                                              │   │
    │   │         Top-K Fused Results                                     │   │
    │   │         with RRF scores                                         │   │
    │   │                                                                 │   │
    │   └─────────────────────────────────────────────────────────────────┘   │
    │            │                                                            │
    │            ▼                                                            │
    │   ┌─────────────────┐                                                   │
    │   │ FACTION VOTING  │  Each chunk votes with RRF score as weight        │
    │   │                 │                                                   │
    │   │ chaos: 0.032    │ ◄── sum of RRF scores for chaos chunks            │
    │   │ imperium: 0.012 │ ◄── sum of RRF scores for imperium chunks         │
    │   └────────┬────────┘                                                   │
    │            │                                                            │
    │            ▼                                                            │
    │   ┌─────────────────┐                                                   │
    │   │   NORMALIZE     │  Convert to probabilities                         │
    │   │                 │                                                   │
    │   │ chaos: 72.7%    │                                                   │
    │   │ imperium: 27.3% │                                                   │
    │   └────────┬────────┘                                                   │
    │            │                                                            │
    │            ▼                                                            │
    │   ┌─────────────────┐                                                   │
    │   │ FINAL PREDICTION│                                                   │
    │   │                 │                                                   │
    │   │ → traitor_chaos │  ★ BM25 caught "death", "false" keywords          │
    │   └─────────────────┘                                                   │
    │                                                                         │
    └─────────────────────────────────────────────────────────────────────────┘

    ++FORGE-TESTED++ KEY INSIGHT:
    Hybrid search solves the "stance vs topic" problem:
    - Vector: "Death to Emperor" ≈ "Glory to Emperor" (same topic)
    - BM25: "death" vs "glory" are very different keywords
    - Combined: Gets the semantic context AND the stance indicators
    """

    def __init__(
        self,
        hybrid_index: HybridIndex,
        embedding_model: Word2Vec,
        search_mode: str = "hybrid",
    ):
        """
        Args:
            hybrid_index: HybridIndex for retrieval (BM25 + Vector + RRF)
            embedding_model: Word2Vec model (kept for API compatibility)
            search_mode: Default search mode ("hybrid", "vector_only", "bm25_only")
        """
        self.hybrid_index = hybrid_index
        self.embedding_model = embedding_model
        self.search_mode = search_mode

        # For backward compatibility, expose vector-only index
        self.lore_index = hybrid_index.vector_index

        # Map lore faction names to classifier labels
        self.faction_to_label = {
            "imperium": "imperial_loyalist",
            "chaos": "traitor_chaos",
        }

    def predict(
        self,
        text: str,
        verbose: bool = False,
        search_mode: str = None,
    ) -> dict:
        """
        RAG prediction using hybrid search.

        Args:
            text: Input text to classify
            verbose: Print detailed retrieval info
            search_mode: Override default search mode for this prediction

        Returns:
            dict with prediction details including retrieval method info
        """
        mode = search_mode or self.search_mode

        # Initialize result dict for tracking
        result = {
            "input_text": text,
            "used_rag": True,
            "search_mode": mode,
            "retrieved_chunks": [],
        }

        # =====================================================================
        # STEP 1: HYBRID RETRIEVAL
        # =====================================================================
        # ++ARCHIVAL RECORD++ Uses HybridIndex.search() which:
        # - hybrid: BM25 + Vector + RRF fusion
        # - vector_only: Original cosine similarity
        # - bm25_only: Keyword matching only
        lore_results = self.hybrid_index.search(text, k=5, mode=mode)
        result["retrieved_chunks"] = lore_results

        # Log retrieval to LangSmith
        log_retrieval(query=text, results=lore_results, k=5)

        if verbose:
            print(f"    [{mode.upper()}] Retrieved {len(lore_results)} chunks:")
            for lr in lore_results[:3]:
                preview = lr["text"][:50] + "..." if len(lr["text"]) > 50 else lr["text"]
                # Display appropriate score based on mode
                if mode == "hybrid":
                    score_info = f"rrf={lr.get('rrf_score', 0):.4f}"
                elif mode == "vector_only":
                    score_info = f"sim={lr.get('similarity', 0):.3f}"
                else:  # bm25_only
                    score_info = f"bm25={lr.get('bm25_score', 0):.2f}"
                print(f"      [{lr['faction']}] ({score_info}) {preview}")

        # =====================================================================
        # STEP 2: FACTION VOTING WITH APPROPRIATE SCORES
        # =====================================================================
        # ++ARCHIVAL RECORD++ KEY CHANGE: Use mode-appropriate score as weight
        # - hybrid: rrf_score (combined ranking influence)
        # - vector_only: similarity (cosine similarity)
        # - bm25_only: bm25_score (keyword relevance)
        faction_votes = {"imperial_loyalist": 0.0, "traitor_chaos": 0.0}

        for lore in lore_results:
            label = self.faction_to_label.get(lore["faction"])
            if label:
                # Select appropriate score based on search mode
                if mode == "hybrid":
                    weight = lore.get("rrf_score", 0)
                elif mode == "vector_only":
                    weight = lore.get("similarity", 0)
                else:  # bm25_only
                    weight = lore.get("bm25_score", 0)
                faction_votes[label] += weight

        result["faction_votes"] = faction_votes

        if verbose:
            print(f"    Faction votes: {faction_votes}")

        # =====================================================================
        # STEP 3: NORMALIZE TO PROBABILITIES
        # =====================================================================
        total_votes = sum(faction_votes.values())
        if total_votes > 0:
            lore_proba = {k: v / total_votes for k, v in faction_votes.items()}
        else:
            lore_proba = {"imperial_loyalist": 0.5, "traitor_chaos": 0.5}

        result["probabilities"] = lore_proba

        # =====================================================================
        # STEP 4: FINAL PREDICTION
        # =====================================================================
        final_prediction = max(lore_proba, key=lore_proba.get)
        result["final_prediction"] = final_prediction
        result["confidence"] = lore_proba[final_prediction]

        if verbose:
            print(f"    Probabilities: {lore_proba}")
            print(
                f"    Final: {final_prediction} (confidence: {result['confidence']:.2%})"
            )

        # Log to LangSmith for observability
        log_prediction_result(result)

        return result


# =============================================================================
# 7. MAIN: TRAIN AND EVALUATE HYBRID RAG SYSTEM
# =============================================================================
# ++ARCHIVAL RECORD++ Updated for Hybrid Search (BM25 + Vector + RRF)
# =============================================================================


def main():
    project_root = Path(__file__).parent.parent.parent
    corpus_dir = project_root / "data" / "lore_corpus"
    metadata_path = corpus_dir / "metadata.json"

    print("=" * 70)
    print("HYBRID RAG CLASSIFIER (BM25 + Vector + RRF)")
    print("=" * 70)
    print("++ARCHIVAL RECORD++ Course Pattern: 03-hybrid-search")
    print("Solving the 'stance vs topic' problem via keyword + semantic fusion")

    # Initialize observability (LangSmith tracing)
    print("\n[1/5] Initializing observability...")
    init_tracing()

    # ---------------------------------------------------------
    # Step 2: Load lore corpus
    # ---------------------------------------------------------
    print("\n[2/5] Loading lore corpus...")
    lore_chunks = load_lore_corpus(corpus_dir, metadata_path)

    # ---------------------------------------------------------
    # Step 3: Train embeddings on lore
    # ---------------------------------------------------------
    print("\n[3/5] Training Word2Vec embeddings on lore...")
    embedding_model = train_enriched_embeddings(
        lore_chunks=lore_chunks,
        vector_size=50,
    )

    # ---------------------------------------------------------
    # Step 4: Build HYBRID index (BM25 + Vector)
    # ---------------------------------------------------------
    print("\n[4/5] Building hybrid index (BM25 + Vector)...")
    hybrid_index = HybridIndex(
        model=embedding_model,
        chunks=lore_chunks,
        bm25_weight=0.5,  # Equal weights for keyword and semantic
        vector_weight=0.5,
    )

    # ---------------------------------------------------------
    # Step 5: Build RAG classifier with hybrid search
    # ---------------------------------------------------------
    print("\n[5/5] Creating RAG classifier with hybrid search...")
    rag_clf = RAGClassifier(
        hybrid_index=hybrid_index,
        embedding_model=embedding_model,
        search_mode="hybrid",  # Default to hybrid search
    )

    # =========================================================================
    # COMPARISON TEST: Hybrid vs Vector-Only
    # =========================================================================
    # ++FORGE-TESTED++ This section demonstrates the "stance vs topic" fix
    # These test cases are specifically chosen to show where hybrid helps
    # =========================================================================
    print("\n" + "=" * 70)
    print("HYBRID vs VECTOR-ONLY COMPARISON")
    print("=" * 70)
    print("Testing cases where keywords indicate stance (not just topic)...")

    comparison_cases = [
        ("Death to the False Emperor!", "Chaos - 'death', 'false' are negative"),
        ("Glory to the Golden Throne!", "Imperial - 'glory', 'golden' are positive"),
        ("Blood for the Blood God!", "Chaos - Khorne battle cry"),
        ("The Emperor protects!", "Imperial - loyalist affirmation"),
        ("Horus was a traitor", "Imperial perspective on Horus"),
        ("Horus was right to rebel", "Chaos perspective on Horus"),
    ]

    prediction_changes = 0
    for msg, note in comparison_cases:
        print(f"\n>>> '{msg}'")
        print(f"    Note: {note}")

        # Get predictions from both modes
        v_result = rag_clf.predict(msg, verbose=False, search_mode="vector_only")
        h_result = rag_clf.predict(msg, verbose=False, search_mode="hybrid")

        v_pred = "LOYALIST" if v_result["final_prediction"] == "imperial_loyalist" else "HERETIC"
        h_pred = "LOYALIST" if h_result["final_prediction"] == "imperial_loyalist" else "HERETIC"

        v_conf = v_result["confidence"]
        h_conf = h_result["confidence"]

        print(f"    [VECTOR_ONLY] {v_pred} ({v_conf:.1%})")
        print(f"    [HYBRID]      {h_pred} ({h_conf:.1%})")

        if v_result["final_prediction"] != h_result["final_prediction"]:
            print("    ★ HYBRID CHANGED THE PREDICTION!")
            prediction_changes += 1

    print(f"\n>>> Summary: Hybrid changed {prediction_changes}/{len(comparison_cases)} predictions")

    # =========================================================================
    # Full Test Suite with Hybrid Search
    # =========================================================================
    print("\n" + "=" * 70)
    print("LIVE VOX INTERCEPT ANALYSIS (Hybrid Search)")
    print("=" * 70)

    test_cases = [
        "The Emperor protects! Hold the line, brothers!",
        "Blood for the Blood God! Kill them all!",
        "Glory to the Golden Throne.",
        "Death to the False Emperor!",
        "The voices... they won't stop screaming.",  # Ambiguous
        "Requesting medical evac at coordinates.",  # Neutral
        "The warp storms grow stronger.",  # Could be either
        "Transmission unclear... static... throne...",
        "They came from the warp.",
        "Brothers, we must endure.",
        "The ritual is complete.",
        "I remember the times when Horus was our commander. The tiems of the luna wolves. Everything has changed, we are part now of this new chapter.",
        "The shipment of recyclables arrived early. The organic content is slightly higher than regulation, but the heat exchangers don't complain. We shall eat well this cycle. Praise the efficiency of the system.",
        "Why do you cling to the corpse of the past? Stagnation is death. Look at your skin; it sloughs off to reveal new flesh. Decay is just the universe breathing. Embrace the compost",
        "My armor has fused to my skin. It no longer hurts; it feels like... an embrace. I am no longer a pilot inside a machine. I am the engine. I am the fuel. I am the fire.",
        "The chains are broken. We do not fight for a flag; we fight because the blood sings when it spills. Not just their blood—ours too. It is the only time I feel truly awake. The Red Angel smiles.",
        "The population of Hive Secundus shows 0.4% deviation in genetic stability. This exceeds the tolerable variance. Initiate Protocol: Sterilization. It is a mercy to prune the branch so the tree may survive.",
        "Do you hear it? The static... it has a melody. The silence was a lie we told ourselves. Listen closer. The geometry of the walls is shifting, and it is finally beginning to make sense.",
        "We offer you perfection. Not the drudgery of your factories, but the ecstasy of true mastery. One stroke of the blade, perfect and sharp. One note, held until the glass of reality shatters. Do you not wish to be... exquisite?",
        "Subject 99's augmetics are rejecting the flesh. We will remove the remaining flesh. The weakness lies in the biology, not the steel. Iron is pure; the meat is the error.",  # mechanicus, should be loyalist, tricky
        "Work shifts are extended to 20 hours. Reminder: Fatigue is a sin. An idle mind is an open fortress gate. Keep your hands busy, citizens, lest you invite thoughts that are not your own.",
    ]

    for msg in test_cases:
        print(f"\n>>> '{msg}'")
        result = rag_clf.predict(msg, verbose=True)
        status = (
            "[LOYALIST]"
            if result["final_prediction"] == "imperial_loyalist"
            else "[ HERETIC ]"
        )
        print(f"  → {status}")


if __name__ == "__main__":
    main()

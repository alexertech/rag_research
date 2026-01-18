"""
Chunker Module for RAG Pipeline

================================================================================
COURSE TOPICS COVERED:
    - RAG Lecture Session 1: "Chunking Strategies" (Ash Tilawat)
    - RAG Lecture Session 2: "Precision Delta" — chunk size affects retrieval quality
    - Key Quote: "Chunking is where RAG lives or dies" — Ash
================================================================================

PURPOSE:
    Break large documents into smaller pieces (chunks) for vector embedding.
    Each chunk becomes a separate searchable unit in the RAG system.

WHY CHUNKING:
    - Documents are too large to embed as single vectors (dilutes meaning)
    - Retrieval needs precision: return relevant SECTIONS, not whole docs
    - Context windows have limits: chunks must fit in LLM prompts

KEY CONCEPTS:
    - chunk_size: Target size in characters per chunk
    - overlap: Characters shared between consecutive chunks (prevents lost context at boundaries)
    - metadata: Information attached to each chunk (source file, faction, position)
"""

import json
from pathlib import Path


def chunk_text(text: str, chunk_size: int = 400, overlap: int = 80) -> list[str]:
    """
    Split text into overlapping chunks using a sliding window approach.

    COURSE TOPIC: RAG Session 1 — "Sliding Window Chunking"
    - 400 chars ≈ ~100 tokens (rule of thumb: 4 chars/token)
    - 80 char overlap (~20%) prevents context loss at boundaries

    HOW IT WORKS:
        Text:    [===========================================================]
        Chunk 1: [===========]
        Chunk 2:         [===========]   <- overlaps with chunk 1
        Chunk 3:                 [===========]
                         ↑
                     overlap region

    Args:
        text: The full document text to chunk
        chunk_size: Target characters per chunk (not a hard limit - we respect word boundaries)
        overlap: Characters to repeat between chunks (preserves context across boundaries)

    Returns:
        List of text chunks
    """
    # Split into words to avoid cutting mid-word
    words = text.split()

    if not words:
        return []

    chunks = []
    current_chunk_words = []
    current_length = 0

    for word in words:
        word_length = len(word) + 1  # +1 for space

        # If adding this word exceeds chunk_size, save current chunk and start new one
        if current_length + word_length > chunk_size and current_chunk_words:
            # Save completed chunk
            chunk_text = " ".join(current_chunk_words)
            chunks.append(chunk_text)

            # Calculate how many words to keep for overlap
            # Walk backwards through words until we have ~overlap characters
            overlap_words = []
            overlap_length = 0
            for w in reversed(current_chunk_words):
                if overlap_length + len(w) + 1 > overlap:
                    break
                overlap_words.insert(0, w)
                overlap_length += len(w) + 1

            # Start new chunk with overlap words
            current_chunk_words = overlap_words
            current_length = overlap_length

        # Add word to current chunk
        current_chunk_words.append(word)
        current_length += word_length

    # Don't forget the last chunk
    if current_chunk_words:
        chunks.append(" ".join(current_chunk_words))

    return chunks


def chunk_document(
    file_path: Path, faction: str, chunk_size: int = 400, overlap: int = 80
) -> list[dict]:
    """
    Chunk a single document and attach metadata to each chunk.

    COURSE TOPIC: RAG Session 1 — "Metadata-Filtered RAG"
    - Metadata enables filtering at retrieval time (e.g., faction_filter="chaos")
    - Key Quote: "Metadata-filtered RAG is usually the right starting point" — Ash

    METADATA PURPOSE:
        When RAG retrieves a chunk, we need to know:
        - Where it came from (source file)
        - What faction it belongs to (for filtering/classification)
        - Its position (for debugging, potential re-ranking)

    Args:
        file_path: Path to the markdown file
        faction: Faction label (e.g., "imperium", "chaos")
        chunk_size: Target chunk size in characters
        overlap: Overlap between chunks

    Returns:
        List of chunk dicts: {"text": str, "source": str, "faction": str, "chunk_index": int}
    """
    text = file_path.read_text(encoding="utf-8")

    # Get raw text chunks
    text_chunks = chunk_text(text, chunk_size, overlap)

    # Attach metadata to each chunk
    chunks_with_metadata = []
    for i, chunk in enumerate(text_chunks):
        chunks_with_metadata.append(
            {
                "text": chunk,
                "source": file_path.name,  # Just filename, not full path
                "faction": faction,
                "chunk_index": i,
            }
        )

    return chunks_with_metadata


def chunk_corpus(
    corpus_dir: Path, metadata_path: Path, chunk_size: int = 400, overlap: int = 80
) -> list[dict]:
    """
    Chunk all documents in the corpus using metadata.json for faction info.

    WORKFLOW:
        1. Read metadata.json to get document list and factions
        2. For each document: chunk it and attach metadata
        3. Return flat list of all chunks from all documents

    Args:
        corpus_dir: Path to lore_corpus directory
        metadata_path: Path to metadata.json
        chunk_size: Target chunk size
        overlap: Overlap between chunks

    Returns:
        List of all chunks from all documents
    """
    # Load corpus metadata
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    all_chunks = []

    for doc in metadata["documents"]:
        doc_path = corpus_dir / doc["file"]
        faction = doc["faction"]

        if not doc_path.exists():
            print(f"Warning: {doc_path} not found, skipping")
            continue

        # Chunk this document
        doc_chunks = chunk_document(doc_path, faction, chunk_size, overlap)
        all_chunks.extend(doc_chunks)

        print(f"Chunked {doc['file']}: {len(doc_chunks)} chunks")

    return all_chunks


# =============================================================================
# CLI Entry Point - For testing/manual runs
# =============================================================================
if __name__ == "__main__":
    # Default paths relative to this project
    project_root = Path(__file__).parent.parent.parent
    corpus_dir = project_root / "data" / "lore_corpus"
    metadata_path = corpus_dir / "metadata.json"

    print("=" * 50)
    print("CHUNKING LORE CORPUS")
    print("=" * 50)

    chunks = chunk_corpus(corpus_dir, metadata_path)

    print("=" * 50)
    print(f"TOTAL CHUNKS: {len(chunks)}")
    print("=" * 50)

    # Preview first few chunks
    print("\nSample chunks:")
    for chunk in chunks[:3]:
        preview = (
            chunk["text"][:100] + "..." if len(chunk["text"]) > 100 else chunk["text"]
        )
        print(
            f"  [{chunk['faction']}] {chunk['source']}#{chunk['chunk_index']}: {preview}"
        )

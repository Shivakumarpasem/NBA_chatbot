"""
RAG module: load NBA recaps, chunk, embed, store in Chroma, retrieve top-k chunks.
Uses sentence-transformers (free, local) and Chroma (free) — no API cost.
"""
import hashlib
from pathlib import Path
from typing import Dict, List, Optional

# Lazy imports to avoid loading heavy deps until needed
_chroma_client = None
_embedding_model = None
_retrieval_cache: Dict[str, List[str]] = {}


def _get_embedding_model():
    """Lazy load sentence-transformers model."""
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedding_model


def _get_chroma_collection(rag_docs_dir: Path, persist_dir: Optional[Path] = None):
    """Get or create Chroma collection for NBA recaps."""
    global _chroma_client
    from chromadb import PersistentClient
    from chromadb.config import Settings

    persist_dir = persist_dir or (rag_docs_dir.parent / ".chroma_db")
    persist_dir.mkdir(parents=True, exist_ok=True)

    client = PersistentClient(path=str(persist_dir), settings=Settings(anonymized_telemetry=False))
    collection = client.get_or_create_collection(name="nba_recaps", metadata={"description": "NBA Finals recaps and bios"})

    return collection


def _split_section_windows(section_lines: List[str], max_chars: int = 900, overlap_chars: int = 150) -> List[str]:
    """
    Convert one section into multiple retrieval-friendly windows.
    This improves recall versus single giant chunks.
    """
    if not section_lines:
        return []
    header = section_lines[0].strip()
    body = "\n".join(section_lines[1:]).strip()
    if not body:
        return []

    # Paragraph-aware windowing.
    paragraphs = [p.strip() for p in body.split("\n\n") if p.strip()]
    windows: List[str] = []
    current = ""

    for para in paragraphs:
        if not current:
            current = para
            continue
        candidate = f"{current}\n\n{para}"
        if len(candidate) <= max_chars:
            current = candidate
        else:
            windows.append(f"{header}\n\n{current}".strip())
            # Light overlap from the end of previous window.
            tail = current[-overlap_chars:] if overlap_chars > 0 else ""
            current = f"{tail}\n\n{para}".strip() if tail else para

    if current:
        windows.append(f"{header}\n\n{current}".strip())

    # Guardrail: avoid tiny windows.
    return [w for w in windows if len(w) > 80]


def _load_and_chunk(docs_path: Path) -> List[str]:
    """Load docs and split by section header + paragraph windows."""
    text = docs_path.read_text(encoding="utf-8")
    chunks: List[str] = []
    current = []
    for line in text.split("\n"):
        if line.strip().startswith("===") and line.strip().endswith("==="):
            if current:
                chunks.extend(_split_section_windows(current))
            current = [line]
        else:
            current.append(line)
    if current:
        chunks.extend(_split_section_windows(current))
    return [c for c in chunks if len(c) > 50]


def build_index(rag_docs_dir: Path) -> None:
    """
    Build the vector index from RAG docs. Call once (or when docs change).
    """
    recaps_file = rag_docs_dir / "nba_recaps.txt"
    if not recaps_file.exists():
        raise FileNotFoundError(f"RAG docs not found: {recaps_file}")

    chunks = _load_and_chunk(recaps_file)
    if not chunks:
        raise ValueError("No chunks extracted from RAG docs")

    model = _get_embedding_model()
    embeddings = model.encode(chunks).tolist()

    collection = _get_chroma_collection(rag_docs_dir)
    # Replace old index entries to avoid stale/duplicate chunks.
    if collection.count() > 0:
        existing = collection.get()
        existing_ids = existing.get("ids", []) if existing else []
        if existing_ids:
            collection.delete(ids=existing_ids)

    # Stable IDs from content hash to make rebuild deterministic.
    ids = [f"chunk_{hashlib.md5(c.encode('utf-8')).hexdigest()[:12]}" for c in chunks]
    collection.upsert(ids=ids, documents=chunks, embeddings=embeddings)
    _retrieval_cache.clear()


def retrieve(rag_docs_dir: Path, query: str, top_k: int = 4) -> List[str]:
    """
    Embed query, search vector DB, return top-k chunks as context.

    Inputs:
        rag_docs_dir: path to data/rag_docs/
        query: user question
        top_k: number of chunks to retrieve

    Outputs:
        List of chunk strings (context for the LLM)
    """
    recaps_file = rag_docs_dir / "nba_recaps.txt"
    if not recaps_file.exists():
        return []

    cache_key = f"{query.strip().lower()}::{top_k}"
    if cache_key in _retrieval_cache:
        return _retrieval_cache[cache_key]

    collection = _get_chroma_collection(rag_docs_dir)
    # Ensure index exists
    if collection.count() == 0:
        build_index(rag_docs_dir)

    model = _get_embedding_model()
    query_embedding = model.encode([query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=min(top_k, collection.count()))
    documents = results.get("documents", [[]])
    out = documents[0] if documents else []
    _retrieval_cache[cache_key] = out
    return out

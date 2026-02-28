"""
RAG service — Milestone 2.
Chunk + embed PDFs into Chroma, retrieve relevant chunks per asset_id.
Asset isolation is enforced: every collection is named asset_{asset_id}.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

# Optional imports — RAG is only active when M2 dependencies are installed.
try:
    import chromadb
    from openai import AsyncOpenAI

    _CHROMA_AVAILABLE = True
except ImportError:
    _CHROMA_AVAILABLE = False
    logger.info("chromadb/openai not installed — RAG disabled (install M2 deps)")

_chroma_client: Optional[object] = None


def _get_chroma_client():
    global _chroma_client
    if not _CHROMA_AVAILABLE:
        return None
    if _chroma_client is None:
        _chroma_client = chromadb.Client()
    return _chroma_client


async def _embed_text(text: str) -> list[float]:
    """Embed a single text string using OpenAI text-embedding-3-small."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    client = AsyncOpenAI(api_key=api_key)
    response = await client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )
    return response.data[0].embedding


async def ingest_pdf(pdf_path: str, asset_id: str) -> int:
    """
    Load a PDF, chunk it, embed each chunk, and store in Chroma under
    the collection for this asset_id.  Returns the number of chunks stored.
    Asset isolation: collection name = "asset_{asset_id}".
    """
    if not _CHROMA_AVAILABLE:
        logger.warning("RAG dependencies not installed; skipping PDF ingest")
        return 0

    from langchain_community.document_loaders import PyPDFLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    if not chunks:
        logger.warning("PDF produced no chunks: %s", pdf_path)
        return 0

    client = _get_chroma_client()
    collection = client.get_or_create_collection(name=f"asset_{asset_id}")

    ids = [f"{asset_id}_chunk_{i}" for i in range(len(chunks))]
    texts = [chunk.page_content for chunk in chunks]

    # Embed all chunks
    embeddings = []
    for text in texts:
        emb = await _embed_text(text)
        embeddings.append(emb)

    collection.add(documents=texts, embeddings=embeddings, ids=ids)
    logger.info("Ingested %d chunks for asset %s", len(chunks), asset_id)
    return len(chunks)


async def retrieve_relevant_chunks(
    query: str, asset_id: str, n_results: int = 3
) -> str:
    """
    Retrieve the top-n most relevant chunks for this query, filtered to
    asset_id only (no cross-asset data leakage).
    Returns an empty string if no PDF has been indexed for this asset.
    """
    if not _CHROMA_AVAILABLE:
        return ""

    client = _get_chroma_client()

    try:
        collection = client.get_collection(name=f"asset_{asset_id}")
    except Exception:
        # No PDF indexed for this asset yet — that's fine
        return ""

    total = collection.count()
    if total == 0:
        return ""

    query_emb = await _embed_text(query)

    results = collection.query(
        query_embeddings=[query_emb],
        n_results=min(n_results, total),
    )

    docs = results.get("documents", [[]])[0]
    if not docs:
        return ""

    return "\n\n---\n\n".join(docs)

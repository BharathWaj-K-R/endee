"""Endee-compatible vector storage layer backed by Chroma for the current implementation."""

from __future__ import annotations

from functools import lru_cache
import os
from typing import TYPE_CHECKING

from langchain_core.embeddings import Embeddings
from langchain.schema import Document

from utils.config import get_settings

if TYPE_CHECKING:
    from langchain_core.vectorstores import VectorStore


class SentenceTransformerEmbeddings(Embeddings):
    """LangChain-compatible wrapper around SentenceTransformers embeddings."""

    def __init__(self, model_name: str) -> None:
        """Load the embedding model once using a compact default model."""
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device="cpu")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of document chunks."""
        embeddings = self.model.encode(
            texts,
            batch_size=8,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return [embedding.tolist() for embedding in embeddings]

    def embed_query(self, text: str) -> list[float]:
        """Embed a single user query."""
        embedding = self.model.encode(
            text,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return embedding.tolist()


class SimpleHashEmbeddings(Embeddings):
    """Lightweight deterministic embeddings for low-memory environments."""

    def __init__(self, dimensions: int = 256) -> None:
        """Initialize the simple embedding size."""
        self.dimensions = dimensions

    def _hash(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for idx, char in enumerate(text):
            vector[(idx + ord(char)) % self.dimensions] += 1.0
        return vector

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of document chunks."""
        return [self._hash(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        """Embed a single user query."""
        return self._hash(text)


class EndeeVectorStoreAdapter:
    """Vector store abstraction designed to be compatible with Endee vector database."""

    def __init__(self, backend: "VectorStore") -> None:
        """Wrap the active vector backend behind an Endee-ready interface."""
        # NOTE:
        # This abstraction ensures that switching to Endee requires changes
        # only in this module, leaving ingestion, retrieval, and generation untouched.
        self.backend = backend

    def add_documents(self, documents: list[Document]) -> None:
        """Insert embedded document chunks into the active collection.

        This module is designed to be compatible with Endee vector database.
        Endee API calls for batch upserts would be inserted here when replacing the backend.
        """
        self.backend.add_documents(documents)

    def similarity_search(
        self,
        query: str,
        top_k: int = 4,
        score_threshold: float | None = None,
    ) -> list[Document]:
        """Run semantic retrieval and optionally filter weak matches.

        This module is designed to be compatible with Endee vector database.
        Endee similarity search API calls would be inserted here when replacing the backend.
        """
        if score_threshold is not None:
            results = self.backend.similarity_search_with_relevance_scores(query, k=top_k)
            filtered_documents = [
                document for document, score in results if score >= score_threshold
            ]
            return filtered_documents

        return self.backend.similarity_search(query, k=top_k)

    def delete_collection(self) -> None:
        """Delete the active collection through the current backend abstraction.

        This module is designed to be compatible with Endee vector database.
        Endee collection deletion API calls would be inserted here when replacing the backend.
        """
        if hasattr(self.backend, "delete_collection"):
            self.backend.delete_collection()


@lru_cache(maxsize=1)
def get_embedding_function() -> SentenceTransformerEmbeddings:
    """Return a shared embedding function instance."""
    settings = get_settings()
    provider = settings.embedding_provider.lower()

    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAI embeddings.")
        return OpenAIEmbeddings(
            api_key=settings.openai_api_key,
            model=settings.openai_embedding_model,
        )

    if provider == "simple":
        return SimpleHashEmbeddings()

    return SentenceTransformerEmbeddings(settings.embedding_model_name)


@lru_cache(maxsize=1)
def get_vector_store() -> EndeeVectorStoreAdapter:
    """Return the active Endee-compatible vector storage layer."""
    from langchain_chroma import Chroma

    settings = get_settings()
    chroma_backend = Chroma(
        collection_name=settings.chroma_collection_name,
        persist_directory=settings.chroma_dir,
        embedding_function=get_embedding_function(),
    )
    return EndeeVectorStoreAdapter(backend=chroma_backend)

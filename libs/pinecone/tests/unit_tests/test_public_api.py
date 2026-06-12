"""
Public-API contract tests for langchain-pinecone.

test_public_signatures_stable  — pins parameter names for every exported class's
    constructor and key public methods. A failure here means a parameter was
    renamed, removed, or added as required (all breaking changes). This requires an
    explicit API-contract decision (see docs/conventions/api-stability.md), not a
    quick fix.

test_critical_defaults_stable  — pins specific defaults callers rely on
    (k=4, text_key="text", batch_size=32, truncate="END"). Changing these is a
    silent breaking change that signature-name tests alone won't catch.

test_pydantic_model_fields_stable  — pins field names for the Pydantic model classes
    (PineconeEmbeddings, PineconeSparseEmbeddings, PineconeRerank).

test_pinecone_alias_is_deprecated  — asserts the deprecated Pinecone alias emits a
    DeprecationWarning on instantiation and remains a subclass of PineconeVectorStore.
"""

import inspect
from typing import Any
from unittest.mock import Mock

import pytest

from langchain_pinecone import (
    Pinecone,
    PineconeRerank,
    PineconeVectorStore,
)
from langchain_pinecone.embeddings import PineconeEmbeddings, PineconeSparseEmbeddings
from langchain_pinecone.vectorstores_sparse import PineconeSparseVectorStore


def _param_names(method: Any) -> set[str]:
    """Return explicit parameter names, excluding self/cls/*args/**kwargs."""
    return {
        name
        for name, p in inspect.signature(method).parameters.items()
        if name not in ("self", "cls")
        and p.kind
        not in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL)
    }


_PVS = PineconeVectorStore
_SPVS = PineconeSparseVectorStore
_PE = PineconeEmbeddings
_SPE = PineconeSparseEmbeddings
_PR = PineconeRerank

_SIGNATURE_CASES: list[tuple[Any, set[str]]] = [
    # ── PineconeVectorStore ─────────────────────────────────────────────────
    (
        _PVS.__init__,
        {
            "index",
            "embedding",
            "text_key",
            "namespace",
            "distance_strategy",
            "pinecone_api_key",
            "index_name",
            "host",
        },
    ),
    (
        _PVS.add_texts,
        {
            "texts",
            "metadatas",
            "ids",
            "namespace",
            "batch_size",
            "embedding_chunk_size",
            "async_req",
            "id_prefix",
        },
    ),
    (
        _PVS.aadd_texts,
        {
            "texts",
            "metadatas",
            "ids",
            "namespace",
            "batch_size",
            "embedding_chunk_size",
            "id_prefix",
        },
    ),
    (_PVS.similarity_search, {"query", "k", "filter", "namespace"}),
    (_PVS.asimilarity_search, {"query", "k", "filter", "namespace"}),
    (_PVS.similarity_search_with_score, {"query", "k", "filter", "namespace"}),
    (_PVS.asimilarity_search_with_score, {"query", "k", "filter", "namespace"}),
    (
        _PVS.max_marginal_relevance_search,
        {"query", "k", "fetch_k", "lambda_mult", "filter", "namespace"},
    ),
    (
        _PVS.amax_marginal_relevance_search,
        {"query", "k", "fetch_k", "lambda_mult", "filter", "namespace"},
    ),
    (_PVS.delete, {"ids", "delete_all", "namespace", "filter"}),
    (
        _PVS.from_texts,
        {
            "texts",
            "embedding",
            "metadatas",
            "ids",
            "batch_size",
            "text_key",
            "namespace",
            "index_name",
            "upsert_kwargs",
            "pool_threads",
            "embeddings_chunk_size",
            "async_req",
            "id_prefix",
        },
    ),
    (
        _PVS.afrom_texts,
        {
            "texts",
            "embedding",
            "metadatas",
            "ids",
            "batch_size",
            "text_key",
            "namespace",
            "index_name",
            "upsert_kwargs",
            "embeddings_chunk_size",
            "id_prefix",
        },
    ),
    (
        _PVS.from_existing_index,
        {"index_name", "embedding", "text_key", "namespace", "pool_threads"},
    ),
    # ── PineconeSparseVectorStore ───────────────────────────────────────────
    (
        _SPVS.__init__,
        {
            "index",
            "embedding",
            "text_key",
            "namespace",
            "distance_strategy",
            "pinecone_api_key",
            "index_name",
            "host",
        },
    ),
    # ── PineconeEmbeddings ──────────────────────────────────────────────────
    (_PE.embed_documents, {"texts"}),
    (_PE.embed_query, {"text"}),
    (_PE.aembed_documents, {"texts"}),
    (_PE.aembed_query, {"text"}),
    # ── PineconeSparseEmbeddings ────────────────────────────────────────────
    (_SPE.embed_documents, {"texts"}),
    (_SPE.embed_query, {"text"}),
    (_SPE.aembed_documents, {"texts"}),
    (_SPE.aembed_query, {"text"}),
    # ── PineconeRerank ──────────────────────────────────────────────────────
    (
        _PR.rerank,
        {"documents", "query", "rank_fields", "model", "top_n", "truncate"},
    ),
    (
        _PR.arerank,
        {"documents", "query", "rank_fields", "model", "top_n", "truncate"},
    ),
    (_PR.compress_documents, {"documents", "query", "callbacks"}),
    (_PR.acompress_documents, {"documents", "query", "callbacks"}),
]


@pytest.mark.parametrize(
    "method,expected_params",
    _SIGNATURE_CASES,
    ids=[m.__qualname__ for m, _ in _SIGNATURE_CASES],
)
def test_public_signatures_stable(method: Any, expected_params: set[str]) -> None:
    actual = _param_names(method)
    assert actual == expected_params


# ── Critical defaults ──────────────────────────────────────────────────────────


def _default(method: Any, param: str) -> Any:
    return inspect.signature(method).parameters[param].default


def test_critical_defaults_stable() -> None:
    # k=4 is the universal search default
    assert _default(_PVS.similarity_search, "k") == 4
    assert _default(_PVS.asimilarity_search, "k") == 4
    assert _default(_PVS.similarity_search_with_score, "k") == 4
    assert _default(_PVS.asimilarity_search_with_score, "k") == 4
    assert _default(_PVS.max_marginal_relevance_search, "k") == 4
    assert _default(_PVS.amax_marginal_relevance_search, "k") == 4

    # MMR-specific defaults
    assert _default(_PVS.max_marginal_relevance_search, "fetch_k") == 20
    assert _default(_PVS.amax_marginal_relevance_search, "fetch_k") == 20
    assert _default(_PVS.max_marginal_relevance_search, "lambda_mult") == 0.5
    assert _default(_PVS.amax_marginal_relevance_search, "lambda_mult") == 0.5

    # text_key="text" is the default metadata field callers index on
    assert _default(_PVS.__init__, "text_key") == "text"
    assert _default(_PVS.from_texts, "text_key") == "text"
    assert _default(_PVS.afrom_texts, "text_key") == "text"
    assert _default(_PVS.from_existing_index, "text_key") == "text"

    # batch_size=32 is the default upsert/embed batch
    assert _default(_PVS.add_texts, "batch_size") == 32
    assert _default(_PVS.aadd_texts, "batch_size") == 32
    assert _default(_PVS.from_texts, "batch_size") == 32
    assert _default(_PVS.afrom_texts, "batch_size") == 32

    # rerank truncate default
    assert _default(_PR.rerank, "truncate") == "END"
    assert _default(_PR.arerank, "truncate") == "END"


# ── Pydantic model fields ──────────────────────────────────────────────────────


def test_pydantic_model_fields_stable() -> None:
    embedding_fields = set(PineconeEmbeddings.model_fields)
    assert {
        "model",
        "batch_size",
        "query_params",
        "document_params",
        "dimension",
        "show_progress_bar",
        "pinecone_api_key",
    } <= embedding_fields

    sparse_embedding_fields = set(PineconeSparseEmbeddings.model_fields)
    assert {
        "model",
        "batch_size",
        "query_params",
        "document_params",
        "dimension",
        "show_progress_bar",
        "pinecone_api_key",
    } <= sparse_embedding_fields

    rerank_fields = set(PineconeRerank.model_fields)
    assert {
        "top_n",
        "model",
        "pinecone_api_key",
        "rank_fields",
        "return_documents",
    } <= rerank_fields


# ── Deprecated alias ───────────────────────────────────────────────────────────


def test_pinecone_alias_is_deprecated() -> None:
    assert issubclass(Pinecone, PineconeVectorStore)

    mock_index = Mock()
    mock_index.config.host = "example.org"
    mock_index.config.api_key = "test-key"
    mock_embedding = Mock()

    with pytest.warns(DeprecationWarning):
        Pinecone(index=mock_index, embedding=mock_embedding)

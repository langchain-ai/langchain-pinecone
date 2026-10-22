"""
Asserts that the conditional import shims in vectorstores.py and embeddings.py
resolve to real, non-None classes on the installed pinecone version.

Both sources use try/except ImportError to pick the correct module path across
pinecone major versions. If both branches of a shim fail simultaneously the
binding goes missing silently; this test file catches that failure mode.
"""

import inspect

import langchain_pinecone.embeddings as _embeddings_mod
import langchain_pinecone.vectorstores as _vectorstores_mod
from langchain_pinecone.vectorstores import ApplyResult, _Index, _IndexAsyncio


def test_vectorstores_index_shim_resolves() -> None:
    assert _Index is not None
    assert inspect.isclass(_Index)


def test_vectorstores_index_asyncio_shim_resolves() -> None:
    assert _IndexAsyncio is not None
    assert inspect.isclass(_IndexAsyncio)


def test_vectorstores_apply_result_shim_resolves() -> None:
    assert ApplyResult is not None
    assert inspect.isclass(ApplyResult)


def test_embeddings_embeddings_list_shim_resolves() -> None:
    EmbeddingsList = _embeddings_mod.EmbeddingsList  # type: ignore[attr-defined]
    assert EmbeddingsList is not None
    assert inspect.isclass(EmbeddingsList)


def test_vectorstores_shim_names_in_module_namespace() -> None:
    assert hasattr(_vectorstores_mod, "_Index")
    assert hasattr(_vectorstores_mod, "_IndexAsyncio")
    assert hasattr(_vectorstores_mod, "ApplyResult")


def test_embeddings_shim_name_in_module_namespace() -> None:
    assert hasattr(_embeddings_mod, "EmbeddingsList")

import os
import time
import uuid
from datetime import datetime
from typing import Any, List

import pytest
from langchain_core.utils import convert_to_secret_str
from pinecone import AwsRegion, CloudProvider, Metric, ServerlessSpec, VectorType
from pytest_mock import MockerFixture  # type: ignore[import-not-found]

from langchain_pinecone.embeddings import PineconeSparseEmbeddings
from langchain_pinecone.vectorstores_sparse import PineconeSparseVectorStore
from tests.integration_tests.test_vectorstores import (
    DEFAULT_SLEEP,
    _apoll_for_results,
    _poll_for_results,
    _sweep_stale_langchain_test_indexes,
)

SPARSE_MODEL_NAME = "pinecone-sparse-english-v0"
_SPARSE_PREFIX = "langchain-test-sparse-"
INDEX_NAME = f"{_SPARSE_PREFIX}{datetime.now().strftime('%Y%m%d%H%M%S')}"

requires_api_key = pytest.mark.skipif(
    not os.environ.get("PINECONE_API_KEY"),
    reason="Pinecone API key not set",
)


class TestPineconeSparseVectorStore:
    index: Any
    pc: Any

    @classmethod
    def setup_class(cls) -> None:
        if not os.environ.get("PINECONE_API_KEY"):
            pytest.skip("PINECONE_API_KEY not set")

        import pinecone

        client = pinecone.Pinecone()
        _sweep_stale_langchain_test_indexes(client, _SPARSE_PREFIX)
        if client.has_index(name=INDEX_NAME):
            client.delete_index(INDEX_NAME)
            time.sleep(DEFAULT_SLEEP)
        client.create_index(
            name=INDEX_NAME,
            metric=Metric.DOTPRODUCT,
            spec=ServerlessSpec(cloud=CloudProvider.AWS, region=AwsRegion.US_WEST_2),
            vector_type=VectorType.SPARSE,
        )
        ready_deadline = time.monotonic() + 300
        while not client.describe_index(INDEX_NAME).status["ready"]:
            if time.monotonic() > ready_deadline:
                raise TimeoutError(f"Index {INDEX_NAME} not ready after 300 s")
            time.sleep(1)
        cls.index = client.Index(INDEX_NAME)
        cls.pc = client

    @classmethod
    def teardown_class(cls) -> None:
        cls.pc.delete_index(INDEX_NAME)

    @pytest.fixture(autouse=True)
    def patch_sparse_model_listing(self, mocker: MockerFixture) -> None:
        mocker.patch(
            "langchain_pinecone.embeddings.PineconeEmbeddings.list_supported_models",
            return_value=[{"model": SPARSE_MODEL_NAME}],
        )

    @pytest.fixture
    def embeddings(self) -> PineconeSparseEmbeddings:
        return PineconeSparseEmbeddings(
            model=SPARSE_MODEL_NAME,
            pinecone_api_key=convert_to_secret_str(
                os.environ.get("PINECONE_API_KEY", "")
            ),
        )

    @pytest.fixture
    def store(self, embeddings: PineconeSparseEmbeddings) -> PineconeSparseVectorStore:
        return PineconeSparseVectorStore(
            index=self.index,
            embedding=embeddings,
        )

    @requires_api_key
    def test_add_texts_and_similarity_search(
        self,
        store: PineconeSparseVectorStore,
    ) -> None:
        unique_id = uuid.uuid4().hex
        sentinel = f"langchain sparse test {unique_id}"
        texts: List[str] = ["foo", "bar", "baz", sentinel]
        namespace = f"sync-search-{unique_id}"

        store.add_texts(texts, namespace=namespace)
        results = _poll_for_results(
            lambda: store.similarity_search(sentinel, k=1, namespace=namespace),
        )
        assert len(results) >= 1
        assert results[0].page_content == sentinel

    @requires_api_key
    def test_similarity_search_with_score(
        self,
        store: PineconeSparseVectorStore,
    ) -> None:
        unique_id = uuid.uuid4().hex
        sentinel = f"langchain sparse score test {unique_id}"
        texts: List[str] = ["foo", "bar", sentinel]
        namespace = f"sync-score-{unique_id}"

        store.add_texts(texts, namespace=namespace)
        results = _poll_for_results(
            lambda: store.similarity_search_with_score(
                sentinel, k=3, namespace=namespace
            ),
            min_count=3,
        )
        assert len(results) >= 1
        docs_and_scores = results
        best_doc, best_score = max(docs_and_scores, key=lambda x: x[1])
        assert best_doc.page_content == sentinel
        for doc, score in docs_and_scores:
            if doc.page_content != sentinel:
                assert best_score >= score

    @requires_api_key
    @pytest.mark.asyncio
    async def test_aadd_texts_and_asimilarity_search(
        self,
        store: PineconeSparseVectorStore,
    ) -> None:
        unique_id = uuid.uuid4().hex
        sentinel = f"langchain sparse async test {unique_id}"
        texts: List[str] = ["foo", "bar", "baz", sentinel]
        namespace = f"async-search-{unique_id}"

        await store.aadd_texts(texts, namespace=namespace)
        results = await _apoll_for_results(
            lambda: store.asimilarity_search(sentinel, k=1, namespace=namespace),
        )
        assert len(results) >= 1
        assert results[0].page_content == sentinel

    @requires_api_key
    @pytest.mark.asyncio
    async def test_asimilarity_search_with_score(
        self,
        store: PineconeSparseVectorStore,
    ) -> None:
        unique_id = uuid.uuid4().hex
        sentinel = f"langchain sparse async score test {unique_id}"
        texts: List[str] = ["foo", "bar", sentinel]
        namespace = f"async-score-{unique_id}"

        await store.aadd_texts(texts, namespace=namespace)
        results = await _apoll_for_results(
            lambda: store.asimilarity_search_with_score(
                sentinel, k=3, namespace=namespace
            ),
            min_count=3,
        )
        assert len(results) >= 1
        docs_and_scores = results
        best_doc, best_score = max(docs_and_scores, key=lambda x: x[1])
        assert best_doc.page_content == sentinel
        for doc, score in docs_and_scores:
            if doc.page_content != sentinel:
                assert best_score >= score

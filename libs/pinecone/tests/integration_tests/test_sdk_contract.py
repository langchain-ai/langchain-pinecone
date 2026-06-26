"""Integration tests asserting the raw Pinecone SDK response shapes that vectorstores.py consumes.

These tests protect the exact access patterns used in langchain_pinecone/vectorstores.py against
structural changes in the Pinecone SDK (e.g. the pinecone-8 upgrade in SDK-001). A structural
change in one of the covered surfaces will produce a clear, targeted failure here rather than
surfacing as confusing empty results or an AttributeError deep in the call stack.

Covered response surfaces (the exact ones the source consumes):
- Index.query(): results["matches"], match["id"]/["score"]/["metadata"]/["values"]
- Pinecone.list_indexes(): iterable, each entry exposes .name
- Index.describe_index_stats(): stats["namespaces"][ns]["vector_count"], stats["total_vector_count"]
- index.config.host and index.config.api_key (used in PineconeVectorStore.__init__ when index=)

Source touchpoints:
- vectorstores.py ~lines 593-641: results["matches"], res["metadata"], res.get("id"), res["score"]
- vectorstores.py ~lines 765-769: item["values"], results["matches"][i]["metadata"]
- vectorstores.py ~lines 231-235: index.config.host, index.config.api_key
- vectorstores.py ~line 912: client.list_indexes(), idx.name
- _poll_for_vector_count helper: stats["namespaces"][ns]["vector_count"], stats["total_vector_count"]
"""

import os
import time
import uuid
from datetime import datetime
from typing import Any, Generator, List

import pinecone  # type: ignore
import pytest  # type: ignore[import-not-found]
from langchain_core.utils import convert_to_secret_str
from pinecone import AwsRegion, CloudProvider, Metric, ServerlessSpec
from pytest_mock import MockerFixture  # type: ignore[import-not-found]

from langchain_pinecone import PineconeEmbeddings, PineconeVectorStore

pytestmark = pytest.mark.skipif(
    not os.environ.get("PINECONE_API_KEY"), reason="Pinecone API key not set"
)

_INDEX_NAME = f"langchain-test-sdk-contract-{datetime.now().strftime('%Y%m%d%H%M%S')}"
_MODEL = "multilingual-e5-large"
_DIMENSION = 1024
_FRESHNESS_MAX_WAIT: float = 60.0
_FRESHNESS_INTERVAL: float = 2.0


def _poll_for_vector_count(
    index: Any,
    namespace: str,
    expected_count: int,
    *,
    max_wait: float = _FRESHNESS_MAX_WAIT,
    interval: float = _FRESHNESS_INTERVAL,
) -> Any:
    """Poll describe_index_stats until namespace has at least expected_count vectors."""
    deadline = time.monotonic() + max_wait
    stats: Any = {}
    while time.monotonic() < deadline:
        stats = index.describe_index_stats()
        count = stats.get("namespaces", {}).get(namespace, {}).get("vector_count", 0)
        if count >= expected_count:
            return stats
        time.sleep(interval)
    return stats


class TestPineconeSdkContract:
    """Assert raw SDK response shapes that langchain_pinecone/vectorstores.py depends on."""

    index: "pinecone.Index"
    pc: "pinecone.Pinecone"

    @classmethod
    def setup_class(cls) -> None:
        import pinecone

        client = pinecone.Pinecone()
        if client.has_index(name=_INDEX_NAME):
            client.delete_index(_INDEX_NAME)
            time.sleep(5)
        client.create_index(
            name=_INDEX_NAME,
            dimension=_DIMENSION,
            metric=Metric.COSINE,
            spec=ServerlessSpec(cloud=CloudProvider.AWS, region=AwsRegion.US_WEST_2),
        )
        deadline = time.monotonic() + 300
        while not client.describe_index(_INDEX_NAME).status["ready"]:
            if time.monotonic() > deadline:
                raise TimeoutError(f"Index {_INDEX_NAME} not ready after 300 s")
            time.sleep(1)
        cls.index = client.Index(_INDEX_NAME)
        cls.pc = client

    @classmethod
    def teardown_class(cls) -> None:
        cls.pc.delete_index(_INDEX_NAME)

    @pytest.fixture(autouse=True)
    def patch_pinecone_model_listing(self, mocker: MockerFixture) -> None:
        mocker.patch(
            "langchain_pinecone.embeddings.PineconeEmbeddings.list_supported_models",
            return_value=[{"model": _MODEL}],
        )

    @pytest.fixture
    def embeddings(self) -> PineconeEmbeddings:
        return PineconeEmbeddings(
            model=_MODEL,
            pinecone_api_key=convert_to_secret_str(
                os.environ.get("PINECONE_API_KEY", "")
            ),
            dimension=_DIMENSION,
        )

    @pytest.fixture
    def _contract_ns(self) -> Generator[str, None, None]:
        """Yield a unique namespace and delete_all it on teardown."""
        ns = f"contract-{uuid.uuid4().hex[:8]}"
        yield ns
        try:
            self.index.delete(delete_all=True, namespace=ns)
        except Exception:
            pass

    def test_query_response_shape(
        self, embeddings: PineconeEmbeddings, _contract_ns: str
    ) -> None:
        """index.query() result exposes matches; each match exposes id, score, metadata, values.

        Mirrors the access pattern in vectorstores.py similarity_search_by_vector_with_score
        (~lines 593-611) and max_marginal_relevance_search_by_vector (~lines 753-769).
        """
        ns = _contract_ns
        texts = ["the quick brown fox", "a lazy dog resting by the river"]
        PineconeVectorStore.from_texts(
            texts=texts,
            embedding=embeddings,
            index_name=_INDEX_NAME,
            namespace=ns,
        )
        vec: List[float] = embeddings.embed_query("quick fox")

        # Poll until query returns non-empty matches — mirrors post-upsert eventual consistency.
        results: Any = {}
        deadline = time.monotonic() + _FRESHNESS_MAX_WAIT
        while time.monotonic() < deadline:
            results = self.index.query(
                vector=vec,
                top_k=2,
                include_metadata=True,
                include_values=True,
                namespace=ns,
            )
            if results["matches"]:  # same subscript the source uses
                break
            time.sleep(_FRESHNESS_INTERVAL)

        # Assert the structure the source reads (same access patterns).
        matches = results["matches"]  # vectorstores.py: for res in results["matches"]
        assert isinstance(matches, list)
        assert len(matches) > 0, "Expected at least one match after upsert"

        for match in matches:
            _ = match.get("id")  # vectorstores.py: res.get("id")
            _ = match["score"]  # vectorstores.py: res["score"]
            _ = match["metadata"]  # vectorstores.py: res["metadata"]
            values = match["values"]  # vectorstores.py (MMR): item["values"]
            assert isinstance(values, list)
            assert len(values) == _DIMENSION

    def test_list_indexes_response_shape(self) -> None:
        """Pinecone.list_indexes() is iterable; entries expose .name.

        Mirrors vectorstores.py ~line 912: for idx in client.list_indexes() / idx.name,
        and the _sweep_stale helper: for idx in client.list_indexes(): name = idx.name.
        """
        indexes = self.pc.list_indexes()
        names: List[str] = [idx.name for idx in indexes]  # source access pattern
        assert _INDEX_NAME in names

    def test_describe_index_stats_response_shape(
        self, embeddings: PineconeEmbeddings, _contract_ns: str
    ) -> None:
        """describe_index_stats() exposes namespaces→<ns>→vector_count and total_vector_count.

        Mirrors the _poll_for_vector_count helper in test_vectorstores.py and
        the setup autouse fixture: stats["namespaces"][ns]["vector_count"],
        stats["total_vector_count"].
        """
        ns = _contract_ns
        texts = ["the quick brown fox", "a lazy dog resting by the river"]
        PineconeVectorStore.from_texts(
            texts=texts,
            embedding=embeddings,
            index_name=_INDEX_NAME,
            namespace=ns,
        )
        stats = _poll_for_vector_count(self.index, ns, expected_count=2)

        # Same access patterns the source and test helpers use.
        _ = stats.get("namespaces", {})  # source: stats.get("namespaces", {})
        total = stats["total_vector_count"]  # helper: stats["total_vector_count"]
        assert total >= 2, "Expected total_vector_count to reflect upserted docs"

        ns_stats = stats["namespaces"].get(ns, {})
        count = ns_stats[
            "vector_count"
        ]  # helper: stats["namespaces"][ns]["vector_count"]
        assert count >= 2, "Expected vector_count to reflect upserted docs in namespace"

    def test_index_config_shape(self) -> None:
        """index.config.host and index.config.api_key are accessible.

        Mirrors vectorstores.py __init__ (~lines 231-235) where an externally-provided
        index object is used to derive host and api_key:
          self._index_host = index.config.host
          ... SecretStr(index.config.api_key) ...
        """
        host: str = self.index.config.host  # vectorstores.py: index.config.host
        api_key = self.index.config.api_key  # vectorstores.py: index.config.api_key

        assert isinstance(host, str)
        assert host, "index.config.host must be a non-empty string"
        assert api_key is not None, "index.config.api_key must not be None"

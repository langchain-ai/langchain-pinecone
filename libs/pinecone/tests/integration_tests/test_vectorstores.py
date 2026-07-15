import asyncio
import os
import time
import uuid
from datetime import datetime
from typing import Any, Awaitable, Callable, Generator, List

import numpy as np
import pinecone  # type: ignore
import pytest  # type: ignore[import-not-found]
from langchain_core.documents import Document
from langchain_core.utils import convert_to_secret_str
from pinecone import AwsRegion, CloudProvider, Metric, ServerlessSpec
from pytest_mock import MockerFixture  # type: ignore[import-not-found]

from langchain_pinecone import PineconeEmbeddings, PineconeVectorStore

pytestmark = pytest.mark.skipif(
    not os.environ.get("PINECONE_API_KEY"), reason="Pinecone API key not set"
)

# unique name of the index for this test run
INDEX_NAME = f"langchain-test-vectorstores-{datetime.now().strftime('%Y%m%d%H%M%S')}"
NAMESPACE_NAME = "langchain-test-namespace"  # name of the namespace
# Pinecone-hosted inference model used to embed test text. Using Pinecone's own
# inference endpoint keeps all integration traffic on the service under test
# (no third-party embedding provider / API key required).
MODEL = "multilingual-e5-large"
DIMENSION = 1024  # dimension of the `multilingual-e5-large` embeddings

DEFAULT_SLEEP = 15

_FRESHNESS_MAX_WAIT: float = 60.0
_FRESHNESS_INTERVAL: float = 2.0

# Name prefix for indexes this suite creates.  The suffix is a YYYYMMDDHHMMSS timestamp
# so stale indexes from cancelled CI runs can be identified and swept by age.
_VECTORSTORE_PREFIX = "langchain-test-vectorstores-"


def _sweep_stale_langchain_test_indexes(
    client: "pinecone.Pinecone",
    prefix: str,
    age_threshold_s: int = 7200,
) -> None:
    """Delete own stale test indexes leaked by cancelled CI runs.

    Matches only names of the form ``<prefix>YYYYMMDDHHMMSS``.  Indexes younger than
    ``age_threshold_s`` seconds (default 2 h) are left alone so parallel CI jobs are
    not disturbed.  Never touches names that do not match (e.g. ``idx-*`` from other
    suites).  All errors are swallowed so this sweep never blocks ``setup_class``.
    """
    try:
        now = datetime.utcnow()
        for idx in client.list_indexes():
            name: str = idx.name
            if not name.startswith(prefix):
                continue
            ts_str = name[len(prefix) :]
            try:
                created = datetime.strptime(ts_str, "%Y%m%d%H%M%S")
            except ValueError:
                continue
            if (now - created).total_seconds() > age_threshold_s:
                try:
                    client.delete_index(name)
                except Exception:
                    pass
    except Exception:
        pass


def _poll_for_results(
    fn: Callable[[], List[Any]],
    *,
    min_count: int = 1,
    max_wait: float = _FRESHNESS_MAX_WAIT,
    interval: float = _FRESHNESS_INTERVAL,
) -> List[Any]:
    """Poll fn() until it returns at least min_count items, or timeout.

    Wraps post-upsert reads that race eventual consistency.  The caller still
    asserts on the returned list so a genuinely wrong result fails the test.
    """
    deadline = time.monotonic() + max_wait
    result: List[Any] = []
    while time.monotonic() < deadline:
        result = fn()
        if len(result) >= min_count:
            return result
        time.sleep(interval)
    return result


async def _apoll_for_results(
    fn: Callable[[], Awaitable[List[Any]]],
    *,
    min_count: int = 1,
    max_wait: float = _FRESHNESS_MAX_WAIT,
    interval: float = _FRESHNESS_INTERVAL,
) -> List[Any]:
    """Async variant of _poll_for_results."""
    deadline = time.monotonic() + max_wait
    result: List[Any] = []
    while time.monotonic() < deadline:
        result = await fn()
        if len(result) >= min_count:
            return result
        await asyncio.sleep(interval)
    return result


def _poll_for_vector_count(
    index: Any,
    namespace: str,
    expected_count: int,
    *,
    max_wait: float = _FRESHNESS_MAX_WAIT,
    interval: float = _FRESHNESS_INTERVAL,
) -> Any:
    """Poll describe_index_stats until namespace has at least expected_count vectors.

    Returns the last seen stats dict for the caller to assert on.
    """
    deadline = time.monotonic() + max_wait
    stats: Any = {}
    while time.monotonic() < deadline:
        stats = index.describe_index_stats()
        count = stats.get("namespaces", {}).get(namespace, {}).get("vector_count", 0)
        if count >= expected_count:
            return stats
        time.sleep(interval)
    return stats


def _poll_for_vector_count_at_most(
    index: Any,
    namespace: str,
    max_count: int,
    *,
    max_wait: float = _FRESHNESS_MAX_WAIT,
    interval: float = _FRESHNESS_INTERVAL,
) -> Any:
    """Poll describe_index_stats until namespace has at most max_count vectors.

    Used to assert post-delete eventual consistency. Returns the last seen stats
    dict for the caller to assert on.
    """
    deadline = time.monotonic() + max_wait
    stats: Any = {}
    while time.monotonic() < deadline:
        stats = index.describe_index_stats()
        count = stats.get("namespaces", {}).get(namespace, {}).get("vector_count", 0)
        if count <= max_count:
            return stats
        time.sleep(interval)
    return stats


class TestPinecone:
    index: "pinecone.Index"
    pc: "pinecone.Pinecone"

    @classmethod
    def setup_class(self) -> None:
        import pinecone

        client = pinecone.Pinecone()
        print(f"client: {client}")  # noqa: T201
        _sweep_stale_langchain_test_indexes(client, _VECTORSTORE_PREFIX)
        if client.has_index(name=INDEX_NAME):  # change to list comprehension
            client.delete_index(INDEX_NAME)
            time.sleep(DEFAULT_SLEEP)  # prevent race with subsequent creation
        client.create_index(
            name=INDEX_NAME,
            dimension=DIMENSION,
            metric=Metric.COSINE,
            spec=ServerlessSpec(cloud=CloudProvider.AWS, region=AwsRegion.US_WEST_2),
        )
        # Wait for the freshly created index to be ready before any test uses it.
        # Without this the first test races the index becoming queryable and its
        # similarity_search comes back empty (the OpenAI embedding round-trip used
        # to mask this by adding latency here).
        _ready_deadline = time.monotonic() + 300
        while not client.describe_index(INDEX_NAME).status["ready"]:
            if time.monotonic() > _ready_deadline:
                raise TimeoutError(f"Index {INDEX_NAME} not ready after 300 s")
            time.sleep(1)

        self.index = client.Index(INDEX_NAME)
        self.pc = client

    @classmethod
    def teardown_class(self) -> None:
        self.pc.delete_index(INDEX_NAME)

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        # delete all the vectors in the index for all possible namespaces
        index_stats = self.index.describe_index_stats()
        if index_stats["total_vector_count"] > 0:
            # Clean up the main namespace and any test-specific namespaces
            namespaces_to_clean = [
                NAMESPACE_NAME,
                f"{INDEX_NAME}-1",
                f"{INDEX_NAME}-2",
            ]
            for ns in namespaces_to_clean:
                try:
                    self.index.delete(delete_all=True, namespace=ns)
                except Exception:
                    # if namespace not found, continue
                    pass
            time.sleep(DEFAULT_SLEEP)  # wait for deletion to propagate

    @pytest.fixture(autouse=True)
    def patch_pinecone_model_listing(self, mocker: MockerFixture) -> None:
        # Avoid a live model-catalog lookup during PineconeEmbeddings construction;
        # the embed calls themselves still hit the live inference endpoint.
        mocker.patch(
            "langchain_pinecone.embeddings.PineconeEmbeddings.list_supported_models",
            return_value=[{"model": MODEL}],
        )

    @pytest.fixture
    def embeddings(self) -> PineconeEmbeddings:
        return PineconeEmbeddings(
            model=MODEL,
            pinecone_api_key=convert_to_secret_str(
                os.environ.get("PINECONE_API_KEY", "")
            ),
            dimension=DIMENSION,
        )

    @pytest.fixture
    def texts(self) -> List[str]:
        return ["foo", "bar", "baz"]

    def test_from_texts(self, texts: List[str], embeddings: PineconeEmbeddings) -> None:
        """Test end to end construction and search."""
        unique_id = uuid.uuid4().hex
        needs = f"foobuu {unique_id} booo"
        texts.insert(0, needs)

        docsearch = PineconeVectorStore.from_texts(
            texts=texts,
            embedding=embeddings,
            index_name=INDEX_NAME,
            namespace=NAMESPACE_NAME,
        )
        output = _poll_for_results(
            lambda: docsearch.similarity_search(
                unique_id, k=1, namespace=NAMESPACE_NAME
            ),
        )
        output[0].id = None  # overwrite ID for ease of comparison
        assert output == [Document(page_content=needs)]

    @pytest.mark.asyncio
    async def test_afrom_texts(
        self, texts: List[str], embeddings: PineconeEmbeddings
    ) -> None:
        """Test end to end construction and search."""
        unique_id = uuid.uuid4().hex
        needs = f"foobuu {unique_id} booo"
        texts.insert(0, needs)
        print(f"texts: {texts}")  # noqa: T201
        docsearch = await PineconeVectorStore.afrom_texts(
            texts=texts,
            embedding=embeddings,
            index_name=INDEX_NAME,
            namespace=f"{NAMESPACE_NAME}-afrom-texts",
        )
        output = await _apoll_for_results(
            lambda: docsearch.asimilarity_search(
                needs,
                k=1,
                namespace=f"{NAMESPACE_NAME}-afrom-texts",
            ),
        )
        print(f"output: {output}")  # noqa: T201
        output[0].id = None  # overwrite ID for ease of comparison
        assert output == [Document(page_content=needs)]

    def test_from_texts_with_metadatas(
        self, texts: List[str], embeddings: PineconeEmbeddings
    ) -> None:
        """Test end to end construction and search."""

        unique_id = uuid.uuid4().hex
        needs = f"foobuu {unique_id} booo"
        texts = [needs] + texts

        metadatas = [{"page": i} for i in range(len(texts))]

        namespace = f"{NAMESPACE_NAME}-md"
        docsearch = PineconeVectorStore.from_texts(
            texts,
            embeddings,
            index_name=INDEX_NAME,
            metadatas=metadatas,
            namespace=namespace,
        )
        output = _poll_for_results(
            lambda: docsearch.similarity_search(needs, k=1, namespace=namespace),
        )

        output[0].id = None
        # TODO: why metadata={"page": 0.0}) instead of {"page": 0}?
        assert output == [Document(page_content=needs, metadata={"page": 0.0})]

    @pytest.mark.asyncio
    async def test_afrom_texts_with_metadatas(
        self, texts: List[str], embeddings: PineconeEmbeddings
    ) -> None:
        """Test end to end construction and search."""

        unique_id = uuid.uuid4().hex
        needs = f"foobuu {unique_id} booo"
        texts = [needs] + texts

        metadatas = [{"page": i} for i in range(len(texts))]

        namespace = f"{NAMESPACE_NAME}-md"
        docsearch = await PineconeVectorStore.afrom_texts(
            texts,
            embeddings,
            index_name=INDEX_NAME,
            metadatas=metadatas,
            namespace=namespace,
        )
        output = await _apoll_for_results(
            lambda: docsearch.asimilarity_search(needs, k=1, namespace=namespace),
        )

        output[0].id = None
        # TODO: why metadata={"page": 0.0}) instead of {"page": 0}?
        assert output == [Document(page_content=needs, metadata={"page": 0.0})]

    @pytest.mark.asyncio
    async def test_aadd_documents(
        self, texts: List[str], embeddings: PineconeEmbeddings
    ) -> None:
        """Test adding documents to existing index."""

        texts_1 = ["foo", "bar", "baz"]
        metadatas = [{"page": i} for i in range(len(texts_1))]
        docsearch = await PineconeVectorStore.afrom_texts(
            texts_1,
            embeddings,
            index_name=INDEX_NAME,
            metadatas=metadatas,
            namespace=f"{INDEX_NAME}-1",
        )

        texts_2 = ["foo2", "bar2", "baz2"]
        metadatas = [{"page": i} for i in range(len(texts_2))]

        docs = [
            Document(page_content=text, metadata=metadata)
            for text, metadata in zip(texts_2, metadatas)
        ]

        # Search with namespace
        await docsearch.aadd_documents(documents=docs, namespace=f"{INDEX_NAME}-2")
        output = await _apoll_for_results(
            lambda: docsearch.asimilarity_search(
                docs[0].page_content, k=1, namespace=f"{INDEX_NAME}-2"
            ),
        )
        # we expect this to return the document object with text "foo2"
        # however, it will include a modified Document object with a new ID
        # so we assert on each attribute
        assert len(output) == 1, f"Expected 1 Document result, got {output}"
        assert output[0].page_content == docs[0].page_content, (
            f"Expected page_content={docs[0].page_content}, got {output[0]}"
        )
        # check metadata fields
        assert output[0].metadata, (
            f"Expected populated metadata dictionary, got {output[0]}"
        )
        assert output[0].metadata.get("page") is not None, (
            f"Expected metadata to have page key, got {output[0]}"
        )
        # note that pinecone returns ints as floats, so we need to cast to int in comparison
        out_page = output[0].metadata.get("page")
        assert isinstance(out_page, float)
        assert int(out_page) == docs[0].metadata.get("page"), (
            f"Expected metadata={docs[0].metadata}, got {output[0]}"
        )
        assert isinstance(output[0].id, str), (
            f"Expected id to be a string, got {output[0]}"
        )

    def test_from_texts_with_scores(self, embeddings: PineconeEmbeddings) -> None:
        """Test end to end construction and search with scores and IDs."""
        texts = ["foo", "bar", "baz"]
        metadatas = [{"page": i} for i in range(len(texts))]
        docsearch = PineconeVectorStore.from_texts(
            texts,
            embeddings,
            index_name=INDEX_NAME,
            metadatas=metadatas,
            namespace=NAMESPACE_NAME,
        )
        output = _poll_for_results(
            lambda: docsearch.similarity_search_with_score(
                "foo", k=3, namespace=NAMESPACE_NAME
            ),
            min_count=3,
        )
        print(f"output: {output}")  # noqa: T201
        sorted_docs_and_scores = sorted(output, key=lambda x: x[0].metadata["page"])
        for document in sorted_docs_and_scores:
            document[0].id = None  # overwrite IDs for ease of comparison
        # TODO: why metadata={"page": 0.0}) instead of {"page": 0}, etc???
        docs_only = [doc for doc, _ in sorted_docs_and_scores]
        assert docs_only == [
            Document(page_content="foo", metadata={"page": 0.0}),
            Document(page_content="bar", metadata={"page": 1.0}),
            Document(page_content="baz", metadata={"page": 2.0}),
        ]
        # "foo" should have the highest score
        assert sorted_docs_and_scores[0][1] > sorted_docs_and_scores[1][1]
        assert sorted_docs_and_scores[0][1] > sorted_docs_and_scores[2][1]

    @pytest.mark.asyncio
    async def test_afrom_texts_with_scores(
        self, embeddings: PineconeEmbeddings
    ) -> None:
        """Test end to end construction and search with scores and IDs."""
        texts = ["foo", "bar", "baz"]
        metadatas = [{"page": i} for i in range(len(texts))]
        print("metadatas", metadatas)  # noqa: T201
        docsearch = await PineconeVectorStore.afrom_texts(
            texts,
            embeddings,
            index_name=INDEX_NAME,
            metadatas=metadatas,
            namespace=NAMESPACE_NAME,
        )
        print(texts)  # noqa: T201
        output = await _apoll_for_results(
            lambda: docsearch.asimilarity_search_with_score(
                "foo", k=3, namespace=NAMESPACE_NAME
            ),
            min_count=3,
        )
        print(f"output: {output}")  # noqa: T201
        sorted_docs_and_scores = sorted(output, key=lambda x: x[0].metadata["page"])
        for document in sorted_docs_and_scores:
            document[0].id = None  # overwrite IDs for ease of comparison
        # TODO: why metadata={"page": 0.0}) instead of {"page": 0}, etc???
        docs_only = [doc for doc, _ in sorted_docs_and_scores]
        assert docs_only == [
            Document(page_content="foo", metadata={"page": 0.0}),
            Document(page_content="bar", metadata={"page": 1.0}),
            Document(page_content="baz", metadata={"page": 2.0}),
        ]
        # "foo" should have the highest score
        assert sorted_docs_and_scores[0][1] > sorted_docs_and_scores[1][1]
        assert sorted_docs_and_scores[0][1] > sorted_docs_and_scores[2][1]
        # clear the namespace
        self.index.delete(delete_all=True, namespace=NAMESPACE_NAME)

    def test_from_existing_index_with_namespaces(
        self, embeddings: PineconeEmbeddings
    ) -> None:
        """Test that namespaces are properly handled."""
        # Create two indexes with the same name but different namespaces
        texts_1 = ["foo", "bar", "baz"]
        metadatas = [{"page": i} for i in range(len(texts_1))]
        PineconeVectorStore.from_texts(
            texts_1,
            embeddings,
            index_name=INDEX_NAME,
            metadatas=metadatas,
            namespace=f"{INDEX_NAME}-1",
        )

        texts_2 = ["foo2", "bar2", "baz2"]
        metadatas = [{"page": i} for i in range(len(texts_2))]

        PineconeVectorStore.from_texts(
            texts_2,
            embeddings,
            index_name=INDEX_NAME,
            metadatas=metadatas,
            namespace=f"{INDEX_NAME}-2",
        )

        # Search with namespace
        docsearch = PineconeVectorStore.from_existing_index(
            index_name=INDEX_NAME,
            embedding=embeddings,
            namespace=f"{INDEX_NAME}-1",
        )
        output = _poll_for_results(
            lambda: docsearch.similarity_search(
                "foo", k=20, namespace=f"{INDEX_NAME}-1"
            ),
            min_count=1,
        )
        # check that we don't get results from the other namespace
        page_contents = sorted(set([o.page_content for o in output]))
        assert all(content in ["foo", "bar", "baz"] for content in page_contents)
        assert all(content not in ["foo2", "bar2", "baz2"] for content in page_contents)

    def test_add_documents_with_ids(
        self, texts: List[str], embeddings: PineconeEmbeddings
    ) -> None:
        ids = [uuid.uuid4().hex for _ in range(len(texts))]
        PineconeVectorStore.from_texts(
            texts=texts,
            ids=ids,
            embedding=embeddings,
            index_name=INDEX_NAME,
            namespace=NAMESPACE_NAME,
        )
        index_stats = _poll_for_vector_count(self.index, NAMESPACE_NAME, len(texts))
        assert index_stats["namespaces"][NAMESPACE_NAME]["vector_count"] == len(texts)

        ids_1 = [uuid.uuid4().hex for _ in range(len(texts))]
        PineconeVectorStore.from_texts(
            texts=[t + "-1" for t in texts],
            ids=ids_1,
            embedding=embeddings,
            index_name=INDEX_NAME,
            namespace=NAMESPACE_NAME,
        )
        index_stats = _poll_for_vector_count(self.index, NAMESPACE_NAME, len(texts) * 2)
        assert (
            index_stats["namespaces"][NAMESPACE_NAME]["vector_count"] == len(texts) * 2
        )
        # only focused on this namespace now
        # assert index_stats["total_vector_count"] == len(texts) * 2

    @pytest.mark.xfail(reason="relevance score just over 1")
    def test_relevance_score_bound(self, embeddings: PineconeEmbeddings) -> None:
        """Ensures all relevance scores are between 0 and 1."""
        texts = ["foo", "bar", "baz"]
        metadatas = [{"page": i} for i in range(len(texts))]
        docsearch = PineconeVectorStore.from_texts(
            texts,
            embeddings,
            index_name=INDEX_NAME,
            metadatas=metadatas,
        )
        # wait for the index to be ready
        output = _poll_for_results(
            lambda: docsearch.similarity_search_with_relevance_scores("foo", k=3),
            min_count=3,
        )
        print(output)  # noqa: T201
        assert all(
            (1 >= score or np.isclose(score, 1)) and score >= 0 for _, score in output
        )

    @pytest.mark.asyncio
    @pytest.mark.xfail(reason="relevance score just over 1")
    async def test_arelevance_score_bound(self, embeddings: PineconeEmbeddings) -> None:
        """Ensures all relevance scores are between 0 and 1."""
        texts = ["foo", "bar", "baz"]
        metadatas = [{"page": i} for i in range(len(texts))]
        docsearch = await PineconeVectorStore.afrom_texts(
            texts,
            embeddings,
            index_name=INDEX_NAME,
            metadatas=metadatas,
        )
        # wait for the index to be ready
        output = await _apoll_for_results(
            lambda: docsearch.asimilarity_search_with_relevance_scores("foo", k=3),
            min_count=3,
        )
        print(output)  # noqa: T201
        assert all(
            (1 >= score or np.isclose(score, 1)) and score >= 0 for _, score in output
        )

    @pytest.mark.skipif(reason="slow to run for benchmark")
    @pytest.mark.parametrize(
        "pool_threads,batch_size,embeddings_chunk_size,data_multiplier",
        [
            (
                1,
                32,
                32,
                1000,
            ),  # simulate single threaded with embeddings_chunk_size = batch_size = 32
            (
                1,
                32,
                1000,
                1000,
            ),  # simulate single threaded with embeddings_chunk_size = 1000
            (
                4,
                32,
                1000,
                1000,
            ),  # simulate 4 threaded with embeddings_chunk_size = 1000
            (20, 64, 5000, 1000),
        ],  # simulate 20 threaded with embeddings_chunk_size = 5000
    )
    def test_from_texts_with_metadatas_benchmark(
        self,
        pool_threads: int,
        batch_size: int,
        embeddings_chunk_size: int,
        data_multiplier: int,
        documents: List[Document],
        embeddings: PineconeEmbeddings,
    ) -> None:
        """Test end to end construction and search."""

        texts = [document.page_content for document in documents] * data_multiplier
        uuids = [uuid.uuid4().hex for _ in range(len(texts))]
        metadatas = [{"page": i} for i in range(len(texts))]
        docsearch = PineconeVectorStore.from_texts(
            texts,
            embeddings,
            ids=uuids,
            metadatas=metadatas,
            index_name=INDEX_NAME,
            namespace=NAMESPACE_NAME,
            pool_threads=pool_threads,
            batch_size=batch_size,
            embeddings_chunk_size=embeddings_chunk_size,
        )

        query = "What did the president say about Ketanji Brown Jackson"
        _ = docsearch.similarity_search(query, k=1, namespace=NAMESPACE_NAME)

    @pytest.fixture
    def mock_pool_not_supported(self, mocker: MockerFixture) -> None:
        """
        This is the error thrown when multiprocessing is not supported.
        See https://github.com/langchain-ai/langchain/issues/11168
        """
        mocker.patch(
            "multiprocessing.synchronize.SemLock.__init__",
            side_effect=OSError(
                "FileNotFoundError: [Errno 2] No such file or directory"
            ),
        )

    @pytest.mark.usefixtures("mock_pool_not_supported")
    def test_that_async_freq_uses_multiprocessing(
        self, texts: List[str], embeddings: PineconeEmbeddings
    ) -> None:
        with pytest.raises(OSError):
            PineconeVectorStore.from_texts(
                texts=texts,
                embedding=embeddings,
                index_name=INDEX_NAME,
                namespace=NAMESPACE_NAME,
            )

    @pytest.mark.usefixtures("mock_pool_not_supported")
    def test_that_async_freq_false_enabled_singlethreading(
        self, texts: List[str], embeddings: PineconeEmbeddings
    ) -> None:
        PineconeVectorStore.from_texts(
            texts=texts,
            embedding=embeddings,
            index_name=INDEX_NAME,
            namespace=NAMESPACE_NAME,
            async_req=False,  # force single-threaded path
        )

    # ------------------------------------------------------------------
    # delete() / adelete() integration tests (TC-011)
    # ------------------------------------------------------------------

    @pytest.fixture
    def _delete_ns(self) -> Generator[str, None, None]:
        """Yield a unique namespace and delete_all it on teardown."""
        ns = f"del-{uuid.uuid4().hex[:8]}"
        yield ns
        try:
            self.index.delete(delete_all=True, namespace=ns)
        except Exception:
            pass

    def test_delete_by_ids(
        self, embeddings: PineconeEmbeddings, _delete_ns: str
    ) -> None:
        """delete(ids=...) removes only the specified vectors; remaining count asserted."""
        ns = _delete_ns
        texts = ["alpha", "beta", "gamma", "delta"]
        doc_ids = [uuid.uuid4().hex for _ in texts]
        docsearch = PineconeVectorStore.from_texts(
            texts=texts,
            ids=doc_ids,
            embedding=embeddings,
            index_name=INDEX_NAME,
            namespace=ns,
        )
        _poll_for_vector_count(self.index, ns, len(texts))

        ids_to_delete = doc_ids[:2]
        docsearch.delete(ids=ids_to_delete, namespace=ns)

        stats = _poll_for_vector_count_at_most(
            self.index, ns, len(texts) - len(ids_to_delete)
        )
        remaining = stats.get("namespaces", {}).get(ns, {}).get("vector_count", 0)
        assert remaining == len(texts) - len(ids_to_delete)

    def test_delete_by_filter(
        self, embeddings: PineconeEmbeddings, _delete_ns: str
    ) -> None:
        """delete(filter=...) removes only vectors matching the metadata predicate."""
        ns = _delete_ns
        texts = ["alpha", "beta", "gamma", "delta"]
        metadatas = [{"page": i} for i in range(len(texts))]
        docsearch = PineconeVectorStore.from_texts(
            texts=texts,
            embedding=embeddings,
            metadatas=metadatas,
            index_name=INDEX_NAME,
            namespace=ns,
        )
        _poll_for_vector_count(self.index, ns, len(texts))

        docsearch.delete(filter={"page": 0}, namespace=ns)

        expected_remaining = len(texts) - 1
        stats = _poll_for_vector_count_at_most(self.index, ns, expected_remaining)
        remaining = stats.get("namespaces", {}).get(ns, {}).get("vector_count", 0)
        assert remaining == expected_remaining

    def test_delete_all(self, embeddings: PineconeEmbeddings, _delete_ns: str) -> None:
        """delete(delete_all=True) empties the entire namespace."""
        ns = _delete_ns
        texts = ["alpha", "beta", "gamma"]
        docsearch = PineconeVectorStore.from_texts(
            texts=texts,
            embedding=embeddings,
            index_name=INDEX_NAME,
            namespace=ns,
        )
        _poll_for_vector_count(self.index, ns, len(texts))

        docsearch.delete(delete_all=True, namespace=ns)

        stats = _poll_for_vector_count_at_most(self.index, ns, 0)
        remaining = stats.get("namespaces", {}).get(ns, {}).get("vector_count", 0)
        assert remaining == 0

    @pytest.mark.asyncio
    async def test_adelete_delete_all(
        self, embeddings: PineconeEmbeddings, _delete_ns: str
    ) -> None:
        """adelete(delete_all=True) empties the namespace asynchronously."""
        ns = _delete_ns
        texts = ["alpha", "beta", "gamma"]
        docsearch = await PineconeVectorStore.afrom_texts(
            texts=texts,
            embedding=embeddings,
            index_name=INDEX_NAME,
            namespace=ns,
        )
        _poll_for_vector_count(self.index, ns, len(texts))

        await docsearch.adelete(delete_all=True, namespace=ns)

        stats = _poll_for_vector_count_at_most(self.index, ns, 0)
        remaining = stats.get("namespaces", {}).get(ns, {}).get("vector_count", 0)
        assert remaining == 0

    def test_max_marginal_relevance_search(
        self, embeddings: PineconeEmbeddings
    ) -> None:
        """Dense MMR returns k distinct documents drawn from the corpus."""
        unique_id = uuid.uuid4().hex[:8]
        ns = f"mmr-dense-{unique_id}"
        corpus = ["alpha one", "beta two", "gamma three", "delta four"]
        docsearch = PineconeVectorStore.from_texts(
            corpus, embeddings, index_name=INDEX_NAME, namespace=ns
        )
        results = _poll_for_results(
            lambda: docsearch.max_marginal_relevance_search(
                "alpha", k=2, fetch_k=4, namespace=ns
            ),
            min_count=2,
        )
        assert len(results) == 2
        texts_returned = [doc.page_content for doc in results]
        assert all(t in corpus for t in texts_returned)
        assert len(set(texts_returned)) == 2

    def test_max_marginal_relevance_search_by_vector(
        self, embeddings: PineconeEmbeddings
    ) -> None:
        """Dense MMR by vector returns k distinct documents drawn from the corpus."""
        unique_id = uuid.uuid4().hex[:8]
        ns = f"mmr-dense-bv-{unique_id}"
        corpus = ["alpha one", "beta two", "gamma three", "delta four"]
        docsearch = PineconeVectorStore.from_texts(
            corpus, embeddings, index_name=INDEX_NAME, namespace=ns
        )
        query_embedding = embeddings.embed_query("alpha")
        results = _poll_for_results(
            lambda: docsearch.max_marginal_relevance_search_by_vector(
                query_embedding, k=2, fetch_k=4, namespace=ns
            ),
            min_count=2,
        )
        assert len(results) == 2
        texts_returned = [doc.page_content for doc in results]
        assert all(t in corpus for t in texts_returned)
        assert len(set(texts_returned)) == 2

    @pytest.mark.asyncio
    async def test_amax_marginal_relevance_search(
        self, embeddings: PineconeEmbeddings
    ) -> None:
        """Async dense MMR returns k distinct documents drawn from the corpus."""
        unique_id = uuid.uuid4().hex[:8]
        ns = f"mmr-dense-async-{unique_id}"
        corpus = ["alpha one", "beta two", "gamma three", "delta four"]
        docsearch = await PineconeVectorStore.afrom_texts(
            corpus, embeddings, index_name=INDEX_NAME, namespace=ns
        )
        results = await _apoll_for_results(
            lambda: docsearch.amax_marginal_relevance_search(
                "alpha", k=2, fetch_k=4, namespace=ns
            ),
            min_count=2,
        )
        assert len(results) == 2
        texts_returned = [doc.page_content for doc in results]
        assert all(t in corpus for t in texts_returned)
        assert len(set(texts_returned)) == 2

    @pytest.mark.asyncio
    async def test_amax_marginal_relevance_search_by_vector(
        self, embeddings: PineconeEmbeddings
    ) -> None:
        """Async dense MMR by vector returns k distinct documents drawn from the corpus."""
        unique_id = uuid.uuid4().hex[:8]
        ns = f"mmr-dense-bv-async-{unique_id}"
        corpus = ["alpha one", "beta two", "gamma three", "delta four"]
        docsearch = await PineconeVectorStore.afrom_texts(
            corpus, embeddings, index_name=INDEX_NAME, namespace=ns
        )
        query_embedding = await embeddings.aembed_query("alpha")
        results = await _apoll_for_results(
            lambda: docsearch.amax_marginal_relevance_search_by_vector(
                query_embedding, k=2, fetch_k=4, namespace=ns
            ),
            min_count=2,
        )
        assert len(results) == 2
        texts_returned = [doc.page_content for doc in results]
        assert all(t in corpus for t in texts_returned)
        assert len(set(texts_returned)) == 2

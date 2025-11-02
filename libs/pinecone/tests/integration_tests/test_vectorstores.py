import asyncio
import time
import uuid
from datetime import datetime
from typing import List

import numpy as np
import pinecone  # type: ignore
import pytest  # type: ignore[import-not-found]
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings  # type: ignore[import-not-found]
from pinecone import AwsRegion, CloudProvider, Metric, ServerlessSpec
from pytest_mock import MockerFixture  # type: ignore[import-not-found]

from langchain_pinecone import PineconeVectorStore

# unique name of the index for this test run
INDEX_NAME = f"langchain-test-vectorstores-{datetime.now().strftime('%Y%m%d%H%M%S')}"
NAMESPACE_NAME = "langchain-test-namespace"  # name of the namespace
DIMENSION = 1536  # dimension of the embeddings

DEFAULT_SLEEP = 15


class TestPinecone:
    index: "pinecone.Index"
    pc: "pinecone.Pinecone"

    @classmethod
    def setup_class(self) -> None:
        import pinecone

        client = pinecone.Pinecone()
        print(f"client: {client}")  # noqa: T201
        if client.has_index(name=INDEX_NAME):  # change to list comprehension
            client.delete_index(INDEX_NAME)
            time.sleep(DEFAULT_SLEEP)  # prevent race with subsequent creation
        client.create_index(
            name=INDEX_NAME,
            dimension=DIMENSION,
            metric=Metric.COSINE,
            spec=ServerlessSpec(cloud=CloudProvider.AWS, region=AwsRegion.US_WEST_2),
        )

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

    @pytest.fixture
    def embedding_openai(self) -> OpenAIEmbeddings:
        return OpenAIEmbeddings()

    @pytest.fixture
    def texts(self) -> List[str]:
        return ["foo", "bar", "baz"]

    def test_from_texts(
        self, texts: List[str], embedding_openai: OpenAIEmbeddings
    ) -> None:
        """Test end to end construction and search."""
        unique_id = uuid.uuid4().hex
        needs = f"foobuu {unique_id} booo"
        texts.insert(0, needs)

        docsearch = PineconeVectorStore.from_texts(
            texts=texts,
            embedding=embedding_openai,
            index_name=INDEX_NAME,
            namespace=NAMESPACE_NAME,
        )
        time.sleep(DEFAULT_SLEEP)  # prevent race condition
        output = docsearch.similarity_search(unique_id, k=1, namespace=NAMESPACE_NAME)
        output[0].id = None  # overwrite ID for ease of comparison
        assert output == [Document(page_content=needs)]

    @pytest.mark.asyncio
    async def test_afrom_texts(
        self, texts: List[str], embedding_openai: OpenAIEmbeddings
    ) -> None:
        """Test end to end construction and search."""
        unique_id = uuid.uuid4().hex
        needs = f"foobuu {unique_id} booo"
        texts.insert(0, needs)
        print(f"texts: {texts}")  # noqa: T201
        docsearch = await PineconeVectorStore.afrom_texts(
            texts=texts,
            embedding=embedding_openai,
            index_name=INDEX_NAME,
            namespace=f"{NAMESPACE_NAME}-afrom-texts",
        )
        await asyncio.sleep(DEFAULT_SLEEP)  # prevent race condition
        output = await docsearch.asimilarity_search(
            needs,
            k=1,
            namespace=f"{NAMESPACE_NAME}-afrom-texts",
        )
        print(f"output: {output}")  # noqa: T201
        output[0].id = None  # overwrite ID for ease of comparison
        assert output == [Document(page_content=needs)]

    def test_from_texts_with_metadatas(
        self, texts: List[str], embedding_openai: OpenAIEmbeddings
    ) -> None:
        """Test end to end construction and search."""

        unique_id = uuid.uuid4().hex
        needs = f"foobuu {unique_id} booo"
        texts = [needs] + texts

        metadatas = [{"page": i} for i in range(len(texts))]

        namespace = f"{NAMESPACE_NAME}-md"
        docsearch = PineconeVectorStore.from_texts(
            texts,
            embedding_openai,
            index_name=INDEX_NAME,
            metadatas=metadatas,
            namespace=namespace,
        )
        time.sleep(DEFAULT_SLEEP)  # prevent race condition
        output = docsearch.similarity_search(needs, k=1, namespace=namespace)

        output[0].id = None
        # TODO: why metadata={"page": 0.0}) instead of {"page": 0}?
        assert output == [Document(page_content=needs, metadata={"page": 0.0})]

    @pytest.mark.asyncio
    async def test_afrom_texts_with_metadatas(
        self, texts: List[str], embedding_openai: OpenAIEmbeddings
    ) -> None:
        """Test end to end construction and search."""

        unique_id = uuid.uuid4().hex
        needs = f"foobuu {unique_id} booo"
        texts = [needs] + texts

        metadatas = [{"page": i} for i in range(len(texts))]

        namespace = f"{NAMESPACE_NAME}-md"
        docsearch = await PineconeVectorStore.afrom_texts(
            texts,
            embedding_openai,
            index_name=INDEX_NAME,
            metadatas=metadatas,
            namespace=namespace,
        )
        await asyncio.sleep(DEFAULT_SLEEP)  # prevent race condition
        output = await docsearch.asimilarity_search(needs, k=1, namespace=namespace)

        output[0].id = None
        # TODO: why metadata={"page": 0.0}) instead of {"page": 0}?
        assert output == [Document(page_content=needs, metadata={"page": 0.0})]

    @pytest.mark.asyncio
    async def test_aadd_documents(
        self, texts: List[str], embedding_openai: OpenAIEmbeddings
    ) -> None:
        """Test adding documents to existing index."""

        texts_1 = ["foo", "bar", "baz"]
        metadatas = [{"page": i} for i in range(len(texts_1))]
        docsearch = await PineconeVectorStore.afrom_texts(
            texts_1,
            embedding_openai,
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
        await asyncio.sleep(DEFAULT_SLEEP)  # prevent race condition
        output = await docsearch.asimilarity_search(
            docs[0].page_content, k=1, namespace=f"{INDEX_NAME}-2"
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

    def test_from_texts_with_scores(self, embedding_openai: OpenAIEmbeddings) -> None:
        """Test end to end construction and search with scores and IDs."""
        texts = ["foo", "bar", "baz"]
        metadatas = [{"page": i} for i in range(len(texts))]
        docsearch = PineconeVectorStore.from_texts(
            texts,
            embedding_openai,
            index_name=INDEX_NAME,
            metadatas=metadatas,
            namespace=NAMESPACE_NAME,
        )
        time.sleep(DEFAULT_SLEEP)  # prevent race condition
        output = docsearch.similarity_search_with_score(
            "foo", k=3, namespace=NAMESPACE_NAME
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
        self, embedding_openai: OpenAIEmbeddings
    ) -> None:
        """Test end to end construction and search with scores and IDs."""
        texts = ["foo", "bar", "baz"]
        metadatas = [{"page": i} for i in range(len(texts))]
        print("metadatas", metadatas)  # noqa: T201
        docsearch = await PineconeVectorStore.afrom_texts(
            texts,
            embedding_openai,
            index_name=INDEX_NAME,
            metadatas=metadatas,
            namespace=NAMESPACE_NAME,
        )
        print(texts)  # noqa: T201
        await asyncio.sleep(DEFAULT_SLEEP)  # prevent race condition
        output = await docsearch.asimilarity_search_with_score(
            "foo", k=3, namespace=NAMESPACE_NAME
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
        self, embedding_openai: OpenAIEmbeddings
    ) -> None:
        """Test that namespaces are properly handled."""
        # Create two indexes with the same name but different namespaces
        texts_1 = ["foo", "bar", "baz"]
        metadatas = [{"page": i} for i in range(len(texts_1))]
        PineconeVectorStore.from_texts(
            texts_1,
            embedding_openai,
            index_name=INDEX_NAME,
            metadatas=metadatas,
            namespace=f"{INDEX_NAME}-1",
        )

        texts_2 = ["foo2", "bar2", "baz2"]
        metadatas = [{"page": i} for i in range(len(texts_2))]

        PineconeVectorStore.from_texts(
            texts_2,
            embedding_openai,
            index_name=INDEX_NAME,
            metadatas=metadatas,
            namespace=f"{INDEX_NAME}-2",
        )

        time.sleep(DEFAULT_SLEEP)  # prevent race condition

        # Search with namespace
        docsearch = PineconeVectorStore.from_existing_index(
            index_name=INDEX_NAME,
            embedding=embedding_openai,
            namespace=f"{INDEX_NAME}-1",
        )
        output = docsearch.similarity_search("foo", k=20, namespace=f"{INDEX_NAME}-1")
        # check that we don't get results from the other namespace
        page_contents = sorted(set([o.page_content for o in output]))
        assert all(content in ["foo", "bar", "baz"] for content in page_contents)
        assert all(content not in ["foo2", "bar2", "baz2"] for content in page_contents)

    def test_add_documents_with_ids(
        self, texts: List[str], embedding_openai: OpenAIEmbeddings
    ) -> None:
        ids = [uuid.uuid4().hex for _ in range(len(texts))]
        PineconeVectorStore.from_texts(
            texts=texts,
            ids=ids,
            embedding=embedding_openai,
            index_name=INDEX_NAME,
            namespace=NAMESPACE_NAME,
        )
        time.sleep(DEFAULT_SLEEP)  # prevent race condition
        index_stats = self.index.describe_index_stats()
        assert index_stats["namespaces"][NAMESPACE_NAME]["vector_count"] == len(texts)

        ids_1 = [uuid.uuid4().hex for _ in range(len(texts))]
        PineconeVectorStore.from_texts(
            texts=[t + "-1" for t in texts],
            ids=ids_1,
            embedding=embedding_openai,
            index_name=INDEX_NAME,
            namespace=NAMESPACE_NAME,
        )
        time.sleep(DEFAULT_SLEEP)  # prevent race condition
        index_stats = self.index.describe_index_stats()
        assert (
            index_stats["namespaces"][NAMESPACE_NAME]["vector_count"] == len(texts) * 2
        )
        # only focused on this namespace now
        # assert index_stats["total_vector_count"] == len(texts) * 2

    @pytest.mark.xfail(reason="relevance score just over 1")
    def test_relevance_score_bound(self, embedding_openai: OpenAIEmbeddings) -> None:
        """Ensures all relevance scores are between 0 and 1."""
        texts = ["foo", "bar", "baz"]
        metadatas = [{"page": i} for i in range(len(texts))]
        docsearch = PineconeVectorStore.from_texts(
            texts,
            embedding_openai,
            index_name=INDEX_NAME,
            metadatas=metadatas,
        )
        # wait for the index to be ready
        time.sleep(DEFAULT_SLEEP)
        output = docsearch.similarity_search_with_relevance_scores("foo", k=3)
        print(output)  # noqa: T201
        assert all(
            (1 >= score or np.isclose(score, 1)) and score >= 0 for _, score in output
        )

    @pytest.mark.asyncio
    @pytest.mark.xfail(reason="relevance score just over 1")
    async def test_arelevance_score_bound(
        self, embedding_openai: OpenAIEmbeddings
    ) -> None:
        """Ensures all relevance scores are between 0 and 1."""
        texts = ["foo", "bar", "baz"]
        metadatas = [{"page": i} for i in range(len(texts))]
        docsearch = await PineconeVectorStore.afrom_texts(
            texts,
            embedding_openai,
            index_name=INDEX_NAME,
            metadatas=metadatas,
        )
        # wait for the index to be ready
        await asyncio.sleep(DEFAULT_SLEEP)  # prevent race condition
        output = await docsearch.asimilarity_search_with_relevance_scores("foo", k=3)
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
        embedding_openai: OpenAIEmbeddings,
    ) -> None:
        """Test end to end construction and search."""

        texts = [document.page_content for document in documents] * data_multiplier
        uuids = [uuid.uuid4().hex for _ in range(len(texts))]
        metadatas = [{"page": i} for i in range(len(texts))]
        docsearch = PineconeVectorStore.from_texts(
            texts,
            embedding_openai,
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
        self, texts: List[str], embedding_openai: OpenAIEmbeddings
    ) -> None:
        with pytest.raises(OSError):
            PineconeVectorStore.from_texts(
                texts=texts,
                embedding=embedding_openai,
                index_name=INDEX_NAME,
                namespace=NAMESPACE_NAME,
            )

    @pytest.mark.usefixtures("mock_pool_not_supported")
    def test_that_async_freq_false_enabled_singlethreading(
        self, texts: List[str], embedding_openai: OpenAIEmbeddings
    ) -> None:
        PineconeVectorStore.from_texts(
            texts=texts,
            embedding=embedding_openai,
            index_name=INDEX_NAME,
            namespace=NAMESPACE_NAME,
            async_req=False,  # force single-threaded path
        )

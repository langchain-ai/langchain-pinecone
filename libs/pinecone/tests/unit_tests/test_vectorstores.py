import logging
from typing import TYPE_CHECKING, Callable, Type
from unittest.mock import ANY, Mock, call

import pytest
from pinecone import PineconeAsyncio, SparseValues  # type: ignore[import-untyped]
from pytest_mock import AsyncMockType, MockerFixture, MockType

from langchain_pinecone._utilities import DistanceStrategy
from langchain_pinecone.embeddings import PineconeEmbeddings, PineconeSparseEmbeddings
from langchain_pinecone.vectorstores import PineconeVectorStore
from langchain_pinecone.vectorstores_sparse import PineconeSparseVectorStore

if TYPE_CHECKING:
    from pytest import FixtureRequest as __FixtureRequest

    class FixtureRequest(__FixtureRequest):
        param: str
else:
    from pytest import FixtureRequest


@pytest.fixture
def mock_embedding(mocker: MockerFixture) -> AsyncMockType:
    """Fixture for mock embedding function."""
    mock_embedding = mocker.AsyncMock(spec=PineconeEmbeddings)
    mock_embedding.embed_documents = mocker.Mock(return_value=[[0.1, 0.2, 0.3]])
    mock_embedding.aembed_documents = mocker.AsyncMock(return_value=[[0.1, 0.2, 0.3]])
    return mock_embedding


@pytest.fixture
def mock_sparse_embedding(mocker: MockerFixture) -> AsyncMockType:
    """Fixture for mock embedding function."""
    mock_embedding = mocker.AsyncMock(spec=PineconeSparseEmbeddings)
    mock_embedding.embed_documents = mocker.Mock(
        return_value=[SparseValues(indices=[0, 28, 218], values=[0.34, 0.239, 0.92])]
    )
    mock_embedding.aembed_documents = mocker.AsyncMock(
        return_value=[SparseValues(indices=[0, 28, 218], values=[0.34, 0.239, 0.92])]
    )
    return mock_embedding


@pytest.fixture
def mock_async_index(mocker: MockerFixture) -> MockType:
    """Fixture for mock async index."""
    # Import the actual _IndexAsyncio class to use as spec
    from pinecone.data import _IndexAsyncio  # type:ignore[import-untyped]

    mock_async_index = mocker.Mock(spec=_IndexAsyncio)
    mock_async_index.config = mocker.Mock()
    mock_async_index.config.host = "example.org"
    mock_async_index.config.api_key = "test"
    mock_async_index.upsert = mocker.AsyncMock(return_value=None)
    mock_async_index.__aenter__ = mocker.AsyncMock(return_value=mock_async_index)
    mock_async_index.__aexit__ = mocker.AsyncMock(return_value=None)
    mock_async_index.describe_index_stats = mocker.Mock(
        return_value={"vector_type": "sparse"}
    )
    return mock_async_index


@pytest.fixture
def mock_index(mocker: MockerFixture) -> MockType:
    """Fixture for mock async index."""
    # Import the actual _IndexAsyncio class to use as spec
    from pinecone.data import _Index

    mock_index = mocker.Mock(spec=_Index)
    mock_index.config = mocker.Mock()
    mock_index.config.host = "example.org"
    mock_index.config.api_key = "test"
    mock_index.upsert = mocker.Mock(return_value=None)
    mock_index.describe_index_stats = mocker.Mock(
        return_value={"vector_type": "sparse"}
    )
    return mock_index


@pytest.fixture
def mock_pinecone_client(mocker: MockerFixture, mock_index: MockType) -> AsyncMockType:
    mock_pinecone_client = mocker.patch(
        "langchain_pinecone.vectorstores.PineconeClient"
    )
    mock_pinecone_client.return_value.Index.return_value = mock_index
    return mock_pinecone_client


@pytest.fixture
def mock_pinecone_async_client(
    mocker: MockerFixture, mock_async_index: AsyncMockType
) -> AsyncMockType:
    mock_pinecone_async_client = mocker.patch(
        "langchain_pinecone.vectorstores.PineconeAsyncioClient"
    )
    instance = mock_pinecone_async_client.return_value
    instance.__aenter__.return_value = instance
    instance.__aexit__.return_value = None
    instance.IndexAsyncio.return_value = mock_async_index
    return mock_pinecone_async_client


@pytest.fixture
def mock_async_client(
    mocker: MockerFixture, mock_async_index: MockType
) -> AsyncMockType:
    mock_async_client = mocker.AsyncMock(spec=PineconeAsyncio)
    mock_async_client.IndexAsyncio.return_value = mock_async_index
    mock_async_client.__aenter__.return_value = mock_async_client
    mock_client_class = mocker.patch(
        "langchain_pinecone.vectorstores.PineconeAsyncioClient",
        return_value=mock_async_client,
    )

    return mock_client_class


def test_id_prefix() -> None:
    """Test integration of the id_prefix parameter."""
    embedding = Mock()
    embedding.embed_documents = Mock(return_value=[0.1, 0.2, 0.3, 0.4, 0.5])
    index = Mock()
    index.upsert = Mock(return_value=None)
    text_key = "testing"
    vectorstore = PineconeVectorStore(index, embedding, text_key)
    texts = ["alpha", "beta", "gamma", "delta", "epsilon"]
    id_prefix = "testing_prefixes"
    vectorstore.add_texts(texts, id_prefix=id_prefix, async_req=False)


def test_sparse_vectorstore__raises_on_dense_embedding(mocker: MockerFixture) -> None:
    with pytest.raises(ValueError):
        PineconeSparseVectorStore(embedding=mocker.Mock(spec=PineconeEmbeddings))


@pytest.mark.parametrize(
    "vectorstore_cls,mock_embedding_obj",
    [
        (PineconeVectorStore, "mock_embedding"),
        (PineconeSparseVectorStore, "mock_sparse_embedding"),
    ],
)
class TestVectorstores:
    def test_initialization(
        self,
        request: FixtureRequest,
        vectorstore_cls: Type[PineconeVectorStore],
        mock_index: Mock,
        mock_embedding_obj: str,
    ) -> None:
        """Test integration vectorstore initialization."""
        # mock index
        mock_embedding = request.getfixturevalue(mock_embedding_obj)
        text_key = "xyz"
        vectorstore_cls(index=mock_index, embedding=mock_embedding, text_key=text_key)

    def test_initialization_with_index_name__caches_host(
        self,
        request: FixtureRequest,
        vectorstore_cls: Type[PineconeVectorStore],
        mock_embedding_obj: str,
        mock_pinecone_client: MockType,
    ) -> None:
        """Test integration vectorstore initialization with index name."""
        # mock index
        mock_embedding = request.getfixturevalue(mock_embedding_obj)
        # Assert no calls to PineconeClient mage
        vectorstore = vectorstore_cls(
            embedding=mock_embedding,
            pinecone_api_key="test-key",
            index_name="test-index",
        )
        # Verify host is properly cached
        assert (
            vectorstore._index_host
            == mock_pinecone_client.return_value.Index.return_value.config.host
        )
        mock_pinecone_client.return_value.Index.assert_called_once_with(
            name="test-index"
        )

    def test_initialization_without_host_or_index_name__raises_valueerror(
        self,
        request: FixtureRequest,
        vectorstore_cls: Type[PineconeVectorStore],
        mock_embedding_obj: str,
    ) -> None:
        """Test integration vectorstore initialization without host or index name."""
        # mock index
        mock_embedding = request.getfixturevalue(mock_embedding_obj)
        with pytest.raises(ValueError):
            vectorstore_cls(
                embedding=mock_embedding,
                pinecone_api_key="test-key",
            )

    def test_host_parameter__avoids_sync_index_creation(
        self,
        request: FixtureRequest,
        vectorstore_cls: Type[PineconeVectorStore],
        mock_embedding_obj: str,
        mock_pinecone_client: MockType,
    ) -> None:
        """Test that providing host parameter avoids creating unnecessary sync index."""
        mock_embedding = request.getfixturevalue(mock_embedding_obj)

        # Create vectorstore with host parameter
        vectorstore = vectorstore_cls(
            pinecone_api_key="test-key",
            host="direct-host.pinecone.io",
            embedding=mock_embedding,
            text_key="text",
        )

        # Verify that PineconeClient was NOT called since host was provided
        mock_pinecone_client.assert_not_called()

        # Verify host is properly cached
        assert vectorstore._index_host == "direct-host.pinecone.io"

        # Verify that _index is None since no sync index was created
        assert vectorstore._index is None

    @pytest.mark.asyncio
    async def test_async_index__uses_cached_host_without_sync_calls(
        self,
        request: FixtureRequest,
        vectorstore_cls: Type[PineconeVectorStore],
        mock_embedding_obj: str,
        mock_pinecone_async_client: AsyncMockType,
    ) -> None:
        """Test that async_index uses cached host without making sync calls."""
        mock_embedding = request.getfixturevalue(mock_embedding_obj)

        # Create vectorstore with host parameter (avoids sync index creation)
        vectorstore = vectorstore_cls(
            pinecone_api_key="test-key",
            host="cached-host.pinecone.io",
            embedding=mock_embedding,
            text_key="text",
        )

        # Access async_index property
        result = await vectorstore.async_index

        # Verify async client was called with the cached host
        mock_pinecone_async_client.return_value.IndexAsyncio.assert_called_once_with(
            host="cached-host.pinecone.io"
        )

        # Verify result is the mock async index
        assert (
            result == mock_pinecone_async_client.return_value.IndexAsyncio.return_value
        )

        # Verify no sync index was created or accessed
        assert vectorstore._index is None

    @pytest.mark.parametrize("mock_index_obj", ["mock_index", "mock_async_index"])
    def test_initialization_with_index__caches_host(
        self,
        request: FixtureRequest,
        vectorstore_cls: Type[PineconeVectorStore],
        mock_embedding_obj: str,
        mock_index_obj: MockType,
    ) -> None:
        """Tests that initializing the vectorstore with an asynchronous index"""
        mock_embedding = request.getfixturevalue(mock_embedding_obj)
        mock_index = request.getfixturevalue(mock_index_obj)

        # Create vectorstore with host parameter (avoids sync index creation)
        vectorstore = vectorstore_cls(
            pinecone_api_key="test-key",
            index=mock_index,
            embedding=mock_embedding,
            text_key="text",
        )

        # Verify host is properly cached
        assert vectorstore._index_host == mock_index.config.host

    @pytest.mark.parametrize("mock_index_obj", ["mock_index", "mock_async_index"])
    def test_initialization_with_index_and_host__ignores_host(
        self,
        request: FixtureRequest,
        vectorstore_cls: Type[PineconeVectorStore],
        mock_embedding_obj: str,
        mock_index_obj: MockType,
    ) -> None:
        """Tests that initializing the vectorstore with an asynchronous index"""
        mock_embedding = request.getfixturevalue(mock_embedding_obj)
        mock_index = request.getfixturevalue(mock_index_obj)

        # Create vectorstore with host parameter (avoids sync index creation)
        vectorstore = vectorstore_cls(
            pinecone_api_key="test-key",
            host="another-unrelated-host.pinecone.io",
            index=mock_index,
            embedding=mock_embedding,
            text_key="text",
        )

        # Verify host is properly cached
        assert vectorstore._index_host == mock_index.config.host

    @pytest.mark.asyncio
    async def test_aadd_texts__calls_index_upsert(
        self,
        request: FixtureRequest,
        vectorstore_cls: Type[PineconeVectorStore],
        mock_embedding_obj: str,
        mock_async_index: AsyncMockType,
    ) -> None:
        """Test that aadd_texts properly calls the async index upsert method."""

        mock_embedding = request.getfixturevalue(mock_embedding_obj)

        # Create vectorstore with mocked components
        vectorstore = vectorstore_cls(
            index=mock_async_index, embedding=mock_embedding, text_key="text"
        )

        # Test adding texts
        texts = ["test document"]
        await vectorstore.aadd_texts(texts)

        # Verify the async embedding was called
        mock_embedding.aembed_documents.assert_called_once_with(texts)

        # Verify the async upsert was called
        mock_async_index.upsert.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialising_with_sync_index__still_uses_async_index(
        self,
        request: FixtureRequest,
        vectorstore_cls: Type[PineconeVectorStore],
        mock_async_client: AsyncMockType,
        mock_index: AsyncMockType,
        mock_embedding_obj: str,
    ) -> None:
        """Test that initializing with a sync index still enables async operations."""
        mock_embedding = request.getfixturevalue(mock_embedding_obj)

        # Create vectorstore with sync index
        vectorstore = vectorstore_cls(
            index=mock_index, embedding=mock_embedding, text_key="text"
        )

        texts = ["test document"]
        await vectorstore.aadd_texts(texts)

        # Verify async client was created with correct params
        mock_async_client.assert_called_once_with(
            api_key=mock_index.config.api_key, source_tag="langchain"
        )

        # Verify the async upsert was called
        mock_async_client.return_value.IndexAsyncio.return_value.upsert.assert_called_once()

    @pytest.mark.asyncio
    async def test_asimilarity_search_with_score(
        self,
        request: FixtureRequest,
        mocker: MockerFixture,
        vectorstore_cls: Type[PineconeVectorStore],
        mock_embedding_obj: str,
        mock_async_index: AsyncMockType,
    ) -> None:
        """Test async similarity search with score functionality."""
        mock_async_index.query = mocker.AsyncMock(
            return_value={
                "matches": [
                    {
                        "metadata": {"text": "test doc", "other": "metadata"},
                        "score": 0.8,
                        "id": "test-id",
                    }
                ]
            }
        )

        mock_embedding = request.getfixturevalue(mock_embedding_obj)

        # Create vectorstore
        vectorstore = vectorstore_cls(
            index=mock_async_index, embedding=mock_embedding, text_key="text"
        )

        # Perform async search
        results = await vectorstore.asimilarity_search_with_score("test query", k=1)

        # Verify results
        assert len(results) == 1
        doc, score = results[0]
        assert doc.page_content == "test doc"
        assert doc.metadata == {"other": "metadata"}
        assert score == 0.8
        assert doc.id == "test-id"

    @pytest.mark.asyncio
    async def test_adelete(
        self,
        request: FixtureRequest,
        mocker: MockerFixture,
        vectorstore_cls: Type[PineconeVectorStore],
        mock_embedding_obj: str,
        mock_async_index: AsyncMockType,
    ) -> None:
        """Test async delete functionality."""
        # Setup the async mock for delete
        mock_async_index.delete = mocker.AsyncMock(return_value=None)

        mock_embedding = request.getfixturevalue(mock_embedding_obj)

        # Create vectorstore
        vectorstore = vectorstore_cls(
            index=mock_async_index, embedding=mock_embedding, text_key="text"
        )

        # Test delete all
        await vectorstore.adelete(delete_all=True)
        mock_async_index.delete.assert_called_with(delete_all=True, namespace=None)

        # Test delete by ids
        test_ids = ["id1", "id2", "id3"]
        await vectorstore.adelete(ids=test_ids)
        assert mock_async_index.delete.call_count == 2  # One more call

        # Test delete by filter
        test_filter = {"metadata_field": "value"}
        await vectorstore.adelete(filter=test_filter)
        mock_async_index.delete.assert_called_with(filter=test_filter, namespace=None)

    @pytest.mark.asyncio
    async def test_sync_req_with_async_req__use_future_parallelism(
        self,
        request: FixtureRequest,
        mocker: MockerFixture,
        vectorstore_cls: Type[PineconeVectorStore],
        mock_embedding_obj: str,
        mock_index: MockType,
    ) -> None:
        mock_embedding = request.getfixturevalue(mock_embedding_obj)

        mock_upsert_return = mocker.Mock()
        mock_index.upsert = mocker.Mock(return_value=mock_upsert_return)

        # Create vectorstore
        vectorstore = vectorstore_cls(
            index=mock_index, embedding=mock_embedding, text_key="text"
        )

        texts = ["test"] * 3
        vectorstore.add_texts(texts, async_req=True)
        mock_embedding.embed_documents.assert_called_once_with(texts)

        mock_index.upsert.assert_called_once()
        mock_upsert_return.get.assert_called_once()

    @pytest.mark.asyncio
    async def test_sync_req_with_async_req__use_future_parallelism_multi(
        self,
        request: FixtureRequest,
        mocker: MockerFixture,
        vectorstore_cls: Type[PineconeVectorStore],
        mock_embedding_obj: str,
        mock_index: MockType,
    ) -> None:
        mock_embedding = request.getfixturevalue(mock_embedding_obj)

        mock_upsert_return = mocker.Mock()
        mock_index.upsert = mocker.Mock(return_value=mock_upsert_return)

        # Create vectorstore
        vectorstore = vectorstore_cls(
            index=mock_index, embedding=mock_embedding, text_key="text"
        )

        texts = ["test"] * 3000  # 3x embedding_chunk_size
        vectorstore.add_texts(texts, async_req=True)

        # When async_req == True, we expect `upsert` to be called 3 times...
        mock_index.upsert.assert_has_calls(
            [call(vectors=ANY, namespace=ANY, async_req=ANY)] * 3  # type: ignore
        )
        # each upsert call will yield a `multiprocessing.pool.ApplyResult` object
        # assert we fetch the future result 3 times
        mock_upsert_return.get.assert_has_calls([call()] * 3)

    @pytest.mark.asyncio
    async def test__closes_pinecone_client(
        self,
        request: FixtureRequest,
        vectorstore_cls: Type[PineconeVectorStore],
        mock_async_client: AsyncMockType,
        mock_embedding_obj: str,
        mock_index: MockType,
    ) -> None:
        """Test the PineconeAsyncio client is closed properly"""
        mock_embedding = request.getfixturevalue(mock_embedding_obj)

        # Create vectorstore
        vectorstore = vectorstore_cls(
            index=mock_index, embedding=mock_embedding, text_key="text"
        )

        await vectorstore.aadd_texts(["test"])

        mock_async_client.return_value.__aexit__.assert_called_once()
        mock_async_client.return_value.IndexAsyncio.return_value.__aexit__.assert_called_once()

    @pytest.mark.asyncio
    async def test_index__closes_only_once_even_multiple_calls(
        self,
        request: FixtureRequest,
        vectorstore_cls: Type[PineconeVectorStore],
        mock_async_client: AsyncMockType,
        mock_embedding_obj: str,
        mock_index: MockType,
    ) -> None:
        """Test the PineconeAsyncio client is closed properly"""
        mock_embedding = request.getfixturevalue(mock_embedding_obj)

        # Create vectorstore
        vectorstore = vectorstore_cls(
            index=mock_index, embedding=mock_embedding, text_key="text"
        )

        await vectorstore.aadd_texts(["test1"] * 2000)  # 2x embedding_chunk_size

        # Even though embeddings are called twice (for each chunk in loop) ...
        mock_embedding.aembed_documents.assert_has_calls([call(["test1"] * 1000)] * 2)

        # ... we're persisting the connection and only closing on completion
        mock_async_client.return_value.IndexAsyncio.return_value.__aexit__.assert_called_once()

    @pytest.mark.asyncio
    async def test_async_context_manager__reuses_single_async_session(
        self,
        request: FixtureRequest,
        vectorstore_cls: Type[PineconeVectorStore],
        mock_embedding_obj: str,
        mock_async_index: AsyncMockType,
    ) -> None:
        """Ensure async context manager keeps a single session across operations."""
        mock_embedding = request.getfixturevalue(mock_embedding_obj)

        vectorstore = vectorstore_cls(
            index=mock_async_index, embedding=mock_embedding, text_key="text"
        )

        async with vectorstore:
            await vectorstore.aadd_texts(["doc1"])
            await vectorstore.aadd_texts(["doc2"])

        mock_async_index.__aenter__.assert_called_once()
        mock_async_index.__aexit__.assert_called_once()

    @pytest.mark.asyncio
    async def test_aclose__closes_active_async_context_once(
        self,
        request: FixtureRequest,
        vectorstore_cls: Type[PineconeVectorStore],
        mock_embedding_obj: str,
        mock_async_index: AsyncMockType,
    ) -> None:
        """Ensure aclose shuts down an opened async index exactly once."""
        mock_embedding = request.getfixturevalue(mock_embedding_obj)

        vectorstore = vectorstore_cls(
            index=mock_async_index, embedding=mock_embedding, text_key="text"
        )

        await vectorstore.__aenter__()
        await vectorstore.aclose()
        await vectorstore.aclose()

        mock_async_index.__aenter__.assert_called_once()
        mock_async_index.__aexit__.assert_called_once()

    @pytest.mark.asyncio
    async def test_ephemeral_async_calls_refresh_index(
        self,
        request: FixtureRequest,
        mocker: MockerFixture,
        vectorstore_cls: Type[PineconeVectorStore],
        mock_embedding_obj: str,
        mock_pinecone_client: MockType,
        mock_async_client: AsyncMockType,
        mock_async_index: AsyncMockType,
    ) -> None:
        """Sequential async calls without context manager recreate sessions as needed."""
        if vectorstore_cls is PineconeSparseVectorStore:
            pytest.skip("Not applicable for sparse vector store.")

        mock_embedding = request.getfixturevalue(mock_embedding_obj)
        mock_embedding.aembed_query = mocker.AsyncMock(return_value=[0.1, 0.2, 0.3])

        mock_async_index.query = mocker.AsyncMock(
            return_value={
                "matches": [
                    {
                        "metadata": {"text": "example text", "source": "unit"},
                        "score": 0.42,
                        "id": "example-id",
                    }
                ]
            }
        )
        mock_async_index.delete = mocker.AsyncMock(return_value=None)

        vectorstore = vectorstore_cls(
            embedding=mock_embedding,
            pinecone_api_key="test-key",
            index_name="test-index",
        )

        ids = await vectorstore.aadd_texts(["example text"])
        assert len(ids) == 1
        assert mock_async_index.__aenter__.call_count == 1
        await vectorstore.asimilarity_search("example", k=1)
        assert mock_async_index.__aenter__.call_count == 2
        await vectorstore.adelete(ids=ids)
        assert mock_async_index.__aenter__.call_count == 3


def test_similarity_search_by_vector_with_score__defaults_to_store_namespace(
    mock_embedding: AsyncMockType,
    mock_index: MockType,
) -> None:
    """Ensure sync similarity search falls back to the configured namespace when None is passed."""
    mock_index.query = Mock(return_value={"matches": []})
    vectorstore = PineconeVectorStore(
        index=mock_index,
        embedding=mock_embedding,
        text_key="text",
        namespace="default-ns",
    )

    vectorstore.similarity_search_by_vector_with_score(
        [0.1, 0.2, 0.3],
        namespace=None,
    )

    assert mock_index.query.call_args.kwargs["namespace"] == "default-ns"


@pytest.mark.asyncio
async def test_asimilarity_search_by_vector_with_score__defaults_to_store_namespace(
    mocker: MockerFixture,
    mock_embedding: AsyncMockType,
    mock_async_index: AsyncMockType,
) -> None:
    """Ensure async similarity search falls back to the configured namespace when None is passed."""
    mock_async_index.query = mocker.AsyncMock(return_value={"matches": []})
    vectorstore = PineconeVectorStore(
        index=mock_async_index,
        embedding=mock_embedding,
        text_key="text",
        namespace="default-ns",
    )

    await vectorstore.asimilarity_search_by_vector_with_score(
        [0.1, 0.2, 0.3],
        namespace=None,
    )

    assert mock_async_index.query.call_args.kwargs["namespace"] == "default-ns"


def test_sparse_similarity_search_by_vector_with_score(
    mocker: MockerFixture,
    mock_index: MockType,
    mock_sparse_embedding: AsyncMockType,
) -> None:
    """Sync sparse similarity search returns (Document, score) with correct fields."""
    mock_index.query = mocker.Mock(
        return_value={
            "matches": [
                {
                    "metadata": {"text": "test doc", "source": "unit"},
                    "score": 0.9,
                    "id": "doc-1",
                }
            ]
        }
    )
    vectorstore = PineconeSparseVectorStore(
        index=mock_index, embedding=mock_sparse_embedding, text_key="text"
    )
    embedding = SparseValues(indices=[0, 1], values=[0.5, 0.3])
    results = vectorstore.similarity_search_by_vector_with_score(embedding)

    assert len(results) == 1
    doc, score = results[0]
    assert doc.page_content == "test doc"
    assert doc.metadata == {"source": "unit"}
    assert doc.id == "doc-1"
    assert score == 0.9
    assert mock_index.query.call_args.kwargs["sparse_vector"] == embedding


@pytest.mark.asyncio
async def test_asparse_similarity_search_by_vector_with_score(
    mocker: MockerFixture,
    mock_async_index: MockType,
    mock_sparse_embedding: AsyncMockType,
) -> None:
    """Async sparse similarity search returns (Document, score) with correct fields."""
    mock_async_index.query = mocker.AsyncMock(
        return_value={
            "matches": [
                {
                    "metadata": {"text": "test doc", "source": "unit"},
                    "score": 0.9,
                    "id": "doc-1",
                }
            ]
        }
    )
    vectorstore = PineconeSparseVectorStore(
        index=mock_async_index, embedding=mock_sparse_embedding, text_key="text"
    )
    embedding = SparseValues(indices=[0, 1], values=[0.5, 0.3])
    results = await vectorstore.asimilarity_search_by_vector_with_score(embedding)

    assert len(results) == 1
    doc, score = results[0]
    assert doc.page_content == "test doc"
    assert doc.metadata == {"source": "unit"}
    assert doc.id == "doc-1"
    assert score == 0.9
    assert mock_async_index.query.call_args.kwargs["sparse_vector"] == embedding


def test_sparse_mmr_search_by_vector(
    mocker: MockerFixture,
    mock_index: MockType,
    mock_sparse_embedding: AsyncMockType,
) -> None:
    """Sync MMR search sets include_values=True and returns MMR-selected documents."""
    sparse_vals = {"indices": [0, 1], "values": [0.5, 0.3]}
    mock_index.query = mocker.Mock(
        return_value={
            "matches": [
                {
                    "metadata": {"text": "doc 1", "source": "a"},
                    "score": 0.9,
                    "id": "id-1",
                    "sparse_values": sparse_vals,
                },
                {
                    "metadata": {"text": "doc 2", "source": "b"},
                    "score": 0.8,
                    "id": "id-2",
                    "sparse_values": sparse_vals,
                },
            ]
        }
    )
    mocker.patch(
        "langchain_pinecone.vectorstores_sparse.sparse_maximal_marginal_relevance",
        return_value=[0],
    )
    vectorstore = PineconeSparseVectorStore(
        index=mock_index, embedding=mock_sparse_embedding, text_key="text"
    )
    embedding = SparseValues(indices=[0, 1], values=[0.5, 0.3])
    results = vectorstore.max_marginal_relevance_search_by_vector(embedding, k=1)

    assert mock_index.query.call_args.kwargs["include_values"] is True
    assert len(results) == 1
    assert results[0].page_content == "doc 1"


@pytest.mark.asyncio
async def test_asparse_mmr_search_by_vector(
    mocker: MockerFixture,
    mock_async_index: MockType,
    mock_sparse_embedding: AsyncMockType,
) -> None:
    """Async MMR search sets include_values=True and returns MMR-selected documents."""
    sparse_vals = {"indices": [0, 1], "values": [0.5, 0.3]}
    mock_async_index.query = mocker.AsyncMock(
        return_value={
            "matches": [
                {
                    "metadata": {"text": "doc 1", "source": "a"},
                    "score": 0.9,
                    "id": "id-1",
                    "sparse_values": sparse_vals,
                },
                {
                    "metadata": {"text": "doc 2", "source": "b"},
                    "score": 0.8,
                    "id": "id-2",
                    "sparse_values": sparse_vals,
                },
            ]
        }
    )
    mocker.patch(
        "langchain_pinecone.vectorstores_sparse.sparse_maximal_marginal_relevance",
        return_value=[0],
    )
    vectorstore = PineconeSparseVectorStore(
        index=mock_async_index, embedding=mock_sparse_embedding, text_key="text"
    )
    embedding = SparseValues(indices=[0, 1], values=[0.5, 0.3])
    results = await vectorstore.amax_marginal_relevance_search_by_vector(embedding, k=1)

    assert mock_async_index.query.call_args.kwargs["include_values"] is True
    assert len(results) == 1
    assert results[0].page_content == "doc 1"


def test_sparse_search_skips_match_without_text_key(
    mocker: MockerFixture,
    mock_index: MockType,
    mock_sparse_embedding: AsyncMockType,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A match missing text_key is skipped and a warning is logged."""
    mock_index.query = mocker.Mock(
        return_value={
            "matches": [
                {"metadata": {"source": "no-text"}, "score": 0.9, "id": "bad"},
                {"metadata": {"text": "valid doc"}, "score": 0.8, "id": "good"},
            ]
        }
    )
    vectorstore = PineconeSparseVectorStore(
        index=mock_index, embedding=mock_sparse_embedding, text_key="text"
    )
    embedding = SparseValues(indices=[0], values=[0.5])
    with caplog.at_level(
        logging.WARNING, logger="langchain_pinecone.vectorstores_sparse"
    ):
        results = vectorstore.similarity_search_by_vector_with_score(embedding)

    assert len(results) == 1
    assert results[0][0].page_content == "valid doc"
    assert "Skipping" in caplog.text


def test_sparse_vectorstore_rejects_dense_index(
    mock_index: MockType,
    mock_sparse_embedding: AsyncMockType,
) -> None:
    """Construction raises ValueError when the index's vector_type is not sparse."""
    mock_index.describe_index_stats.return_value = {"vector_type": "dense"}
    with pytest.raises(ValueError, match="Sparse Indexes"):
        PineconeSparseVectorStore(
            index=mock_index, embedding=mock_sparse_embedding, text_key="text"
        )


# --- TC-005: PineconeVectorStore constructor entrypoints ---


def test_get_pinecone_index_returns_named(
    mocker: MockerFixture, mock_index: MockType
) -> None:
    """get_pinecone_index returns the named Index when the name is present in the project."""
    mock_client_cls = mocker.patch("langchain_pinecone.vectorstores.PineconeClient")
    mock_client = mock_client_cls.return_value
    mock_client.Index.return_value = mock_index

    existing = mocker.Mock()
    existing.name = "my-index"
    mock_client.list_indexes.return_value.index_list = {"indexes": [existing]}

    result = PineconeVectorStore.get_pinecone_index(
        "my-index", pinecone_api_key="test-key"
    )

    assert result is mock_index
    mock_client.Index.assert_called_once_with("my-index")


def test_get_pinecone_index_no_indexes_raises(mocker: MockerFixture) -> None:
    """get_pinecone_index raises ValueError when the project has no active indexes."""
    mock_client_cls = mocker.patch("langchain_pinecone.vectorstores.PineconeClient")
    mock_client_cls.return_value.list_indexes.return_value.index_list = {"indexes": []}

    with pytest.raises(ValueError, match="No active indexes"):
        PineconeVectorStore.get_pinecone_index("any-name", pinecone_api_key="test-key")


def test_get_pinecone_index_unknown_name_raises(mocker: MockerFixture) -> None:
    """get_pinecone_index raises ValueError listing the existing indexes when name not found."""
    mock_client_cls = mocker.patch("langchain_pinecone.vectorstores.PineconeClient")
    existing = mocker.Mock()
    existing.name = "other-index"
    mock_client_cls.return_value.list_indexes.return_value.index_list = {
        "indexes": [existing]
    }

    with pytest.raises(ValueError, match="other-index"):
        PineconeVectorStore.get_pinecone_index("my-index", pinecone_api_key="test-key")


def test_from_texts_orchestration(
    mocker: MockerFixture, mock_index: MockType, mock_embedding: AsyncMockType
) -> None:
    """from_texts wires get_pinecone_index → constructor → add_texts and returns the store."""
    mocker.patch.object(
        PineconeVectorStore, "get_pinecone_index", return_value=mock_index
    )
    mock_embedding.embed_documents = mocker.Mock(
        return_value=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    )

    texts = ["text one", "text two"]
    result = PineconeVectorStore.from_texts(
        texts, mock_embedding, index_name="my-index", async_req=False
    )

    assert isinstance(result, PineconeVectorStore)
    mock_embedding.embed_documents.assert_called_once_with(texts)
    mock_index.upsert.assert_called_once()


def test_from_existing_index_builds_store(
    mocker: MockerFixture, mock_index: MockType, mock_embedding: AsyncMockType
) -> None:
    """from_existing_index constructs the store from a resolved index without embedding calls."""
    mocker.patch.object(
        PineconeVectorStore, "get_pinecone_index", return_value=mock_index
    )

    result = PineconeVectorStore.from_existing_index("my-index", mock_embedding)

    assert isinstance(result, PineconeVectorStore)
    mock_embedding.embed_documents.assert_not_called()
    mock_embedding.aembed_documents.assert_not_called()


async def test_afrom_texts_orchestration(
    mocker: MockerFixture,
    mock_embedding: AsyncMockType,
    mock_pinecone_client: MockType,
    mock_async_client: AsyncMockType,
    mock_async_index: AsyncMockType,
) -> None:
    """afrom_texts constructs via index_name and delegates to aadd_texts."""
    mock_embedding.aembed_documents = mocker.AsyncMock(
        return_value=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    )

    texts = ["text one", "text two"]
    result = await PineconeVectorStore.afrom_texts(
        texts,
        mock_embedding,
        index_name="my-index",
        pinecone_api_key="test-key",
    )

    assert isinstance(result, PineconeVectorStore)
    mock_embedding.aembed_documents.assert_called_once_with(texts)
    mock_async_index.upsert.assert_called_once()


# --- TC-006: PineconeVectorStore edge paths and relevance scoring ---


def test_search_skips_match_without_text_key(
    mocker: MockerFixture,
    mock_index: MockType,
    mock_embedding: AsyncMockType,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """similarity_search_by_vector_with_score skips a match missing text_key and logs a warning."""
    mock_index.query = mocker.Mock(
        return_value={
            "matches": [
                {
                    "metadata": {"text": "valid doc", "extra": "x"},
                    "score": 0.9,
                    "id": "good",
                },
                {"metadata": {"source": "no-text"}, "score": 0.5, "id": "bad"},
            ]
        }
    )
    vectorstore = PineconeVectorStore(
        index=mock_index, embedding=mock_embedding, text_key="text"
    )
    with caplog.at_level(logging.WARNING, logger="langchain_pinecone.vectorstores"):
        results = vectorstore.similarity_search_by_vector_with_score([0.1, 0.2, 0.3])

    assert len(results) == 1
    doc, score = results[0]
    assert doc.page_content == "valid doc"
    assert score == 0.9
    assert "Skipping" in caplog.text


async def test_asearch_skips_match_without_text_key(
    mocker: MockerFixture,
    mock_async_index: AsyncMockType,
    mock_embedding: AsyncMockType,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """asimilarity_search_by_vector_with_score skips a match missing text_key and logs a warning."""
    mock_async_index.query = mocker.AsyncMock(
        return_value={
            "matches": [
                {
                    "metadata": {"text": "valid doc", "extra": "x"},
                    "score": 0.9,
                    "id": "good",
                },
                {"metadata": {"source": "no-text"}, "score": 0.5, "id": "bad"},
            ]
        }
    )
    vectorstore = PineconeVectorStore(
        index=mock_async_index, embedding=mock_embedding, text_key="text"
    )
    with caplog.at_level(logging.WARNING, logger="langchain_pinecone.vectorstores"):
        results = await vectorstore.asimilarity_search_by_vector_with_score(
            [0.1, 0.2, 0.3]
        )

    assert len(results) == 1
    doc, score = results[0]
    assert doc.page_content == "valid doc"
    assert score == 0.9
    assert "Skipping" in caplog.text


@pytest.mark.parametrize(
    "strategy,expected_fn",
    [
        (DistanceStrategy.COSINE, PineconeVectorStore._cosine_relevance_score_fn),
        (
            DistanceStrategy.MAX_INNER_PRODUCT,
            PineconeVectorStore._max_inner_product_relevance_score_fn,
        ),
        (
            DistanceStrategy.EUCLIDEAN_DISTANCE,
            PineconeVectorStore._euclidean_relevance_score_fn,
        ),
    ],
)
def test_select_relevance_score_fn_per_strategy(
    mock_index: MockType,
    mock_embedding: AsyncMockType,
    strategy: DistanceStrategy,
    expected_fn: Callable[[float], float],
) -> None:
    """_select_relevance_score_fn returns the expected callable for each DistanceStrategy."""
    vectorstore = PineconeVectorStore(
        index=mock_index,
        embedding=mock_embedding,
        text_key="text",
        distance_strategy=strategy,
    )
    fn = vectorstore._select_relevance_score_fn()
    assert fn is expected_fn


def test_select_relevance_score_fn_raises_for_unknown_strategy(
    mock_index: MockType,
    mock_embedding: AsyncMockType,
) -> None:
    """_select_relevance_score_fn raises ValueError for an unrecognized strategy."""
    vectorstore = PineconeVectorStore(
        index=mock_index, embedding=mock_embedding, text_key="text"
    )
    vectorstore.distance_strategy = "UNKNOWN"  # type: ignore[assignment]
    with pytest.raises(ValueError, match="Unknown distance strategy"):
        vectorstore._select_relevance_score_fn()


def test_cosine_relevance_score_fn() -> None:
    """_cosine_relevance_score_fn maps Pinecone's [-1, 1] score to [0, 1] via (score + 1) / 2."""
    assert PineconeVectorStore._cosine_relevance_score_fn(-1.0) == pytest.approx(0.0)
    assert PineconeVectorStore._cosine_relevance_score_fn(1.0) == pytest.approx(1.0)
    assert PineconeVectorStore._cosine_relevance_score_fn(0.0) == pytest.approx(0.5)


def test_sync_query_rejects_applyresult(
    mocker: MockerFixture,
    mock_index: MockType,
    mock_embedding: AsyncMockType,
) -> None:
    """similarity_search_by_vector_with_score raises ValueError when index.query returns ApplyResult."""
    try:
        from pinecone.db_data.index import ApplyResult
    except ImportError:
        from pinecone.data.index import ApplyResult

    mock_index.query = mocker.Mock(return_value=object.__new__(ApplyResult))
    vectorstore = PineconeVectorStore(
        index=mock_index, embedding=mock_embedding, text_key="text"
    )
    with pytest.raises(ValueError, match="asynchronous result from synchronous call"):
        vectorstore.similarity_search_by_vector_with_score([0.1, 0.2, 0.3])


def test_mmr_sync_query_rejects_applyresult(
    mocker: MockerFixture,
    mock_index: MockType,
    mock_embedding: AsyncMockType,
) -> None:
    """max_marginal_relevance_search_by_vector raises ValueError when index.query returns ApplyResult."""
    try:
        from pinecone.db_data.index import ApplyResult
    except ImportError:
        from pinecone.data.index import ApplyResult

    mock_index.query = mocker.Mock(return_value=object.__new__(ApplyResult))
    vectorstore = PineconeVectorStore(
        index=mock_index, embedding=mock_embedding, text_key="text"
    )
    with pytest.raises(ValueError, match="asynchronous result from synchronous call"):
        vectorstore.max_marginal_relevance_search_by_vector([0.1, 0.2, 0.3])


def test_delete_requires_an_argument(
    mock_index: MockType,
    mock_embedding: AsyncMockType,
) -> None:
    """delete() raises ValueError when none of ids, delete_all, or filter is provided."""
    vectorstore = PineconeVectorStore(
        index=mock_index, embedding=mock_embedding, text_key="text"
    )
    with pytest.raises(ValueError, match="ids, delete_all, or filter"):
        vectorstore.delete()


# --- FIX-003: Namespace fallback regression tests (issue #63) ---
# Regression introduced in v0.2.11 (PR #60): kwargs.get("namespace", self._namespace)
# did not fall back when namespace=None was explicitly passed through the call chain.
# Fixed in PR #83 by splitting to kwargs.get("namespace") + if-None check.


def test_similarity_search_with_score__honors_constructor_namespace(
    mocker: MockerFixture,
    mock_embedding: AsyncMockType,
    mock_index: MockType,
) -> None:
    """similarity_search_with_score with no query-time namespace uses constructor namespace."""
    mock_embedding.embed_query = mocker.Mock(return_value=[0.1, 0.2, 0.3])
    mock_index.query = mocker.Mock(return_value={"matches": []})
    vectorstore = PineconeVectorStore(
        index=mock_index,
        embedding=mock_embedding,
        text_key="text",
        namespace="constructor-ns",
    )

    vectorstore.similarity_search_with_score("test query")

    assert mock_index.query.call_args.kwargs["namespace"] == "constructor-ns"


@pytest.mark.asyncio
async def test_asimilarity_search_with_score__honors_constructor_namespace(
    mocker: MockerFixture,
    mock_embedding: AsyncMockType,
    mock_async_index: AsyncMockType,
) -> None:
    """asimilarity_search_with_score with no query-time namespace uses constructor namespace."""
    mock_embedding.aembed_query = mocker.AsyncMock(return_value=[0.1, 0.2, 0.3])
    mock_async_index.query = mocker.AsyncMock(return_value={"matches": []})
    vectorstore = PineconeVectorStore(
        index=mock_async_index,
        embedding=mock_embedding,
        text_key="text",
        namespace="constructor-ns",
    )

    await vectorstore.asimilarity_search_with_score("test query")

    assert mock_async_index.query.call_args.kwargs["namespace"] == "constructor-ns"


def test_similarity_search__honors_constructor_namespace(
    mocker: MockerFixture,
    mock_embedding: AsyncMockType,
    mock_index: MockType,
) -> None:
    """similarity_search with no query-time namespace uses constructor namespace.

    This is the exact bug scenario: as_retriever(search_type="similarity") passes
    namespace=None through the call chain, which must fall back to self._namespace.
    """
    mock_embedding.embed_query = mocker.Mock(return_value=[0.1, 0.2, 0.3])
    mock_index.query = mocker.Mock(return_value={"matches": []})
    vectorstore = PineconeVectorStore(
        index=mock_index,
        embedding=mock_embedding,
        text_key="text",
        namespace="constructor-ns",
    )

    vectorstore.similarity_search("test query")

    assert mock_index.query.call_args.kwargs["namespace"] == "constructor-ns"


@pytest.mark.asyncio
async def test_asimilarity_search__honors_constructor_namespace(
    mocker: MockerFixture,
    mock_embedding: AsyncMockType,
    mock_async_index: AsyncMockType,
) -> None:
    """asimilarity_search with no query-time namespace uses constructor namespace."""
    mock_embedding.aembed_query = mocker.AsyncMock(return_value=[0.1, 0.2, 0.3])
    mock_async_index.query = mocker.AsyncMock(return_value={"matches": []})
    vectorstore = PineconeVectorStore(
        index=mock_async_index,
        embedding=mock_embedding,
        text_key="text",
        namespace="constructor-ns",
    )

    await vectorstore.asimilarity_search("test query")

    assert mock_async_index.query.call_args.kwargs["namespace"] == "constructor-ns"


# --- FIX-002: from_texts forwards pinecone_api_key to get_pinecone_index (issue #110) ---


def test_from_texts_forwards_pinecone_api_key(
    mocker: MockerFixture, mock_index: MockType, mock_embedding: AsyncMockType
) -> None:
    """from_texts passes the pinecone_api_key kwarg through to get_pinecone_index."""
    mock_get_index = mocker.patch.object(
        PineconeVectorStore, "get_pinecone_index", return_value=mock_index
    )
    mock_embedding.embed_documents = mocker.Mock(return_value=[[0.1, 0.2, 0.3]])

    PineconeVectorStore.from_texts(
        ["text"],
        mock_embedding,
        index_name="my-index",
        pinecone_api_key="caller-supplied-key",
        async_req=False,
    )

    mock_get_index.assert_called_once_with(
        "my-index", 4, pinecone_api_key="caller-supplied-key"
    )


def test_from_texts_raises_when_no_api_key(
    mocker: MockerFixture, mock_embedding: AsyncMockType
) -> None:
    """from_texts raises ValueError when neither pinecone_api_key kwarg nor env var is set."""
    mocker.patch.dict("os.environ", {"PINECONE_API_KEY": ""})

    with pytest.raises(ValueError, match="Pinecone API key"):
        PineconeVectorStore.from_texts(
            ["text"],
            mock_embedding,
            index_name="my-index",
        )


def test_similarity_score_threshold_retriever__honors_constructor_namespace(
    mocker: MockerFixture,
    mock_embedding: AsyncMockType,
    mock_index: MockType,
) -> None:
    """as_retriever(similarity_score_threshold) with no query-time namespace uses constructor namespace.

    The threshold retriever calls similarity_search_with_score internally;
    verifies the namespace fallback survives the full retriever → vectorstore chain.
    """
    mock_embedding.embed_query = mocker.Mock(return_value=[0.1, 0.2, 0.3])
    mock_index.query = mocker.Mock(return_value={"matches": []})
    vectorstore = PineconeVectorStore(
        index=mock_index,
        embedding=mock_embedding,
        text_key="text",
        namespace="constructor-ns",
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"score_threshold": 0.5, "k": 5},
    )
    retriever.invoke("test query")

    assert mock_index.query.call_args.kwargs["namespace"] == "constructor-ns"

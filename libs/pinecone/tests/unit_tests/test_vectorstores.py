from typing import TYPE_CHECKING, Type
from unittest.mock import ANY, Mock, call

import pytest
from pinecone import PineconeAsyncio, SparseValues  # type: ignore[import-untyped]
from pytest_mock import AsyncMockType, MockerFixture, MockType

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

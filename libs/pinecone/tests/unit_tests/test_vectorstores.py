from typing import TYPE_CHECKING, Type
from unittest.mock import Mock

import pytest
from pinecone import SparseValues  # type: ignore[import-untyped]
from pytest_mock import AsyncMockType, MockerFixture

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
    mock_embedding = mocker.AsyncMock()
    mock_embedding.aembed_documents = mocker.AsyncMock(return_value=[[0.1, 0.2, 0.3]])
    return mock_embedding


@pytest.fixture
def mock_sparse_embedding(mocker: MockerFixture) -> AsyncMockType:
    """Fixture for mock embedding function."""
    mock_embedding = mocker.AsyncMock(spec=PineconeSparseEmbeddings)
    mock_embedding.aembed_documents = mocker.AsyncMock(
        return_value=[SparseValues(indices=[0, 28, 218], values=[0.34, 0.239, 0.92])]
    )
    return mock_embedding


@pytest.fixture
def mock_async_index(mocker: MockerFixture) -> AsyncMockType:
    """Fixture for mock async index."""
    # Import the actual _IndexAsyncio class to use as spec
    from pinecone.data import _IndexAsyncio  # type:ignore[import-untyped]

    mock_async_index = mocker.AsyncMock(spec=_IndexAsyncio)
    mock_async_index.config = mocker.Mock()
    mock_async_index.config.host = "example.org"
    mock_async_index.__aenter__ = mocker.AsyncMock(return_value=mock_async_index)
    mock_async_index.__aexit__ = mocker.AsyncMock(return_value=None)
    mock_async_index.upsert = mocker.AsyncMock(return_value=None)
    mock_async_index.describe_index_stats = mocker.Mock(
        return_value={"vector_type": "sparse"}
    )
    return mock_async_index


@pytest.fixture
def mock_index(mocker: MockerFixture) -> AsyncMockType:
    """Fixture for mock async index."""
    # Import the actual _IndexAsyncio class to use as spec
    from pinecone.data import _Index

    mock_index = mocker.AsyncMock(spec=_Index)
    mock_index.config = mocker.Mock()
    mock_index.config.host = "example.org"
    mock_index.upsert = mocker.AsyncMock(return_value=None)
    mock_index.describe_index_stats = mocker.Mock(
        return_value={"vector_type": "sparse"}
    )
    return mock_index


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
    vectorstore.add_texts(texts, id_prefix=id_prefix)


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
        mocker: MockerFixture,
        vectorstore_cls: Type[PineconeVectorStore],
        mock_index: AsyncMockType,
        mock_embedding_obj: str,
        mock_async_index: AsyncMockType,
    ) -> None:
        """Test that initializing with a sync index still enables async operations."""

        # Mock the async client
        mock_async_client = mocker.patch(
            "langchain_pinecone.vectorstores.PineconeAsyncioClient"
        )

        mock_embedding = request.getfixturevalue(mock_embedding_obj)

        # Create vectorstore with sync index
        vectorstore = vectorstore_cls(
            index=mock_index, embedding=mock_embedding, text_key="text"
        )

        # Mock the async index creation to return our mock
        mock_async_client.return_value.IndexAsyncio.return_value = mock_async_index

        texts = ["test document"]
        await vectorstore.aadd_texts(texts)

        # Verify async client was created with correct params
        mock_async_client.assert_called_once_with(
            api_key=mock_index.config.api_key, source_tag="langchain"
        )

        # Verify the async upsert was called
        mock_async_index.upsert.assert_called_once()

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

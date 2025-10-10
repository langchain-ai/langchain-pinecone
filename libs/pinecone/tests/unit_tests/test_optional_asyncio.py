import pytest
from langchain_core.utils import convert_to_secret_str
from pytest_mock import MockerFixture

from langchain_pinecone.embeddings import PineconeEmbeddings
from langchain_pinecone.rerank import PineconeRerank
from langchain_pinecone.vectorstores import PineconeVectorStore


def test_vectorstore_sync_works_without_asyncio_extra(mocker: MockerFixture) -> None:
    # Simulate missing asyncio extra
    mocker.patch("langchain_pinecone.vectorstores.PineconeAsyncioClient", None)

    # Mock sync index and embedding
    mock_index = mocker.Mock()
    mock_index.config = mocker.Mock()
    mock_index.config.host = "example.org"
    mock_index.config.api_key = "test"
    mock_index.upsert = mocker.Mock(return_value=None)

    mock_embedding = mocker.Mock()
    mock_embedding.embed_documents = mocker.Mock(return_value=[[0.1, 0.2, 0.3]])

    vs = PineconeVectorStore(
        index=mock_index, embedding=mock_embedding, text_key="text"
    )

    # Sync path should work without asyncio client
    vs.add_texts(["hello"], async_req=False)
    mock_index.upsert.assert_called_once()


@pytest.mark.asyncio
async def test_vectorstore_async_raises_without_asyncio_extra(
    mocker: MockerFixture,
) -> None:
    mocker.patch("langchain_pinecone.vectorstores.PineconeAsyncioClient", None)

    mock_async_index = mocker.Mock()
    mock_async_index.config = mocker.Mock(host="example.org", api_key="test")

    mock_embedding = mocker.Mock()
    mock_embedding.aembed_documents = mocker.AsyncMock(return_value=[[0.1, 0.2, 0.3]])

    vs = PineconeVectorStore(
        index=mock_async_index, embedding=mock_embedding, text_key="text"
    )

    with pytest.raises(ImportError):
        await vs.async_index


def test_embeddings_sync_works_without_asyncio_extra(mocker: MockerFixture) -> None:
    mocker.patch("langchain_pinecone.embeddings.PineconeAsyncioClient", None)
    mocker.patch(
        "langchain_pinecone.embeddings.PineconeEmbeddings.list_supported_models",
        return_value=[{"model": "multilingual-e5-large"}],
    )

    emb = PineconeEmbeddings(
        model="multilingual-e5-large", pinecone_api_key=convert_to_secret_str("test")
    )
    # Sync methods should work
    mock_client = mocker.patch.object(emb, "_client")
    mock_client.inference.embed.return_value = [{"values": [0.1, 0.2]}]
    assert isinstance(emb.embed_query("hi"), list)


@pytest.mark.asyncio
async def test_embeddings_async_raises_without_asyncio_extra(
    mocker: MockerFixture,
) -> None:
    mocker.patch("langchain_pinecone.embeddings.PineconeAsyncioClient", None)
    mocker.patch(
        "langchain_pinecone.embeddings.PineconeEmbeddings.list_supported_models",
        return_value=[{"model": "multilingual-e5-large"}],
    )

    emb = PineconeEmbeddings(
        model="multilingual-e5-large", pinecone_api_key=convert_to_secret_str("test")
    )
    with pytest.raises(ImportError):
        await emb.aembed_query("hi")


@pytest.mark.asyncio
async def test_rerank_async_raises_without_asyncio_extra(mocker: MockerFixture) -> None:
    mocker.patch("langchain_pinecone.rerank.PineconeAsyncio", None)
    mocker.patch(
        "langchain_pinecone.rerank.PineconeRerank.list_supported_models",
        return_value=[{"model": "bge-reranker-v2-m3"}],
    )

    rr = PineconeRerank(pinecone_api_key=convert_to_secret_str("test"))
    with pytest.raises(ImportError):
        await rr._get_async_client()

    # Public API should surface empty result while logging the guidance message
    result = await rr.arerank(["doc"], query="q")
    assert result == []

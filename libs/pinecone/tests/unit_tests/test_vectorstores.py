from unittest.mock import AsyncMock, Mock

import pytest

from langchain_pinecone.vectorstores import PineconeVectorStore


def test_initialization() -> None:
    """Test integration vectorstore initialization."""
    # mock index
    index = Mock()
    embedding = Mock()
    text_key = "xyz"
    PineconeVectorStore(index, embedding, text_key)


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


@pytest.mark.asyncio
async def test_async_id_prefix() -> None:
    """Test integration of the id_prefix parameter."""
    embedding = AsyncMock()
    embedding.embed_documents = AsyncMock(return_value=[0.1, 0.2, 0.3, 0.4, 0.5])
    index = AsyncMock()
    index.upsert = Mock(return_value=None)
    text_key = "testing"
    vectorstore = PineconeVectorStore(
        index_name="testing",
        embedding=embedding,
        text_key=text_key,
        pinecone_api_key="test",
    )
    texts = ["alpha", "beta", "gamma", "delta", "epsilon"]
    id_prefix = "testing_prefixes"
    await vectorstore.aadd_texts(texts, id_prefix=id_prefix)

from unittest.mock import Mock, patch

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
    vectorstore.add_texts(texts, id_prefix=id_prefix, async_req=False)


def test_initialization_grpc_import_error() -> None:
    embedding = Mock()
    # Mock grpcio modules not installed
    with patch.dict("sys.modules", {"google": None}):
        with pytest.raises(ImportError) as error:
            PineconeVectorStore(
                embedding=embedding,
                index_name="abc",
                pinecone_api_key="xyz",
                use_grpc=True,
            )

    assert "Install grpc extras to use the Pinecone GRPC client" in str(error.value)

import os
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document
from pinecone import Pinecone
from pydantic import SecretStr

from langchain_pinecone.pinecone_rerank import PineconeRerank


class TestPineconeRerank:
    @pytest.fixture
    def mock_pinecone_client(self):
        """Fixture to provide a mocked Pinecone client."""
        mock_client = MagicMock(spec=Pinecone)
        mock_client.inference = MagicMock()
        return mock_client

    @pytest.fixture
    def mock_rerank_response(self):
        """Fixture to provide a mocked rerank API response."""
        mock_result1 = MagicMock()
        mock_result1.id = "doc0"
        mock_result1.score = 0.9
        mock_result1.document = {"id": "doc0", "content": "Document 1 content"}

        mock_result2 = MagicMock()
        mock_result2.id = "doc1"
        mock_result2.score = 0.7
        mock_result2.document = {"id": "doc1", "content": "Document 2 content"}

        return [mock_result1, mock_result2]

    def test_initialization_with_api_key(self, mock_pinecone_client):
        """Test initialization with API key environment variable."""
        with patch.dict(os.environ, {"PINECONE_API_KEY": "fake-api-key"}):
            with patch(
                "langchain_pinecone.pinecone_rerank.Pinecone",
                return_value=mock_pinecone_client,
            ) as mock_pinecone_constructor:
                reranker = PineconeRerank(model="test-model")
                mock_pinecone_constructor.assert_called_once_with(
                    api_key="fake-api-key"
                )
                assert reranker.client == mock_pinecone_client
                assert reranker.model == "test-model"
                assert reranker.top_n == 3  # Default value

    def test_initialization_with_client(self, mock_pinecone_client):
        """Test initialization with a provided Pinecone client instance."""
        reranker = PineconeRerank(client=mock_pinecone_client, model="test-model")
        assert reranker.client == mock_pinecone_client
        assert reranker.model == "test-model"

    def test_initialization_missing_model(self):
        """Test initialization fails if model is not specified."""
        with pytest.raises(ValueError, match="Did not find `model`!"):
            PineconeRerank(pinecone_api_key=SecretStr("fake-key"))

    def test_initialization_invalid_client_type(self):
        """Test initialization fails with invalid client type."""
        with pytest.raises(
            ValueError, match="The 'client' parameter must be an instance of"
        ):
            PineconeRerank(client="not a pinecone client", model="test-model")

    def test_validate_environment_with_api_key(self, mock_pinecone_client):
        """Test validate_environment creates client with API key."""
        with patch.dict(os.environ, {"PINECONE_API_KEY": "fake-api-key"}):
            with patch(
                "langchain_pinecone.pinecone_rerank.Pinecone",
                return_value=mock_pinecone_client,
            ) as mock_pinecone_constructor:
                reranker = PineconeRerank(model="test-model", client=None)
                reranker.validate_environment()
                mock_pinecone_constructor.assert_called_once_with(
                    api_key="fake-api-key"
                )
                assert reranker.client == mock_pinecone_client

    def test_validate_environment_with_client(self, mock_pinecone_client):
        """Test validate_environment keeps provided client."""
        reranker = PineconeRerank(client=mock_pinecone_client, model="test-model")
        reranker.validate_environment()
        assert reranker.client == mock_pinecone_client

    def test_validate_model_specified(self):
        """Test validate_model_specified passes when model is set."""
        reranker = PineconeRerank(
            model="test-model", pinecone_api_key=SecretStr("fake-key")
        )
        reranker.validate_model_specified()  # Should not raise error

    def test_validate_model_specified_missing(self):
        """Test validate_model_specified fails when model is missing."""
        with pytest.raises(ValueError, match="Did not find `model`!"):
            PineconeRerank(pinecone_api_key=SecretStr("fake-key"))

    @pytest.mark.parametrize(
        "document_input, expected_output",
        [
            ("just a string", {"id": "doc0", "content": "just a string"}),
            (
                Document(page_content="doc content", metadata={"source": "test"}),
                {"id": "doc0", "content": "doc content"},
            ),
            (
                {"id": "custom-id", "content": "dict content"},
                {"id": "custom-id", "content": "dict content"},
            ),
            (
                {"content": "dict content without id"},
                {"id": "doc0", "content": "dict content without id"},
            ),
        ],
    )
    def test__document_to_dict(self, document_input, expected_output):
        """Test _document_to_dict handles different input types."""
        reranker = PineconeRerank(
            model="test-model", pinecone_api_key=SecretStr("fake-key")
        )
        result = reranker._document_to_dict(document_input, 0)
        assert result == expected_output

    def test_rerank_empty_documents(self, mock_pinecone_client):
        """Test rerank returns empty list for empty documents."""
        reranker = PineconeRerank(client=mock_pinecone_client, model="test-model")
        results = reranker.rerank([], "query")
        assert results == []
        mock_pinecone_client.inference.rerank.assert_not_called()

    def test_rerank_calls_api_and_formats_results(
        self, mock_pinecone_client, mock_rerank_response
    ):
        """Test rerank calls API with correct args and formats results."""
        mock_pinecone_client.inference.rerank.return_value = mock_rerank_response

        reranker = PineconeRerank(
            client=mock_pinecone_client,
            model="test-model",
            top_n=2,
            rank_fields=["text"],
            return_documents=True,
        )
        documents = ["doc1 content", "doc2 content", "doc3 content"]
        query = "test query"

        results = reranker.rerank(documents, query)

        mock_pinecone_client.inference.rerank.assert_called_once_with(
            model="test-model",
            query=query,
            documents=[
                {"id": "doc0", "content": "doc1 content"},
                {"id": "doc1", "content": "doc2 content"},
                {"id": "doc2", "content": "doc3 content"},
            ],
            rank_fields=["text"],
            top_n=2,
            return_documents=True,
            parameters={"truncate": "END"},
        )

        assert len(results) == 2
        assert results[0]["id"] == "doc0"
        assert results[0]["score"] == 0.9
        assert results[0]["index"] == 0
        assert results[0]["document"] == {"id": "doc0", "content": "Document 1 content"}

        assert results[1]["id"] == "doc1"
        assert results[1]["score"] == 0.7
        assert results[1]["index"] == 1
        assert results[1]["document"] == {"id": "doc1", "content": "Document 2 content"}

    def test_compress_documents(self, mock_pinecone_client, mock_rerank_response):
        """Test compress_documents calls rerank and formats output as Documents."""
        reranker = PineconeRerank(
            client=mock_pinecone_client, model="test-model", return_documents=True
        )
        documents = [
            Document(page_content="Document 1 content", metadata={"source": "a"}),
            Document(page_content="Document 2 content", metadata={"source": "b"}),
            Document(page_content="Document 3 content", metadata={"source": "c"}),
        ]
        query = "test query"

        with patch.object(
            reranker, "rerank", return_value=mock_rerank_response
        ) as mock_rerank:
            compressed_docs = reranker.compress_documents(documents, query)

            mock_rerank.assert_called_once_with(documents, query)

            assert len(compressed_docs) == 2
            assert isinstance(compressed_docs[0], Document)
            assert compressed_docs[0].page_content == "Document 1 content"
            assert compressed_docs[0].metadata["source"] == "a"
            assert compressed_docs[0].metadata["relevance_score"] == 0.9

            assert isinstance(compressed_docs[1], Document)
            assert compressed_docs[1].page_content == "Document 2 content"
            assert compressed_docs[1].metadata["source"] == "b"
            assert compressed_docs[1].metadata["relevance_score"] == 0.7

    def test_compress_documents_no_return_documents(self, mock_pinecone_client):
        """Test compress_documents when return_documents is False."""
        reranker = PineconeRerank(
            client=mock_pinecone_client, model="test-model", return_documents=False
        )
        documents = [
            Document(page_content="Document 1 content", metadata={"source": "a"}),
            Document(page_content="Document 2 content", metadata={"source": "b"}),
        ]
        query = "test query"

        # Mock rerank to return results without the 'document' field
        mock_result1 = MagicMock()
        mock_result1.id = "doc0"
        mock_result1.score = 0.9
        del mock_result1.document  # Simulate no document returned

        mock_result2 = MagicMock()
        mock_result2.id = "doc1"
        mock_result2.score = 0.7
        del mock_result2.document  # Simulate no document returned

        mock_rerank_response_no_docs = [mock_result1, mock_result2]

        with patch.object(
            reranker, "rerank", return_value=mock_rerank_response_no_docs
        ) as mock_rerank:
            compressed_docs = reranker.compress_documents(documents, query)

            mock_rerank.assert_called_once_with(documents, query)

            assert len(compressed_docs) == 2
            assert isinstance(compressed_docs[0], Document)
            assert compressed_docs[0].page_content == "Document 1 content"
            assert compressed_docs[0].metadata["source"] == "a"
            assert compressed_docs[0].metadata["relevance_score"] == 0.9

            assert isinstance(compressed_docs[1], Document)
            assert compressed_docs[1].page_content == "Document 2 content"
            assert compressed_docs[1].metadata["source"] == "b"
            assert compressed_docs[1].metadata["relevance_score"] == 0.7

    def test_compress_documents_index_none(
        self, mock_pinecone_client, mock_rerank_response
    ):
        """Test compress_documents handles results where index is None."""
        reranker = PineconeRerank(
            client=mock_pinecone_client, model="test-model", return_documents=True
        )
        documents = [
            Document(page_content="Document 1 content", metadata={"source": "a"}),
        ]
        query = "test query"

        # Mock rerank to return a result with an ID that doesn't match any input document ID
        mock_result_unknown = MagicMock()
        mock_result_unknown.id = "unknown-doc"
        mock_result_unknown.score = 0.5
        mock_result_unknown.document = {
            "id": "unknown-doc",
            "content": "Unknown content",
        }

        mock_rerank_response_unknown = [mock_result_unknown]

        with patch.object(
            reranker, "rerank", return_value=mock_rerank_response_unknown
        ) as mock_rerank:
            compressed_docs = reranker.compress_documents(documents, query)

            mock_rerank.assert_called_once_with(documents, query)

            assert (
                len(compressed_docs) == 0
            )  # No documents should be added if index is None

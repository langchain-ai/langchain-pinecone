import os
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document
from pinecone import Pinecone  # type: ignore
from pydantic import SecretStr

from langchain_pinecone.rerank import PineconeRerank


class TestPineconeRerank:
    @pytest.fixture
    def mock_pinecone_client(self) -> MagicMock:
        """Fixture to provide a mocked Pinecone client."""
        mock_client = MagicMock(spec=Pinecone)
        mock_client.inference = MagicMock()
        return mock_client

    @pytest.fixture
    def mock_rerank_response(self) -> MagicMock:
        """Fixture to provide a mocked rerank API response."""
        mock_result1 = MagicMock()
        mock_result1.id = "doc0"
        mock_result1.index = 0
        mock_result1.score = 0.9
        mock_result1.document = {"id": "doc0", "text": "Document 1 content"}

        mock_result2 = MagicMock()
        mock_result2.id = "doc1"
        mock_result2.index = 1
        mock_result2.score = 0.7
        mock_result2.document = {"id": "doc1", "text": "Document 2 content"}

        mock_response = MagicMock()
        mock_response.data = [mock_result1, mock_result2]
        return mock_response

    def test_initialization_with_api_key(self, mock_pinecone_client: MagicMock) -> None:
        """Test initialization with API key environment variable."""
        with patch.dict(os.environ, {"PINECONE_API_KEY": "fake-api-key"}):
            with patch(
                "langchain_pinecone.rerank.Pinecone",
                return_value=mock_pinecone_client,
            ) as mock_pinecone_constructor:
                reranker = PineconeRerank(model="test-model")
                mock_pinecone_constructor.assert_called_once_with(
                    api_key="fake-api-key"
                )
                assert reranker.client == mock_pinecone_client
                assert reranker.model == "test-model"
                assert reranker.top_n == 3  # Default value

    def test_initialization_with_client(self, mock_pinecone_client: MagicMock) -> None:
        """Test initialization with a provided Pinecone client instance."""
        reranker = PineconeRerank(client=mock_pinecone_client, model="test-model")
        assert reranker.client == mock_pinecone_client
        assert reranker.model == "test-model"

    def test_initialization_missing_model(self) -> None:
        """Test initialization fails if model is not specified."""
        with pytest.raises(ValueError, match="Did not find `model`!"):
            PineconeRerank(pinecone_api_key=SecretStr("fake-key"))

    def test_initialization_invalid_client_type(self) -> None:
        """Test initialization fails with invalid client type."""
        with pytest.raises(
            ValueError, match="The 'client' parameter must be an instance of"
        ):
            PineconeRerank(client="not a pinecone client", model="test-model")

    def test_client_creation_with_api_key(
        self, mock_pinecone_client: MagicMock
    ) -> None:
        """Test client is created with API key when not provided."""
        with patch.dict(os.environ, {"PINECONE_API_KEY": "fake-api-key"}):
            with patch(
                "langchain_pinecone.rerank.Pinecone", return_value=mock_pinecone_client
            ) as mock_pinecone_constructor:
                # Initialize with no client
                reranker = PineconeRerank(model="test-model")
                # Verify client was created
                mock_pinecone_constructor.assert_called_with(api_key="fake-api-key")
                assert reranker.client == mock_pinecone_client

    def test_client_preserved_when_provided(
        self, mock_pinecone_client: MagicMock
    ) -> None:
        """Test client is preserved when explicitly provided."""
        reranker = PineconeRerank(client=mock_pinecone_client, model="test-model")
        assert reranker.client == mock_pinecone_client

    def test_model_required(self) -> None:
        """Test model is required for initialization."""
        with pytest.raises(ValueError, match="Did not find `model`!"):
            PineconeRerank(pinecone_api_key=SecretStr("fake-key"))

    @pytest.mark.parametrize(
        "document_input, expected_output",
        [
            ("just a string", {"id": "doc0", "text": "just a string"}),
            (
                Document(page_content="doc content", metadata={"source": "test"}),
                {"id": "doc0", "text": "doc content"},
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
    def test__document_to_dict(
        self, document_input: Any, expected_output: Dict[str, Any]
    ) -> None:
        """Test _document_to_dict handles different input types."""
        reranker = PineconeRerank(
            model="test-model", pinecone_api_key=SecretStr("fake-key")
        )
        result = reranker._document_to_dict(document_input, 0)
        assert result == expected_output

    def test_rerank_empty_documents(self, mock_pinecone_client: MagicMock) -> None:
        """Test rerank returns empty list for empty documents."""
        reranker = PineconeRerank(client=mock_pinecone_client, model="test-model")
        results = reranker.rerank([], "query")
        assert results == []
        mock_pinecone_client.inference.rerank.assert_not_called()

    def test_rerank_calls_api_and_formats_results(
        self, mock_pinecone_client: MagicMock, mock_rerank_response: MagicMock
    ) -> None:
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
                {"id": "doc0", "text": "doc1 content"},
                {"id": "doc1", "text": "doc2 content"},
                {"id": "doc2", "text": "doc3 content"},
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
        assert results[0]["document"] == {"id": "doc0", "text": "Document 1 content"}

        assert results[1]["id"] == "doc1"
        assert results[1]["score"] == 0.7
        assert results[1]["index"] == 1
        assert results[1]["document"] == {"id": "doc1", "text": "Document 2 content"}

    def test_compress_documents(
        self, mock_pinecone_client: MagicMock, mock_rerank_response: MagicMock
    ) -> None:
        """Test compress_documents calls rerank and formats output as Documents."""
        # Setup reranker
        reranker = PineconeRerank(
            client=mock_pinecone_client, model="test-model", return_documents=True
        )

        # Prepare documents and query
        documents = [
            Document(page_content="Document 1 content", metadata={"source": "a"}),
            Document(page_content="Document 2 content", metadata={"source": "b"}),
            Document(page_content="Document 3 content", metadata={"source": "c"}),
        ]
        query = "test query"

        # Patch the class's rerank method
        with patch("langchain_pinecone.rerank.PineconeRerank.rerank") as mock_rerank:
            # Configure mock to return formatted results
            mock_rerank.return_value = [
                {
                    "id": "doc0",
                    "index": 0,
                    "score": 0.9,
                    "document": {"id": "doc0", "text": "Document 1 content"},
                },
                {
                    "id": "doc1",
                    "index": 1,
                    "score": 0.7,
                    "document": {"id": "doc1", "text": "Document 2 content"},
                },
            ]

            # Call the method under test
            compressed_docs = reranker.compress_documents(documents, query)

            # Verify rerank was called
            mock_rerank.assert_called_once_with(documents, query)

            # Verify results
            assert len(compressed_docs) == 2
            assert isinstance(compressed_docs[0], Document)
            assert compressed_docs[0].page_content == "Document 1 content"
            assert compressed_docs[0].metadata["source"] == "a"
            assert compressed_docs[0].metadata["relevance_score"] == 0.9

            assert isinstance(compressed_docs[1], Document)
            assert compressed_docs[1].page_content == "Document 2 content"
            assert compressed_docs[1].metadata["source"] == "b"
            assert compressed_docs[1].metadata["relevance_score"] == 0.7

    def test_compress_documents_no_return_documents(
        self, mock_pinecone_client: MagicMock
    ) -> None:
        """Test compress_documents when return_documents is False."""
        # Setup reranker
        reranker = PineconeRerank(
            client=mock_pinecone_client, model="test-model", return_documents=False
        )

        # Prepare documents and query
        documents = [
            Document(page_content="Document 1 content", metadata={"source": "a"}),
            Document(page_content="Document 2 content", metadata={"source": "b"}),
        ]
        query = "test query"

        # Patch the class's rerank method
        with patch("langchain_pinecone.rerank.PineconeRerank.rerank") as mock_rerank:
            # Configure mock to return results without documents
            mock_rerank.return_value = [
                {"id": "doc0", "index": 0, "score": 0.9},
                {"id": "doc1", "index": 1, "score": 0.7},
            ]

            # Call the method under test
            compressed_docs = reranker.compress_documents(documents, query)

            # Verify rerank was called
            mock_rerank.assert_called_once_with(documents, query)

            # Verify results
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
        self, mock_pinecone_client: MagicMock
    ) -> None:
        """Test compress_documents handles results where index is None."""
        # Setup reranker
        reranker = PineconeRerank(
            client=mock_pinecone_client, model="test-model", return_documents=True
        )

        # Prepare documents and query
        documents = [
            Document(page_content="Document 1 content", metadata={"source": "a"}),
        ]
        query = "test query"

        # Patch the class's rerank method
        with patch("langchain_pinecone.rerank.PineconeRerank.rerank") as mock_rerank:
            # Configure mock to return a result with index=None
            mock_rerank.return_value = [
                {
                    "id": "unknown-doc",
                    "index": None,
                    "score": 0.5,
                    "document": {"id": "unknown-doc", "text": "Unknown content"},
                }
            ]

            # Call the method under test
            compressed_docs = reranker.compress_documents(documents, query)

            # Verify rerank was called
            mock_rerank.assert_called_once_with(documents, query)

            # Verify no documents were returned since index is None
            assert len(compressed_docs) == 0

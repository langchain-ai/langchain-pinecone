import os
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.documents import Document
from pinecone import Pinecone, PineconeAsyncio
from pydantic import SecretStr

from langchain_pinecone.rerank import PineconeRerank


# helper function for testing shared assertions in later rerank tests
def check_rerank_call_and_results(
    mock_client: MagicMock,
    mock_rerank_response: MagicMock,
    expected_model: str,
    expected_parameters: Dict[str, Any],
    results: list[dict[str, Any]],
) -> None:
    mock_client.inference.rerank.assert_called_once_with(
        model=expected_model,
        query="test query",
        documents=[
            {"id": "doc_0", "text": "doc_1 content"},
            {"id": "doc_1", "text": "doc_2 content"},
            {"id": "doc_2", "text": "doc_3 content"},
        ],
        rank_fields=["text"],
        top_n=2,
        return_documents=True,
        parameters=expected_parameters,
    )

    assert len(results) == 2
    assert results[0]["id"] == "doc_0"
    assert results[0]["score"] == 0.9
    assert results[0]["index"] == 0
    assert results[0]["document"] == {"id": "doc_0", "text": "Document 1 content"}
    assert results[1]["id"] == "doc_1"
    assert results[1]["score"] == 0.7
    assert results[1]["index"] == 1
    assert results[1]["document"] == {"id": "doc_1", "text": "Document 2 content"}


class TestPineconeRerank:
    @pytest.fixture
    def mock_pinecone_client(self) -> MagicMock:
        """Fixture to provide a mocked Pinecone client."""
        mock_client = MagicMock(spec=Pinecone)
        mock_client.inference = MagicMock()
        return mock_client

    @pytest.fixture
    def mock_pinecone_async_client(self) -> MagicMock:
        """Fixture to provide a mocked Pinecone async client."""
        mock_client = MagicMock(spec=PineconeAsyncio)
        mock_client.inference = AsyncMock()
        return mock_client

    @pytest.fixture
    def mock_rerank_response(self) -> MagicMock:
        """Fixture to provide a mocked rerank API response."""
        mock_result1 = MagicMock()
        mock_result1.id = "doc_0"
        mock_result1.index = 0
        mock_result1.score = 0.9

        # Create document object with id attribute and to_dict method
        doc1 = MagicMock()
        doc1.id = "doc_0"
        doc1.to_dict.return_value = {"id": "doc_0", "text": "Document 1 content"}
        mock_result1.document = doc1

        mock_result2 = MagicMock()
        mock_result2.id = "doc_1"
        mock_result2.index = 1
        mock_result2.score = 0.7

        # Create document object with id attribute and to_dict method
        doc2 = MagicMock()
        doc2.id = "doc_1"
        doc2.to_dict.return_value = {"id": "doc_1", "text": "Document 2 content"}
        mock_result2.document = doc2

        mock_response = MagicMock()
        mock_response.data = [mock_result1, mock_result2]
        return mock_response

    @pytest.mark.asyncio
    async def test_initialization_with_api_key(
        self, mock_pinecone_client: MagicMock
    ) -> None:
        """Test initialization with API key environment variable."""
        with patch.dict(os.environ, {"PINECONE_API_KEY": "fake-api-key"}):
            with patch(
                "langchain_pinecone.rerank.Pinecone",
                return_value=mock_pinecone_client,
            ) as mock_pinecone_constructor:
                reranker = PineconeRerank(model="test-model")

                # Force client initialization by calling _get_sync_client
                client = reranker._get_sync_client()

                mock_pinecone_constructor.assert_called_once_with(
                    api_key="fake-api-key"
                )
                assert client == mock_pinecone_client
                assert reranker.model == "test-model"
                assert reranker.top_n == 3  # Default value

    @pytest.mark.asyncio
    async def test_initialization_with_client(
        self, mock_pinecone_client: MagicMock
    ) -> None:
        """Test initialization with a provided Pinecone client instance."""
        reranker = PineconeRerank(client=mock_pinecone_client, model="test-model")
        assert reranker.client == mock_pinecone_client
        assert reranker.model == "test-model"

    @pytest.mark.asyncio
    async def test_initialization_missing_model(self) -> None:
        """Test default model is used when model is not specified."""
        # Instead of raising an error, now we check for default model value
        with patch.dict(os.environ, {"PINECONE_API_KEY": "fake-api-key"}):
            reranker = PineconeRerank(pinecone_api_key=SecretStr("fake-key"))
            assert reranker.model == "bge-reranker-v2-m3"  # Default model

    def test_initialization_invalid_client_type(self) -> None:
        """Test initialization fails with invalid client type."""
        # Mock an invalid object that's not a Pinecone instance
        invalid_client = MagicMock()

        # Use the _get_sync_client method which checks the type
        reranker = PineconeRerank(model="test-model")
        reranker.client = invalid_client  # Directly set an invalid client

        # Now when we try to use _get_sync_client, it should verify the client type
        with pytest.raises(
            TypeError, match="The 'client' parameter must be an instance of Pinecone"
        ):
            reranker._get_sync_client()

    @pytest.mark.asyncio
    async def test_client_creation_with_api_key(
        self, mock_pinecone_client: MagicMock
    ) -> None:
        """Test client is created with API key when not provided."""
        with patch.dict(os.environ, {"PINECONE_API_KEY": "fake-api-key"}):
            with patch(
                "langchain_pinecone.rerank.Pinecone", return_value=mock_pinecone_client
            ) as mock_pinecone_constructor:
                # Initialize with no client
                reranker = PineconeRerank(model="test-model")

                # Force client initialization by calling _get_sync_client
                client = reranker._get_sync_client()

                # Verify client was created
                mock_pinecone_constructor.assert_called_with(api_key="fake-api-key")
                assert client == mock_pinecone_client

    @pytest.mark.asyncio
    async def test_client_preserved_when_provided(
        self, mock_pinecone_client: MagicMock
    ) -> None:
        """Test client is preserved when explicitly provided."""
        reranker = PineconeRerank(client=mock_pinecone_client, model="test-model")
        assert reranker.client == mock_pinecone_client

    @pytest.mark.asyncio
    async def test_model_required(self) -> None:
        """Test default model is used when model is not specified."""
        # Instead of raising an error, now we check for default model value
        with patch.dict(os.environ, {"PINECONE_API_KEY": "fake-api-key"}):
            reranker = PineconeRerank(pinecone_api_key=SecretStr("fake-key"))
            assert reranker.model == "bge-reranker-v2-m3"  # Default model

    @pytest.mark.parametrize(
        "document_input, expected_output",
        [
            ("just a string", {"id": "doc_0", "text": "just a string"}),
            (
                Document(page_content="doc content", metadata={"source": "test"}),
                {"id": "doc_0", "text": "doc content", "source": "test"},
            ),
            (
                {"id": "custom-id", "content": "dict content"},
                {"id": "custom-id", "content": "dict content"},
            ),
            (
                {"content": "dict content without id"},
                {"id": "doc_0", "content": "dict content without id"},
            ),
        ],
    )
    @pytest.mark.asyncio
    async def test__document_to_dict(
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

    @pytest.mark.parametrize(
        "model,expected_parameters",
        [
            ("cohere-rerank-3.5", {}),  # Test 'cohere' disables 'truncate'
            ("test-model", {"truncate": "END"}),  # Test default includes 'truncate'
        ],
    )
    def test_rerank_models(
        self,
        mock_pinecone_client: MagicMock,
        mock_rerank_response: MagicMock,
        model: str,
        expected_parameters: Dict[str, Any],
    ) -> None:
        mock_pinecone_client.inference.rerank.return_value = mock_rerank_response
        reranker = PineconeRerank(
            client=mock_pinecone_client,
            model=model,
            top_n=2,
            rank_fields=["text"],
            return_documents=True,
        )
        documents = ["doc_1 content", "doc_2 content", "doc_3 content"]
        query = "test query"

        results = reranker.rerank(documents, query)
        check_rerank_call_and_results(
            mock_pinecone_client,
            mock_rerank_response,
            model,
            expected_parameters,
            results,
        )

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
                    "id": "doc_0",
                    "index": 0,
                    "score": 0.9,
                    "document": {"id": "doc_0", "text": "Document 1 content"},
                },
                {
                    "id": "doc_1",
                    "index": 1,
                    "score": 0.7,
                    "document": {"id": "doc_1", "text": "Document 2 content"},
                },
            ]

            # Call the method under test
            compressed_docs = reranker.compress_documents(documents, query)

            # Verify rerank was called
            mock_rerank.assert_called_once_with(documents=documents, query=query)

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
                {"id": "doc_0", "index": 0, "score": 0.9},
                {"id": "doc_1", "index": 1, "score": 0.7},
            ]

            # Call the method under test
            compressed_docs = reranker.compress_documents(documents, query)

            # Verify rerank was called
            mock_rerank.assert_called_once_with(documents=documents, query=query)

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
            mock_rerank.assert_called_once_with(documents=documents, query=query)

            # Verify no documents were returned since index is None
            assert len(compressed_docs) == 0

    def test_rerank_with_dict_documents(
        self, mock_pinecone_client: MagicMock, mock_rerank_response: MagicMock
    ) -> None:
        """Test rerank handles dict documents and returns correct IDs and scores."""
        docs_dict = [
            {
                "id": "doc_1",
                "text": "Article about renewable energy.",
                "title": "Renewable Energy",
            },
            {
                "id": "doc_2",
                "text": "Report on economic growth.",
                "title": "Economic Growth",
            },
            {
                "id": "doc_3",
                "text": "News on climate policy changes.",
                "title": "Climate Policy",
            },
        ]
        mock_pinecone_client.inference.rerank.return_value = mock_rerank_response
        reranker = PineconeRerank(
            client=mock_pinecone_client,
            model="test-model",
            rank_fields=["text"],
            return_documents=True,
        )
        results = reranker.rerank(docs_dict, "Latest news on climate change.")
        mock_pinecone_client.inference.rerank.assert_called_once_with(
            model="test-model",
            query="Latest news on climate change.",
            documents=docs_dict,
            rank_fields=["text"],
            top_n=3,
            return_documents=True,
            parameters={"truncate": "END"},
        )
        assert results[0]["id"] == mock_rerank_response.data[0].id
        assert results[1]["id"] == mock_rerank_response.data[1].id
        for res in results:
            assert isinstance(res["score"], float)
        assert all(res["id"] is not None for res in results)

    @pytest.mark.asyncio
    async def test_arerank_empty_documents(
        self, mock_pinecone_async_client: MagicMock
    ) -> None:
        """Test arerank returns empty list for empty documents."""
        reranker = PineconeRerank(
            async_client=mock_pinecone_async_client, model="test-model"
        )
        results = await reranker.arerank([], "query")
        assert results == []
        mock_pinecone_async_client.inference.rerank.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "model,expected_parameters",
        [
            ("cohere-rerank-3.5", {}),  # Test 'cohere' disables 'truncate'
            ("test-model", {"truncate": "END"}),  # Test default includes 'truncate'
        ],
    )
    async def test_arerank_models(
        self,
        mock_pinecone_async_client: MagicMock,
        mock_rerank_response: MagicMock,
        model: str,
        expected_parameters: Dict[str, Any],
    ) -> None:
        mock_pinecone_async_client.inference.rerank.return_value = mock_rerank_response
        reranker = PineconeRerank(
            async_client=mock_pinecone_async_client,
            model=model,
            top_n=2,
            rank_fields=["text"],
            return_documents=True,
        )
        documents = ["doc_1 content", "doc_2 content", "doc_3 content"]
        query = "test query"

        results = await reranker.arerank(documents, query)
        check_rerank_call_and_results(
            mock_pinecone_async_client,
            mock_rerank_response,
            model,
            expected_parameters,
            results,
        )

    async def test_acompress_documents(
        self, mock_pinecone_async_client: MagicMock, mock_rerank_response: MagicMock
    ) -> None:
        """Test acompress_documents calls arerank and formats output as Documents."""
        # Setup reranker
        reranker = PineconeRerank(
            async_client=mock_pinecone_async_client,
            model="test-model",
            return_documents=True,
        )

        # Prepare documents and query
        documents = [
            Document(page_content="Document 1 content", metadata={"source": "a"}),
            Document(page_content="Document 2 content", metadata={"source": "b"}),
            Document(page_content="Document 3 content", metadata={"source": "c"}),
        ]
        query = "test query"

        # Patch the class's arerank method
        with patch(
            "langchain_pinecone.rerank.PineconeRerank.arerank", new_callable=AsyncMock
        ) as mock_arerank:
            # Configure mock_arerank to return a value similar to mock_rerank_response
            # but ensure 'index' corresponds to original document indices
            mock_arerank.return_value = [
                {
                    "id": "doc_0",
                    "index": 0,  # Corresponds to documents[0]
                    "score": 0.9,
                    "document": {"id": "doc_0", "text": "Document 1 content"},
                },
                {
                    "id": "doc_1",
                    "index": 1,  # Corresponds to documents[1]
                    "score": 0.7,
                    "document": {"id": "doc_1", "text": "Document 2 content"},
                },
            ]

            # Call the method under test
            compressed_docs = await reranker.acompress_documents(documents, query)

            # Verify arerank was called
            mock_arerank.assert_called_once_with(documents=documents, query=query)

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

    async def test_acompress_documents_no_return_documents(
        self, mock_pinecone_async_client: MagicMock
    ) -> None:
        """Test acompress_documents when return_documents is False."""
        # Setup reranker
        reranker = PineconeRerank(
            async_client=mock_pinecone_async_client,
            model="test-model",
            return_documents=False,
        )

        # Prepare documents and query
        documents = [
            Document(page_content="Document 1 content", metadata={"source": "a"}),
            Document(page_content="Document 2 content", metadata={"source": "b"}),
        ]
        query = "test query"

        # Patch the class's arerank method
        with patch(
            "langchain_pinecone.rerank.PineconeRerank.arerank", new_callable=AsyncMock
        ) as mock_arerank:
            # Configure mock_arerank to return results
            mock_arerank.return_value = [
                {
                    "id": "doc_0",
                    "index": 0,
                    "score": 0.9,
                    "document": {"id": "doc_0", "text": "Document 1 content"},
                },
                {
                    "id": "doc_1",
                    "index": 1,
                    "score": 0.7,
                    "document": {"id": "doc_1", "text": "Document 2 content"},
                },
            ]

            # Call the method under test
            compressed_docs = await reranker.acompress_documents(documents, query)

            # Verify arerank was called
            mock_arerank.assert_called_once_with(documents=documents, query=query)

            # Verify results (metadata only, no full document objects)
            assert len(compressed_docs) == 2
            assert isinstance(compressed_docs[0], Document)
            assert compressed_docs[0].page_content == "Document 1 content"
            assert compressed_docs[0].metadata["source"] == "a"
            assert compressed_docs[0].metadata["relevance_score"] == 0.9

            assert isinstance(compressed_docs[1], Document)
            assert compressed_docs[1].page_content == "Document 2 content"
            assert compressed_docs[1].metadata["source"] == "b"
            assert compressed_docs[1].metadata["relevance_score"] == 0.7

    async def test_acompress_documents_index_none(
        self, mock_pinecone_async_client: MagicMock
    ) -> None:
        """Test acompress_documents handles results where index is None."""
        # Setup reranker
        reranker = PineconeRerank(
            async_client=mock_pinecone_async_client,
            model="test-model",
            return_documents=True,
        )

        # Prepare documents and query
        documents = [
            Document(page_content="Document 1 content", metadata={"source": "a"}),
        ]
        query = "test query"

        # Patch the class's arerank method
        with patch(
            "langchain_pinecone.rerank.PineconeRerank.arerank", new_callable=AsyncMock
        ) as mock_arerank:
            # Configure mock to return a result with index=None
            mock_arerank.return_value = [
                {
                    "id": "unknown-doc",
                    "index": None,
                    "score": 0.5,
                    "document": {"id": "unknown-doc", "text": "Unknown content"},
                }
            ]

            # Call the method under test
            compressed_docs = await reranker.acompress_documents(documents, query)

            # Verify arerank was called
            mock_arerank.assert_called_once_with(documents=documents, query=query)

            # Verify no documents were returned since index is None
            assert len(compressed_docs) == 0

    @pytest.mark.asyncio
    async def test_arerank_with_dict_documents(
        self, mock_pinecone_async_client: MagicMock, mock_rerank_response: MagicMock
    ) -> None:
        """Test arerank handles dict documents and returns correct IDs and scores."""
        docs_dict = [
            {
                "id": "doc_1",
                "text": "Article about renewable energy.",
                "title": "Renewable Energy",
            },
            {
                "id": "doc_2",
                "text": "Report on economic growth.",
                "title": "Economic Growth",
            },
            {
                "id": "doc_3",
                "text": "News on climate policy changes.",
                "title": "Climate Policy",
            },
        ]
        mock_pinecone_async_client.inference.rerank.return_value = mock_rerank_response
        reranker = PineconeRerank(
            async_client=mock_pinecone_async_client,
            model="test-model",
            rank_fields=["text"],
            return_documents=True,
        )
        results = await reranker.arerank(docs_dict, "Latest news on climate change.")
        mock_pinecone_async_client.inference.rerank.assert_called_once_with(
            model="test-model",
            query="Latest news on climate change.",
            documents=docs_dict,
            rank_fields=["text"],
            top_n=3,
            return_documents=True,
            parameters={"truncate": "END"},
        )
        assert results[0]["id"] == mock_rerank_response.data[0].id
        assert results[1]["id"] == mock_rerank_response.data[1].id
        for res in results:
            assert isinstance(res["score"], float)
        assert all(res["id"] is not None for res in results)

    @pytest.mark.asyncio
    async def test_async_client_initialization(
        self, mock_pinecone_async_client: MagicMock
    ) -> None:
        """Test async client initialization works correctly."""
        with patch.dict(os.environ, {"PINECONE_API_KEY": "fake-api-key"}):
            with patch(
                "langchain_pinecone.rerank.PineconeAsyncio",
                return_value=mock_pinecone_async_client,
            ) as mock_pinecone_async_constructor:
                reranker = PineconeRerank(model="test-model")

                # Force client initialization by calling _get_async_client
                client = await reranker._get_async_client()

                mock_pinecone_async_constructor.assert_called_once_with(
                    api_key="fake-api-key"
                )
                assert client == mock_pinecone_async_client

    @pytest.mark.asyncio
    async def test_async_client_invalid_type(self) -> None:
        """Test initialization fails with invalid async client type."""
        # Mock an invalid object that's not a PineconeAsyncio instance
        invalid_client = MagicMock()

        # Use the _get_async_client method which checks the type
        reranker = PineconeRerank(model="test-model")
        reranker.async_client = invalid_client  # Directly set an invalid client

        # Now when we try to use _get_async_client, it should verify the client type
        with pytest.raises(
            TypeError,
            match="The 'async_client' parameter must be an instance of PineconeAsyncio",
        ):
            await reranker._get_async_client()

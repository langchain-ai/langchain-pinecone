import os

import pytest
from langchain_core.documents import Document

from langchain_pinecone import PineconeRerank

# Ensure environment variables are set for integration tests
pytestmark = pytest.mark.skipif(
    not os.environ.get("PINECONE_API_KEY"), reason="Pinecone API key not set"
)


def test_pinecone_rerank_basic() -> None:
    """Test basic reranking functionality."""
    reranker = PineconeRerank(model="bge-reranker-v2-m3")
    query = "What is the capital of France?"
    documents = [
        Document(page_content="Paris is the capital of France."),
        Document(page_content="Berlin is the capital of Germany."),
        Document(page_content="The Eiffel Tower is in Paris."),
    ]

    compressed_docs = reranker.compress_documents(documents=documents, query=query)

    assert len(compressed_docs) > 0
    assert isinstance(compressed_docs[0], Document)
    # Basic check for score presence, actual score value depends on model
    assert "relevance_score" in compressed_docs[0].metadata
    # Check if the most relevant document is ranked higher (basic assumption)
    assert "Paris is the capital of France." in compressed_docs[0].page_content


def test_pinecone_rerank_top_n() -> None:
    """Test reranking with a specific top_n value."""
    reranker = PineconeRerank(model="bge-reranker-v2-m3", top_n=1)
    query = "What is the capital of France?"
    documents = [
        Document(page_content="Paris is the capital of France."),
        Document(page_content="Berlin is the capital of Germany."),
        Document(page_content="The Eiffel Tower is in Paris."),
    ]

    compressed_docs = reranker.compress_documents(documents=documents, query=query)

    assert len(compressed_docs) == 1


def test_pinecone_rerank_rank_fields() -> None:
    """Test reranking using specific rank_fields."""
    # Test ranking by the 'text' field explicitly (was 'content')
    reranker = PineconeRerank(model="bge-reranker-v2-m3", rank_fields=["text"])
    query = "Latest news on climate change."
    documents = [
        Document(
            page_content="Article about renewable energy.", metadata={"id": "doc1"}
        ),
        Document(page_content="Report on economic growth.", metadata={"id": "doc2"}),
        Document(
            page_content="News on climate policy changes.", metadata={"id": "doc3"}
        ),
    ]

    compressed_docs = reranker.compress_documents(documents=documents, query=query)

    # Ensure we got results back
    assert len(compressed_docs) > 0
    assert isinstance(compressed_docs[0], Document)
    assert "relevance_score" in compressed_docs[0].metadata

    # Verify that the most relevant document contains climate-related content
    # (the exact ordering might vary with model updates, so check more broadly)
    climate_related = False
    for doc in compressed_docs:
        if "climate policy" in doc.page_content:
            climate_related = True
            break
    assert climate_related, "Expected to find climate-related content in top results"


def test_pinecone_rerank_with_parameters() -> None:
    """Test reranking with additional model parameters."""
    # Note: The specific parameters depend on the model. 'truncate' is common.
    reranker = PineconeRerank(model="bge-reranker-v2-m3")
    query = "Explain the concept of quantum entanglement."
    documents = [
        Document(page_content="Quantum entanglement is a physical phenomenon..."),
        Document(page_content="Classical mechanics describes motion..."),
    ]

    # Get reranking results
    compressed_docs = reranker.compress_documents(documents=documents, query=query)

    assert len(compressed_docs) > 0
    assert isinstance(compressed_docs[0], Document)
    assert "relevance_score" in compressed_docs[0].metadata

    # Check that quantum entanglement document is found
    quantum_found = False
    for doc in compressed_docs:
        if "Quantum entanglement" in doc.page_content:
            quantum_found = True
            break
    assert quantum_found, "Expected to find quantum entanglement document in results"


def test_pinecone_rerank_rank_fields_dict() -> None:
    """Integration test rerank with dict docs and rank_fields"""
    # Skip if no API key
    reranker = PineconeRerank(model="bge-reranker-v2-m3", rank_fields=["text"])
    docs_dict = [
        {
            "id": "doc1",
            "text": "Article about renewable energy.",
            "title": "Renewable Energy",
        },
        {
            "id": "doc2",
            "text": "Report on economic growth.",
            "title": "Economic Growth",
        },
        {
            "id": "doc3",
            "text": "News on climate policy changes.",
            "title": "Climate Policy",
        },
    ]
    results = reranker.rerank(docs_dict, "Latest news on climate change.")

    assert len(results) == 3, "Expected 3 results for top_n=3 (default)"

    # Verify properties of the first (most relevant) result
    assert results[0]["id"] == "doc3", "ID of the top result should be 'doc3'"
    assert results[0]["document"]["text"] == "News on climate policy changes."
    assert results[0]["document"]["title"] == "Climate Policy"

    # Verify properties for all results
    for res in results:
        assert res["id"] is not None, (
            f"ID should not be None, but was for result: {res}"
        )
        assert isinstance(res["id"], str), (
            f"ID should be a string, got {type(res['id'])}"
        )
        assert isinstance(res["score"], float), (
            f"Score should be a float, got {type(res['score'])}"
        )
        assert res["score"] >= 0.0 and res["score"] <= 1.0, (
            f"Score should be between 0 and 1, got {res['score']}"
        )
        assert "document" in res, "'document' field should be in the result"
        assert isinstance(res["document"], dict), (
            f"'document' field should be a dictionary, currently: {type(res['document'])}"
        )
        assert "id" in res["document"], "'id' should be in res['document']"
        assert res["document"]["id"] == res["id"], (
            "ID in result root should match ID in result document"
        )
        assert "text" in res["document"], "'text' should be in res['document']"

    # Verify the order of IDs (important for relevance ranking)
    assert [res["id"] for res in results] == ["doc3", "doc1", "doc2"], (
        "Documents not in expected order of relevance"
    )


# Async Tests


@pytest.mark.asyncio
async def test_async_pinecone_rerank_basic() -> None:
    """Test basic asynchronous reranking functionality."""
    reranker = PineconeRerank(model="bge-reranker-v2-m3")
    query = "What is the capital of France?"
    documents = [
        Document(page_content="Paris is the capital of France."),
        Document(page_content="Berlin is the capital of Germany."),
        Document(page_content="The Eiffel Tower is in Paris."),
    ]

    compressed_docs = await reranker.acompress_documents(
        documents=documents, query=query
    )

    assert len(compressed_docs) > 0
    assert isinstance(compressed_docs[0], Document)
    assert "relevance_score" in compressed_docs[0].metadata
    assert "Paris is the capital of France." in compressed_docs[0].page_content


@pytest.mark.asyncio
async def test_async_pinecone_rerank_top_n() -> None:
    """Test asynchronous reranking with a specific top_n value."""
    reranker = PineconeRerank(model="bge-reranker-v2-m3", top_n=1)
    query = "What is the capital of France?"
    documents = [
        Document(page_content="Paris is the capital of France."),
        Document(page_content="Berlin is the capital of Germany."),
        Document(page_content="The Eiffel Tower is in Paris."),
    ]

    compressed_docs = await reranker.acompress_documents(
        documents=documents, query=query
    )

    assert len(compressed_docs) == 1
    assert "Paris is the capital of France." in compressed_docs[0].page_content


@pytest.mark.asyncio
async def test_async_pinecone_rerank_rank_fields() -> None:
    """Test asynchronous reranking using specific rank_fields."""
    reranker = PineconeRerank(model="bge-reranker-v2-m3", rank_fields=["text"])
    query = "Latest news on climate change."
    documents = [
        Document(
            page_content="Article about renewable energy.", metadata={"id": "doc1"}
        ),
        Document(page_content="Report on economic growth.", metadata={"id": "doc2"}),
        Document(
            page_content="News on climate policy changes.", metadata={"id": "doc3"}
        ),
    ]

    compressed_docs = await reranker.acompress_documents(
        documents=documents, query=query
    )

    assert len(compressed_docs) > 0
    assert isinstance(compressed_docs[0], Document)
    assert "relevance_score" in compressed_docs[0].metadata
    climate_related = False
    for doc in compressed_docs:
        if "climate policy" in doc.page_content:
            climate_related = True
            break
    assert climate_related, "Expected to find climate-related content in top results"


@pytest.mark.asyncio
async def test_async_pinecone_rerank_with_parameters() -> None:
    """Test asynchronous reranking with additional model parameters."""
    reranker = PineconeRerank(model="bge-reranker-v2-m3")
    query = "Explain the concept of quantum entanglement."
    documents = [
        Document(page_content="Quantum entanglement is a physical phenomenon..."),
        Document(page_content="Classical mechanics describes motion..."),
    ]

    compressed_docs = await reranker.acompress_documents(
        documents=documents, query=query
    )

    assert len(compressed_docs) > 0
    assert isinstance(compressed_docs[0], Document)
    assert "relevance_score" in compressed_docs[0].metadata
    quantum_found = False
    for doc in compressed_docs:
        if "Quantum entanglement" in doc.page_content:
            quantum_found = True
            break
    assert quantum_found, "Expected to find quantum entanglement document in results"


@pytest.mark.asyncio
async def test_async_pinecone_rerank_rank_fields_dict() -> None:
    """Async integration test rerank with dict docs and rank_fields"""
    reranker = PineconeRerank(model="bge-reranker-v2-m3", rank_fields=["text"])
    docs_dict = [
        {
            "id": "doc1",
            "text": "Article about renewable energy.",
            "title": "Renewable Energy",
        },
        {
            "id": "doc2",
            "text": "Report on economic growth.",
            "title": "Economic Growth",
        },
        {
            "id": "doc3",
            "text": "News on climate policy changes.",
            "title": "Climate Policy",
        },
    ]
    results = await reranker.arerank(docs_dict, "Latest news on climate change.")

    assert len(results) == 3, "Expected 3 results for top_n=3 (default)"

    assert results[0]["id"] == "doc3", "ID of the top result should be 'doc3'"
    assert results[0]["document"]["text"] == "News on climate policy changes."
    assert results[0]["document"]["title"] == "Climate Policy"

    for res in results:
        assert res["id"] is not None, (
            f"ID should not be None, but was for result: {res}"
        )
        assert isinstance(res["id"], str), (
            f"ID should be a string, got {type(res['id'])}"
        )
        assert isinstance(res["score"], float), (
            f"Score should be a float, got {type(res['score'])}"
        )
        assert res["score"] >= 0.0 and res["score"] <= 1.0, (
            f"Score should be between 0 and 1, got {res['score']}"
        )
        assert "document" in res, "'document' field should be in the result"
        assert isinstance(res["document"], dict), (
            f"'document' field should be a dictionary, currently: {type(res['document'])}"
        )
        assert "id" in res["document"], "'id' should be in res['document']"
        assert res["document"]["id"] == res["id"], (
            "ID in result root should match ID in result document"
        )
        assert "text" in res["document"], "'text' should be in res['document']"

    assert [res["id"] for res in results] == ["doc3", "doc1", "doc2"], (
        "Documents not in expected order of relevance"
    )

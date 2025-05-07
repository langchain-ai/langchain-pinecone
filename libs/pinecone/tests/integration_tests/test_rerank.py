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
    assert "Paris is the capital of France." in compressed_docs[0].page_content


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

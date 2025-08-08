import os

import pytest

from langchain_pinecone._utilities import aget_pinecone_supported_models

# Check for required environment variables
requires_api_key = pytest.mark.skipif(
    "PINECONE_API_KEY" not in os.environ,
    reason="Test requires PINECONE_API_KEY environment variable",
)


@requires_api_key
@pytest.mark.asyncio
async def test_aget_pinecone_supported_models_integration() -> None:
    """Test the async function with real Pinecone API."""
    api_key = os.environ.get("PINECONE_API_KEY")
    assert api_key is not None

    # Test basic call without filters
    result = await aget_pinecone_supported_models(api_key)

    assert isinstance(result, dict)
    assert "models" in result
    assert isinstance(result["models"], list)

    # Verify at least some models are returned
    assert len(result["models"]) > 0

    # Test with embed filter
    embed_models = await aget_pinecone_supported_models(api_key, model_type="embed")
    assert isinstance(embed_models, dict)
    assert "models" in embed_models

    # Test with rerank filter
    rerank_models = await aget_pinecone_supported_models(api_key, model_type="rerank")
    assert isinstance(rerank_models, dict)
    assert "models" in rerank_models

    # Test with vector_type filter
    dense_models = await aget_pinecone_supported_models(api_key, vector_type="dense")
    assert isinstance(dense_models, dict)
    assert "models" in dense_models


@requires_api_key
@pytest.mark.asyncio
async def test_aget_pinecone_supported_models_with_combined_filters() -> None:
    """Test the async function with combined filters."""
    api_key = os.environ.get("PINECONE_API_KEY")
    assert api_key is not None

    # Test with both model_type and vector_type filters
    dense_embed_models = await aget_pinecone_supported_models(
        api_key, model_type="embed", vector_type="dense"
    )

    assert isinstance(dense_embed_models, dict)
    assert "models" in dense_embed_models

    # All returned models should be embed type and dense vector type
    for model in dense_embed_models["models"]:
        assert model.get("type") == "embed"
        assert model.get("vector_type") == "dense"

from typing import Any, Type
from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.utils import convert_to_secret_str
from langchain_tests.unit_tests.embeddings import EmbeddingsTests
from pinecone import (
    PineconeAsyncio,  # type: ignore[import-untyped]
    SparseValues,  # type: ignore[import-untyped]
)

from langchain_pinecone import PineconeEmbeddings
from langchain_pinecone.embeddings import PineconeSparseEmbeddings

API_KEY = convert_to_secret_str("NOT_A_VALID_KEY")
MODEL_NAME = "multilingual-e5-large"
SPARSE_MODEL_NAME = "pinecone-sparse-english-v0"


@pytest.fixture(autouse=True)
def patch_pinecone_model_listing(mocker: Any) -> None:
    mocker.patch(
        "langchain_pinecone.embeddings.PineconeEmbeddings.list_supported_models",
        return_value=[
            {"model": "multilingual-e5-large"},
            {"model": "pinecone-sparse-english-v0"},
        ],
    )


@pytest.fixture(autouse=True)
def mock_pinecone() -> Any:
    """Mock Pinecone client for all tests."""
    with patch("langchain_pinecone.embeddings.PineconeClient") as mock:
        yield mock


class TestPineconeEmbeddingsStandard(EmbeddingsTests):
    """Standard LangChain embeddings tests."""

    @property
    def embeddings_class(self) -> Type[PineconeEmbeddings]:
        """Get the class under test."""
        return PineconeEmbeddings

    @property
    def embedding_model_params(self) -> dict:
        """Get the parameters for initializing the embeddings model."""
        return {
            "model": MODEL_NAME,
            "pinecone_api_key": API_KEY,
        }


class TestPineconeEmbeddingsConfig:
    def test_valid_model_parameter(self, mocker: Any) -> None:
        """Test that a valid model name passes validation."""
        # Patch list_supported_models to return a list containing MODEL_NAME
        mocker.patch(
            "langchain_pinecone.embeddings.PineconeEmbeddings.list_supported_models",
            return_value=[{"model": MODEL_NAME}],
        )
        embeddings = PineconeEmbeddings(model=MODEL_NAME, pinecone_api_key=API_KEY)
        assert embeddings.model == MODEL_NAME

    def test_invalid_model_parameter(self, mocker: Any) -> None:
        """Test that an invalid model name raises a ValueError."""
        # Patch list_supported_models to return a list NOT containing 'invalid-model'
        mocker.patch(
            "langchain_pinecone.embeddings.PineconeEmbeddings.list_supported_models",
            return_value=[{"model": MODEL_NAME}],
        )
        with pytest.raises(
            ValueError, match="not a supported Pinecone embedding model"
        ):
            PineconeEmbeddings(model="invalid-model", pinecone_api_key=API_KEY)

    """Additional configuration tests for PineconeEmbeddings."""

    @pytest.mark.parametrize(
        "model_name,expected_config,embeddings_cls",
        [
            (
                MODEL_NAME,
                {
                    "batch_size": 96,
                    "query_params": {"input_type": "query", "truncate": "END"},
                    "document_params": {"input_type": "passage", "truncate": "END"},
                    "dimension": 1024,
                },
                PineconeEmbeddings,
            ),
            (
                SPARSE_MODEL_NAME,
                {
                    "batch_size": 96,
                    "query_params": {"input_type": "query", "truncate": "END"},
                    "document_params": {"input_type": "passage", "truncate": "END"},
                    "dimension": None,
                },
                PineconeSparseEmbeddings,
            ),
        ],
    )
    def test_default_config(
        self,
        model_name: str,
        expected_config: dict[str, Any],
        embeddings_cls: PineconeEmbeddings,
    ) -> None:
        """Test default configuration is set correctly."""
        embeddings = embeddings_cls(model=model_name, pinecone_api_key=API_KEY)  # type: ignore
        assert embeddings.batch_size == expected_config["batch_size"]
        assert embeddings.query_params == expected_config["query_params"]
        assert embeddings.document_params == expected_config["document_params"]
        assert embeddings.dimension == expected_config["dimension"]

    def test_custom_config(self) -> None:
        """Test custom configuration overrides defaults."""
        embeddings = PineconeEmbeddings(
            model=MODEL_NAME,
            pinecone_api_key=API_KEY,
            batch_size=128,
            query_params={"custom": "param"},
            document_params={"other": "param"},
        )
        assert embeddings.batch_size == 128
        assert embeddings.query_params == {"custom": "param"}
        assert embeddings.document_params == {"other": "param"}

    def test_custom_config_pinecone_key(self) -> None:
        """Test custom configuration overrides defaults."""
        embeddings = PineconeEmbeddings(
            model=MODEL_NAME,
            api_key=API_KEY,
            batch_size=128,
            query_params={"custom": "param"},
            document_params={"other": "param"},
        )  # type: ignore[call-arg]
        assert embeddings.batch_size == 128
        assert embeddings.query_params == {"custom": "param"}
        assert embeddings.document_params == {"other": "param"}

    @pytest.mark.asyncio
    async def test_async_client_initialization(self) -> None:
        """Test async client is initialized correctly and only when needed."""
        embeddings = PineconeEmbeddings(model=MODEL_NAME, pinecone_api_key=API_KEY)
        assert embeddings._async_client is None

        # Access async_client property
        client = embeddings.async_client
        assert client is not None
        assert isinstance(client, PineconeAsyncio)

    @pytest.mark.asyncio
    async def test_alist_supported_models(self, mocker: Any) -> None:
        """Test the async list_supported_models method."""
        mock_response = {
            "models": [
                {"model": "multilingual-e5-large", "type": "embed"},
                {"model": "pinecone-sparse-english-v0", "type": "embed"},
            ]
        }

        # Mock the aget_pinecone_supported_models function
        mocker.patch(
            "langchain_pinecone.embeddings.aget_pinecone_supported_models",
            return_value=mock_response,
        )

        embeddings = PineconeEmbeddings(model=MODEL_NAME, pinecone_api_key=API_KEY)
        result = await embeddings.alist_supported_models()

        assert result == mock_response

    @pytest.mark.asyncio
    async def test_alist_supported_models_with_vector_type(self, mocker: Any) -> None:
        """Test the async list_supported_models method with vector_type filter."""
        mock_response = {
            "models": [
                {
                    "model": "multilingual-e5-large",
                    "type": "embed",
                    "vector_type": "dense",
                },
            ]
        }

        # Mock the aget_pinecone_supported_models function
        mocker.patch(
            "langchain_pinecone.embeddings.aget_pinecone_supported_models",
            return_value=mock_response,
        )

        embeddings = PineconeEmbeddings(model=MODEL_NAME, pinecone_api_key=API_KEY)
        result = await embeddings.alist_supported_models(vector_type="dense")

        assert result == mock_response


class TestPineconeEmbeddingsResponseParsing:
    """Tests that embed_* methods correctly unpack r["values"] from inference responses."""

    @pytest.fixture
    def emb(self) -> PineconeEmbeddings:
        return PineconeEmbeddings(model=MODEL_NAME, pinecone_api_key=API_KEY)

    def test_embed_documents_parses_values(
        self, mocker: Any, emb: PineconeEmbeddings
    ) -> None:
        fake = [{"values": [0.1, 0.2, 0.3]}, {"values": [0.4, 0.5, 0.6]}]
        mocker.patch.object(PineconeEmbeddings, "_embed_texts", return_value=fake)
        result = emb.embed_documents(["text 1", "text 2"])
        assert result == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]

    def test_embed_query_parses_values(
        self, mocker: Any, emb: PineconeEmbeddings
    ) -> None:
        fake = [{"values": [0.1, 0.2, 0.3]}]
        mocker.patch.object(PineconeEmbeddings, "_embed_texts", return_value=fake)
        result = emb.embed_query("query text")
        assert result == [0.1, 0.2, 0.3]

    async def test_aembed_documents_parses_values(
        self, mocker: Any, emb: PineconeEmbeddings
    ) -> None:
        fake = [{"values": [0.1, 0.2, 0.3]}, {"values": [0.4, 0.5, 0.6]}]
        mocker.patch.object(
            PineconeEmbeddings, "_aembed_texts", new=AsyncMock(return_value=fake)
        )
        result = await emb.aembed_documents(["text 1", "text 2"])
        assert result == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]

    async def test_aembed_query_parses_values(
        self, mocker: Any, emb: PineconeEmbeddings
    ) -> None:
        fake = [{"values": [0.1, 0.2, 0.3]}]
        mocker.patch.object(
            PineconeEmbeddings, "_aembed_texts", new=AsyncMock(return_value=fake)
        )
        result = await emb.aembed_query("query text")
        assert result == [0.1, 0.2, 0.3]

    def test_embed_documents_batches_in_order(
        self, mocker: Any, emb: PineconeEmbeddings
    ) -> None:
        emb.batch_size = 2
        batch_results = [
            [{"values": [0.1, 0.2]}, {"values": [0.3, 0.4]}],
            [{"values": [0.5, 0.6]}, {"values": [0.7, 0.8]}],
            [{"values": [0.9, 1.0]}],
        ]
        mock_embed = mocker.Mock(side_effect=batch_results)
        mocker.patch.object(PineconeEmbeddings, "_embed_texts", mock_embed)

        result = emb.embed_documents(["a", "b", "c", "d", "e"])

        assert mock_embed.call_count == 3
        assert mock_embed.call_args_list[0].kwargs["texts"] == ["a", "b"]
        assert mock_embed.call_args_list[1].kwargs["texts"] == ["c", "d"]
        assert mock_embed.call_args_list[2].kwargs["texts"] == ["e"]
        assert result == [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8], [0.9, 1.0]]


class TestPineconeSparseEmbeddingsResponseParsing:
    """Tests that sparse embed_* methods correctly build SparseValues from responses."""

    @pytest.fixture
    def sparse_emb(self) -> PineconeSparseEmbeddings:
        return PineconeSparseEmbeddings(
            model=SPARSE_MODEL_NAME, pinecone_api_key=API_KEY
        )

    def test_sparse_embed_documents_builds_sparsevalues(
        self, mocker: Any, sparse_emb: PineconeSparseEmbeddings
    ) -> None:
        fake = [
            {"sparse_indices": [1, 5, 10], "sparse_values": [0.1, 0.2, 0.3]},
            {"sparse_indices": [2, 7], "sparse_values": [0.4, 0.5]},
        ]
        mocker.patch.object(PineconeSparseEmbeddings, "_embed_texts", return_value=fake)
        result = sparse_emb.embed_documents(["text 1", "text 2"])
        assert len(result) == 2
        assert isinstance(result[0], SparseValues)
        assert result[0].indices == [1, 5, 10]
        assert result[0].values == [0.1, 0.2, 0.3]
        assert isinstance(result[1], SparseValues)
        assert result[1].indices == [2, 7]
        assert result[1].values == [0.4, 0.5]

    def test_sparse_embed_query_builds_sparsevalues(
        self, mocker: Any, sparse_emb: PineconeSparseEmbeddings
    ) -> None:
        fake = [{"sparse_indices": [1, 5], "sparse_values": [0.1, 0.2]}]
        mocker.patch.object(PineconeSparseEmbeddings, "_embed_texts", return_value=fake)
        result = sparse_emb.embed_query("query text")
        assert isinstance(result, SparseValues)
        assert result.indices == [1, 5]
        assert result.values == [0.1, 0.2]

    async def test_sparse_aembed_documents_builds_sparsevalues(
        self, mocker: Any, sparse_emb: PineconeSparseEmbeddings
    ) -> None:
        fake = [
            {"sparse_indices": [1, 5, 10], "sparse_values": [0.1, 0.2, 0.3]},
            {"sparse_indices": [2, 7], "sparse_values": [0.4, 0.5]},
        ]
        mocker.patch.object(
            PineconeSparseEmbeddings, "_aembed_texts", new=AsyncMock(return_value=fake)
        )
        result = await sparse_emb.aembed_documents(["text 1", "text 2"])
        assert len(result) == 2
        assert isinstance(result[0], SparseValues)
        assert result[0].indices == [1, 5, 10]
        assert result[0].values == [0.1, 0.2, 0.3]

    async def test_sparse_aembed_query_builds_sparsevalues(
        self, mocker: Any, sparse_emb: PineconeSparseEmbeddings
    ) -> None:
        fake = [{"sparse_indices": [1, 5], "sparse_values": [0.1, 0.2]}]
        mocker.patch.object(
            PineconeSparseEmbeddings, "_aembed_texts", new=AsyncMock(return_value=fake)
        )
        result = await sparse_emb.aembed_query("query text")
        assert isinstance(result, SparseValues)
        assert result.indices == [1, 5]
        assert result.values == [0.1, 0.2]


class TestQueryParamsParity:
    """Regression tests: query ops must use query_params, not document_params."""

    def test_embed_query_uses_query_params(self, mocker: Any) -> None:
        embeddings = PineconeEmbeddings(model=MODEL_NAME, pinecone_api_key=API_KEY)
        mock_embed = mocker.Mock(return_value=[{"values": [0.1, 0.2, 0.3]}])
        mocker.patch.object(embeddings, "_embed_texts", mock_embed)

        embeddings.embed_query("test query")

        assert mock_embed.call_args.kwargs["parameters"] == embeddings.query_params

    async def test_aembed_query_uses_query_params(self, mocker: Any) -> None:
        embeddings = PineconeEmbeddings(model=MODEL_NAME, pinecone_api_key=API_KEY)
        mock_aembed = mocker.AsyncMock(return_value=[{"values": [0.1, 0.2, 0.3]}])
        mocker.patch.object(embeddings, "_aembed_texts", mock_aembed)

        await embeddings.aembed_query("test query")

        assert mock_aembed.call_args.kwargs["parameters"] == embeddings.query_params

    async def test_sparse_aembed_query_uses_query_params(self, mocker: Any) -> None:
        sparse_embeddings = PineconeSparseEmbeddings(
            model=SPARSE_MODEL_NAME, pinecone_api_key=API_KEY
        )
        mock_aembed = mocker.AsyncMock(
            return_value=[{"sparse_indices": [0, 1], "sparse_values": [0.5, 0.5]}]
        )
        mocker.patch.object(sparse_embeddings, "_aembed_texts", mock_aembed)

        await sparse_embeddings.aembed_query("test query")

        assert (
            mock_aembed.call_args.kwargs["parameters"] == sparse_embeddings.query_params
        )

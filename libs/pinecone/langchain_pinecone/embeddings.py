import logging
from typing import Any, Dict, Iterable, List, Optional

from langchain_core.embeddings import Embeddings
from langchain_core.utils import secret_from_env
from pinecone import Pinecone as PineconeClient  # type: ignore[import-untyped]
from pinecone import (
    PineconeAsyncio as PineconeAsyncioClient,  # type: ignore[import-untyped]
)
from pinecone import SparseValues
from pinecone.data.features.inference.inference import (  # type: ignore[import-untyped]
    EmbeddingsList,
)
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    SecretStr,
    model_validator,
)
from typing_extensions import Self

logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 64


class PineconeEmbeddings(BaseModel, Embeddings):
    """PineconeEmbeddings embedding model.

    Example:
        .. code-block:: python

            from langchain_pinecone import PineconeEmbeddings
            from langchain_pinecone import PineconeVectorStore
            from langchain_core.documents import Document

            # Initialize embeddings with a specific model
            embeddings = PineconeEmbeddings(model="multilingual-e5-large")

            # Embed a single query
            query_embedding = embeddings.embed_query("What is machine learning?")

            # Embed multiple documents
            docs = ["Document 1 content", "Document 2 content"]
            doc_embeddings = embeddings.embed_documents(docs)

            # Use with PineconeVectorStore
            from pinecone import Pinecone

            pc = Pinecone(api_key="your-api-key")
            index = pc.Index("your-index-name")

            vectorstore = PineconeVectorStore(
                index=index,
                embedding=embeddings
            )

            # Add documents to vector store
            vectorstore.add_documents([
                Document(page_content="Hello, world!"),
                Document(page_content="This is a test.")
            ])

            # Search for similar documents
            results = vectorstore.similarity_search("hello", k=2)
    """

    # Clients
    _client: PineconeClient = PrivateAttr(default=None)
    _async_client: Optional[PineconeAsyncioClient] = PrivateAttr(default=None)
    model: str
    """Model to use for example 'multilingual-e5-large'."""
    # Config
    batch_size: Optional[int] = None
    """Batch size for embedding documents."""
    query_params: Dict = Field(default_factory=dict)
    """Parameters for embedding query."""
    document_params: Dict = Field(default_factory=dict)
    """Parameters for embedding document"""
    #
    dimension: Optional[int] = None
    #
    show_progress_bar: bool = False
    pinecone_api_key: SecretStr = Field(
        default_factory=secret_from_env(
            "PINECONE_API_KEY",
            error_message="Pinecone API key not found. Please set the PINECONE_API_KEY "
            "environment variable or pass it via `pinecone_api_key`.",
        ),
        alias="api_key",
    )
    """Pinecone API key. 
    
    If not provided, will look for the PINECONE_API_KEY environment variable."""

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        protected_namespaces=(),
    )

    @property
    def async_client(self) -> PineconeAsyncioClient:
        """Lazily initialize the async client."""
        return PineconeAsyncioClient(
            api_key=self.pinecone_api_key.get_secret_value(), source_tag="langchain"
        )

    @model_validator(mode="before")
    @classmethod
    def set_default_config(cls, values: dict) -> Any:
        """Set default configuration based on model."""
        default_config_map = {
            "multilingual-e5-large": {
                "batch_size": 96,
                "query_params": {"input_type": "query", "truncation": "END"},
                "document_params": {"input_type": "passage", "truncation": "END"},
                "dimension": 1024,
            },
        }
        model = values.get("model")
        if model in default_config_map:
            config = default_config_map[model]
            for key, value in config.items():
                if key not in values:
                    values[key] = value
        return values

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """Validate that Pinecone version and credentials exist in environment."""
        api_key_str = self.pinecone_api_key.get_secret_value()
        client = PineconeClient(api_key=api_key_str, source_tag="langchain")
        self._client = client

        # Ensure async_client is lazily initialized
        return self

    def _get_batch_iterator(self, texts: List[str]) -> tuple[Iterable, int]:
        if self.batch_size is None:
            batch_size = DEFAULT_BATCH_SIZE
        else:
            batch_size = self.batch_size

        if self.show_progress_bar:
            try:
                from tqdm.auto import tqdm  # type: ignore
            except ImportError as e:
                raise ImportError(
                    "Must have tqdm installed if `show_progress_bar` is set to True. "
                    "Please install with `pip install tqdm`."
                ) from e

            _iter = tqdm(range(0, len(texts), batch_size))
        else:
            _iter = range(0, len(texts), batch_size)

        return _iter, batch_size

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs."""
        embeddings: List[List[float]] = []

        _iter, batch_size = self._get_batch_iterator(texts)
        for i in _iter:
            response = self._embed_texts(
                model=self.model,
                parameters=self.document_params,
                texts=texts[i : i + batch_size],
            )
            embeddings.extend([r["values"] for r in response])

        return embeddings

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings: List[List[float]] = []
        _iter, batch_size = self._get_batch_iterator(texts)
        for i in _iter:
            response = await self._aembed_texts(
                model=self.model,
                parameters=self.document_params,
                texts=texts[i : i + batch_size],
            )
            embeddings.extend([r["values"] for r in response])
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        return self._embed_texts(
            model=self.model, parameters=self.query_params, texts=[text]
        )[0]["values"]

    async def aembed_query(self, text: str) -> List[float]:
        """Asynchronously embed query text."""
        embeddings = await self._aembed_texts(
            model=self.model,
            parameters=self.document_params,
            texts=[text],
        )
        return embeddings[0]["values"]

    def _embed_texts(
        self, texts: List[str], model: str, parameters: dict
    ) -> EmbeddingsList:
        return self._client.inference.embed(
            model=model, inputs=texts, parameters=parameters
        )

    async def _aembed_texts(
        self, texts: List[str], model: str, parameters: dict
    ) -> EmbeddingsList:
        async with self.async_client as aclient:
            embeddings: EmbeddingsList = await aclient.inference.embed(
                model=model, inputs=texts, parameters=parameters
            )
            return embeddings


class PineconeSparseEmbeddings(PineconeEmbeddings):
    """PineconeSparseEmbeddings embedding model.

    Example:
        .. code-block:: python

            from langchain_pinecone import PineconeSparseEmbeddings
            from langchain_pinecone import PineconeVectorStore
            from langchain_core.documents import Document

            # Initialize sparse embeddings
            sparse_embeddings = PineconeSparseEmbeddings(model="pinecone-sparse-english-v0")

            # Embed a single query (returns SparseValues)
            query_embedding = sparse_embeddings.embed_query("What is machine learning?")
            # query_embedding contains SparseValues with indices and values

            # Embed multiple documents
            docs = ["Document 1 content", "Document 2 content"]
            doc_embeddings = sparse_embeddings.embed_documents(docs)

            # Use with an index configured for sparse vectors
            from pinecone import Pinecone

            pc = Pinecone(api_key="your-api-key")

            # Create index with sparse embeddings support
            if not pc.has_index("sparse-index"):
                pc.create_index_for_model(
                    name="sparse-index",
                    cloud="aws",
                    region="us-east-1",
                    embed={
                        "model": "pinecone-sparse-english-v0",
                        "field_map": {"text": "chunk_text"},
                        "metric": "dotproduct",
                        "read_parameters": {},
                        "write_parameters": {}
                    }
                )

            index = pc.Index("sparse-index")

            # IMPORTANT: Use PineconeSparseVectorStore for sparse vectors
            # The regular PineconeVectorStore won't work with sparse embeddings
            from langchain_pinecone.vectorstores_sparse import PineconeSparseVectorStore

            # Initialize sparse vector store with sparse embeddings
            vector_store = PineconeSparseVectorStore(
                index=index,
                embedding=sparse_embeddings
            )

            # Add documents
            from uuid import uuid4

            documents = [
                Document(page_content="Machine learning is awesome", metadata={"source": "article"}),
                Document(page_content="Neural networks power modern AI", metadata={"source": "book"})
            ]

            # Generate unique IDs for each document
            uuids = [str(uuid4()) for _ in range(len(documents))]

            # Add documents to the vector store
            vector_store.add_documents(documents=documents, ids=uuids)

            # Search for similar documents
            results = vector_store.similarity_search("machine learning", k=2)
    """

    @model_validator(mode="before")
    @classmethod
    def set_default_config(cls, values: dict) -> Any:
        """Set default configuration based on model."""
        default_config_map = {
            "pinecone-sparse-english-v0": {
                "batch_size": 96,
                "query_params": {"input_type": "query", "truncation": "END"},
                "document_params": {"input_type": "passage", "truncation": "END"},
                "dimension": None,
            },
        }
        model = values.get("model")
        if model in default_config_map:
            config = default_config_map[model]
            for key, value in config.items():
                if key not in values:
                    values[key] = value
        return values

    def embed_documents(self, texts: List[str]) -> List[SparseValues]:
        """Embed search docs with sparse embeddings."""
        embeddings: List[SparseValues] = []

        _iter, batch_size = self._get_batch_iterator(texts)
        for i in _iter:
            response = self._embed_texts(
                model=self.model,
                parameters=self.document_params,
                texts=texts[i : i + batch_size],
            )
            for r in response:
                embeddings.append(
                    SparseValues(indices=r["sparse_indices"], values=r["sparse_values"])
                )

        return embeddings

    async def aembed_documents(self, texts: List[str]) -> List[SparseValues]:
        """Asynchronously embed search docs with sparse embeddings."""
        embeddings: List[SparseValues] = []
        _iter, batch_size = self._get_batch_iterator(texts)
        for i in _iter:
            response = await self._aembed_texts(
                model=self.model,
                parameters=self.document_params,
                texts=texts[i : i + batch_size],
            )
            for r in response:
                embeddings.append(
                    SparseValues(indices=r["sparse_indices"], values=r["sparse_values"])
                )
        return embeddings

    def embed_query(self, text: str) -> SparseValues:
        """Embed query text with sparse embeddings."""
        response = self._embed_texts(
            model=self.model, parameters=self.query_params, texts=[text]
        )[0]
        return SparseValues(
            indices=response["sparse_indices"], values=response["sparse_values"]
        )

    async def aembed_query(self, text: str) -> SparseValues:
        """Asynchronously embed query text with sparse embeddings."""
        embeddings = await self._aembed_texts(
            model=self.model,
            parameters=self.query_params,
            texts=[text],
        )
        response = embeddings[0]
        return SparseValues(
            indices=response["sparse_indices"], values=response["sparse_values"]
        )

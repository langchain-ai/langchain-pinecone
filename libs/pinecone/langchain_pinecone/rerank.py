from __future__ import annotations

import logging
from copy import deepcopy
from typing import Any, Dict, List, Optional, Sequence, Union

from langchain_core.callbacks.base import Callbacks
from langchain_core.documents import BaseDocumentCompressor, Document
from langchain_core.utils import secret_from_env
from pinecone import Pinecone, PineconeAsyncio
from pydantic import ConfigDict, Field, SecretStr

logger = logging.getLogger(__name__)


class PineconeRerank(BaseDocumentCompressor):
    """Document compressor that uses `Pinecone Rerank API`."""

    client: Optional[Pinecone] = None
    """Pinecone client to use for compressing documents."""
    async_client: Optional[PineconeAsyncio] = None
    """Pinecone client to use for compressing documents."""
    top_n: Optional[int] = 3
    """Number of documents to return."""
    model: str = Field(
        default="bge-reranker-v2-m3",
        description="Model to use for reranking. Default is 'bge-reranker-v2-m3'.",
    )
    """Model to use for reranking."""
    pinecone_api_key: Optional[SecretStr] = Field(
        default_factory=secret_from_env("PINECONE_API_KEY", default=None)
    )
    """Pinecone API key. Must be specified directly or via environment variable 
    PINECONE_API_KEY."""
    rank_fields: Optional[List[str]] = None
    """Fields to use for reranking when documents are dictionaries."""
    return_documents: bool = True
    """Whether to return the documents in the reranking results."""

    model_config = ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=True,
    )

    def _get_api_key(self) -> Optional[str]:
        """Get the API key from SecretStr or directly."""
        if isinstance(self.pinecone_api_key, SecretStr):
            return self.pinecone_api_key.get_secret_value()
        return self.pinecone_api_key

    def _get_sync_client(self) -> Pinecone:
        """Get or create the sync client."""
        if self.client is None:
            self.client = Pinecone(api_key=self._get_api_key())
        elif not isinstance(self.client, Pinecone):
            raise TypeError(
                "The 'client' parameter must be an instance of Pinecone.\n"
                "You may create the Pinecone object like:\n\n"
                "from pinecone import Pinecone\nclient = Pinecone(api_key=...)"
            )
        return self.client

    async def _get_async_client(self) -> PineconeAsyncio:
        """Get or create the async client."""
        if self.async_client is None:
            self.async_client = PineconeAsyncio(api_key=self._get_api_key())
        elif not isinstance(self.async_client, PineconeAsyncio):
            raise TypeError(
                "The 'async_client' parameter must be an instance of PineconeAsyncio.\n"
                "You may create the PineconeAsyncio object like:\n\n"
                "from pinecone import PineconeAsyncio\nasync_client = PineconeAsyncio(api_key=...)"
            )
        return self.async_client

    def _document_to_dict(
        self,
        document: Union[str, Document, dict],
        index: int,
    ) -> dict:
        if isinstance(document, Document):
            doc_id_from_meta = document.metadata.get("id")
            if isinstance(doc_id_from_meta, str) and doc_id_from_meta:
                doc_id = doc_id_from_meta
            else:  # Generate ID if not valid
                doc_id = f"doc_{index}"

            doc_data = {
                "id": doc_id,
                "text": document.page_content,
                **document.metadata,
            }
            return doc_data
        elif isinstance(document, dict):
            current_id = document.get("id")
            if not isinstance(current_id, str) or not current_id:
                document["id"] = f"doc_{index}"  # Generate and set ID if not valid
            return document
        else:
            return {"id": f"doc_{index}", "text": str(document)}

    def _rerank_params(self, model: str, truncate: str) -> dict:
        """Returns the parameters for the rerank API call."""
        parameters = {}
        # Only include truncate parameter for models that support it
        if model != "cohere-rerank-3.5":
            parameters["truncate"] = truncate
        return parameters

    def rerank(
        self,
        documents: Sequence[Union[str, Document, dict]],
        query: str,
        *,
        rank_fields: Optional[List[str]] = None,
        model: Optional[str] = None,
        top_n: Optional[int] = None,
        truncate: str = "END",
    ) -> List[Dict[str, Any]]:
        """Returns an ordered list of documents ordered by their relevance to the provided query."""
        if len(documents) == 0:  # to avoid empty API call
            return []

        # Convert documents to dict format
        docs = [
            self._document_to_dict(document=doc, index=i)
            for i, doc in enumerate(documents)
        ]

        try:
            client = self._get_sync_client()

            # Use self.model if model is None
            model_to_use = model if model is not None else self.model
            if model_to_use is None:  # This should never happen due to validator
                raise ValueError("No model specified for reranking")

            rerank_result = client.inference.rerank(
                model=model_to_use,
                query=query,
                documents=docs,
                rank_fields=rank_fields or self.rank_fields or ["text"],
                top_n=top_n or self.top_n,
                return_documents=self.return_documents,
                parameters=self._rerank_params(model=model_to_use, truncate=truncate),
            )

            result_dicts = []
            for result_item_data in rerank_result.data:
                result_dict = {
                    "id": result_item_data.document.id,
                    "index": result_item_data.index,
                    "score": result_item_data.score,
                }

                if self.return_documents:
                    result_dict["document"] = result_item_data.document.to_dict()

                result_dicts.append(result_dict)

            return result_dicts

        except Exception as e:
            logger.error(f"Rerank error: {e}")
            return []

    async def arerank(
        self,
        documents: Sequence[Union[str, Document, dict]],
        query: str,
        *,
        rank_fields: Optional[List[str]] = None,
        model: Optional[str] = None,
        top_n: Optional[int] = None,
        truncate: str = "END",
    ) -> List[Dict[str, Any]]:
        """Async rerank documents using Pinecone's reranking API."""
        if len(documents) == 0:  # to avoid empty API call
            return []

        docs = [
            self._document_to_dict(document=doc, index=i)
            for i, doc in enumerate(documents)
        ]

        try:
            client = await self._get_async_client()

            # Use self.model if model is None
            model_to_use = model if model is not None else self.model
            if model_to_use is None:  # This should never happen due to validator
                raise ValueError("No model specified for reranking")

            rerank_result = await client.inference.rerank(
                model=model_to_use,
                query=query,
                documents=docs,
                rank_fields=rank_fields or self.rank_fields or ["text"],
                top_n=top_n or self.top_n,
                return_documents=self.return_documents,
                parameters=self._rerank_params(model=model_to_use, truncate=truncate),
            )

            result_dicts = []
            for result_item_data in rerank_result.data:
                result_dict = {
                    "id": result_item_data.document.id,
                    "index": result_item_data.index,
                    "score": result_item_data.score,
                }

                if self.return_documents:
                    result_dict["document"] = result_item_data.document.to_dict()

                result_dicts.append(result_dict)

            return result_dicts
        except Exception as e:
            logger.error(f"Async rerank error: {e}")
            return []

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        """Compress documents using Pinecone's rerank API."""
        if not documents:
            return []

        compressed = []
        reranked_results = self.rerank(documents=documents, query=query)

        if not reranked_results:
            return []

        for res in reranked_results:
            if res["index"] is not None:
                doc_index = res["index"]
                if 0 <= doc_index < len(documents):
                    doc = documents[doc_index]
                    doc_copy = Document(
                        doc.page_content, metadata=deepcopy(doc.metadata)
                    )
                    doc_copy.metadata["relevance_score"] = res["score"]
                    compressed.append(doc_copy)

        return compressed

    async def acompress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        """Async compress documents using Pinecone's rerank API."""
        if not documents:
            return []

        compressed = []
        reranked_results = await self.arerank(documents=documents, query=query)

        if not reranked_results:
            return []

        for res in reranked_results:
            if res["index"] is not None:
                doc_index = res["index"]
                if 0 <= doc_index < len(documents):
                    doc = documents[doc_index]
                    doc_copy = Document(
                        doc.page_content, metadata=deepcopy(doc.metadata)
                    )
                    doc_copy.metadata["relevance_score"] = res["score"]
                    compressed.append(doc_copy)

        return compressed

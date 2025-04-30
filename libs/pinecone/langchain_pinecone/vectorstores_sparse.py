from __future__ import annotations

import asyncio
import logging
import uuid
from typing import (
    TYPE_CHECKING,
    Any,
    Iterable,
    List,
    Optional,
    Tuple,
    TypeVar,
)

from langchain_core.documents import Document
from langchain_core.utils import batch_iterate
from langchain_core.vectorstores import VectorStore
from pinecone import SparseValues, Vector  # type: ignore[import-untyped]

from langchain_pinecone._utilities import (
    DistanceStrategy,
    sparse_maximal_marginal_relevance,
)
from langchain_pinecone.embeddings import PineconeSparseEmbeddings
from langchain_pinecone.vectorstores import PineconeVectorStore

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

VST = TypeVar("VST", bound=VectorStore)


class PineconeSparseVectorStore(PineconeVectorStore):
    """Pinecone sparse vector store integration.

    This class extends PineconeVectorStore to support sparse vector representations.
    It requires a Pinecone sparse index and PineconeSparseEmbeddings.

    Setup:
        ```python
        # Install required packages
        pip install langchain-pinecone pinecone-client
        ```

    Key init args - indexing params:
        text_key (str): The metadata key where the document text will be stored.
        namespace (str): Pinecone namespace to use.
        distance_strategy (DistanceStrategy): Strategy for computing distances.

    Key init args - client params:
        index (pinecone.Index): A Pinecone sparse index.
        embedding (PineconeSparseEmbeddings): A sparse embeddings model.
        pinecone_api_key (str): The Pinecone API key.
        index_name (str): The name of the Pinecone index.

    See full list of supported init args and their descriptions in the params section.

    Instantiate:
        ```python
        from pinecone import Pinecone
        from langchain_pinecone import PineconeSparseVectorStore
        from langchain_pinecone.embeddings import PineconeSparseEmbeddings

        # Initialize Pinecone client
        pc = Pinecone(api_key="your-api-key")

        # Get your sparse index
        index = pc.Index("your-sparse-index-name")

        # Initialize embedding function
        embeddings = PineconeSparseEmbeddings()

        # Create vector store
        vectorstore = PineconeSparseVectorStore(
            index=index,
            embedding=embeddings,
            text_key="content",
            namespace="my-namespace"
        )
        ```

    Add Documents:
        ```python
        from langchain_core.documents import Document

        docs = [
            Document(page_content="This is a sparse vector example"),
            Document(page_content="Another document for testing")
        ]

        # Option 1: Add from Document objects
        vectorstore.add_documents(docs)

        # Option 2: Add from texts
        texts = ["Text 1", "Text 2"]
        metadatas = [{"source": "source1"}, {"source": "source2"}]
        vectorstore.add_texts(texts, metadatas=metadatas)
        ```

    Update Documents:
        Update documents by re-adding them with the same IDs.
        ```python
        ids = ["id1", "id2"]
        texts = ["Updated text 1", "Updated text 2"]
        metadatas = [{"source": "updated_source1"}, {"source": "updated_source2"}]

        vectorstore.add_texts(texts, metadatas=metadatas, ids=ids)
        ```

    Delete Documents:
        ```python
        # Delete by IDs
        vectorstore.delete(ids=["id1", "id2"])

        # Delete by filter
        vectorstore.delete(filter={"source": "source1"})

        # Delete all documents in a namespace
        vectorstore.delete(delete_all=True, namespace="my-namespace")
        ```

    Search:
        ```python
        # Search for similar documents
        docs = vectorstore.similarity_search("query text", k=5)

        # Search with filters
        docs = vectorstore.similarity_search(
            "query text",
            k=5,
            filter={"source": "source1"}
        )

        # Maximal marginal relevance search for diversity
        docs = vectorstore.max_marginal_relevance_search(
            "query text",
            k=5,
            fetch_k=20,
            lambda_mult=0.5
        )
        ```

    Search with score:
        ```python
        # Search with relevance scores
        docs_and_scores = vectorstore.similarity_search_with_score(
            "query text",
            k=5
        )

        for doc, score in docs_and_scores:
            print(f"Score: {score}, Document: {doc.page_content}")
        ```

    Use as Retriever:
        ```python
        # Create a retriever
        retriever = vectorstore.as_retriever()

        # Customize retriever
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 0.5},
            filter={"source": "source1"}
        )

        # Use the retriever
        docs = retriever.get_relevant_documents("query text")
        ```
    """

    def __init__(
        self,
        index: Optional[Any] = None,
        embedding: Optional[PineconeSparseEmbeddings] = None,
        text_key: Optional[str] = "text",
        namespace: Optional[str] = None,
        distance_strategy: Optional[DistanceStrategy] = DistanceStrategy.COSINE,
        *,
        pinecone_api_key: Optional[str] = None,
        index_name: Optional[str] = None,
    ):
        if index and index.describe_index_stats()["vector_type"] != "sparse":
            raise ValueError(
                "PineconeSparseVectorStore can only be used with Sparse Indexes"
            )
        super().__init__(
            index,
            embedding,
            text_key,
            namespace,
            distance_strategy,
            pinecone_api_key=pinecone_api_key,
            index_name=index_name,
        )

    @property
    def embeddings(self) -> PineconeSparseEmbeddings:
        if not self._embedding:
            raise ValueError(
                "Must provide a PineconeSparseEmbeddings to the PineconeSparseVectorStore"
            )
        if not isinstance(self._embedding, PineconeSparseEmbeddings):
            raise ValueError(
                "PineconeSparseVectorStore can only be used with PineconeSparseEmbeddings"
            )
        return self._embedding

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        namespace: Optional[str] = None,
        batch_size: int = 32,
        embedding_chunk_size: int = 1000,
        *,
        id_prefix: Optional[str] = None,
        **kwargs: Any,
    ) -> List[str]:
        if namespace is None:
            namespace = self._namespace

        texts = list(texts)
        ids = ids or [str(uuid.uuid4()) for _ in texts]
        if id_prefix:
            ids = [
                id_prefix + "#" + id if id_prefix + "#" not in id else id for id in ids
            ]
        metadatas = metadatas or [{} for _ in texts]
        for metadata, text in zip(metadatas, texts):
            metadata[self._text_key] = text

        # For loops to avoid memory issues and optimize when using HTTP based embeddings
        # The first loop runs the embeddings, it benefits when using OpenAI embeddings
        for i in range(0, len(texts), embedding_chunk_size):
            chunk_texts = texts[i : i + embedding_chunk_size]
            chunk_ids = ids[i : i + embedding_chunk_size]
            chunk_metadatas = metadatas[i : i + embedding_chunk_size]
            embeddings = self.embeddings.embed_documents(chunk_texts)
            vectors = [
                Vector(id=chunk_id, sparse_values=value, metadata=metadata)
                for (chunk_id, value, metadata) in zip(
                    chunk_ids, embeddings, chunk_metadatas
                )
            ]
            self.index.upsert(
                vectors=vectors,
                namespace=namespace,
                **kwargs,
            )
        return ids

    async def aadd_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        namespace: Optional[str] = None,
        batch_size: int = 32,
        embedding_chunk_size: int = 1000,
        *,
        id_prefix: Optional[str] = None,
        **kwargs: Any,
    ) -> list[str]:
        """Asynchronously run more texts through the embeddings and add to the vectorstore.

        Upsert optimization is done by chunking the embeddings and upserting them.
        This is done to avoid memory issues and optimize using HTTP based embeddings.
        For OpenAI embeddings, use pool_threads>4 when constructing the pinecone.Index,
        embedding_chunk_size>1000 and batch_size~64 for best performance.
        Args:
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.
            ids: Optional list of ids to associate with the texts.
            namespace: Optional pinecone namespace to add the texts to.
            batch_size: Batch size to use when adding the texts to the vectorstore.
            embedding_chunk_size: Chunk size to use when embedding the texts.
            id_prefix: Optional string to use as an ID prefix when upserting vectors.

        Returns:
            List of ids from adding the texts into the vectorstore.

        """
        if namespace is None:
            namespace = self._namespace

        texts = list(texts)
        ids = ids or [str(uuid.uuid4()) for _ in texts]
        if id_prefix:
            ids = [
                id_prefix + "#" + id if id_prefix + "#" not in id else id for id in ids
            ]
        metadatas = metadatas or [{} for _ in texts]
        for metadata, text in zip(metadatas, texts):
            metadata[self._text_key] = text

        # For loops to avoid memory issues and optimize when using HTTP based embeddings
        for i in range(0, len(texts), embedding_chunk_size):
            chunk_texts = texts[i : i + embedding_chunk_size]
            chunk_ids = ids[i : i + embedding_chunk_size]
            chunk_metadatas = metadatas[i : i + embedding_chunk_size]
            embeddings = await self.embeddings.aembed_documents(chunk_texts)
            vector_tuples = zip(chunk_ids, embeddings, chunk_metadatas)

            async with self.async_index as idx:
                # Split into batches and upsert asynchronously
                tasks = []
                for batch_vector_tuples in batch_iterate(batch_size, vector_tuples):
                    task = idx.upsert(
                        vectors=[
                            Vector(
                                id=chunk_id,
                                sparse_values=sparse_values,
                                metadata=metadata,
                            )
                            for chunk_id, sparse_values, metadata in batch_vector_tuples
                        ],
                        namespace=namespace,
                        **kwargs,
                    )
                    tasks.append(task)

                # Wait for all upserts to complete
                await asyncio.gather(*tasks)

        return ids

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[dict] = None,
        namespace: Optional[str] = None,
    ) -> List[Tuple[Document, float]]:
        """Return pinecone documents most similar to query, along with scores.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter: Dictionary of argument(s) to filter on metadata
            namespace: Namespace to search in. Default will search in '' namespace.

        Returns:
            List of Documents most similar to the query and score for each
        """
        return self.similarity_search_by_vector_with_score(
            self.embeddings.embed_query(query), k=k, filter=filter, namespace=namespace
        )

    async def asimilarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[dict] = None,
        namespace: Optional[str] = None,
    ) -> list[tuple[Document, float]]:
        """Asynchronously return pinecone documents most similar to query, along with scores.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter: Dictionary of argument(s) to filter on metadata
            namespace: Namespace to search in. Default will search in '' namespace.

        Returns:
            List of Documents most similar to the query and score for each
        """
        return await self.asimilarity_search_by_vector_with_score(
            (await self.embeddings.aembed_query(query)),
            k=k,
            filter=filter,
            namespace=namespace,
        )

    def similarity_search_by_vector_with_score(
        self,
        embedding: SparseValues,
        *,
        k: int = 4,
        filter: Optional[dict] = None,
        namespace: Optional[str] = None,
    ) -> List[Tuple[Document, float]]:
        """Return pinecone documents most similar to embedding, along with scores."""

        if namespace is None:
            namespace = self._namespace
        docs = []
        results = self.index.query(
            sparse_vector=embedding,
            top_k=k,
            include_metadata=True,
            namespace=namespace,
            filter=filter,
        )
        for res in results["matches"]:
            metadata = res["metadata"]
            id = res.get("id")
            if self._text_key in metadata:
                text = metadata.pop(self._text_key)
                score = res["score"]
                docs.append(
                    (Document(id=id, page_content=text, metadata=metadata), score)
                )
            else:
                logger.warning(
                    f"Found document with no `{self._text_key}` key. Skipping."
                )
        return docs

    async def asimilarity_search_by_vector_with_score(
        self,
        embedding: SparseValues,
        *,
        k: int = 4,
        filter: Optional[dict] = None,
        namespace: Optional[str] = None,
    ) -> List[Tuple[Document, float]]:
        """Return pinecone documents most similar to embedding, along with scores asynchronously."""
        if namespace is None:
            namespace = self._namespace

        docs = []
        async with self.async_index as idx:
            results = await idx.query(
                sparse_vector=embedding,
                top_k=k,
                include_metadata=True,
                namespace=namespace,
                filter=filter,
            )

        for res in results["matches"]:
            metadata = res["metadata"]
            id = res.get("id")
            if self._text_key in metadata:
                text = metadata.pop(self._text_key)
                score = res["score"]
                docs.append(
                    (Document(id=id, page_content=text, metadata=metadata), score)
                )
            else:
                logger.warning(
                    f"Found document with no `{self._text_key}` key. Skipping."
                )
        return docs

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[dict] = None,
        namespace: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return pinecone documents most similar to query.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter: Dictionary of argument(s) to filter on metadata
            namespace: Namespace to search in. Default will search in '' namespace.

        Returns:
            List of Documents most similar to the query and score for each
        """
        docs_and_scores = self.similarity_search_with_score(
            query, k=k, filter=filter, namespace=namespace, **kwargs
        )
        return [doc for doc, _ in docs_and_scores]

    async def asimilarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[dict] = None,
        namespace: Optional[str] = None,
        **kwargs: Any,
    ) -> list[Document]:
        docs_and_scores = await self.asimilarity_search_with_score(
            query, k=k, filter=filter, namespace=namespace, **kwargs
        )
        return [doc for doc, _ in docs_and_scores]

    def max_marginal_relevance_search_by_vector(
        self,
        embedding: SparseValues,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[dict] = None,
        namespace: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs selected using the maximal marginal relevance.

        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            fetch_k: Number of Documents to fetch to pass to MMR algorithm.
            lambda_mult: Number between 0 and 1 that determines the degree
                        of diversity among the results with 0 corresponding
                        to maximum diversity and 1 to minimum diversity.
                        Defaults to 0.5.
            filter: Dictionary of argument(s) to filter on metadata
            namespace: Namespace to search in. Default will search in '' namespace.

        Returns:
            List of Documents selected by maximal marginal relevance.
        """
        if namespace is None:
            namespace = self._namespace
        results = self.index.query(
            sparse_vector=embedding,
            top_k=fetch_k,
            include_values=True,
            include_metadata=True,
            namespace=namespace,
            filter=filter,
        )
        mmr_selected = sparse_maximal_marginal_relevance(
            query_embedding=embedding,
            embedding_list=[
                SparseValues.from_dict(item["sparse_values"])
                for item in results["matches"]  # type: ignore
            ],
            k=k,
            lambda_mult=lambda_mult,
        )
        selected = [results["matches"][i]["metadata"] for i in mmr_selected]
        return [
            Document(page_content=metadata.pop((self._text_key)), metadata=metadata)
            for metadata in selected
        ]

    async def amax_marginal_relevance_search_by_vector(
        self,
        embedding: SparseValues,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[dict] = None,
        namespace: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs selected using the maximal marginal relevance asynchronously.

        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            fetch_k: Number of Documents to fetch to pass to MMR algorithm.
            lambda_mult: Number between 0 and 1 that determines the degree
                        of diversity among the results with 0 corresponding
                        to maximum diversity and 1 to minimum diversity.
                        Defaults to 0.5.
            filter: Dictionary of argument(s) to filter on metadata
            namespace: Namespace to search in. Default will search in '' namespace.

        Returns:
            List of Documents selected by maximal marginal relevance.
        """
        if namespace is None:
            namespace = self._namespace

        async with self.async_index as idx:
            results = await idx.query(
                sparse_vector=embedding,
                top_k=fetch_k,
                include_values=True,
                include_metadata=True,
                namespace=namespace,
                filter=filter,
            )

        mmr_selected = sparse_maximal_marginal_relevance(
            query_embedding=embedding,
            embedding_list=[
                SparseValues.from_dict(item["sparse_values"])
                for item in results["matches"]  # type: ignore
            ],
            k=k,
            lambda_mult=lambda_mult,
        )
        selected = [results["matches"][i]["metadata"] for i in mmr_selected]
        return [
            Document(page_content=metadata.pop(self._text_key), metadata=metadata)
            for metadata in selected
        ]

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[dict] = None,
        namespace: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs selected using the maximal marginal relevance.

        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            fetch_k: Number of Documents to fetch to pass to MMR algorithm.
            lambda_mult: Number between 0 and 1 that determines the degree
                        of diversity among the results with 0 corresponding
                        to maximum diversity and 1 to minimum diversity.
                        Defaults to 0.5.
            filter: Dictionary of argument(s) to filter on metadata
            namespace: Namespace to search in. Default will search in '' namespace.

        Returns:
            List of Documents selected by maximal marginal relevance.
        """
        embedding = self.embeddings.embed_query(query)
        return self.max_marginal_relevance_search_by_vector(
            embedding, k, fetch_k, lambda_mult, filter, namespace
        )

    async def amax_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[dict] = None,
        namespace: Optional[str] = None,
        **kwargs: Any,
    ) -> list[Document]:
        embedding = await self.embeddings.aembed_query(query)
        return await self.amax_marginal_relevance_search_by_vector(
            embedding,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            filter=filter,
            namespace=namespace,
        )

    def delete(
        self,
        ids: Optional[List[str]] = None,
        delete_all: Optional[bool] = None,
        namespace: Optional[str] = None,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> None:
        """Delete by vector IDs or filter.
        Args:
            ids: List of ids to delete.
            delete_all: Whether delete all vectors in the index.
            filter: Dictionary of conditions to filter vectors to delete.
            namespace: Namespace to search in. Default will search in '' namespace.
        """

        if namespace is None:
            namespace = self._namespace

        if delete_all:
            self.index.delete(delete_all=True, namespace=namespace, **kwargs)
        elif ids is not None:
            chunk_size = 1000
            for i in range(0, len(ids), chunk_size):
                chunk = ids[i : i + chunk_size]
                self.index.delete(ids=chunk, namespace=namespace, **kwargs)
        elif filter is not None:
            self.index.delete(filter=filter, namespace=namespace, **kwargs)
        else:
            raise ValueError("Either ids, delete_all, or filter must be provided.")

        return None

    async def adelete(
        self,
        ids: Optional[List[str]] = None,
        delete_all: Optional[bool] = None,
        namespace: Optional[str] = None,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> None:
        if namespace is None:
            namespace = self._namespace

        if delete_all:
            async with self.async_index as idx:
                await idx.delete(delete_all=True, namespace=namespace, **kwargs)
        elif ids is not None:
            chunk_size = 1000
            async with self.async_index as idx:
                tasks = []
                for i in range(0, len(ids), chunk_size):
                    chunk = ids[i : i + chunk_size]
                    tasks.append(idx.delete(ids=chunk, namespace=namespace, **kwargs))
                await asyncio.gather(*tasks)
        elif filter is not None:
            async with self.async_index as idx:
                await idx.delete(filter=filter, namespace=namespace, **kwargs)
        else:
            raise ValueError("Either ids, delete_all, or filter must be provided.")

        return None

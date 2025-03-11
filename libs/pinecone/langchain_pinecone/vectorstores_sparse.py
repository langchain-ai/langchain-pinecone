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
from pinecone import SparseValues, Vector

from langchain_pinecone._utilities import DistanceStrategy, maximal_marginal_relevance
from langchain_pinecone.embeddings import PineconeSparseEmbeddings
from langchain_pinecone.vectorstores import PineconeVectorStore

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

VST = TypeVar("VST", bound=VectorStore)


class PineconeSparseVectorStore(PineconeVectorStore):
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
        mmr_selected = maximal_marginal_relevance(
            np.array([embedding], dtype=np.float32),
            [item["values"] for item in results["matches"]],
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
        embedding: List[float],
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
                vector=embedding,
                top_k=fetch_k,
                include_values=True,
                include_metadata=True,
                namespace=namespace,
                filter=filter,
            )

        mmr_selected = maximal_marginal_relevance(
            np.array([embedding], dtype=np.float32),
            [item["values"] for item in results["matches"]],
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
        embedding = self._embedding.embed_query(query)
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
        embedding = await self._embedding.aembed_query(query)
        return await self.amax_marginal_relevance_search_by_vector(
            embedding,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            filter=filter,
            namespace=namespace,
        )

    @classmethod
    def get_pinecone_index(
        cls,
        index_name: Optional[str],
        pool_threads: int = 4,
        *,
        pinecone_api_key: Optional[str] = None,
    ) -> Index:
        """Return a Pinecone Index instance.

        Args:
            index_name: Name of the index to use.
            pool_threads: Number of threads to use for index upsert.
            pinecone_api_key: The api_key of Pinecone.
        Returns:
            Pinecone Index instance."""
        _pinecone_api_key = pinecone_api_key or os.environ.get("PINECONE_API_KEY") or ""
        client = PineconeClient(
            api_key=_pinecone_api_key, pool_threads=pool_threads, source_tag="langchain"
        )
        indexes = client.list_indexes()
        index_names = [i.name for i in indexes.index_list["indexes"]]

        if index_name in index_names:
            index = client.Index(index_name)
        elif len(index_names) == 0:
            raise ValueError(
                "No active indexes found in your Pinecone project, "
                "are you sure you're using the right Pinecone API key and Environment? "
                "Please double check your Pinecone dashboard."
            )
        else:
            raise ValueError(
                f"Index '{index_name}' not found in your Pinecone project. "
                f"Did you mean one of the following indexes: {', '.join(index_names)}"
            )
        return index

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        batch_size: int = 32,
        text_key: str = "text",
        namespace: Optional[str] = None,
        index_name: Optional[str] = None,
        upsert_kwargs: Optional[dict] = None,
        pool_threads: int = 4,
        embeddings_chunk_size: int = 1000,
        *,
        id_prefix: Optional[str] = None,
        **kwargs: Any,
    ) -> PineconeVectorStore:
        """Construct Pinecone wrapper from raw documents.

        This is a user-friendly interface that:
            1. Embeds documents.
            2. Adds the documents to a provided Pinecone index

        This is intended to be a quick way to get started.

        The `pool_threads` affects the speed of the upsert operations.

        Setup: set the `PINECONE_API_KEY` environment variable to your Pinecone API key.

        Example:
            .. code-block:: python

                from langchain_pinecone import PineconeVectorStore, PineconeEmbeddings

                embeddings = PineconeEmbeddings(model="multilingual-e5-large")

                index_name = "my-index"
                vectorstore = PineconeVectorStore.from_texts(
                    texts,
                    index_name=index_name,
                    embedding=embedding,
                    namespace=namespace,
                )
        """
        pinecone_index = cls.get_pinecone_index(index_name, pool_threads)
        pinecone = cls(pinecone_index, embedding, text_key, namespace, **kwargs)

        pinecone.add_texts(
            texts,
            metadatas=metadatas,
            ids=ids,
            namespace=namespace,
            batch_size=batch_size,
            embedding_chunk_size=embeddings_chunk_size,
            id_prefix=id_prefix,
            **(upsert_kwargs or {}),
        )
        return pinecone

    @classmethod
    async def afrom_texts(
        cls: type[PineconeVectorStore],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        batch_size: int = 32,
        text_key: str = "text",
        namespace: Optional[str] = None,
        index_name: Optional[str] = None,
        upsert_kwargs: Optional[dict] = None,
        embeddings_chunk_size: int = 1000,
        *,
        id_prefix: Optional[str] = None,
        **kwargs: Any,
    ) -> PineconeVectorStore:
        pinecone = cls(
            index_name=index_name,
            embedding=embedding,
            text_key=text_key,
            namespace=namespace,
            **kwargs,
        )

        await pinecone.aadd_texts(
            texts,
            metadatas=metadatas,
            ids=ids,
            namespace=namespace,
            batch_size=batch_size,
            embedding_chunk_size=embeddings_chunk_size,
            id_prefix=id_prefix,
            **(upsert_kwargs or {}),
        )

        return pinecone

    @classmethod
    def from_existing_index(
        cls,
        index_name: str,
        embedding: Embeddings,
        text_key: str = "text",
        namespace: Optional[str] = None,
        pool_threads: int = 4,
    ) -> PineconeVectorStore:
        """Load pinecone vectorstore from index name."""
        pinecone_index = cls.get_pinecone_index(index_name, pool_threads)
        return cls(pinecone_index, embedding, text_key, namespace)

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

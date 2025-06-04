import time
from datetime import datetime
from typing import AsyncGenerator

import pytest
from langchain_core.documents import Document
from pinecone import (
    AwsRegion,
    CloudProvider,
    Metric,
    Pinecone,
    ServerlessSpec,
    SparseValues,
)

from langchain_pinecone import PineconeEmbeddings, PineconeVectorStore
from langchain_pinecone.embeddings import PineconeSparseEmbeddings
from tests.integration_tests.test_vectorstores import DEFAULT_SLEEP

DIMENSION = 1024
# unique name of the index for this test run
INDEX_NAME = f"langchain-test-embeddings-{datetime.now().strftime('%Y%m%d%H%M%S')}"
MODEL = "multilingual-e5-large"
SPARSE_MODEL_NAME = "pinecone-sparse-english-v0"
NAMESPACE_NAME = "test_namespace"


@pytest.fixture(scope="function")
async def embd_client() -> AsyncGenerator[PineconeEmbeddings, None]:
    client = PineconeEmbeddings(model=MODEL)
    yield client
    await client.async_client.close()


@pytest.fixture(scope="function")
async def sparse_embd_client() -> AsyncGenerator[PineconeSparseEmbeddings, None]:
    client = PineconeSparseEmbeddings(model=SPARSE_MODEL_NAME)
    yield client
    await client.async_client.close()


def test_embed_query(embd_client: PineconeEmbeddings) -> None:
    out = embd_client.embed_query("Hello, world!")
    assert isinstance(out, list)
    assert len(out) == DIMENSION


def test_sparse_embed_query(sparse_embd_client: PineconeSparseEmbeddings) -> None:
    out = sparse_embd_client.embed_query("Hello, world!")
    assert isinstance(out, SparseValues)
    assert len(out.indices) == 2
    assert len(out.values) == 2


@pytest.mark.asyncio
async def test_aembed_query(embd_client: PineconeEmbeddings) -> None:
    out = await embd_client.aembed_query("Hello, world!")
    assert isinstance(out, list)
    assert len(out) == DIMENSION


def test_embed_documents(embd_client: PineconeEmbeddings) -> None:
    out = embd_client.embed_documents(["Hello, world!", "This is a test."])
    assert isinstance(out, list)
    assert len(out) == 2
    assert len(out[0]) == DIMENSION


@pytest.mark.asyncio
async def test_aembed_documents(embd_client: PineconeEmbeddings) -> None:
    out = await embd_client.aembed_documents(["Hello, world!", "This is a test."])
    assert isinstance(out, list)
    assert len(out) == 2
    assert len(out[0]) == DIMENSION


def test_vector_store(embd_client: PineconeEmbeddings) -> None:
    # setup index if it doesn't exist
    pc = Pinecone()
    if pc.has_index(name=INDEX_NAME):  # change to list comprehension
        pc.delete_index(INDEX_NAME)
        time.sleep(DEFAULT_SLEEP)  # prevent race with subsequent creation
    pc.create_index(
        name=INDEX_NAME,
        dimension=DIMENSION,
        metric=Metric.COSINE,
        spec=ServerlessSpec(cloud=CloudProvider.AWS, region=AwsRegion.US_WEST_2),
    )
    # now test connecting directly and adding docs
    vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding=embd_client)
    vectorstore.add_documents(
        [Document("Hello, world!"), Document("This is a test.")],
        namespace=NAMESPACE_NAME,
    )
    time.sleep(DEFAULT_SLEEP)  # Increase wait time to ensure indexing is complete
    resp = vectorstore.similarity_search(query="hello", namespace=NAMESPACE_NAME)
    assert len(resp) == 2
    # delete index
    pc.delete_index(INDEX_NAME)

# langchain-pinecone

This package contains the LangChain integration with Pinecone.

## Installation

```bash
pip install -qU langchain langchain-pinecone langchain-openai
```

And you should configure credentials by setting the following environment variables:

- `PINECONE_API_KEY`
- `OPENAI_API_KEY` (optional, for embeddings to use)

## Development

### Running Tests

The test suite includes both unit tests and integration tests. To run the tests:

```bash
# Run unit tests only
make test

# Run integration tests (requires environment variables)
make integration_test
```

#### Required Environment Variables for Tests

Integration tests require the following environment variables:

- `PINECONE_API_KEY`: Required for all integration tests
- `OPENAI_API_KEY`: Optional, required only for OpenAI embedding tests

You can set these environment variables before running the tests:

```bash
export PINECONE_API_KEY="your-api-key"
export OPENAI_API_KEY="your-openai-key"  # Optional
```

If these environment variables are not set, the integration tests that require them will be skipped.

## Usage

### Initialization

Before initializing our vector store, let's connect to a Pinecone index. If one named `index_name` doesn't exist, it will be created.

```python
from pinecone import ServerlessSpec

index_name = "langchain-test-index"  # change if desired

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )

index = pc.Index(index_name)
```

Initialize embedding model:

```python
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
```

The `PineconeVectorStore` class exposes the connection to the Pinecone vector store.

```python
from langchain_pinecone import PineconeVectorStore

vector_store = PineconeVectorStore(index=index, embedding=embeddings)
```

### Manage vector store

Once you have created your vector store, we can interact with it by adding and deleting different items.

#### Add items to vector store

We can add items to our vector store by using the `add_documents` function.

```python
from uuid import uuid4

from langchain_core.documents import Document

document_1 = Document(
    page_content="I had chocalate chip pancakes and scrambled eggs for breakfast this morning.",
    metadata={"source": "tweet"},
)

document_2 = Document(
    page_content="The weather forecast for tomorrow is cloudy and overcast, with a high of 62 degrees.",
    metadata={"source": "news"},
)

document_3 = Document(
    page_content="Building an exciting new project with LangChain - come check it out!",
    metadata={"source": "tweet"},
)

document_4 = Document(
    page_content="Robbers broke into the city bank and stole $1 million in cash.",
    metadata={"source": "news"},
)

document_5 = Document(
    page_content="Wow! That was an amazing movie. I can't wait to see it again.",
    metadata={"source": "tweet"},
)

document_6 = Document(
    page_content="Is the new iPhone worth the price? Read this review to find out.",
    metadata={"source": "website"},
)

document_7 = Document(
    page_content="The top 10 soccer players in the world right now.",
    metadata={"source": "website"},
)

document_8 = Document(
    page_content="LangGraph is the best framework for building stateful, agentic applications!",
    metadata={"source": "tweet"},
)

document_9 = Document(
    page_content="The stock market is down 500 points today due to fears of a recession.",
    metadata={"source": "news"},
)

document_10 = Document(
    page_content="I have a bad feeling I am going to get deleted :(",
    metadata={"source": "tweet"},
)

documents = [
    document_1,
    document_2,
    document_3,
    document_4,
    document_5,
    document_6,
    document_7,
    document_8,
    document_9,
    document_10,
]
uuids = [str(uuid4()) for _ in range(len(documents))]
vector_store.add_documents(documents=documents, ids=uuids)
```

#### Delete items from vector store

```
vector_store.delete(ids=[uuids[-1]])
```

### Query vector store

Once your vector store has been created and the relevant documents have been added you will most likely wish to query it during the running of your chain or agent. 

#### Query directly

Performing a simple similarity search can be done as follows:

```python
results = vector_store.similarity_search(
    "LangChain provides abstractions to make working with LLMs easy",
    k=2,
    filter={"source": "tweet"},
)
for res in results:
    print(f"* {res.page_content} [{res.metadata}]")
```

#### Similarity search with score

You can also search with score:

```python
results = vector_store.similarity_search_with_score(
    "Will it be hot tomorrow?", k=1, filter={"source": "news"}
)
for res, score in results:
    print(f"* [SIM={score:3f}] {res.page_content} [{res.metadata}]")
```

### Query by turning into retriever

You can also transform the vector store into a retriever for easier usage in your chains.

```python
retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 1, "score_threshold": 0.4},
)
retriever.invoke("Stealing from the bank is a crime", filter={"source": "news"})
```

### List Supported Pinecone Models (Dynamic)

You can dynamically fetch the list of supported embedding and reranker models from Pinecone using the following methods:

```python
from langchain_pinecone import PineconeEmbeddings, PineconeRerank

# List all supported embedding models
embedding_models = PineconeEmbeddings.list_supported_models()
print("Embedding models:", [m["model"] for m in embedding_models])

# List all supported reranker models
reranker_models = PineconeRerank.list_supported_models()
print("Reranker models:", [m["model"] for m in reranker_models])

# You can also filter by vector type (e.g., 'dense' or 'sparse')
sparse_embedding_models = PineconeEmbeddings.list_supported_models(vector_type="sparse")
print("Sparse embedding models:", [m["model"] for m in sparse_embedding_models])
```

## Async Model Listing

For async applications, you can use the async versions of the model listing functions:

```python
import asyncio
from langchain_pinecone import PineconeEmbeddings, PineconeRerank

async def list_models_async():
    # List all supported embedding models asynchronously
    embedding_models = await PineconeEmbeddings().alist_supported_models()
    print("Embedding models:", [m["model"] for m in embedding_models])
    
    # List all supported reranker models asynchronously
    reranker_models = await PineconeRerank().alist_supported_models()
    print("Reranker models:", [m["model"] for m in reranker_models])
    
    # Filter by vector type asynchronously
    dense_embedding_models = await PineconeEmbeddings().alist_supported_models(vector_type="dense")
    print("Dense embedding models:", [m["model"] for m in dense_embedding_models])

# Run the async function
asyncio.run(list_models_async())
```

You can also use the low-level async function directly:

```python
import asyncio
from langchain_pinecone._utilities import aget_pinecone_supported_models

async def get_models_directly():
    api_key = "your-pinecone-api-key"
    
    # Get all models
    all_models = await aget_pinecone_supported_models(api_key)
    
    # Get only embedding models
    embed_models = await aget_pinecone_supported_models(api_key, model_type="embed")
    
    # Get only dense embedding models
    dense_models = await aget_pinecone_supported_models(api_key, model_type="embed", vector_type="dense")
    
    return all_models, embed_models, dense_models
```

This ensures your application always uses valid, up-to-date model names from Pinecone.
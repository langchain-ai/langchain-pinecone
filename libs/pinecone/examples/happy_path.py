import marimo

__generated_with = "0.9.0"
app = marimo.App(width="medium")


@app.cell
def _imports():
    import os
    import time
    from datetime import datetime

    import marimo as mo

    return datetime, mo, os, time


@app.cell
def _header(mo):
    mo.md(
        r"""
        # langchain-pinecone Happy-Path Demo

        A hands-on confidence check for the `langchain-pinecone` integration running against
        the **locally-installed editable package** (not a PyPI release).

        **Requirements:** set the `PINECONE_API_KEY` environment variable before running.
        The core path uses Pinecone-hosted inference so no additional API keys are needed.
        An LLM key (`ANTHROPIC_API_KEY` or `OPENAI_API_KEY`) is optional and enables the
        RAG-chain cell.

        Sections:
        1. Pinecone client setup + index lifecycle
        2. `PineconeEmbeddings` (dense)
        3. `PineconeVectorStore` — build, add, search
        4. `PineconeVectorStore.as_retriever()` + optional RAG chain
        5. `PineconeSparseVectorStore` (sparse)
        6. `PineconeRerank`
        7. Teardown (deletes every index created here)
        """
    )
    return ()


@app.cell
def _check_key(mo, os):
    _api_key = os.environ.get("PINECONE_API_KEY", "")
    if not _api_key:
        mo.stop(
            True,
            mo.md(
                "**`PINECONE_API_KEY` is not set.** "
                "Export it in your shell before running:\n\n"
                "```bash\nexport PINECONE_API_KEY=<your-key>\n```"
            ),
        )
    api_key = _api_key
    return (api_key,)


@app.cell
def _setup(api_key, datetime, mo):
    import pinecone
    from pinecone import AwsRegion, CloudProvider, Metric, ServerlessSpec, VectorType

    pc = pinecone.Pinecone(api_key=api_key)

    _ts = datetime.now().strftime("%Y%m%d%H%M%S")
    dense_index_name = f"lc-demo-{_ts}"
    sparse_index_name = f"lc-demo-sparse-{_ts}"

    mo.md(
        f"Pinecone client ready. Index names for this session:\n\n"
        f"- Dense: `{dense_index_name}`\n"
        f"- Sparse: `{sparse_index_name}`"
    )
    return (
        AwsRegion,
        CloudProvider,
        Metric,
        ServerlessSpec,
        VectorType,
        dense_index_name,
        pc,
        pinecone,
        sparse_index_name,
    )


# ---------------------------------------------------------------------------
# Section 1: PineconeEmbeddings (dense)
# ---------------------------------------------------------------------------


@app.cell
def _embeddings_header(mo):
    mo.md("## 1. PineconeEmbeddings (dense)")
    return ()


@app.cell
def _embeddings(api_key, mo):
    from langchain_pinecone import PineconeEmbeddings

    embeddings = PineconeEmbeddings(
        model="multilingual-e5-large",
        pinecone_api_key=api_key,
        dimension=1024,
    )

    _query_vec = embeddings.embed_query("What is LangChain?")
    _doc_vecs = embeddings.embed_documents(
        ["LangChain is a framework for LLM apps.", "Pinecone is a vector database."]
    )

    mo.md(
        f"- `embed_query` length: **{len(_query_vec)}**\n"
        f"- `embed_documents` shapes: **{[len(v) for v in _doc_vecs]}**"
    )
    return (embeddings,)


# ---------------------------------------------------------------------------
# Section 2: PineconeVectorStore — build, add, search
# ---------------------------------------------------------------------------


@app.cell
def _vectorstore_header(mo):
    mo.md("## 2. PineconeVectorStore — build, add, search")
    return ()


@app.cell
def _create_dense_index(AwsRegion, CloudProvider, Metric, ServerlessSpec, dense_index_name, mo, pc, time):
    pc.create_index(
        name=dense_index_name,
        dimension=1024,
        metric=Metric.COSINE,
        spec=ServerlessSpec(cloud=CloudProvider.AWS, region=AwsRegion.US_WEST_2),
    )

    # Poll until ready
    _deadline = time.monotonic() + 120
    while time.monotonic() < _deadline:
        _status = pc.describe_index(dense_index_name).status
        if _status.get("ready"):
            break
        time.sleep(2)

    mo.md(f"Dense index `{dense_index_name}` is **ready**.")
    return ()


@app.cell
def _vectorstore_build(api_key, dense_index_name, embeddings, mo):
    from langchain_pinecone import PineconeVectorStore

    _texts = [
        "LangChain is an open-source framework for building LLM-powered apps.",
        "Pinecone is a managed vector database optimised for ML workloads.",
        "Retrieval-augmented generation (RAG) combines search with LLMs.",
        "langchain-pinecone is the official LangChain integration for Pinecone.",
    ]
    _metadatas = [{"source": f"doc-{i}"} for i in range(len(_texts))]

    store = PineconeVectorStore.from_texts(
        _texts,
        embeddings,
        index_name=dense_index_name,
        pinecone_api_key=api_key,
        metadatas=_metadatas,
    )

    mo.md(f"Inserted **{len(_texts)}** documents into `{dense_index_name}`.")
    return (store,)


@app.cell
def _vectorstore_search(mo, store, time):
    # Brief wait for eventual consistency
    time.sleep(5)

    _query = "How does RAG work?"
    _results = store.similarity_search(_query, k=2)
    _scored = store.similarity_search_with_score(_query, k=2)

    mo.md(
        "### similarity_search\n\n"
        + "\n".join(f"- `{r.page_content[:80]}`" for r in _results)
        + "\n\n### similarity_search_with_score\n\n"
        + "\n".join(f"- score={s:.4f}: `{r.page_content[:60]}`" for r, s in _scored)
    )
    return ()


# ---------------------------------------------------------------------------
# Section 3: Retriever + optional RAG chain
# ---------------------------------------------------------------------------


@app.cell
def _retriever_header(mo):
    mo.md("## 3. Retriever + optional RAG chain")
    return ()


@app.cell
def _retriever(mo, os, store):
    retriever = store.as_retriever(search_kwargs={"k": 2})
    _docs = retriever.invoke("What is Pinecone used for?")

    mo.md(
        "Retrieved docs via `as_retriever`:\n\n"
        + "\n".join(f"- `{d.page_content[:80]}`" for d in _docs)
    )

    # Optional RAG chain — only if an LLM provider is importable
    _anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
    _openai_key = os.environ.get("OPENAI_API_KEY", "")

    _rag_output = None
    if _anthropic_key:
        try:
            from langchain_anthropic import ChatAnthropic
            from langchain_core.output_parsers import StrOutputParser
            from langchain_core.prompts import ChatPromptTemplate
            from langchain_core.runnables import RunnablePassthrough

            _prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "human",
                        "Answer using only this context:\n{context}\n\nQuestion: {question}",
                    )
                ]
            )
            _llm = ChatAnthropic(model="claude-haiku-4-5-20251001", api_key=_anthropic_key)
            _chain = (
                {"context": retriever, "question": RunnablePassthrough()}
                | _prompt
                | _llm
                | StrOutputParser()
            )
            _rag_output = _chain.invoke("What is LangChain?")
        except ImportError:
            pass

    elif _openai_key:
        try:
            from langchain_core.output_parsers import StrOutputParser
            from langchain_core.prompts import ChatPromptTemplate
            from langchain_core.runnables import RunnablePassthrough
            from langchain_openai import ChatOpenAI

            _prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "human",
                        "Answer using only this context:\n{context}\n\nQuestion: {question}",
                    )
                ]
            )
            _llm = ChatOpenAI(model="gpt-4o-mini", api_key=_openai_key)
            _chain = (
                {"context": retriever, "question": RunnablePassthrough()}
                | _prompt
                | _llm
                | StrOutputParser()
            )
            _rag_output = _chain.invoke("What is LangChain?")
        except ImportError:
            pass

    if _rag_output:
        mo.md(f"**RAG chain answer:** {_rag_output}")
    else:
        mo.md(
            "*RAG chain skipped — set `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` "
            "(and install the matching provider package) to enable this cell.*"
        )

    return (retriever,)


# ---------------------------------------------------------------------------
# Section 4: PineconeSparseVectorStore
# ---------------------------------------------------------------------------


@app.cell
def _sparse_header(mo):
    mo.md("## 4. PineconeSparseVectorStore")
    return ()


@app.cell
def _create_sparse_index(Metric, ServerlessSpec, VectorType, AwsRegion, CloudProvider, mo, pc, sparse_index_name, time):
    pc.create_index(
        name=sparse_index_name,
        spec=ServerlessSpec(cloud=CloudProvider.AWS, region=AwsRegion.US_WEST_2),
        metric=Metric.DOTPRODUCT,
        vector_type=VectorType.SPARSE,
    )

    _deadline = time.monotonic() + 120
    while time.monotonic() < _deadline:
        _status = pc.describe_index(sparse_index_name).status
        if _status.get("ready"):
            break
        time.sleep(2)

    mo.md(f"Sparse index `{sparse_index_name}` is **ready**.")
    return ()


@app.cell
def _sparse_store(api_key, mo, pc, sparse_index_name, time):
    from langchain_pinecone import PineconeSparseVectorStore
    from langchain_pinecone.embeddings import PineconeSparseEmbeddings

    sparse_embeddings = PineconeSparseEmbeddings(
        model="pinecone-sparse-english-v0",
        pinecone_api_key=api_key,
    )

    _sparse_index = pc.Index(sparse_index_name)
    sparse_store = PineconeSparseVectorStore(
        index=_sparse_index,
        embedding=sparse_embeddings,
    )

    _sparse_texts = [
        "Vector databases store high-dimensional embeddings efficiently.",
        "Sparse vectors use token-level bag-of-words representations.",
        "BM25 is a classic sparse retrieval algorithm.",
        "Dense and sparse retrieval can be combined for hybrid search.",
    ]
    sparse_store.add_texts(_sparse_texts)

    time.sleep(5)
    _sparse_results = sparse_store.similarity_search("sparse retrieval algorithm", k=2)

    mo.md(
        "Sparse similarity_search results:\n\n"
        + "\n".join(f"- `{r.page_content[:80]}`" for r in _sparse_results)
    )
    return (sparse_store,)


# ---------------------------------------------------------------------------
# Section 5: PineconeRerank
# ---------------------------------------------------------------------------


@app.cell
def _rerank_header(mo):
    mo.md("## 5. PineconeRerank")
    return ()


@app.cell
def _rerank(api_key, mo, retriever):
    from langchain_pinecone import PineconeRerank

    reranker = PineconeRerank(
        model="bge-reranker-v2-m3",
        pinecone_api_key=api_key,
        top_n=2,
    )

    _query = "What is a vector database?"
    _docs = retriever.invoke(_query)
    _reranked = reranker.compress_documents(_docs, _query)

    mo.md(
        "Reranked results (top 2):\n\n"
        + "\n".join(
            f"- score={d.metadata.get('relevance_score', 'n/a'):.4f}: "
            f"`{d.page_content[:70]}`"
            for d in _reranked
        )
    )
    return ()


# ---------------------------------------------------------------------------
# Section 6: Teardown — delete all indexes created by this notebook
# ---------------------------------------------------------------------------


@app.cell
def _teardown_header(mo):
    mo.md("## 6. Teardown")
    return ()


@app.cell
def _teardown(dense_index_name, mo, pc, sparse_index_name):
    _deleted = []
    for _name in [dense_index_name, sparse_index_name]:
        try:
            if pc.has_index(_name):
                pc.delete_index(_name)
                _deleted.append(_name)
        except Exception:
            pass

    mo.md(
        "Deleted indexes:\n\n"
        + ("\n".join(f"- `{n}`" for n in _deleted) if _deleted else "*(none to delete)*")
    )
    return ()


if __name__ == "__main__":
    app.run()

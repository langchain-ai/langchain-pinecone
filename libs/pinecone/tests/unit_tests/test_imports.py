from langchain_pinecone import __all__

EXPECTED_ALL = [
    "PineconeEmbeddings",
    "PineconeSparseEmbeddings",
    "PineconeVectorStore",
    "PineconeSparseVectorStore",
    "Pinecone",
    "PineconeRerank",
]


def test_all_imports() -> None:
    assert sorted(EXPECTED_ALL) == sorted(__all__)

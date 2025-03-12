from enum import Enum
from typing import List, Union

import numpy as np
from pinecone import SparseValues  # type: ignore[import-untyped]

Matrix = Union[List[List[float]], List[np.ndarray], np.ndarray]


class DistanceStrategy(str, Enum):
    """Enumerator of the Distance strategies for calculating distances
    between vectors."""

    EUCLIDEAN_DISTANCE = "EUCLIDEAN_DISTANCE"
    MAX_INNER_PRODUCT = "MAX_INNER_PRODUCT"
    COSINE = "COSINE"


def maximal_marginal_relevance(
    query_embedding: np.ndarray,
    embedding_list: list,
    lambda_mult: float = 0.5,
    k: int = 4,
) -> List[int]:
    """Calculate maximal marginal relevance."""
    if min(k, len(embedding_list)) <= 0:
        return []
    if query_embedding.ndim == 1:
        query_embedding = np.expand_dims(query_embedding, axis=0)
    similarity_to_query = cosine_similarity(query_embedding, embedding_list)[0]
    most_similar = int(np.argmax(similarity_to_query))
    idxs = [most_similar]
    selected = np.array([embedding_list[most_similar]])
    while len(idxs) < min(k, len(embedding_list)):
        best_score = -np.inf
        idx_to_add = -1
        similarity_to_selected = cosine_similarity(embedding_list, selected)
        for i, query_score in enumerate(similarity_to_query):
            if i in idxs:
                continue
            redundant_score = max(similarity_to_selected[i])
            equation_score = (
                lambda_mult * query_score - (1 - lambda_mult) * redundant_score
            )
            if equation_score > best_score:
                best_score = equation_score
                idx_to_add = i
        idxs.append(idx_to_add)
        selected = np.append(selected, [embedding_list[idx_to_add]], axis=0)
    return idxs


def cosine_similarity(X: Matrix, Y: Matrix) -> np.ndarray:
    """Row-wise cosine similarity between two equal-width matrices."""
    if len(X) == 0 or len(Y) == 0:
        return np.array([])

    X = np.array(X)
    Y = np.array(Y)
    if X.shape[1] != Y.shape[1]:
        raise ValueError(
            f"Number of columns in X and Y must be the same. X has shape {X.shape} "
            f"and Y has shape {Y.shape}."
        )
    try:
        import simsimd as simd

        X = np.array(X, dtype=np.float32)
        Y = np.array(Y, dtype=np.float32)
        Z = 1 - np.array(simd.cdist(X, Y, metric="cosine"))
        return Z
    except ImportError:
        X_norm = np.linalg.norm(X, axis=1)
        Y_norm = np.linalg.norm(Y, axis=1)
        # Ignore divide by zero errors run time warnings as those are handled below.
        with np.errstate(divide="ignore", invalid="ignore"):
            similarity = np.dot(X, Y.T) / np.outer(X_norm, Y_norm)
        similarity[np.isnan(similarity) | np.isinf(similarity)] = 0.0
        return similarity


def sparse_maximal_marginal_relevance(
    query_embedding: SparseValues,
    embedding_list: List[SparseValues],
    lambda_mult: float = 0.5,
    k: int = 4,
) -> List[int]:
    """Calculate maximal marginal relevance for sparse vectors.

    Args:
        query_embedding: A sparse vector representation of the query
        embedding_list: A list of sparse vector representations to compare against
        lambda_mult: Controls the weight given to query similarity vs diversity
        k: The number of results to return

    Returns:
        A list of indices of the selected items in order of relevance
    """
    if min(k, len(embedding_list)) <= 0:
        return []

    # Calculate similarity between the query and all embeddings
    similarity_to_query = sparse_cosine_similarity(query_embedding, embedding_list)

    # Select the most similar embedding first
    most_similar = int(np.argmax(similarity_to_query))
    idxs = [most_similar]
    selected = [embedding_list[most_similar]]

    # Iteratively select the next best embedding
    while len(idxs) < min(k, len(embedding_list)):
        best_score = -np.inf
        idx_to_add = -1

        # For each candidate embedding
        for i, query_score in enumerate(similarity_to_query):
            if i in idxs:
                continue

            # Calculate similarity to already selected embeddings
            redundant_scores = []
            for selected_embedding in selected:
                # Calculate similarity between this candidate and each selected embedding
                sim = sparse_cosine_similarity(embedding_list[i], [selected_embedding])[
                    0
                ]
                redundant_scores.append(sim)

            redundant_score = max(redundant_scores) if redundant_scores else 0.0

            # Calculate the MMR score
            equation_score = (
                lambda_mult * query_score - (1 - lambda_mult) * redundant_score
            )

            if equation_score > best_score:
                best_score = equation_score
                idx_to_add = i

        idxs.append(idx_to_add)
        selected.append(embedding_list[idx_to_add])

    return idxs


def sparse_cosine_similarity(X: SparseValues, Y: List[SparseValues]) -> np.ndarray:
    """Calculate cosine similarity between sparse vectors without converting to dense.

    Args:
        X: A single sparse vector
        Y: A list of sparse vectors

    Returns:
        A numpy array of similarity scores
    """
    if not Y:
        return np.array([])

    # Calculate the norm of X
    x_norm = np.sqrt(sum(val * val for val in X.values))
    if x_norm == 0:
        return np.zeros(len(Y))

    similarities = np.zeros(len(Y))

    # Create a dictionary for faster lookup of X values
    x_dict = {idx: val for idx, val in zip(X.indices, X.values)}

    for i, y_vec in enumerate(Y):
        # Calculate the dot product
        dot_product = 0.0
        y_norm = 0.0

        for idx, val in zip(y_vec.indices, y_vec.values):
            y_norm += val * val
            if idx in x_dict:
                dot_product += x_dict[idx] * val

        y_norm = np.sqrt(y_norm)

        # Calculate cosine similarity
        if y_norm == 0:
            similarities[i] = 0.0
        else:
            similarities[i] = dot_product / (x_norm * y_norm)

    return similarities

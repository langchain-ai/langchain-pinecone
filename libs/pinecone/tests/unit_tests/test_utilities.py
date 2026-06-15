import sys
import unittest
import unittest.mock
import warnings

import numpy as np
from pinecone import SparseValues  # type: ignore[import-untyped]

from langchain_pinecone._utilities import (
    cosine_similarity,
    maximal_marginal_relevance,
    sparse_cosine_similarity,
    sparse_maximal_marginal_relevance,
)


class TestSparseUtilities(unittest.TestCase):
    def test_sparse_cosine_similarity_basic(self) -> None:
        """Test basic sparse cosine similarity calculation."""
        # Create sparse vectors with known cosine similarity
        x = SparseValues(indices=[0, 1, 2], values=[1.0, 2.0, 3.0])
        y = SparseValues(indices=[0, 1, 2], values=[1.0, 2.0, 3.0])

        # Identical vectors should have cosine similarity of 1.0
        result = sparse_cosine_similarity(x, [y])
        self.assertAlmostEqual(result[0], 1.0, places=6)

        # Orthogonal vectors should have cosine similarity of 0.0
        x = SparseValues(indices=[0, 1], values=[1.0, 0.0])
        y = SparseValues(indices=[1, 2], values=[0.0, 1.0])
        result = sparse_cosine_similarity(x, [y])
        self.assertAlmostEqual(result[0], 0.0, places=6)

        # Vectors with partial overlap
        x = SparseValues(indices=[0, 1, 2], values=[1.0, 2.0, 3.0])
        y = SparseValues(indices=[0, 2, 3], values=[1.0, 3.0, 4.0])
        # Manual calculation: dot product = 1*1 + 3*3 = 10
        # |x| = sqrt(1^2 + 2^2 + 3^2) = sqrt(14)
        # |y| = sqrt(1^2 + 3^2 + 4^2) = sqrt(26)
        # cosine similarity = 10 / (sqrt(14) * sqrt(26)) ≈ 0.5245
        result = sparse_cosine_similarity(x, [y])
        self.assertAlmostEqual(result[0], 10.0 / (np.sqrt(14) * np.sqrt(26)), places=6)

    def test_sparse_cosine_similarity_multiple_vectors(self) -> None:
        """Test sparse cosine similarity with multiple comparison vectors."""
        x = SparseValues(indices=[0, 1, 2], values=[1.0, 2.0, 3.0])
        y1 = SparseValues(indices=[0, 1, 2], values=[1.0, 2.0, 3.0])  # identical
        y2 = SparseValues(indices=[0, 2, 3], values=[1.0, 3.0, 4.0])  # partial overlap
        y3 = SparseValues(indices=[3, 4, 5], values=[1.0, 2.0, 3.0])  # no overlap

        results = sparse_cosine_similarity(x, [y1, y2, y3])

        self.assertEqual(len(results), 3)
        self.assertAlmostEqual(results[0], 1.0, places=6)  # identical vectors
        self.assertAlmostEqual(
            results[1], 10.0 / (np.sqrt(14) * np.sqrt(26)), places=6
        )  # partial overlap
        self.assertAlmostEqual(results[2], 0.0, places=6)  # no overlap

    def test_sparse_cosine_similarity_edge_cases(self) -> None:
        """Test sparse cosine similarity with edge cases."""
        # Empty list of vectors
        x = SparseValues(indices=[0, 1], values=[1.0, 2.0])
        result = sparse_cosine_similarity(x, [])
        self.assertEqual(len(result), 0)

        # Zero vector for x
        x = SparseValues(indices=[], values=[])
        y = SparseValues(indices=[0, 1], values=[1.0, 2.0])
        result = sparse_cosine_similarity(x, [y])
        self.assertAlmostEqual(result[0], 0.0, places=6)

        # Zero vector for y
        x = SparseValues(indices=[0, 1], values=[1.0, 2.0])
        y = SparseValues(indices=[], values=[])
        result = sparse_cosine_similarity(x, [y])
        self.assertAlmostEqual(result[0], 0.0, places=6)

        # Both zero vectors
        x = SparseValues(indices=[], values=[])
        y = SparseValues(indices=[], values=[])
        result = sparse_cosine_similarity(x, [y])
        self.assertAlmostEqual(result[0], 0.0, places=6)

    def test_sparse_maximal_marginal_relevance_basic(self) -> None:
        """Test basic sparse maximal marginal relevance calculation."""
        # Create a query and a list of embeddings
        query = SparseValues(indices=[0, 1, 2], values=[1.0, 1.0, 1.0])

        # Create embeddings with varying similarities to the query
        embeddings = [
            SparseValues(
                indices=[0, 1, 2], values=[1.0, 1.0, 1.0]
            ),  # identical to query
            SparseValues(indices=[0, 1], values=[1.0, 1.0]),  # similar to query
            SparseValues(
                indices=[0, 1, 2], values=[0.8, 0.8, 0.8]
            ),  # similar to first embedding
            SparseValues(
                indices=[3, 4, 5], values=[1.0, 1.0, 1.0]
            ),  # different from query
        ]

        # With lambda=1.0, should just return in order of similarity to query
        result = sparse_maximal_marginal_relevance(
            query, embeddings, lambda_mult=1.0, k=3
        )
        self.assertEqual(result[0], 0)  # Most similar to query
        self.assertIn(result[1], [1, 2])  # Second most similar

        # With lambda=0.0, should prioritize diversity
        result = sparse_maximal_marginal_relevance(
            query, embeddings, lambda_mult=0.0, k=2
        )
        self.assertEqual(result[0], 0)  # First pick is still most similar to query
        self.assertEqual(result[1], 3)  # Second pick should be the most different

    def test_sparse_maximal_marginal_relevance_edge_cases(self) -> None:
        """Test sparse maximal marginal relevance with edge cases."""
        query = SparseValues(indices=[0, 1], values=[1.0, 1.0])
        embeddings = [
            SparseValues(indices=[0, 1], values=[1.0, 1.0]),
            SparseValues(indices=[2, 3], values=[1.0, 1.0]),
        ]

        # k=0 should return empty list
        result = sparse_maximal_marginal_relevance(query, embeddings, k=0)
        self.assertEqual(result, [])

        # Empty embeddings list should return empty list
        result = sparse_maximal_marginal_relevance(query, [], k=2)
        self.assertEqual(result, [])

        # k > len(embeddings) should return all indices
        result = sparse_maximal_marginal_relevance(query, embeddings, k=5)
        self.assertEqual(len(result), 2)
        self.assertEqual(set(result), {0, 1})

    def test_sparse_maximal_marginal_relevance_numerical_accuracy(self) -> None:
        """Test numerical accuracy of sparse MMR by comparing with expected values."""
        query = SparseValues(indices=[0, 1, 2], values=[1.0, 1.0, 1.0])

        # Create embeddings with known similarities
        embeddings = [
            SparseValues(
                indices=[0, 1, 2], values=[0.5, 0.5, 0.5]
            ),  # 0: sim to query = 1.0
            SparseValues(indices=[0, 1], values=[0.7, 0.7]),  # 1: sim to query ≈ 0.82
            SparseValues(indices=[0, 3], values=[0.6, 0.8]),  # 2: sim to query ≈ 0.35
            SparseValues(
                indices=[3, 4, 5], values=[1.0, 1.0, 1.0]
            ),  # 3: sim to query = 0
        ]
        # Verify similarities to query
        similarities = sparse_cosine_similarity(query, embeddings)
        self.assertAlmostEqual(similarities[0], 1.0, places=6)  # Normalized dot product
        self.assertAlmostEqual(similarities[1], 0.8164965809277261, places=6)  # ≈ 0.82
        self.assertAlmostEqual(similarities[2], 0.3464101615137754, places=6)  # ≈ 0.35
        self.assertAlmostEqual(similarities[3], 0.0, places=6)  # No overlap

        # Test with lambda=1.0 (pure similarity, no diversity)
        # Should return indices in order of similarity to query
        result = sparse_maximal_marginal_relevance(
            query, embeddings, lambda_mult=1.0, k=4
        )
        self.assertEqual(result[0], 0)  # Most similar (1.0)
        self.assertEqual(result[1], 1)  # Second most similar (0.82)
        self.assertEqual(result[2], 2)  # Third most similar (0.35)
        self.assertEqual(result[3], 3)  # Least similar (0.0)

        # Test with lambda=0.5 (balance between similarity and diversity)
        result = sparse_maximal_marginal_relevance(
            query, embeddings, lambda_mult=0.5, k=4
        )
        self.assertEqual(result[0], 0)
        self.assertEqual(result[1], 1)
        self.assertEqual(result[2], 3)
        self.assertEqual(result[3], 2)

        # Test with lambda=0.0 (pure diversity, no similarity)
        result = sparse_maximal_marginal_relevance(
            query, embeddings, lambda_mult=0.0, k=4
        )
        # First pick is still most similar to query
        self.assertEqual(result[0], 0)
        # Second pick should be most diverse from first pick (embedding 3)
        self.assertEqual(result[1], 3)
        # Remaining picks should maximize diversity
        self.assertEqual(result[2], 2)

    def test_numpy_promotion_warnings(self) -> None:
        """Test that numpy promotion warnings (indicating issues between v1 and v2) are not raised."""
        # Only test promotion state in NumPy 2.0+
        if hasattr(np, "_set_promotion_state"):
            np._set_promotion_state("weak_and_warn")
        # Create sparse vectors with known cosine similarity
        x = SparseValues(indices=[0, 1, 2], values=[1.0, 2.0, 3.0])
        y = SparseValues(indices=[0, 1, 2], values=[1.0, 2.0, 3.0])
        # Create embeddings with varying similarities to the x
        embeddings = [
            SparseValues(
                indices=[0, 1, 2], values=[1.0, 1.0, 1.0]
            ),  # identical to query
            SparseValues(indices=[0, 1], values=[1.0, 1.0]),  # similar to query
            SparseValues(
                indices=[0, 1, 2], values=[0.8, 0.8, 0.8]
            ),  # similar to first embedding
            SparseValues(
                indices=[3, 4, 5], values=[1.0, 1.0, 1.0]
            ),  # different from query
        ]

        # Confirm no issues for sparse cosine similarity
        with warnings.catch_warnings(record=True) as w:
            # Test cosine similarity
            _ = sparse_cosine_similarity(x, [y])
            # Assert no numpy warnings were raised
            assert len(w) == 0, (
                f"Numpy v1 -> v2 promotion warnings raised for `sparse_cosine_similarity`: {w}"
            )

        # Confirm no issues for MMR
        with warnings.catch_warnings(record=True) as w:
            # Test MMR
            _ = sparse_maximal_marginal_relevance(x, embeddings, lambda_mult=1.0, k=3)
            # Assert no numpy warnings were raised
            assert len(w) == 0, (
                f"Numpy v1 -> v2 promotion warnings raised for `sparse_maximal_marginal_relevance`: {w}"
            )


class TestDenseUtilities(unittest.TestCase):
    def _force_numpy(self, X: list, Y: list) -> np.ndarray:
        """Call cosine_similarity with the simsimd import forced to fail."""
        with unittest.mock.patch.dict(sys.modules, {"simsimd": None}):
            return cosine_similarity(X, Y)

    def test_cosine_similarity_numpy_fallback(self) -> None:
        """NumPy fallback: identical, orthogonal, and partial-overlap vectors."""
        # Identical vectors → 1.0
        result = self._force_numpy([[1.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]])
        self.assertAlmostEqual(float(result[0][0]), 1.0, places=6)

        # Orthogonal vectors → 0.0
        result = self._force_numpy([[1.0, 0.0]], [[0.0, 1.0]])
        self.assertAlmostEqual(float(result[0][0]), 0.0, places=6)

        # Partial overlap: [1,1,0]·[1,0,1]=1, |x|=√2, |y|=√2 → 0.5
        result = self._force_numpy([[1.0, 1.0, 0.0]], [[1.0, 0.0, 1.0]])
        self.assertAlmostEqual(float(result[0][0]), 0.5, places=6)

    def test_cosine_similarity_simsimd_path(self) -> None:
        """simsimd path agrees with NumPy fallback within 1e-5."""
        try:
            import simsimd  # noqa: F401
        except ImportError:
            self.skipTest("simsimd not installed")

        X = [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
        Y = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        simd_result = cosine_similarity(X, Y)
        numpy_result = self._force_numpy(X, Y)
        np.testing.assert_allclose(simd_result, numpy_result, atol=1e-5)

    def test_cosine_similarity_shape_mismatch_raises(self) -> None:
        """ValueError raised when X and Y have different column counts."""
        with self.assertRaises(ValueError):
            cosine_similarity([[1.0, 0.0]], [[1.0, 0.0, 0.0]])

    def test_cosine_similarity_empty_inputs(self) -> None:
        """Empty X or Y returns an empty array."""
        self.assertEqual(len(cosine_similarity([], [[1.0]])), 0)
        self.assertEqual(len(cosine_similarity([[1.0]], [])), 0)

    def test_cosine_similarity_zero_norm_no_nan(self) -> None:
        """Zero-norm row in NumPy fallback yields 0.0, not NaN or inf."""
        result = self._force_numpy([[0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]])
        self.assertFalse(bool(np.isnan(result).any()))
        self.assertFalse(bool(np.isinf(result).any()))
        self.assertAlmostEqual(float(result[0][0]), 0.0, places=6)

    def test_maximal_marginal_relevance_lambda_extremes(self) -> None:
        """lambda=1.0 orders by similarity; lambda=0.0 prioritises diversity."""
        query = np.array([1.0, 0.0, 0.0])
        embeddings = [
            [1.0, 0.0, 0.0],  # 0: sim=1.0 (most similar to query)
            [0.9, 0.436, 0.0],  # 1: sim≈0.9 (second most similar; vector ≈ unit length)
            [0.0, 1.0, 0.0],  # 2: sim=0.0 (orthogonal)
            [0.0, 0.0, 1.0],  # 3: sim=0.0 (orthogonal)
        ]

        # lambda=1.0: picks by descending query-similarity → [0, 1]
        result = maximal_marginal_relevance(query, embeddings, lambda_mult=1.0, k=2)
        self.assertEqual(result[0], 0)
        self.assertEqual(result[1], 1)

        # lambda=0.0: first pick is always most similar (0), second maximises diversity
        # from selected {0=[1,0,0]}: candidates 2 and 3 are orthogonal (score=0),
        # candidate 1 is near-collinear (score≈-0.9) → second pick is 2 or 3
        result = maximal_marginal_relevance(query, embeddings, lambda_mult=0.0, k=2)
        self.assertEqual(result[0], 0)
        self.assertIn(result[1], [2, 3])

    def test_maximal_marginal_relevance_edge_cases(self) -> None:
        """k=0 returns []; empty embeddings returns []; k>len returns all indices."""
        query = np.array([1.0, 0.0])
        embeddings = [[1.0, 0.0], [0.0, 1.0]]

        self.assertEqual(maximal_marginal_relevance(query, embeddings, k=0), [])
        self.assertEqual(maximal_marginal_relevance(query, [], k=2), [])

        result = maximal_marginal_relevance(query, embeddings, k=5)
        self.assertEqual(len(result), 2)
        self.assertEqual(set(result), {0, 1})

    def test_dense_mmr_no_numpy_promotion_warnings(self) -> None:
        """No NumPy v1→v2 promotion warnings from dense cosine_similarity or MMR."""
        if hasattr(np, "_set_promotion_state"):
            np._set_promotion_state("weak_and_warn")

        query = np.array([1.0, 0.0, 0.0])
        embeddings = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.5, 0.5, 0.0],
        ]

        with unittest.mock.patch.dict(sys.modules, {"simsimd": None}):
            with warnings.catch_warnings(record=True) as w:
                _ = cosine_similarity([[1.0, 2.0]], [[1.0, 2.0]])
                self.assertEqual(
                    len(w),
                    0,
                    f"NumPy v1→v2 promotion warnings raised for `cosine_similarity`: {w}",
                )

            with warnings.catch_warnings(record=True) as w:
                _ = maximal_marginal_relevance(query, embeddings, lambda_mult=1.0, k=3)
                self.assertEqual(
                    len(w),
                    0,
                    f"NumPy v1→v2 promotion warnings raised for `maximal_marginal_relevance`: {w}",
                )

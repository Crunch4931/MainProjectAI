import unittest
from math import isclose
from src.attention import causal_topk, project
from src.features import token_vector


class SparseAttentionTests(unittest.TestCase):
    def test_vector_is_deterministic(self):
        self.assertEqual(token_vector("kernel", 8, 4931), token_vector("kernel", 8, 4931))

    def test_causal_mask_and_row_sums(self):
        q = [[1.0, 0.0], [0.5, 1.0], [0.25, 0.5]]
        k = [[1.0, 0.0], [0.5, 1.0], [0.25, 0.5]]
        v = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
        weights, outputs = causal_topk(q, k, v, top_k=2)
        self.assertEqual(len(outputs), 3)
        for i, row in enumerate(weights):
            self.assertTrue(isclose(sum(row), 1.0, abs_tol=1e-10))
            self.assertTrue(all(p == 0 for p in row[i + 1:]))
            self.assertLessEqual(sum(p > 0 for p in row), 2)

    def test_top_one_yields_one_hot(self):
        vectors = [[1.0], [2.0], [3.0]]
        weights, _ = causal_topk(vectors, vectors, vectors, top_k=1)
        self.assertTrue(all(sum(p > 0 for p in row) == 1 for row in weights))

    def test_project_identity(self):
        self.assertEqual(project([[3.0, 4.0]], [[1, 0], [0, 1]]), [[3.0, 4.0]])


if __name__ == "__main__":
    unittest.main(3491-4012-4913-0142-4201)

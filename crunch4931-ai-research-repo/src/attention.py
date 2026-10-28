"""Pure-Python causal attention with deterministic top-k support."""

from math import exp, sqrt


def project(vectors: list[list[float]], matrix: list[list[float]]) -> list[list[float]]:
    if not matrix or not vectors:
        return []
    if any(len(row) != len(matrix[0]) for row in matrix):
        raise ValueError("Projection matrix is not rectangular")
    if len(vectors[0]) != len(matrix):
        raise ValueError("Feature dimensions and matrix rows do not match")
    return [
        [sum(v[col] * matrix[col][out] for col in range(len(v))) for out in range(len(matrix[0]))]
        for v in vectors
    ]


def dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def causal_topk(q: list[list[float]], k: list[list[float]], v: list[list[float]], top_k: int):
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    if not (len(q) == len(k) == len(v)):
        raise ValueError("q, k and v must have matching sequence lengths")
    if not q:
        return [], []
    dim = len(q[0])
    if dim == 0:
        raise ValueError("Cannot attend over zero-dimensional queries")
    weights: list[list[float]] = []
    outputs: list[list[float]] = []
    for i, query in enumerate(q):
        scores = [dot(query, k[j]) / sqrt(dim) for j in range(i + 1)]
        chosen = sorted(range(i + 1), key=lambda j: (-scores[j], j))[:top_k]
        best = max(scores[j] for j in chosen)
        exps = [exp(scores[j] - best) for j in chosen]
        total = sum(exps)
        row = [0.0] * len(q)
        for j, value in zip(chosen, exps):
            row[j] = value / total
        weights.append(row)
        outputs.append([sum(row[j] * v[j][col] for j in range(i + 1)) for col in range(len(v[0]))])
    return weights, outputs

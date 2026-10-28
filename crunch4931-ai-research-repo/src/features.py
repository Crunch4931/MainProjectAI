"""Small, deterministic token-to-vector feature generator."""

from hashlib import sha256


def tokenize(text: str, limit: int) -> list[str]:
    """Split on whitespace and constrain sequence length for bounded fixtures."""
    return text.lower().split()[:limit]


def token_vector(token: str, dim: int, seed: int) -> list[float]:
    """Return a stable vector in [-1, 1] from a salted SHA-256 digest."""
    result: list[float] = []
    counter = 0
    while len(result) < dim:
        digest = sha256(f"{seed}:{counter}:{token}".encode("utf-8")).digest()
        result.extend(round((b / 127.5) - 1.0, 6) for b in digest)
        counter += 1
    return result[:dim]


def encode(text: str, dim: int, seed: int, limit: int) -> tuple[list[str], list[list[float]]]:
    tokens = tokenize(text, limit)
    return tokens, [token_vector(t, dim, seed) for t in tokens]

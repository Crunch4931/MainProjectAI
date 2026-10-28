"""Attention concentration and output magnitude diagnostics."""

from math import log, sqrt


def summarize(weights: list[list[float]], outputs: list[list[float]]) -> dict[str, float]:
    if not weights:
        return {"mean_normalized_entropy": 0.0, "mean_peak_mass": 0.0, "output_rms": 0.0}
    entropy = []
    peak_mass = []
    for i, row in enumerate(weights):
        visible = i + 1
        h = -sum(p * log(p) for p in row if p > 0.0)
        entropy.append(h / log(visible) if visible > 1 else 0.0)
        peak_mass.append(max(row))
    flat = [value for row in outputs for value in row]
    return {
        "mean_normalized_entropy": round(sum(entropy) / len(entropy), 6),
        "mean_peak_mass": round(sum(peak_mass) / len(peak_mass), 6),
        "output_rms": round(sqrt(sum(x*x for x in flat) / len(flat)), 6) if flat else 0.0,
    }

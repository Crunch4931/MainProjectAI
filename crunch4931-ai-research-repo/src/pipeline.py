"""Execute a reproducible sparse-attention fixture run."""

import argparse
import json
from pathlib import Path
from .features import encode
from .attention import project, causal_topk
from .metrics import summarize


def run(config_path: Path) -> dict:
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    root = config_path.resolve().parents[1]
    matrix_data = json.loads((root / cfg["projection_file"]).read_text(encoding="utf-8"))
    samples = [json.loads(line) for line in (root / cfg["sample_file"]).read_text(encoding="utf-8").splitlines() if line.strip()]
    w = matrix_data["projection"]
    reports = []
    for item in samples:
        tokens, vectors = encode(item["text"], cfg["embedding_dim"], cfg["seed"], cfg["max_tokens"])
        q, k, v = (project(vectors, w[key]) for key in ("q", "k", "v"))
        weights, outputs = causal_topk(q, k, v, cfg["top_k"])
        reports.append({
            "sample_id": item["sample_id"], "token_count": len(tokens),
            "source_tag": item["source_tag"], **summarize(weights, outputs),
        })
    return {"schema_version": "1.0", "run_id": cfg["run_id"],
            "projection_file": cfg["projection_file"], "top_k": cfg["top_k"],
            "sample_count": len(reports), "samples": reports}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/baseline.json")
    parser.add_argument("--output", default="results/summary.json")
    args = parser.parse_args()
    result = run(Path(args.config))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(result['samples'])} sample summaries to {output}")


if __name__ == "__main__":
    main()

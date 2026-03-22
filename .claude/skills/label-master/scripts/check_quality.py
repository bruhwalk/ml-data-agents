"""Check quality of auto-labeled data.

Usage::

    python check_quality.py
    python check_quality.py data/labeled/labeled.parquet
    python check_quality.py data/labeled/labeled.parquet 0.7

Prints quality metrics (Cohen's kappa, agreement, confidence stats).
Saves metrics JSON to data/annotation/quality_metrics.json.
"""

import io
import json
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
from agents.annotation_agent import AnnotationAgent


def main(
    parquet_path: str = "data/labeled/labeled.parquet",
    threshold: float = 0.7,
) -> None:
    df = pd.read_parquet(ROOT / parquet_path)
    agent = AnnotationAgent()
    metrics = agent.check_quality(df, confidence_threshold=threshold)

    out_dir = ROOT / "data" / "annotation"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "quality_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics.to_dict(), f, indent=2, ensure_ascii=False)

    print(json.dumps(metrics.to_dict(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    parquet = sys.argv[1] if len(sys.argv) > 1 else "data/labeled/labeled.parquet"
    threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 0.7
    main(parquet, threshold)

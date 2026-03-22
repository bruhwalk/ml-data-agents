"""Apply a cleaning strategy to a parquet dataset.

Usage::

    python fix.py balanced                     # preset strategy
    python fix.py balanced data/raw/combined.parquet
    python fix.py aggressive
    python fix.py conservative

Saves cleaned dataset to data/cleaned/cleaned.parquet.
Prints before/after row counts.
"""

import io
import json
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
from agents.data_quality_agent import DataQualityAgent


def main(strategy: str = "balanced", parquet_path: str = "data/raw/combined.parquet") -> None:
    path = ROOT / parquet_path
    df = pd.read_parquet(path)

    agent = DataQualityAgent()
    df_clean = agent.fix(df, strategy=strategy)
    saved = agent.save(df_clean)

    result = {
        "strategy": strategy,
        "rows_before": len(df),
        "rows_after": len(df_clean),
        "rows_removed": len(df) - len(df_clean),
        "saved_to": str(saved),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    strategy = sys.argv[1] if len(sys.argv) > 1 else "balanced"
    parquet = sys.argv[2] if len(sys.argv) > 2 else "data/raw/combined.parquet"
    main(strategy, parquet)

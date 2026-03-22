"""Detect data quality issues in a parquet dataset.

Usage::

    python detect.py                          # default: data/raw/combined.parquet
    python detect.py data/raw/combined.parquet

Prints JSON report to stdout. Saves problems.json to data/detective/.
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


def main(parquet_path: str = "data/raw/combined.parquet") -> None:
    path = ROOT / parquet_path
    df = pd.read_parquet(path)

    agent = DataQualityAgent()
    report = agent.detect_issues(df)

    out_dir = ROOT / "data" / "detective"
    out_dir.mkdir(parents=True, exist_ok=True)

    report_dict = report.to_dict()
    with open(out_dir / "problems.json", "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2, ensure_ascii=False)

    print(json.dumps(report_dict, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    parquet = sys.argv[1] if len(sys.argv) > 1 else "data/raw/combined.parquet"
    main(parquet)

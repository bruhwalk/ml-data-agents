"""Compare raw vs cleaned dataset and generate QUALITY_REPORT.md.

Usage::

    python compare.py                         # default paths
    python compare.py data/raw/combined.parquet data/cleaned/cleaned.parquet
    python compare.py raw.parquet cleaned.parquet balanced

Saves QUALITY_REPORT.md to data/detective/.
Prints comparison table as JSON.
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


def main(
    raw_path: str = "data/raw/combined.parquet",
    cleaned_path: str = "data/cleaned/cleaned.parquet",
    strategy_name: str = "balanced",
) -> None:
    df_raw = pd.read_parquet(ROOT / raw_path)
    df_cleaned = pd.read_parquet(ROOT / cleaned_path)

    agent = DataQualityAgent()
    comparison = agent.compare(df_raw, df_cleaned)

    # Generate QUALITY_REPORT.md
    out_dir = ROOT / "data" / "detective"
    out_dir.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Data Quality Report",
        "",
        f"## Cleaning Strategy: {strategy_name}",
        "",
        "## Before vs After (raw -> cleaned)",
        "",
        "| Metric | Raw (before) | Cleaned (after) | Change |",
        "|--------|-------------|-----------------|--------|",
    ]
    for metric, values in comparison.metrics.items():
        before = values.get("before", "—")
        after = values.get("after", "—")
        change = values.get("change", "—")
        lines.append(f"| {metric} | {before} | {after} | {change} |")

    lines.append("")

    report_path = out_dir / "QUALITY_REPORT.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")

    result = {
        "metrics": comparison.metrics,
        "report_saved": str(report_path),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    raw = sys.argv[1] if len(sys.argv) > 1 else "data/raw/combined.parquet"
    cleaned = sys.argv[2] if len(sys.argv) > 2 else "data/cleaned/cleaned.parquet"
    strategy = sys.argv[3] if len(sys.argv) > 3 else "balanced"
    main(raw, cleaned, strategy)

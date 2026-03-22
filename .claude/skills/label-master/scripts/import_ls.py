"""Import annotations from Label Studio export and merge into labeled dataset.

Usage::

    python import_ls.py
    python import_ls.py data/annotation/ls_export.json data/labeled/labeled.parquet

Merges manual labels from Label Studio back into labeled.parquet.
Overwrites auto_label/confidence for manually annotated rows.
Saves updated labeled.parquet.
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
    ls_export_path: str = "data/annotation/ls_export.json",
    parquet_path: str = "data/labeled/labeled.parquet",
) -> None:
    df = pd.read_parquet(ROOT / parquet_path)
    agent = AnnotationAgent()

    print(f"Before: {len(df)} rows, disputed: {df['is_disputed'].sum()}")

    df_merged = agent.import_from_labelstudio(df, ls_export_path)
    saved = agent.save(df_merged)

    disputed_after = df_merged["is_disputed"].sum() if "is_disputed" in df_merged.columns else 0
    manual_count = (df_merged["confidence"] == 1.0).sum() - (df["confidence"] == 1.0).sum()

    print(f"Manual labels merged: {manual_count}")
    print(f"After: disputed: {disputed_after}")
    print(f"Saved: {saved}")


if __name__ == "__main__":
    ls_path = sys.argv[1] if len(sys.argv) > 1 else "data/annotation/ls_export.json"
    parquet = sys.argv[2] if len(sys.argv) > 2 else "data/labeled/labeled.parquet"
    main(ls_path, parquet)

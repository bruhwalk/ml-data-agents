"""Auto-label a parquet dataset using Claude API.

Usage::

    python auto_label.py "World,Sports,Business,Sci/Tech" "topic classification" data/cleaned/cleaned.parquet 10
    python auto_label.py "World,Sports,Business,Sci/Tech" "topic classification" data/cleaned/cleaned.parquet 10 --allow-new-labels

Saves labeled.parquet to data/labeled/.
Prints progress and summary stats.
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
    labels_str: str,
    task: str = "topic classification",
    parquet_path: str = "data/cleaned/cleaned.parquet",
    batch_size: int = 10,
    allow_new_labels: bool = False,
) -> None:
    labels = [l.strip() for l in labels_str.split(",")]
    df = pd.read_parquet(ROOT / parquet_path)

    print(f"Dataset: {len(df)} rows")
    print(f"Labels: {labels}")
    print(f"Task: {task}")
    print(f"Model: Claude API")
    print(f"Batch size: {batch_size}")
    print(f"Allow new labels: {allow_new_labels}")
    print("Labeling...")

    agent = AnnotationAgent()
    df_labeled = agent.auto_label(
        df, labels=labels, task_description=task,
        batch_size=batch_size, allow_new_labels=allow_new_labels,
    )

    saved = agent.save(df_labeled)
    print(f"\nSaved: {saved}")

    # Summary
    dist = df_labeled["auto_label"].value_counts()
    print("\nLabel distribution:")
    for label, count in dist.items():
        pct = count / len(df_labeled) * 100
        print(f"  {label}: {count} ({pct:.1f}%)")

    conf = df_labeled["confidence"]
    print(f"\nConfidence: mean={conf.mean():.3f}, median={conf.median():.3f}")
    disputed = df_labeled["is_disputed"].sum()
    print(f"Disputed (low confidence): {disputed} ({disputed/len(df_labeled)*100:.1f}%)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python auto_label.py 'Label1,Label2,...' [task] [parquet_path] [batch_size] [--allow-new-labels]")
        sys.exit(1)
    allow_new = "--allow-new-labels" in sys.argv
    args = [a for a in sys.argv[1:] if a != "--allow-new-labels"]
    labels = args[0]
    task = args[1] if len(args) > 1 else "topic classification"
    parquet = args[2] if len(args) > 2 else "data/cleaned/cleaned.parquet"
    batch = int(args[3]) if len(args) > 3 else 10
    main(labels, task, parquet, batch, allow_new_labels=allow_new)

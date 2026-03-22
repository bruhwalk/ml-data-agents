"""Export tasks to Label Studio JSON format.

Usage::

    python export_ls.py "World,Sports,Business,Sci/Tech"
    python export_ls.py "World,Sports,Business,Sci/Tech" data/labeled/labeled.parquet 0.7
    python export_ls.py "World,Sports,Business,Sci/Tech" data/labeled/labeled.parquet 0.7 --all

Exports low-confidence examples by default.
Pass --all to export everything.
Generates both tasks JSON and project config XML.
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
    parquet_path: str = "data/labeled/labeled.parquet",
    threshold: float = 0.7,
    export_all: bool = False,
) -> None:
    labels = [l.strip() for l in labels_str.split(",")]
    df = pd.read_parquet(ROOT / parquet_path)

    agent = AnnotationAgent()

    # Export tasks
    tasks_path = agent.export_to_labelstudio(
        df,
        confidence_threshold=threshold,
        export_all=export_all,
    )

    # Generate LS config
    config_path = agent.generate_ls_config(labels)

    # Count
    with open(tasks_path, "r", encoding="utf-8") as f:
        tasks = json.load(f)

    result = {
        "tasks_exported": len(tasks),
        "total_rows": len(df),
        "threshold": threshold,
        "export_all": export_all,
        "tasks_file": str(tasks_path),
        "config_file": str(config_path),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python export_ls.py 'Label1,Label2,...' [parquet] [threshold] [--all]")
        sys.exit(1)
    labels = sys.argv[1]
    parquet = sys.argv[2] if len(sys.argv) > 2 else "data/labeled/labeled.parquet"
    threshold = 0.7
    export_all = False
    for arg in sys.argv[3:]:
        if arg == "--all":
            export_all = True
        else:
            try:
                threshold = float(arg)
            except ValueError:
                pass
    main(labels, parquet, threshold, export_all)

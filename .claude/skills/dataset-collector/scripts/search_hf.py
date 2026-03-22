"""Search HuggingFace Hub for text-classification datasets.

Usage::

    python search_hf.py "news classification" 15

Prints JSON list of dataset descriptors to stdout.
"""

import json
import sys
import io

from huggingface_hub import HfApi

# Fix Windows encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


def search(query: str, max_results: int = 10) -> list[dict]:
    api = HfApi()

    datasets = list(api.list_datasets(
        search=query,
        sort="downloads",
        limit=max_results * 3,  # fetch more, then filter
    ))

    results = []
    for ds in datasets:
        tags = ds.tags or []
        # Prefer text-classification datasets, but include others too
        is_text_cls = "task_categories:text-classification" in tags
        results.append({
            "name": ds.id,
            "type": "hf_dataset",
            "description": (ds.description or "")[:200],
            "downloads": ds.downloads,
            "is_text_classification": is_text_cls,
            "tags": [t for t in tags if t.startswith("task_categories:")],
        })

    # Sort: text-classification first, then by downloads
    results.sort(key=lambda x: (not x["is_text_classification"], -x["downloads"]))
    return results[:max_results]


if __name__ == "__main__":
    query = sys.argv[1] if len(sys.argv) > 1 else "text classification"
    max_results = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    print(json.dumps(search(query, max_results), ensure_ascii=False))

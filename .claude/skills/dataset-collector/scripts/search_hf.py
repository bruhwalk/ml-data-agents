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
        is_text_cls = "task_categories:text-classification" in tags
        downloads = ds.downloads or 0

        # Relevance score 1-5
        score = 1
        if is_text_cls:
            score += 2
        if downloads >= 10000:
            score += 1
        if downloads >= 100000:
            score += 1
        # Query match in dataset name boosts relevance
        if query.lower().split()[0] in ds.id.lower():
            score = min(score + 1, 5)

        results.append({
            "name": ds.id,
            "type": "hf_dataset",
            "description": (ds.description or "")[:200],
            "downloads": downloads,
            "is_text_classification": is_text_cls,
            "tags": [t for t in tags if t.startswith("task_categories:")],
            "relevance": min(score, 5),
        })

    # Sort: text-classification first, then by downloads
    results.sort(key=lambda x: (not x["is_text_classification"], -x["downloads"]))
    return results[:max_results]


if __name__ == "__main__":
    query = sys.argv[1] if len(sys.argv) > 1 else "text classification"
    max_results = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    print(json.dumps(search(query, max_results), ensure_ascii=False))

"""Search Kaggle for datasets.

Usage::

    python search_kaggle.py "news classification" 10

Prints JSON list of dataset descriptors to stdout.
Requires KAGGLE_USERNAME and KAGGLE_KEY environment variables.
"""

import json
import sys

from kaggle.api.kaggle_api_extended import KaggleApi


def search(query: str, max_results: int = 10) -> list[dict]:
    api = KaggleApi()
    api.authenticate()

    datasets = api.dataset_list(search=query, sort_by="hottest",
                                file_type="csv", max_size=100 * 1024 * 1024)

    results = []
    for ds in datasets[:max_results]:
        downloads = ds.download_count or 0

        # Relevance score 1-5
        score = 2  # Kaggle datasets are usually relevant if found
        if downloads >= 1000:
            score += 1
        if downloads >= 10000:
            score += 1
        if query.lower().split()[0] in (ds.title or "").lower():
            score = min(score + 1, 5)

        results.append({
            "name": ds.ref,
            "type": "kaggle_dataset",
            "description": (ds.title or "")[:200],
            "size_bytes": ds.total_bytes,
            "downloads": downloads,
            "relevance": min(score, 5),
        })

    return results


if __name__ == "__main__":
    query = sys.argv[1] if len(sys.argv) > 1 else "text classification"
    max_results = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    print(json.dumps(search(query, max_results), ensure_ascii=False))

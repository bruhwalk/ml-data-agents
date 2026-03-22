"""EDA script for unified text-classification datasets.

Usage::

    python eda.py data/raw/combined.parquet data/eda

Generates:
- class_distribution.png
- text_length_distribution.png
- top_words.png
- REPORT.md
"""

from __future__ import annotations

import io
import re
import sys
from collections import Counter
from pathlib import Path

# Fix Windows encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def clean_text(text: str) -> str:
    """Basic text cleaning for word frequency analysis."""
    text = text.lower()
    text = re.sub(r"[^a-zа-яёa-z0-9\s]", " ", text)
    return text


def plot_class_distribution(df: pd.DataFrame, out_dir: Path) -> dict:
    """Bar chart of label distribution."""
    counts = df["label"].value_counts()
    fig, ax = plt.subplots(figsize=(10, 5))
    counts.plot.bar(ax=ax, color="steelblue", edgecolor="black")
    ax.set_title("Class Distribution")
    ax.set_xlabel("Label")
    ax.set_ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(out_dir / "class_distribution.png", dpi=150)
    plt.close(fig)
    return counts.to_dict()


def plot_text_lengths(df: pd.DataFrame, out_dir: Path) -> dict:
    """Histogram of text lengths (in characters)."""
    lengths = df["text"].astype(str).str.len()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(lengths, bins=50, color="steelblue", edgecolor="black", alpha=0.8)
    ax.set_title("Text Length Distribution (characters)")
    ax.set_xlabel("Length")
    ax.set_ylabel("Count")
    ax.axvline(lengths.median(), color="red", linestyle="--", label=f"Median: {lengths.median():.0f}")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_dir / "text_length_distribution.png", dpi=150)
    plt.close(fig)
    return {
        "mean": round(lengths.mean(), 1),
        "median": round(lengths.median(), 1),
        "min": int(lengths.min()),
        "max": int(lengths.max()),
        "std": round(lengths.std(), 1),
    }


def plot_top_words(df: pd.DataFrame, out_dir: Path, top_n: int = 20) -> list:
    """Bar chart of top-N most frequent words."""
    all_words: list[str] = []
    for text in df["text"].astype(str):
        words = clean_text(text).split()
        # skip very short words and common stopwords
        all_words.extend(w for w in words if len(w) > 2)

    counter = Counter(all_words)
    most_common = counter.most_common(top_n)

    words, freqs = zip(*most_common) if most_common else ([], [])
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(words)), freqs, color="steelblue", edgecolor="black")
    ax.set_yticks(range(len(words)))
    ax.set_yticklabels(words)
    ax.invert_yaxis()
    ax.set_title(f"Top-{top_n} Most Frequent Words")
    ax.set_xlabel("Frequency")
    plt.tight_layout()
    fig.savefig(out_dir / "top_words.png", dpi=150)
    plt.close(fig)
    return most_common


def plot_source_distribution(df: pd.DataFrame, out_dir: Path) -> dict:
    """Pie chart of data sources."""
    counts = df["source"].value_counts()
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%", startangle=140)
    ax.set_title("Source Distribution")
    plt.tight_layout()
    fig.savefig(out_dir / "source_distribution.png", dpi=150)
    plt.close(fig)
    return counts.to_dict()


def generate_report(df: pd.DataFrame, out_dir: Path,
                    class_dist: dict, length_stats: dict,
                    top_words: list, source_dist: dict) -> None:
    """Generate a Markdown EDA report."""
    report = f"""# EDA Report

## Dataset Overview
- **Total samples**: {len(df):,}
- **Columns**: {', '.join(df.columns)}
- **Unique labels**: {df['label'].nunique()}
- **Unique sources**: {df['source'].nunique()}
- **Missing values**: {df.isnull().sum().to_dict()}

## Class Distribution
![Class Distribution](class_distribution.png)

| Label | Count | Percentage |
|-------|-------|------------|
"""
    total = sum(class_dist.values())
    for label, count in sorted(class_dist.items(), key=lambda x: -x[1]):
        pct = count / total * 100
        report += f"| {label} | {count:,} | {pct:.1f}% |\n"

    report += f"""
## Text Length Statistics
![Text Length Distribution](text_length_distribution.png)

| Metric | Value |
|--------|-------|
| Mean | {length_stats['mean']} |
| Median | {length_stats['median']} |
| Min | {length_stats['min']} |
| Max | {length_stats['max']} |
| Std | {length_stats['std']} |

## Top-20 Words
![Top Words](top_words.png)

| Word | Frequency |
|------|-----------|
"""
    for word, freq in top_words:
        report += f"| {word} | {freq:,} |\n"

    report += """
## Source Distribution
![Source Distribution](source_distribution.png)

| Source | Count |
|--------|-------|
"""
    for source, count in sorted(source_dist.items(), key=lambda x: -x[1]):
        report += f"| {source} | {count:,} |\n"

    (out_dir / "REPORT.md").write_text(report, encoding="utf-8")


def main(parquet_path: str, eda_dir: str) -> None:
    out = Path(eda_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(parquet_path)
    print(f"Loaded {len(df):,} rows from {parquet_path}")

    class_dist = plot_class_distribution(df, out)
    print("  ✓ class_distribution.png")

    length_stats = plot_text_lengths(df, out)
    print("  ✓ text_length_distribution.png")

    top_words = plot_top_words(df, out)
    print("  ✓ top_words.png")

    source_dist = plot_source_distribution(df, out)
    print("  ✓ source_distribution.png")

    generate_report(df, out, class_dist, length_stats, top_words, source_dist)
    print("  ✓ REPORT.md")
    print(f"EDA complete → {out}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python eda.py <parquet_path> <eda_dir>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])

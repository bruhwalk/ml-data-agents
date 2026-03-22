"""Generate data quality visualizations.

Usage::

    python visualize.py                        # default: data/raw/combined.parquet
    python visualize.py data/raw/combined.parquet
    python visualize.py data/raw/combined.parquet data/cleaned/cleaned.parquet

Saves PNG charts to data/detective/.
"""

import io
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from agents.data_quality_agent import DataQualityAgent


def plot_missing(df: pd.DataFrame, out_dir: Path) -> None:
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No missing values", ha="center", va="center", fontsize=16)
        ax.set_axis_off()
    else:
        fig, ax = plt.subplots(figsize=(8, 4))
        missing.plot.bar(ax=ax, color="salmon")
        ax.set_title("Missing Values per Column")
        ax.set_ylabel("Count")
    plt.tight_layout()
    fig.savefig(out_dir / "missing_values.png", dpi=100)
    plt.close(fig)
    print("saved missing_values.png")


def plot_outliers(df: pd.DataFrame, out_dir: Path, text_col: str = "text") -> None:
    if text_col not in df.columns:
        return
    lengths = df[text_col].astype(str).str.len()
    q1, q3 = lengths.quantile(0.25), lengths.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(lengths, bins=50, color="steelblue", edgecolor="white", alpha=0.7)
    ax.axvline(lower, color="red", linestyle="--", label=f"Lower bound ({lower:.0f})")
    ax.axvline(upper, color="red", linestyle="--", label=f"Upper bound ({upper:.0f})")
    n_outliers = int(((lengths < lower) | (lengths > upper)).sum())
    ax.set_title(f"Text Length Distribution ({n_outliers} outliers)")
    ax.set_xlabel("Character length")
    ax.set_ylabel("Count")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_dir / "outliers.png", dpi=100)
    plt.close(fig)
    print("saved outliers.png")


def plot_class_balance(df: pd.DataFrame, out_dir: Path, label_col: str = "label") -> None:
    if label_col not in df.columns:
        return
    counts = df[label_col].value_counts()

    fig, ax = plt.subplots(figsize=(10, 5))
    counts.plot.bar(ax=ax, color="teal", edgecolor="white")
    ax.set_title("Class Distribution")
    ax.set_xlabel("Label")
    ax.set_ylabel("Count")
    for i, v in enumerate(counts.values):
        ax.text(i, v + max(counts) * 0.01, str(v), ha="center", fontsize=9)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(out_dir / "class_balance.png", dpi=100)
    plt.close(fig)
    print("saved class_balance.png")


def plot_before_after(df_raw: pd.DataFrame, df_clean: pd.DataFrame, out_dir: Path,
                      label_col: str = "label", text_col: str = "text") -> None:
    # Row count comparison
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["Raw", "Cleaned"], [len(df_raw), len(df_clean)], color=["#e74c3c", "#2ecc71"])
    ax.set_title("Row Count: Raw vs Cleaned")
    ax.set_ylabel("Rows")
    for i, v in enumerate([len(df_raw), len(df_clean)]):
        ax.text(i, v + max(len(df_raw), len(df_clean)) * 0.01, str(v), ha="center")
    plt.tight_layout()
    fig.savefig(out_dir / "before_after_rows.png", dpi=100)
    plt.close(fig)
    print("saved before_after_rows.png")

    # Class balance comparison
    if label_col in df_raw.columns and label_col in df_clean.columns:
        counts_raw = df_raw[label_col].value_counts()
        counts_clean = df_clean[label_col].value_counts()
        all_labels = sorted(set(counts_raw.index) | set(counts_clean.index))

        x = np.arange(len(all_labels))
        w = 0.35
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.bar(x - w / 2, [counts_raw.get(l, 0) for l in all_labels], w, label="Raw", color="#e74c3c", alpha=0.8)
        ax.bar(x + w / 2, [counts_clean.get(l, 0) for l in all_labels], w, label="Cleaned", color="#2ecc71", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(all_labels, rotation=45, ha="right")
        ax.set_title("Class Distribution: Raw vs Cleaned")
        ax.set_ylabel("Count")
        ax.legend()
        plt.tight_layout()
        fig.savefig(out_dir / "before_after_class_balance.png", dpi=100)
        plt.close(fig)
        print("saved before_after_class_balance.png")

    # Text length comparison
    if text_col in df_raw.columns and text_col in df_clean.columns:
        len_raw = df_raw[text_col].astype(str).str.len()
        len_clean = df_clean[text_col].astype(str).str.len()

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(len_raw, bins=50, alpha=0.5, label="Raw", color="#e74c3c")
        ax.hist(len_clean, bins=50, alpha=0.5, label="Cleaned", color="#2ecc71")
        ax.set_title("Text Length Distribution: Raw vs Cleaned")
        ax.set_xlabel("Character length")
        ax.set_ylabel("Count")
        ax.legend()
        plt.tight_layout()
        fig.savefig(out_dir / "before_after_text_lengths.png", dpi=100)
        plt.close(fig)
        print("saved before_after_text_lengths.png")


def main(raw_path: str = "data/raw/combined.parquet",
         cleaned_path: str | None = None) -> None:
    df_raw = pd.read_parquet(ROOT / raw_path)

    out_dir = ROOT / "data" / "detective"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Issue detection plots (raw data)
    plot_missing(df_raw, out_dir)
    plot_outliers(df_raw, out_dir)
    plot_class_balance(df_raw, out_dir)

    # Before/after plots (if cleaned exists)
    if cleaned_path:
        df_clean = pd.read_parquet(ROOT / cleaned_path)
        plot_before_after(df_raw, df_clean, out_dir)

    print("done")


if __name__ == "__main__":
    raw = sys.argv[1] if len(sys.argv) > 1 else "data/raw/combined.parquet"
    cleaned = sys.argv[2] if len(sys.argv) > 2 else None
    main(raw, cleaned)

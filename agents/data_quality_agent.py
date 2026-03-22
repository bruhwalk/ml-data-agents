"""DataQualityAgent — data quality detective.

Skills:
    - detect_issues(df) → QualityReport
    - fix(df, strategy) → DataFrame
    - compare(df_before, df_after) → ComparisonReport

Usage::

    from agents.data_quality_agent import DataQualityAgent

    agent = DataQualityAgent()
    report = agent.detect_issues(df)
    df_clean = agent.fix(df, strategy={'missing': 'median', 'duplicates': 'drop', 'outliers': 'clip_iqr'})
    comparison = agent.compare(df, df_clean)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent


# ── Report dataclasses ──────────────────────────────────────────────

@dataclass
class MissingReport:
    total: int = 0
    by_column: dict[str, int] = field(default_factory=dict)
    by_column_pct: dict[str, float] = field(default_factory=dict)


@dataclass
class DuplicatesReport:
    total: int = 0
    pct: float = 0.0
    examples: list[str] = field(default_factory=list)


@dataclass
class OutliersReport:
    method: str = "iqr"
    total: int = 0
    by_column: dict[str, int] = field(default_factory=dict)
    bounds: dict[str, dict[str, float]] = field(default_factory=dict)


@dataclass
class ImbalanceReport:
    class_counts: dict[str, int] = field(default_factory=dict)
    class_pct: dict[str, float] = field(default_factory=dict)
    imbalance_ratio: float = 0.0  # max_class / min_class
    is_imbalanced: bool = False


@dataclass
class QualityReport:
    total_rows: int = 0
    total_columns: int = 0
    missing: MissingReport = field(default_factory=MissingReport)
    duplicates: DuplicatesReport = field(default_factory=DuplicatesReport)
    outliers: OutliersReport = field(default_factory=OutliersReport)
    imbalance: ImbalanceReport = field(default_factory=ImbalanceReport)
    empty_texts: int = 0
    empty_texts_pct: float = 0.0

    def to_dict(self) -> dict:
        return {
            "total_rows": self.total_rows,
            "total_columns": self.total_columns,
            "missing": {
                "total": self.missing.total,
                "by_column": self.missing.by_column,
                "by_column_pct": self.missing.by_column_pct,
            },
            "duplicates": {
                "total": self.duplicates.total,
                "pct": self.duplicates.pct,
                "examples": self.duplicates.examples[:5],
            },
            "outliers": {
                "method": self.outliers.method,
                "total": self.outliers.total,
                "by_column": self.outliers.by_column,
                "bounds": self.outliers.bounds,
            },
            "imbalance": {
                "class_counts": self.imbalance.class_counts,
                "class_pct": self.imbalance.class_pct,
                "imbalance_ratio": self.imbalance.imbalance_ratio,
                "is_imbalanced": self.imbalance.is_imbalanced,
            },
            "empty_texts": self.empty_texts,
            "empty_texts_pct": self.empty_texts_pct,
        }


@dataclass
class ComparisonReport:
    metrics: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for metric, values in self.metrics.items():
            rows.append({
                "Metric": metric,
                "Before": values.get("before"),
                "After": values.get("after"),
                "Change": values.get("change"),
            })
        return pd.DataFrame(rows)


# ── Strategies ──────────────────────────────────────────────────────

STRATEGIES = {
    "aggressive": {
        "missing": "drop",
        "duplicates": "drop",
        "outliers": "clip_iqr",
        "empty_texts": "drop",
        "description": "Drop all rows with issues. Maximizes data cleanliness at the cost of size.",
    },
    "conservative": {
        "missing": "fill_empty",
        "duplicates": "keep",
        "outliers": "keep",
        "empty_texts": "keep",
        "description": "Keep as much data as possible. Minimal intervention.",
    },
    "balanced": {
        "missing": "fill_empty",
        "duplicates": "drop",
        "outliers": "clip_zscore",
        "empty_texts": "drop",
        "description": "Remove clear problems (duplicates, empty), keep borderline cases.",
    },
}


# ── Agent ───────────────────────────────────────────────────────────

class DataQualityAgent:
    """Detects and fixes data quality issues.

    Three core skills:
        - ``detect_issues(df)`` — find missing values, duplicates, outliers, imbalance
        - ``fix(df, strategy)`` — apply cleaning strategy
        - ``compare(df_before, df_after)`` — compare metrics before/after
    """

    def __init__(self, text_column: str = "text", label_column: str = "label"):
        self.text_col = text_column
        self.label_col = label_column

    # ── Skill: detect_issues ────────────────────────────────────────

    def detect_issues(self, df: pd.DataFrame) -> QualityReport:
        """Analyze a DataFrame and return a QualityReport with all detected issues."""
        report = QualityReport(
            total_rows=len(df),
            total_columns=len(df.columns),
        )

        # Missing values (label column excluded — AnnotationAgent will handle labeling)
        missing_counts = df.drop(columns=[self.label_col], errors="ignore").isnull().sum()
        report.missing = MissingReport(
            total=int(missing_counts.sum()),
            by_column={col: int(v) for col, v in missing_counts.items() if v > 0},
            by_column_pct={
                col: round(v / len(df) * 100, 2)
                for col, v in missing_counts.items() if v > 0
            },
        )

        # Duplicates (by text column)
        if self.text_col in df.columns:
            dup_mask = df.duplicated(subset=[self.text_col], keep=False)
            n_dups = int(df.duplicated(subset=[self.text_col], keep="first").sum())
            dup_examples = (
                df[dup_mask][self.text_col].head(5).tolist() if n_dups > 0 else []
            )
            report.duplicates = DuplicatesReport(
                total=n_dups,
                pct=round(n_dups / len(df) * 100, 2) if len(df) > 0 else 0,
                examples=dup_examples,
            )

        # Outliers by text length (IQR method)
        if self.text_col in df.columns:
            lengths = df[self.text_col].astype(str).str.len()
            q1 = lengths.quantile(0.25)
            q3 = lengths.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outlier_mask = (lengths < lower) | (lengths > upper)
            report.outliers = OutliersReport(
                method="iqr",
                total=int(outlier_mask.sum()),
                by_column={"text_length": int(outlier_mask.sum())},
                bounds={"text_length": {
                    "q1": round(float(q1), 1),
                    "q3": round(float(q3), 1),
                    "iqr": round(float(iqr), 1),
                    "lower": round(float(lower), 1),
                    "upper": round(float(upper), 1),
                }},
            )

        # Class imbalance
        if self.label_col in df.columns:
            counts = df[self.label_col].value_counts()
            total = counts.sum()
            ratio = float(counts.max() / counts.min()) if counts.min() > 0 else float("inf")
            report.imbalance = ImbalanceReport(
                class_counts={str(k): int(v) for k, v in counts.items()},
                class_pct={str(k): round(v / total * 100, 2) for k, v in counts.items()},
                imbalance_ratio=round(ratio, 2),
                is_imbalanced=ratio > 3.0,
            )

        # Empty texts
        if self.text_col in df.columns:
            empty_mask = (
                df[self.text_col].isnull()
                | (df[self.text_col].astype(str).str.strip() == "")
            )
            report.empty_texts = int(empty_mask.sum())
            report.empty_texts_pct = (
                round(report.empty_texts / len(df) * 100, 2) if len(df) > 0 else 0
            )

        return report

    # ── Skill: fix ──────────────────────────────────────────────────

    def fix(self, df: pd.DataFrame, strategy: dict[str, str] | str = "balanced") -> pd.DataFrame:
        """Apply a cleaning strategy to the DataFrame.

        Args:
            df: Input DataFrame.
            strategy: Either a preset name ('aggressive', 'conservative', 'balanced')
                      or a dict with keys: missing, duplicates, outliers, empty_texts.

        Returns:
            Cleaned DataFrame.
        """
        if isinstance(strategy, str):
            if strategy not in STRATEGIES:
                raise ValueError(
                    f"Unknown strategy: {strategy}. Choose from: {list(STRATEGIES.keys())}"
                )
            strat = STRATEGIES[strategy]
        else:
            strat = strategy

        result = df.copy()

        # Fix missing values (label column is excluded — AnnotationAgent will handle it)
        missing_action = strat.get("missing", "keep")
        exclude_cols = {self.label_col}
        if missing_action == "drop":
            cols_to_check = [c for c in result.columns if c not in exclude_cols]
            result = result.dropna(subset=cols_to_check)
        elif missing_action == "fill_empty":
            if self.text_col in result.columns:
                result[self.text_col] = result[self.text_col].fillna("")
        elif missing_action == "median":
            for col in result.select_dtypes(include=[np.number]).columns:
                if col not in exclude_cols:
                    result[col] = result[col].fillna(result[col].median())
            if self.text_col in result.columns:
                result[self.text_col] = result[self.text_col].fillna("")

        # Fix duplicates
        dup_action = strat.get("duplicates", "keep")
        if dup_action == "drop" and self.text_col in result.columns:
            result = result.drop_duplicates(subset=[self.text_col], keep="first")

        # Fix outliers
        outlier_action = strat.get("outliers", "keep")
        if outlier_action != "keep" and self.text_col in result.columns:
            lengths = result[self.text_col].astype(str).str.len()

            if outlier_action == "clip_iqr":
                q1 = lengths.quantile(0.25)
                q3 = lengths.quantile(0.75)
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                result = result[(lengths >= lower) & (lengths <= upper)]

            elif outlier_action == "clip_zscore":
                mean = lengths.mean()
                std = lengths.std()
                z_scores = np.abs((lengths - mean) / std) if std > 0 else pd.Series(0, index=lengths.index)
                result = result[z_scores <= 3]

            elif outlier_action == "drop":
                q1 = lengths.quantile(0.25)
                q3 = lengths.quantile(0.75)
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                result = result[(lengths >= lower) & (lengths <= upper)]

        # Fix empty texts
        empty_action = strat.get("empty_texts", "keep")
        if empty_action == "drop" and self.text_col in result.columns:
            result = result[
                result[self.text_col].astype(str).str.strip() != ""
            ]
            result = result[result[self.text_col].notna()]

        result = result.reset_index(drop=True)
        return result

    # ── Skill: compare ──────────────────────────────────────────────

    def compare(self, df_before: pd.DataFrame, df_after: pd.DataFrame) -> ComparisonReport:
        """Compare data quality metrics before and after cleaning."""
        report_before = self.detect_issues(df_before)
        report_after = self.detect_issues(df_after)

        def _change(before: Any, after: Any) -> str:
            if isinstance(before, (int, float)) and isinstance(after, (int, float)):
                diff = after - before
                if diff == 0:
                    return "0"
                sign = "+" if diff > 0 else ""
                return f"{sign}{diff}"
            return "—"

        metrics = {
            "Total rows": {
                "before": report_before.total_rows,
                "after": report_after.total_rows,
                "change": _change(report_before.total_rows, report_after.total_rows),
            },
            "Missing values": {
                "before": report_before.missing.total,
                "after": report_after.missing.total,
                "change": _change(report_before.missing.total, report_after.missing.total),
            },
            "Duplicates": {
                "before": report_before.duplicates.total,
                "after": report_after.duplicates.total,
                "change": _change(report_before.duplicates.total, report_after.duplicates.total),
            },
            "Outliers (text length)": {
                "before": report_before.outliers.total,
                "after": report_after.outliers.total,
                "change": _change(report_before.outliers.total, report_after.outliers.total),
            },
            "Empty texts": {
                "before": report_before.empty_texts,
                "after": report_after.empty_texts,
                "change": _change(report_before.empty_texts, report_after.empty_texts),
            },
            "Imbalance ratio": {
                "before": report_before.imbalance.imbalance_ratio,
                "after": report_after.imbalance.imbalance_ratio,
                "change": _change(
                    report_before.imbalance.imbalance_ratio,
                    report_after.imbalance.imbalance_ratio,
                ),
            },
            "Unique classes": {
                "before": len(report_before.imbalance.class_counts),
                "after": len(report_after.imbalance.class_counts),
                "change": _change(
                    len(report_before.imbalance.class_counts),
                    len(report_after.imbalance.class_counts),
                ),
            },
        }

        return ComparisonReport(metrics=metrics)

    # ── Preset strategies info ──────────────────────────────────────

    @staticmethod
    def list_strategies() -> dict[str, dict]:
        """Return available preset strategies with descriptions."""
        return STRATEGIES

    # ── Save cleaned data ───────────────────────────────────────────

    def save(self, df: pd.DataFrame, path: str | Path = "data/cleaned/cleaned.parquet") -> Path:
        """Save the cleaned DataFrame."""
        p = ROOT_DIR / path
        p.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(p, index=False)
        return p

"""AnnotationAgent — LLM-based labeling, quality assessment, and Label Studio export.

Skills:
    - auto_label(df, labels, task) → DataFrame with auto_label + confidence
    - generate_spec(df, task, labels) → Path to annotation_spec.md
    - check_quality(df_labeled) → QualityMetrics
    - export_to_labelstudio(df, threshold) → Path to JSON
    - import_from_labelstudio(df, ls_export_path) → DataFrame with manual labels merged

Usage::

    from agents.annotation_agent import AnnotationAgent

    agent = AnnotationAgent()
    df_labeled = agent.auto_label(df, labels=["World", "Sports", "Business", "Sci/Tech"])
    metrics = agent.check_quality(df_labeled)
    agent.export_to_labelstudio(df_labeled, confidence_threshold=0.7)
    df_final = agent.import_from_labelstudio(df_labeled, "data/annotation/ls_export.json")
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")


@dataclass
class QualityMetrics:
    """Quality assessment of auto-labeling."""
    cohens_kappa: float | None = None
    agreement_pct: float = 0.0
    label_distribution: dict[str, int] = field(default_factory=dict)
    label_distribution_pct: dict[str, float] = field(default_factory=dict)
    confidence_mean: float = 0.0
    confidence_median: float = 0.0
    confidence_std: float = 0.0
    low_confidence_count: int = 0
    low_confidence_pct: float = 0.0
    total: int = 0

    def to_dict(self) -> dict:
        return {
            "cohens_kappa": self.cohens_kappa,
            "agreement_pct": self.agreement_pct,
            "label_distribution": self.label_distribution,
            "label_distribution_pct": self.label_distribution_pct,
            "confidence_mean": self.confidence_mean,
            "confidence_median": self.confidence_median,
            "confidence_std": self.confidence_std,
            "low_confidence_count": self.low_confidence_count,
            "low_confidence_pct": self.low_confidence_pct,
            "total": self.total,
        }


class AnnotationAgent:
    """LLM-based labeling, quality assessment, and Label Studio export.

    Skills:
        - ``auto_label(df, labels, task)`` — classify texts via Claude API
        - ``generate_spec(df, task, labels)`` — annotation specification
        - ``check_quality(df_labeled)`` — Cohen's kappa, distribution, confidence
        - ``export_to_labelstudio(df, threshold)`` — export low-confidence to LS
        - ``import_from_labelstudio(df, path)`` — merge manual labels back
    """

    def __init__(
        self,
        text_column: str = "text",
        label_column: str = "label",
        model: str = "claude-sonnet-4-20250514",
    ):
        self.text_col = text_column
        self.label_col = label_column
        self.model = model
        self._client = None

    def _get_client(self):
        """Lazy-load Anthropic client."""
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic()  # uses ANTHROPIC_API_KEY from env
        return self._client

    # ── Skill: auto_label ──────────────────────────────────────────

    def auto_label(
        self,
        df: pd.DataFrame,
        labels: list[str],
        task_description: str = "topic classification",
        batch_size: int = 10,
        confidence_threshold: float = 0.7,
        allow_new_labels: bool = False,
    ) -> pd.DataFrame:
        """Classify texts using Claude API. Adds auto_label + confidence columns.

        Args:
            df: Input DataFrame with text column.
            labels: Candidate label names.
            task_description: Description of the classification task.
            batch_size: Number of texts per API call.
            confidence_threshold: Below this, examples are flagged as disputed.
            allow_new_labels: If True, LLM may propose new label values when
                highly confident (>0.9) that a text doesn't fit existing labels.

        Returns:
            DataFrame with 'auto_label', 'confidence', 'is_disputed' columns.
        """
        client = self._get_client()
        texts = df[self.text_col].astype(str).tolist()

        labels_str = ", ".join(labels)

        system_prompt = (
            f"You are a text classifier for {task_description}.\n"
            f"Available labels: {labels_str}\n\n"
            "For each text, respond with a JSON array where each element has:\n"
            '  - "label": one of the available labels (exact match)\n'
            '  - "confidence": float 0.0-1.0 (how certain you are)\n\n'
        )

        if allow_new_labels:
            system_prompt += (
                "IMPORTANT: If you are HIGHLY confident (confidence > 0.9) that a text "
                "does NOT fit ANY of the available labels, you may propose a NEW label.\n"
                "New labels must be:\n"
                "  - Concise (1-2 words), similar in style to existing labels\n"
                "  - Truly necessary — only when none of the existing labels fit\n"
                "  - Marked with \"is_new\": true in the response\n\n"
                "Format for new labels:\n"
                '  {"label": "NewLabel", "confidence": 0.95, "is_new": true}\n\n'
            )

        system_prompt += (
            "Respond ONLY with the JSON array, no other text.\n"
            "Always assign exactly ONE label per text.\n"
            "If a text could belong to multiple categories, pick the primary one "
            "and lower the confidence accordingly."
        )

        auto_labels = []
        confidences = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            batch_prompt = "Classify these texts:\n\n"
            for j, text in enumerate(batch):
                truncated = text[:500] if len(text) > 500 else text
                batch_prompt += f"[{j}] {truncated}\n\n"

            message = client.messages.create(
                model=self.model,
                max_tokens=2048,
                system=system_prompt,
                messages=[{"role": "user", "content": batch_prompt}],
            )

            response_text = message.content[0].text.strip()
            parsed = self._parse_llm_response(response_text, len(batch), labels, allow_new_labels)

            for item in parsed:
                auto_labels.append(item["label"])
                confidences.append(item["confidence"])

        result = df.copy()
        result["auto_label"] = auto_labels
        result["confidence"] = confidences
        result["is_disputed"] = result["confidence"] < confidence_threshold
        return result

    def _parse_llm_response(
        self, response: str, expected_count: int, valid_labels: list[str],
        allow_new_labels: bool = False,
    ) -> list[dict]:
        """Parse JSON array from LLM response with fallback."""
        try:
            match = re.search(r'\[.*\]', response, re.DOTALL)
            if match:
                items = json.loads(match.group())
                if isinstance(items, list) and len(items) == expected_count:
                    result = []
                    for item in items:
                        label = item.get("label", valid_labels[0])
                        conf = float(item.get("confidence", 0.5))
                        conf = max(0.0, min(1.0, conf))
                        is_new = bool(item.get("is_new", False))

                        if label not in valid_labels:
                            if allow_new_labels and is_new and conf >= 0.9:
                                # Accept the new label proposed by LLM
                                pass
                            else:
                                # Unknown label not allowed — fall back
                                label = valid_labels[0]
                                conf = min(conf, 0.3)

                        result.append({"label": label, "confidence": conf})
                    return result
        except (json.JSONDecodeError, ValueError, KeyError):
            pass

        # Fallback
        return [{"label": valid_labels[0], "confidence": 0.1}] * expected_count

    # ── Skill: generate_spec ───────────────────────────────────────

    def generate_spec(
        self,
        df: pd.DataFrame,
        task: str,
        labels: list[str],
        examples_per_class: int = 3,
        output_path: str | Path = "data/annotation/annotation_spec.md",
    ) -> Path:
        """Generate annotation specification with class definitions and examples.

        Args:
            df: DataFrame with text and label columns.
            task: Task description.
            labels: List of label names.
            examples_per_class: Number of examples per class.
            output_path: Where to save the spec.

        Returns:
            Path to saved annotation_spec.md.
        """
        label_col = "auto_label" if "auto_label" in df.columns else self.label_col
        p = ROOT_DIR / output_path
        p.parent.mkdir(parents=True, exist_ok=True)

        lines = [
            f"# Annotation Specification: {task}",
            "",
            "## Task Description",
            "",
            f"Classify each text into exactly one of the following categories.",
            f"Task: **{task}**",
            "",
            "## Label Definitions",
            "",
        ]

        for label in labels:
            lines.append(f"### {label}")
            lines.append("")
            lines.append(f"Texts that belong to the **{label}** category.")
            lines.append("")

            mask = df[label_col] == label
            subset = df[mask]
            if "confidence" in subset.columns:
                subset = subset.sort_values("confidence", ascending=False)
            examples = subset[self.text_col].head(examples_per_class).tolist()

            if examples:
                lines.append("**Examples:**")
                lines.append("")
                for j, ex in enumerate(examples, 1):
                    text_preview = ex[:200].replace("\n", " ")
                    lines.append(f'{j}. "{text_preview}"')
                lines.append("")

        lines.extend([
            "## Edge Cases",
            "",
            "- If a text covers multiple topics, choose the **primary** topic.",
            "- If a text is too short to determine the topic, label as the most likely category.",
            "- If a text is in a language other than expected, still classify by topic if possible.",
            "- Advertisements or spam should be labeled based on their primary subject matter.",
            "",
            "## Annotation Guidelines",
            "",
            "1. Read the full text before assigning a label.",
            "2. Focus on the **main topic**, not secondary mentions.",
            "3. When uncertain, choose the label that best fits the majority of the content.",
            f"4. Available labels: {', '.join(labels)}",
            "",
        ])

        p.write_text("\n".join(lines), encoding="utf-8")
        return p

    # ── Skill: check_quality ───────────────────────────────────────

    def check_quality(
        self,
        df: pd.DataFrame,
        confidence_threshold: float = 0.7,
    ) -> QualityMetrics:
        """Compute quality metrics for auto-labeled data.

        Args:
            df: DataFrame with 'auto_label' and 'confidence' columns.
            confidence_threshold: Threshold for low-confidence flagging.

        Returns:
            QualityMetrics dataclass.
        """
        metrics = QualityMetrics(total=len(df))

        if "auto_label" in df.columns:
            counts = df["auto_label"].value_counts()
            total = counts.sum()
            metrics.label_distribution = {str(k): int(v) for k, v in counts.items()}
            metrics.label_distribution_pct = {
                str(k): round(v / total * 100, 2) for k, v in counts.items()
            }

        if "confidence" in df.columns:
            conf = df["confidence"]
            metrics.confidence_mean = round(float(conf.mean()), 4)
            metrics.confidence_median = round(float(conf.median()), 4)
            metrics.confidence_std = round(float(conf.std()), 4)
            low_mask = conf < confidence_threshold
            metrics.low_confidence_count = int(low_mask.sum())
            metrics.low_confidence_pct = (
                round(metrics.low_confidence_count / len(df) * 100, 2)
                if len(df) > 0 else 0
            )

        if "auto_label" in df.columns and self.label_col in df.columns:
            mask = df[self.label_col].notna() & df["auto_label"].notna()
            if mask.sum() > 0:
                from sklearn.metrics import cohen_kappa_score
                y_true = df.loc[mask, self.label_col].astype(str)
                y_pred = df.loc[mask, "auto_label"].astype(str)
                try:
                    kappa = cohen_kappa_score(y_true, y_pred)
                    metrics.cohens_kappa = round(float(kappa), 4)
                except ValueError:
                    metrics.cohens_kappa = None
                agreement = (y_true == y_pred).sum()
                metrics.agreement_pct = round(agreement / mask.sum() * 100, 2)

        return metrics

    # ── Skill: export_to_labelstudio ───────────────────────────────

    def export_to_labelstudio(
        self,
        df: pd.DataFrame,
        confidence_threshold: float = 0.7,
        export_all: bool = False,
        output_path: str | Path = "data/annotation/labelstudio_tasks.json",
    ) -> Path:
        """Export tasks to Label Studio JSON format.

        By default exports only low-confidence (disputed) examples.

        Args:
            df: DataFrame with text, auto_label, confidence columns.
            confidence_threshold: Export rows below this confidence.
            export_all: If True, export all rows.
            output_path: Where to save the JSON.

        Returns:
            Path to saved JSON file.
        """
        p = ROOT_DIR / output_path
        p.parent.mkdir(parents=True, exist_ok=True)

        if export_all:
            subset = df
        elif "confidence" in df.columns:
            subset = df[df["confidence"] < confidence_threshold]
        else:
            subset = df

        tasks = []
        for idx, row in subset.iterrows():
            task = {
                "data": {
                    "text": str(row.get(self.text_col, "")),
                    "source": str(row.get("source", "")),
                    "original_label": str(row.get(self.label_col, "")),
                    "auto_label": str(row.get("auto_label", "")),
                    "confidence": round(float(row.get("confidence", 0)), 3),
                    "index": int(idx),
                },
            }
            if "auto_label" in row and pd.notna(row["auto_label"]):
                task["predictions"] = [
                    {
                        "model_version": self.model,
                        "result": [
                            {
                                "from_name": "label",
                                "to_name": "text",
                                "type": "choices",
                                "value": {"choices": [str(row["auto_label"])]},
                            }
                        ],
                        "score": round(float(row.get("confidence", 0)), 3),
                    }
                ]
            tasks.append(task)

        with open(p, "w", encoding="utf-8") as f:
            json.dump(tasks, f, indent=2, ensure_ascii=False)

        return p

    # ── Generate Label Studio project config ───────────────────────

    def generate_ls_config(
        self,
        labels: list[str],
        output_path: str | Path = "data/annotation/labelstudio_config.xml",
    ) -> Path:
        """Generate Label Studio labeling config XML."""
        p = ROOT_DIR / output_path
        p.parent.mkdir(parents=True, exist_ok=True)

        choices = "\n".join(f'    <Choice value="{label}" />' for label in labels)
        config = f"""<View>
  <Text name="text" value="$text" />
  <Choices name="label" toName="text" choice="single" showInLine="true">
{choices}
  </Choices>
</View>"""
        p.write_text(config, encoding="utf-8")
        return p

    # ── Import from Label Studio ───────────────────────────────────

    def import_from_labelstudio(
        self,
        df: pd.DataFrame,
        ls_export_path: str | Path = "data/annotation/ls_export.json",
    ) -> pd.DataFrame:
        """Merge manual labels from Label Studio export back into the DataFrame.

        Reads LS export JSON, extracts human labels, and updates
        auto_label/confidence for those rows.

        Args:
            df: Original labeled DataFrame (with auto_label, confidence).
            ls_export_path: Path to Label Studio JSON export.

        Returns:
            DataFrame with manual labels merged in.
        """
        p = ROOT_DIR / ls_export_path if not Path(ls_export_path).is_absolute() else Path(ls_export_path)

        with open(p, "r", encoding="utf-8") as f:
            ls_data = json.load(f)

        result = df.copy()

        for task in ls_data:
            data = task.get("data", {})
            idx = data.get("index")
            if idx is None:
                continue

            annotations = task.get("annotations", [])
            if not annotations:
                continue

            # Take the latest annotation
            annotation = annotations[-1]
            results = annotation.get("result", [])
            for r in results:
                if r.get("type") == "choices" and r.get("from_name") == "label":
                    choices = r.get("value", {}).get("choices", [])
                    if choices and idx in result.index:
                        result.at[idx, "auto_label"] = choices[0]
                        result.at[idx, "confidence"] = 1.0  # human label = full confidence
                        result.at[idx, "is_disputed"] = False

        return result

    # ── Save labeled data ──────────────────────────────────────────

    def save(
        self,
        df: pd.DataFrame,
        path: str | Path = "data/labeled/labeled.parquet",
    ) -> Path:
        """Save the labeled DataFrame."""
        p = ROOT_DIR / path
        p.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(p, index=False)
        return p

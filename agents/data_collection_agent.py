"""DataCollectionAgent — universal text-classification dataset collector.

Skills:
    - scrape(url, selector) → DataFrame
    - fetch_api(endpoint, params) → DataFrame
    - load_dataset(name, source='hf'|'kaggle') → DataFrame
    - merge(sources: list[DataFrame]) → DataFrame

Usage::

    from agents.data_collection_agent import DataCollectionAgent

    agent = DataCollectionAgent(config='config.yaml')
    df = agent.run(sources=[
        {'type': 'hf_dataset', 'name': 'imdb'},
        {'type': 'scrape', 'url': '...', 'selector': '...'},
    ])
    # → pd.DataFrame: text, label, source, collected_at
"""

from __future__ import annotations

import datetime as dt
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = ROOT_DIR / ".claude" / "skills" / "dataset-collector" / "scripts"


# ── Config dataclasses ──────────────────────────────────────────────

@dataclass
class GeneralCfg:
    output_dir: str = "data/raw"
    eda_dir: str = "data/eda"
    unified_file: str = "data/raw/combined.parquet"
    max_samples_per_source: int = 5000
    validation_sample_size: int = 10


@dataclass
class HuggingFaceCfg:
    enabled: bool = True
    max_results: int = 15


@dataclass
class KaggleCfg:
    enabled: bool = True
    max_results: int = 15
    username: str = field(default_factory=lambda: os.getenv("KAGGLE_USERNAME", ""))
    key: str = field(default_factory=lambda: os.getenv("KAGGLE_KEY", ""))

    @property
    def available(self) -> bool:
        return self.enabled and bool(self.username) and bool(self.key)


@dataclass
class ScrapingCfg:
    enabled: bool = True
    timeout: int = 30
    user_agent: str = "Mozilla/5.0 (compatible; DataCollectionAgent/1.0)"
    max_pages: int = 5


@dataclass
class RssCfg:
    enabled: bool = True
    max_entries: int = 500


@dataclass
class Config:
    general: GeneralCfg = field(default_factory=GeneralCfg)
    huggingface: HuggingFaceCfg = field(default_factory=HuggingFaceCfg)
    kaggle: KaggleCfg = field(default_factory=KaggleCfg)
    scraping: ScrapingCfg = field(default_factory=ScrapingCfg)
    rss: RssCfg = field(default_factory=RssCfg)


def _merge_cfg(dc_class: type, data: dict[str, Any]) -> Any:
    valid = {f.name for f in dc_class.__dataclass_fields__.values()}
    return dc_class(**{k: v for k, v in data.items() if k in valid})


def load_config(path: str | Path | None = None) -> Config:
    if path is None:
        path = ROOT_DIR / "config.yaml"
    path = Path(path)
    if not path.exists():
        return Config()
    with open(path, "r", encoding="utf-8") as f:
        raw: dict = yaml.safe_load(f) or {}
    return Config(
        general=_merge_cfg(GeneralCfg, raw.get("general", {})),
        huggingface=_merge_cfg(HuggingFaceCfg, raw.get("huggingface", {})),
        kaggle=_merge_cfg(KaggleCfg, raw.get("kaggle", {})),
        scraping=_merge_cfg(ScrapingCfg, raw.get("scraping", {})),
        rss=_merge_cfg(RssCfg, raw.get("rss", {})),
    )


# ── Agent ───────────────────────────────────────────────────────────

class DataCollectionAgent:
    """Collects text-classification data from heterogeneous sources.

    Exposes four core skills:
        - ``scrape(url, selector)`` — scrape a web page
        - ``fetch_api(endpoint, params)`` — call a JSON API
        - ``load_dataset(name, source)`` — load from HuggingFace or Kaggle
        - ``merge(sources)`` — concatenate and unify schema
    """

    SCHEMA_COLUMNS = ["text", "label", "source", "collected_at"]

    def __init__(self, config: str | Path | Config | None = None):
        if isinstance(config, Config):
            self.cfg = config
        else:
            self.cfg = load_config(config)

        self.output_dir = ROOT_DIR / self.cfg.general.output_dir
        self.eda_dir = ROOT_DIR / self.cfg.general.eda_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.eda_dir.mkdir(parents=True, exist_ok=True)

    # ── Skill: scrape ───────────────────────────────────────────────

    def scrape(self, url: str, selector: str,
               label_selector: str | None = None,
               label: str | None = None) -> pd.DataFrame:
        """Scrape a web page and return a unified DataFrame."""
        import requests
        from bs4 import BeautifulSoup

        headers = {"User-Agent": self.cfg.scraping.user_agent}
        resp = requests.get(url, headers=headers, timeout=self.cfg.scraping.timeout)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")
        elements = soup.select(selector)

        rows = []
        for el in elements:
            text = el.get_text(strip=True)
            if not text:
                continue
            row_label = label or ""
            if label_selector:
                lbl_el = el.select_one(label_selector)
                if lbl_el:
                    row_label = lbl_el.get_text(strip=True)
            rows.append({"text": text, "label": row_label})

        return self._to_unified(pd.DataFrame(rows), source=url)

    # ── Skill: fetch_api ────────────────────────────────────────────

    def fetch_api(self, endpoint: str, params: dict | None = None,
                  text_field: str = "text", label_field: str = "label",
                  results_key: str | None = None) -> pd.DataFrame:
        """Fetch data from a JSON API and return a unified DataFrame."""
        import requests

        resp = requests.get(endpoint, params=params,
                            timeout=self.cfg.scraping.timeout)
        resp.raise_for_status()
        data = resp.json()

        if results_key:
            data = data[results_key]

        df = pd.DataFrame(data)
        rename = {}
        if text_field != "text" and text_field in df.columns:
            rename[text_field] = "text"
        if label_field != "label" and label_field in df.columns:
            rename[label_field] = "label"
        if rename:
            df = df.rename(columns=rename)

        return self._to_unified(df, source=endpoint)

    # ── Skill: load_dataset ─────────────────────────────────────────

    def load_dataset(self, name: str, source: str = "hf",
                     split: str = "train",
                     text_field: str = "text",
                     label_field: str = "label",
                     limit: int | None = None) -> pd.DataFrame:
        """Load a dataset from HuggingFace or Kaggle."""
        if source == "hf":
            return self._load_hf(name, split, text_field, label_field, limit)
        elif source == "kaggle":
            return self._load_kaggle(name, text_field, label_field, limit)
        else:
            raise ValueError(f"Unknown dataset source: {source}")

    # ── Skill: merge ────────────────────────────────────────────────

    def merge(self, sources: list[pd.DataFrame]) -> pd.DataFrame:
        """Merge multiple DataFrames into one with unified schema."""
        combined = pd.concat(
            [s.reset_index(drop=True) for s in sources], ignore_index=True
        )
        for col in self.SCHEMA_COLUMNS:
            if col not in combined.columns:
                combined[col] = None
        return combined[self.SCHEMA_COLUMNS]

    # ── High-level API ──────────────────────────────────────────────

    def search_sources(self, query: str) -> list[dict[str, Any]]:
        """Search for datasets across all enabled backends."""
        results: list[dict] = []
        if self.cfg.huggingface.enabled:
            results.extend(self._search_hf(query))
        if self.cfg.kaggle.available:
            results.extend(self._search_kaggle(query))
        return results

    def validate_source(self, source: dict[str, Any]) -> bool:
        """Try to fetch a small sample. Return True on success."""
        try:
            df = self._collect_one(source, limit=self.cfg.general.validation_sample_size)
            return df is not None and len(df) > 0
        except Exception:
            return False

    def collect(self, sources: list[dict[str, Any]]) -> pd.DataFrame:
        """Collect data from selected sources and return merged DataFrame."""
        frames: list[pd.DataFrame] = []
        for src in sources:
            df = self._collect_one(src, limit=self.cfg.general.max_samples_per_source)
            if df is not None and len(df) > 0:
                frames.append(df)
        if not frames:
            raise RuntimeError("No data collected from any source.")
        return self.merge(frames)

    def save(self, df: pd.DataFrame, filename: str = "combined.parquet") -> Path:
        """Save the unified dataset to data/raw/."""
        path = self.output_dir / filename
        df.to_parquet(path, index=False)
        return path

    def run(self, sources: list[dict[str, Any]]) -> pd.DataFrame:
        """End-to-end: collect → merge → save."""
        df = self.collect(sources)
        self.save(df)
        return df

    # ── Additional collection methods ───────────────────────────────

    def fetch_rss(self, url: str, label: str | None = None) -> pd.DataFrame:
        """Parse an RSS/Atom feed and return a unified DataFrame."""
        import feedparser

        feed = feedparser.parse(url)
        rows = []
        for entry in feed.entries[: self.cfg.rss.max_entries]:
            text = entry.get("title", "")
            summary = entry.get("summary", "")
            if summary:
                text = f"{text}. {summary}"
            if not text.strip():
                continue
            rows.append({
                "text": text.strip(),
                "label": label or entry.get("category", "unknown"),
            })
        return self._to_unified(pd.DataFrame(rows), source=url)

    # ── Private helpers ─────────────────────────────────────────────

    def _to_unified(self, df: pd.DataFrame, source: str) -> pd.DataFrame:
        if "source" not in df.columns:
            df["source"] = source
        if "collected_at" not in df.columns:
            df["collected_at"] = dt.datetime.utcnow().isoformat()
        for col in self.SCHEMA_COLUMNS:
            if col not in df.columns:
                df[col] = None
        return df[self.SCHEMA_COLUMNS]

    def _collect_one(self, src: dict[str, Any],
                     limit: int | None = None) -> pd.DataFrame | None:
        src_type = src.get("type", "")
        if src_type == "hf_dataset":
            return self.load_dataset(
                name=src["name"], source="hf",
                split=src.get("split", "train"),
                text_field=src.get("text_field", "text"),
                label_field=src.get("label_field", "label"),
                limit=limit,
            )
        elif src_type == "kaggle_dataset":
            return self.load_dataset(
                name=src["name"], source="kaggle",
                text_field=src.get("text_field", "text"),
                label_field=src.get("label_field", "label"),
                limit=limit,
            )
        elif src_type == "scrape":
            return self.scrape(
                url=src["url"], selector=src["selector"],
                label_selector=src.get("label_selector"),
                label=src.get("label"),
            )
        elif src_type == "api":
            return self.fetch_api(
                endpoint=src["endpoint"], params=src.get("params"),
                text_field=src.get("text_field", "text"),
                label_field=src.get("label_field", "label"),
                results_key=src.get("results_key"),
            )
        elif src_type == "rss":
            return self.fetch_rss(url=src["url"], label=src.get("label"))
        else:
            raise ValueError(f"Unknown source type: {src_type}")

    def _search_hf(self, query: str) -> list[dict]:
        script = SCRIPTS_DIR / "search_hf.py"
        result = subprocess.run(
            [sys.executable, str(script), query,
             str(self.cfg.huggingface.max_results)],
            capture_output=True, text=True, check=True,
        )
        return json.loads(result.stdout)

    def _search_kaggle(self, query: str) -> list[dict]:
        script = SCRIPTS_DIR / "search_kaggle.py"
        result = subprocess.run(
            [sys.executable, str(script), query,
             str(self.cfg.kaggle.max_results)],
            capture_output=True, text=True, check=True,
        )
        return json.loads(result.stdout)

    def _load_hf(self, name: str, split: str, text_field: str,
                 label_field: str, limit: int | None) -> pd.DataFrame:
        from datasets import load_dataset

        ds = load_dataset(name, split=split)
        if limit:
            ds = ds.select(range(min(limit, len(ds))))
        df = ds.to_pandas()

        rename = {}
        if text_field != "text" and text_field in df.columns:
            rename[text_field] = "text"
        if label_field != "label" and label_field in df.columns:
            rename[label_field] = "label"
            if "label" in df.columns:
                df = df.drop(columns=["label"])
        if rename:
            df = df.rename(columns=rename)

        if "label" in df.columns and pd.api.types.is_numeric_dtype(df["label"]):
            if hasattr(ds, "features") and label_field in ds.features:
                feat = ds.features[label_field]
                if hasattr(feat, "names"):
                    df["label"] = df["label"].map(
                        lambda x: feat.names[x] if 0 <= x < len(feat.names) else x
                    )

        return self._to_unified(df, source=f"huggingface:{name}")

    def _load_kaggle(self, name: str, text_field: str,
                     label_field: str, limit: int | None) -> pd.DataFrame:
        import tempfile

        tmp = Path(tempfile.mkdtemp())
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", name,
             "-p", str(tmp), "--unzip"],
            check=True, capture_output=True,
        )
        csvs = list(tmp.glob("**/*.csv"))
        if not csvs:
            raise FileNotFoundError(f"No CSV found in Kaggle dataset {name}")

        df = pd.read_csv(csvs[0], nrows=limit)
        rename = {}
        if text_field != "text" and text_field in df.columns:
            rename[text_field] = "text"
        if label_field != "label" and label_field in df.columns:
            rename[label_field] = "label"
        if rename:
            df = df.rename(columns=rename)

        return self._to_unified(df, source=f"kaggle:{name}")

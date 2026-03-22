# Dataset Collector

Universal text-classification dataset collector. Searches, validates, collects, and unifies datasets from multiple source types into a single parquet file with standardized schema.

## Trigger

User asks to collect / find / gather a dataset for any text classification task.
Examples:
- "собери датасет для классификации топиков в новостях"
- "find a dataset for spam detection"
- "нужны данные для определения тональности отзывов"

## Autonomy rules

- **FULLY AUTONOMOUS** on all steps EXCEPT Step 4 (dataset selection).
- Steps 1, 2, 3: do silently, do NOT comment, do NOT ask questions.
- **Step 4 is MANDATORY and BLOCKING**: you MUST show the table and STOP. Do NOT proceed to Step 5 until the user replies with their selection. NEVER skip Step 4.
- Steps 5, 6, 7: after user selects — do everything automatically, no more questions.
- If something fails — fix it or skip it, move on.

## Workflow

### Step 1 — Understand the task

Extract from the user's message:
- **domain** (news, reviews, emails, social media, …)
- **task** (topic classification, sentiment, spam detection, …)
- **desired classes** if mentioned, otherwise leave open
- **language** (default: English, Russian)

**Save task type to config.yaml** — update `task.type` so all downstream agents know the goal:
```python
import yaml
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
config["task"]["type"] = "News topic text classification"  # derived from user's message
with open("config.yaml", "w") as f:
    yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
```

### Step 2 — Search for sources (SILENT)

Find as many candidate sources as possible (goal: up to **20** validated sources). 
Types:
1. `hf_dataset` — HuggingFace Hub
   `kaggle_dataset` — Kaggle (only if KAGGLE_USERNAME + KAGGLE_KEY in .env)
2. `scrape` — web scraping
   `api` — public JSON API
3. `rss` — RSS/Atom feed - least priority

You MUST always find type of sources like:
`hf_dataset`/`kaggle_dataset` and `scrape`/`api`.
You MUST provide at least 2 data sources (one is an open dataset from HuggingFace/Kaggle, and one is scraping or an API)!

**How to search (do all in parallel, no comments):**

A) HuggingFace — run the search script with **multiple queries** (synonyms, related terms):
```bash
python .claude/skills/dataset-collector/scripts/search_hf.py "<query>" 15
python .claude/skills/dataset-collector/scripts/search_hf.py "<synonym query>" 15
```

B) Kaggle — if credentials exist:
```bash
python .claude/skills/dataset-collector/scripts/search_kaggle.py "<query>" 15
```

C) WebSearch — find websites/APIs/RSS feeds relevant to the domain.

D) Use your knowledge of well-known datasets and data sources (e.g. ag_news, imdb, yelp_review_full, dbpedia_14, yahoo_answers, 20newsgroups, etc.).

### Step 3 — Validate sources (SILENT)

For each candidate, try to fetch a small sample (5–10 rows). Drop any source that fails.

If fewer than 2 source **types** remain, search for more — still silently.

### Step 4 — Present to user (HUMAN-IN-THE-LOOP)

Show a formatted table of **validated** sources, **ranked by relevance** (most relevant first). Maximum **20** sources.

For each source, assess:
- **Relevance** (1–5 stars): how well does this dataset match the user's request? Consider domain match, task match, language match, label quality.
- **Risk** (Low / Medium / High): how likely is collection to fail or produce bad data?
  - **Low**: well-known open dataset, stable API, reliable HF dataset
  - **Medium**: less popular dataset, RSS feed (content may change), API with rate limits
  - **High**: web scraping (site may block), unstable API, unknown dataset quality

```
| #  | Name                    | Type       | Relevance | Risk   | Description                         | Est. Size |
|----|-------------------------|------------|-----------|--------|-------------------------------------|-----------|
| 1  | fancyzhx/ag_news        | hf_dataset | ★★★★★     | Low    | News topics: World/Sports/Biz/Tech  | 120k      |
| 2  | SetFit/20_newsgroups    | hf_dataset | ★★★★☆     | Low    | 20 newsgroup categories             | 18k       |
| 3  | BBC News RSS (4 cats)   | rss        | ★★★★☆     | Medium | BBC Sport/Tech/Business/World       | ~170      |
| 4  | some-scrape-site.com    | scrape     | ★★★☆☆     | High   | News articles, needs parsing        | ~500      |
```

Rules:
- MUST include at least 2 different source types
- Sort by relevance (highest first), then by risk (lowest first)
- Show up to 20 sources
- User selects by numbers: "1, 2, 5" or "все" / "all"
- Multiple datasets can be selected
- **STOP HERE. Wait for the user's response. Do NOT run any collection code until the user picks sources.**
- After user responds — proceed to Step 5 fully autonomously, no more questions

### Step 5 — Collect data (SILENT)

For each selected source, use `DataCollectionAgent` skills or write inline code:

```python
from agents.data_collection_agent import DataCollectionAgent

agent = DataCollectionAgent(config='config.yaml')

# Skill: load_dataset
df_hf = agent.load_dataset('fancyzhx/ag_news', source='hf', limit=5000)

# Skill: scrape
df_web = agent.scrape(url='...', selector='...')

# Skill: fetch_api
df_api = agent.fetch_api(endpoint='...', params={...})

# Skill: merge
df = agent.merge([df_hf, df_web, df_api])
```

**Unified schema** — every row:

| Column       | Type | Description            |
|-------------|------|------------------------|
| `text`       | str  | The text content       |
| `label`      | str  | Class label (may be null) |
| `source`     | str  | Source identifier      |
| `collected_at` | str | ISO timestamp        |

**IMPORTANT: Labels are OPTIONAL.** Datasets without labels are perfectly fine — the AnnotationAgent will label/re-label everything later. Do NOT skip good text datasets just because they lack labels. Sources like RSS feeds, APIs, and web scraping often provide text without labels — collect them anyway.

Save:
- `data/raw/combined.parquet` — merged dataset
- `data/raw/<source_name>.parquet` — per-source files

### Step 6 — EDA (SILENT)

Follow the **EDA Generator** skill (`.claude/skills/eda-generator/SKILL.md`):

1. Generate `notebooks/eda.ipynb` tailored to the collected data and task
2. Execute it:
   ```bash
   python .claude/skills/eda-generator/scripts/run_notebook.py notebooks/eda.ipynb
   ```
3. Verify outputs exist: `notebooks/eda.ipynb` (with results), `data/eda/*.png`, `data/eda/REPORT.md`

### Step 7 — Summary

Show a short final summary:
```
✅ Dataset collected!
   Sources: 2 (HuggingFace: ag_news, RSS: BBC News)
   Total rows: 5,500
   Classes: 4 (World, Sports, Business, Sci/Tech)
   Saved: data/raw/combined.parquet
   EDA: data/eda/REPORT.md
   Notebook: notebooks/eda.ipynb (executed with results)
```

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/search_hf.py` | Search HuggingFace Hub |
| `scripts/search_kaggle.py` | Search Kaggle |

## Config

Reads `config.yaml`:
- `general.max_samples_per_source` — cap per source (default 5000)
- `general.validation_sample_size` — rows for validation (default 10)
- `huggingface.enabled` / `kaggle.enabled` / `scraping.enabled` / `rss.enabled`

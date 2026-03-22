# EDA Generator

Generates a task-specific EDA notebook, executes it, and saves with results.

## Trigger

Called after dataset collection is complete. Input: path to parquet file + task description.

## Autonomy rules

- **FULLY AUTONOMOUS**. No human interaction. Do everything silently.
- Generate notebook → execute → save with outputs. No questions.

## Workflow

### Step 1 — Read the dataset

```python
import pandas as pd
df = pd.read_parquet("data/raw/combined.parquet")
```

Understand what's in the data:
- Column names and types
- Number of rows
- Unique labels
- Unique sources
- Sample rows

### Step 2 — Generate EDA notebook

Create `notebooks/eda.ipynb` with cells **tailored to the specific task and data**. The notebook MUST include:

1. **Title + description** (markdown) — mention the specific task (e.g., "EDA: News Topic Classification")
2. **Load data** — read combined.parquet
3. **Dataset overview** — shape, dtypes, nulls, sample rows
4. **Class distribution** — bar chart + pie chart + table with counts and percentages
5. **Text length analysis** — histogram of char lengths, histogram of word counts, stats table (mean/median/min/max/std)
6. **Text length by class** — overlapping histograms or box plots per label
7. **Top-20 words** — horizontal bar chart (exclude stopwords if possible)
8. **Top words per class** — separate top-10 for each label
9. **Source distribution** — pie chart showing data from each source
10. **Summary** — key findings as text

**Important:**
- Use `matplotlib` for plots (it works headless)
- Save every plot to `data/eda/` as PNG (e.g., `plt.savefig("../data/eda/class_distribution.png")`)
- Also generate `data/eda/REPORT.md` with stats embedded as a markdown table
- Adapt analysis to the actual data — if labels are in Russian, handle encoding; if there are many classes, adjust chart layout

### Step 3 — Execute the notebook

Run the notebook and save it with outputs:

```bash
python .claude/skills/eda-generator/scripts/run_notebook.py notebooks/eda.ipynb
```

This executes every cell in-place and saves the notebook with all outputs (plots, tables, print statements).

### Step 4 — Verify

Check that these files exist and are non-empty:
- `notebooks/eda.ipynb` (with outputs)
- `data/eda/class_distribution.png`
- `data/eda/text_length_distribution.png`
- `data/eda/top_words.png`
- `data/eda/REPORT.md`

If any are missing, fix the notebook and re-run.

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/run_notebook.py` | Execute a notebook in-place and save with outputs |

---
name: data-detective
description: "Detect and fix data quality issues: missing values, duplicates, outliers, class imbalance, label normalization"
user-invocable: true
---

# Data Detective

Detects ALL data quality problems, proposes fixes, lets the user pick a strategy, then saves raw-vs-cleaned comparison.

## Trigger

User asks to check / clean / fix / analyze data quality. Or this skill is called after dataset collection.
Examples:
- "проверь качество данных"
- "почисти датасет"
- "find data quality issues"
- "run data detective on combined.parquet"

## Autonomy rules

- **Steps 1, 2**: FULLY AUTONOMOUS — deep analysis of ALL possible issues. No questions.
- **Step 3 is MANDATORY and BLOCKING**: present ALL found problems with proposed fixes per strategy, wait for user to pick.
- **Steps 4, 5, 6**: after user picks — save, generate report and notebook automatically.

## Workflow

### Step 1 — Deep issue discovery (SILENT)

Load the dataset and task type from config:

```python
import pandas as pd
import yaml
from agents.data_quality_agent import DataQualityAgent

df = pd.read_parquet("data/raw/combined.parquet")
with open("config.yaml") as f:
    config = yaml.safe_load(f)
task_type = config["task"]["type"]  # e.g. "News topic text classification"

agent = DataQualityAgent()
report = agent.detect_issues(df)
```

Use `task_type` to contextualize the analysis — what quality issues matter most for this specific task.

**IMPORTANT: Missing labels are NOT a problem.** The next pipeline stage (AnnotationAgent) will label/re-label all data. So:
- `label` column nulls — IGNORE, do not count as missing values
- Class imbalance — report it for visibility, but do NOT treat as something to fix
- Rows without labels — KEEP, never drop them for missing label

Focus missing-value detection on the `text` column and metadata columns only.

**Go beyond the standard checks.** As an LLM, you can analyze the data deeply and find ALL possible problems. The standard `detect_issues()` covers the basics, but YOU must also look for:

1. **Missing values** — nulls in `text` and metadata columns (NOT `label` — see above)
2. **Duplicates** — duplicate texts
3. **Outliers** — text length outliers (IQR method)
4. **Class imbalance** — report for visibility only, NOT a problem to fix
5. **Empty texts** — blank or whitespace-only text fields
6. **Inconsistent labels** — same category with different casing/naming from different sources (e.g., "Business" vs "business", "Sports" vs "sport", "Sci/Tech" vs "technology")
7. **Near-duplicates** — texts that are almost identical (e.g., same article from different sources with minor edits)
8. **Encoding issues** — mojibake, broken unicode, HTML entities in text
9. **Noise in text** — excessive whitespace, control characters, URLs, email addresses that shouldn't be there
10. **Language mixing** — texts in unexpected languages
11. **Too short / too long texts** — texts that are too short to be meaningful or suspiciously long
12. **Source quality** — are some sources contributing mostly garbage?
13. **Any other anomalies** you notice from inspecting the data

Run detection and save findings:
```bash
python .claude/skills/data-detective/scripts/detect.py data/raw/combined.parquet
```

### Step 2 — Apply ALL strategies and compare (SILENT)

**Label normalization is MANDATORY** before applying strategies. If inconsistent labels detected, normalize:

```python
label_map = {
    "business": "Business",
    "sport": "Sports",
    "world": "World",
    "technology": "Sci/Tech",
    "science": "Sci/Tech",
}
df["label"] = df["label"].map(lambda x: label_map.get(x, x))
```

Apply every strategy and collect comparison:

```python
results = {}
for name in ["aggressive", "conservative", "balanced"]:
    df_fixed = agent.fix(df, strategy=name)
    results[name] = {
        "df": df_fixed,
        "comparison": agent.compare(df, df_fixed),
        "report": agent.detect_issues(df_fixed),
    }
```

Do NOT save any cleaned.parquet yet — the user hasn't chosen.

### Step 3 — Present ALL findings and propose fixes (HUMAN-IN-THE-LOOP)

Show the user EVERYTHING you found:

1. **All detected problems** — comprehensive table with every issue, count, severity, examples
2. **Label normalization** — show mapping if any labels were merged
3. **What each strategy will fix** — for every problem found, show how each strategy handles it:

```
Problems detected:

| #  | Problem                | Count    | Severity | Examples                          |
|----|------------------------|----------|----------|-----------------------------------|
| 1  | Inconsistent labels    | 6 pairs  | HIGH     | business↔Business, sport↔Sports   |
| 2  | Text length outliers   | 315      | MEDIUM   | 3 chars min, 2841 chars max       |
| 3  | Class imbalance        | 19.96x   | HIGH     | Sci/Tech: 27.5%, politics: 1.4%   |
| 4  | Near-duplicate texts   | 23       | LOW      | "Reuters - ..." appears 4 times   |
| ...                                                                                    |

Label normalization (will be applied regardless of strategy):
  business → Business, sport → Sports, technology → Sci/Tech, ...

How each strategy handles these problems:

| Metric              | Original | Aggressive | Conservative | Balanced |
|---------------------|----------|------------|--------------|----------|
| Total rows          | 10,900   | 10,585     | 10,900       | 10,870   |
| Outliers            | 315      | 0          | 315          | 12       |
| Imbalance ratio     | 2.4x     | 2.3x       | 2.4x         | 2.4x    |
| Rows removed        | —        | 315 (2.9%) | 0 (0%)       | 30 (0.3%)|

Recommendation:
  For [this specific ML task], I recommend [strategy] because:
  - [specific reason referencing actual numbers]
  - [trade-off explanation]

Pick a strategy (1/2/3) or describe your own:
```

**STOP HERE. Wait for user's response.**

### Step 4 — Save chosen strategy (SILENT)

```bash
python .claude/skills/data-detective/scripts/fix.py balanced data/raw/combined.parquet
```

Save ONLY:
- `data/cleaned/cleaned.parquet` — ONE cleaned dataset

### Step 5 — Generate raw-vs-cleaned report and notebook (SILENT)

**QUALITY_REPORT.md** — save to `data/detective/QUALITY_REPORT.md` with side-by-side comparison of raw vs cleaned ONLY (not all strategies — just before/after the chosen one):

```markdown
# Data Quality Report

## Problems Detected
[all problems from Step 3]

## Label Normalization
[mapping applied]

## Cleaning Strategy: [chosen]
[why this strategy was chosen]

## Before vs After (raw → cleaned)

| Metric              | Raw (before) | Cleaned (after) | Change       |
|---------------------|--------------|-----------------|--------------|
| Total rows          | 10,900       | 10,870          | -30 (-0.3%)  |
| Missing values      | 0            | 0               | 0            |
| Duplicates          | 0            | 0               | 0            |
| Outliers            | 315          | 12              | -303 (-96%)  |
| Empty texts         | 0            | 0               | 0            |
| Imbalance ratio     | 19.96x       | 2.4x            | -17.56x      |
| Unique classes      | 10           | 5               | -5 (merged)  |

## Justification
[detailed explanation of why this strategy is best for this ML task]
```

**Charts** — generate via script:
```bash
python .claude/skills/data-detective/scripts/visualize.py data/raw/combined.parquet data/cleaned/cleaned.parquet
```
Saves to `data/detective/`: `missing_values.png`, `outliers.png`, `class_balance.png`, `before_after_rows.png`, `before_after_class_balance.png`, `before_after_text_lengths.png`

**Comparison report**:
```bash
python .claude/skills/data-detective/scripts/compare.py data/raw/combined.parquet data/cleaned/cleaned.parquet balanced
```
Saves `data/detective/QUALITY_REPORT.md`

**Notebook** — delegate to `quality-report` skill (`.claude/skills/quality-report/SKILL.md`):
1. Generate `notebooks/data_quality.ipynb` with:
   - All issue visualizations
   - Raw vs cleaned comparison (side-by-side)
   - **Markdown cell with full justification**: what was found, what was fixed, why this strategy, how it affects downstream ML
2. Execute with `run_notebook.py`
3. Verify outputs

### Step 6 — Summary

```
Data quality check complete!
   Label normalization: business→Business, sport→Sports, ...
   Problems found: [list all]
   Strategy applied: balanced

   Raw vs Cleaned:
     Rows: 10,900 → 10,870 (-30)
     Outliers: 315 → 12
     Classes: 10 → 5 (merged)
     Imbalance: 19.96x → 2.4x

   Saved: data/cleaned/cleaned.parquet
   Report: data/detective/QUALITY_REPORT.md
   Charts: data/detective/*.png
   Notebook: notebooks/data_quality.ipynb (with results and justification)
```

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/detect.py` | Detect issues, save problems.json |
| `scripts/fix.py` | Apply cleaning strategy, save cleaned.parquet |
| `scripts/compare.py` | Compare raw vs cleaned, generate QUALITY_REPORT.md |
| `scripts/visualize.py` | Generate all PNG charts to data/detective/ |

## Config

Input dataset: `data/raw/combined.parquet` (output of Dataset Collector)
Output dataset: `data/cleaned/cleaned.parquet` (input for Label Master)
Artifacts: `data/detective/` (problems.json, QUALITY_REPORT.md, all PNG charts)

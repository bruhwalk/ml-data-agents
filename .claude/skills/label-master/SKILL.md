---
name: label-master
description: "Auto-label data with LLM (Claude API), assess quality, export disputed examples to Label Studio"
user-invocable: true
---

# Label Master

Auto-labels data using Claude API, generates annotation spec, checks quality, and exports disputed examples to Label Studio for manual review.

## Trigger

User asks to label / annotate / classify data. Or this skill is called after data cleaning.
Examples:
- "разметь данные"
- "label the dataset"
- "запусти авторазметку"

## Autonomy rules

- **Step 1**: FULLY AUTONOMOUS — analyze data. No questions.
- **Step 2 is MANDATORY and BLOCKING**: show analysis, ask user what to do (re-label / label unlabeled / keep as-is).
- **Steps 3, 4, 5**: FULLY AUTONOMOUS — label, assess quality, export.
- **Step 6 is MANDATORY and BLOCKING**: ask user if they want to open Label Studio for disputed examples.
- **Step 7**: If user said yes — start Label Studio. If no — save and finish.

## Workflow

### Step 1 — Analyze data and propose taxonomy (SILENT)

Load cleaned dataset and task type from config:

```python
import pandas as pd
import yaml

df = pd.read_parquet("data/cleaned/cleaned.parquet")
with open("config.yaml") as f:
    config = yaml.safe_load(f)
task_type = config["task"]["type"]  # e.g. "News topic text classification"
```

Analyze:
- What labels exist in the `label` column
- How many unique labels, their distribution
- How many rows have labels vs missing labels
- Are labels consistent (casing, naming)
- Extract 2-3 example texts per existing label

**Propose a new taxonomy.** Based on the task type from config AND analysis of the actual texts, the LLM should:
1. Read a sample of texts (50-100) to understand what topics/categories actually appear
2. Consider whether existing labels are sufficient or need refinement
3. Propose NEW classes that better fit the task — the taxonomy does NOT have to match the original labels
4. For example, for "News topic text classification" the LLM might propose splitting "Sci/Tech" into "Science" and "Technology", or adding "Health", "Entertainment", etc.
5. Each proposed class must have a clear definition

### Step 2 — Confirm task with user (HUMAN-IN-THE-LOOP)

Present findings, proposed taxonomy, and ask what to do:

```
Task: News topic text classification (from config.yaml)

Dataset analysis:
   Total rows: 10,585
   Labeled: 9,685 (91.5%)
   Unlabeled: 900 (8.5%)
   Existing labels: 5 (World, Sports, Business, Sci/Tech, Politics)

Proposed taxonomy (based on data analysis):

| # | Label        | Definition                                    | Existing match     |
|---|-------------|-----------------------------------------------|---------------------|
| 1 | World       | International news, diplomacy, conflicts      | World               |
| 2 | Sports      | Sports events, results, athletes              | Sports              |
| 3 | Business    | Business, finance, economy, markets           | Business            |
| 4 | Science     | Scientific research, discoveries              | was part of Sci/Tech|
| 5 | Technology  | Tech industry, gadgets, software, AI          | was part of Sci/Tech|
| 6 | Politics    | Domestic politics, elections, government       | Politics            |
| 7 | Health      | NEW — medical news, public health, pharma     | —                   |

Why this taxonomy:
  - Split Sci/Tech into Science + Technology (distinct topics in the data)
  - Added Health (found 47 health-related texts currently labeled as Sci/Tech or World)

What would you like to do?
  1. Re-label ALL texts with the proposed taxonomy
  2. Re-label ALL texts but adjust the taxonomy first
  3. Label only UNLABELED texts (900 rows) with the proposed taxonomy
  4. Keep labels as-is (just fill missing labels programmatically)
```

**STOP HERE. Wait for user's response.**

User may:
- Choose option 1, 2, 3, or 4
- Adjust labels (add/remove/rename/merge)
- Change confidence threshold

### Step 3 — Auto-label with Claude API (SILENT)

Based on user's choice:

**Option 1 (re-label all with proposed taxonomy):** Run LLM classification on every text using the confirmed/adjusted taxonomy.
**Option 2 (re-label all with adjusted taxonomy):** Same as 1 but user changed the classes first.
**Option 3 (label unlabeled only):** Only classify rows where `label` is null.
**Option 4 (keep as-is):** Copy `label` → `auto_label`, set `confidence=1.0`, fill nulls with most common label.

For options 1 and 2, use the auto_label script:

```bash
python .claude/skills/label-master/scripts/auto_label.py "World,Sports,Business,Sci/Tech,Politics" "topic classification" data/cleaned/cleaned.parquet 10
```

For option 3 (label unlabeled only), use the label_unlabeled script:

```bash
python .claude/skills/label-master/scripts/label_unlabeled.py "World,Sports,Business,Sci/Tech,Politics" "topic classification" data/cleaned/cleaned.parquet 10
```

Both scripts read `allow_new_labels` from `config.yaml` automatically. To override, pass `--allow-new-labels`.

The Claude API classifies each text and assigns:
- `auto_label` — the predicted label
- `confidence` — 0.0 to 1.0 (LLM's self-assessed certainty)
- `is_disputed` — True if confidence < threshold (default 0.7)

**IMPORTANT:** This calls Claude API. With 10k texts at batch_size=10, that's ~1000 API calls. Show progress.

### Step 4 — Generate spec and check quality (SILENT)

Generate annotation specification:

```python
from agents.annotation_agent import AnnotationAgent
agent = AnnotationAgent()
agent.generate_spec(df_labeled, task="topic_classification", labels=confirmed_labels)
```

Saves `data/annotation/annotation_spec.md` with:
- Task description
- Class definitions
- 3+ examples per class (highest confidence)
- Edge cases and annotation guidelines

Check quality:

```bash
python .claude/skills/label-master/scripts/check_quality.py data/labeled/labeled.parquet 0.7
```

### Step 5 — Export disputed examples (SILENT)

Export low-confidence examples to Label Studio format:

```bash
python .claude/skills/label-master/scripts/export_ls.py "World,Sports,Business,Sci/Tech,Politics" data/labeled/labeled.parquet 0.7
```

Generates:
- `data/annotation/labelstudio_tasks.json` — tasks with pre-annotations
- `data/annotation/labelstudio_config.xml` — project config

### Step 6 — Ask about Label Studio (HUMAN-IN-THE-LOOP)

Show results and ask:

```
Labeling complete!
   Total labeled: 10,585
   Confidence: mean=0.87, median=0.91
   Disputed (confidence < 0.7): 342 (3.2%)
   Label distribution: World=25%, Sports=22%, Business=24%, Sci/Tech=28%, Politics=1%

   Disputed examples exported to: data/annotation/labelstudio_tasks.json (342 tasks)

Would you like to manually review disputed examples in Label Studio now? (yes/no)
```

**STOP HERE. Wait for user's response.**

### Step 7a — If YES: Start Label Studio

```bash
python .claude/skills/label-master/scripts/start_ls.py "Topic Classification" "World,Sports,Business,Sci/Tech,Politics"
```

This will:
1. Start Label Studio on localhost:8080 (if not already running)
2. Create a project with the label taxonomy
3. Import disputed tasks with pre-annotations from Claude
4. Open browser to the project

Tell the user:

```
Label Studio is open at http://localhost:8080

Instructions:
  1. Review the pre-annotated examples
  2. Fix any incorrect labels
  3. When done, export annotations:
     Project → Export → JSON
     Save to: data/annotation/ls_export.json

Then tell me "ready" or "done" and I'll merge the manual labels back.
```

**STOP HERE. Wait for user to finish labeling.**

When user says they're done:

```bash
python .claude/skills/label-master/scripts/import_ls.py data/annotation/ls_export.json data/labeled/labeled.parquet
```

This merges manual labels back into `labeled.parquet` (confidence=1.0 for manually reviewed).

### Step 7b — If NO: Save and finish

Save as-is. Disputed examples keep their auto_label with low confidence.

### Step 8 — Generate notebook and summary (SILENT)

Create `notebooks/annotation.ipynb` with:
1. Label distribution bar chart
2. Confidence distribution histogram
3. Disputed examples table (text, auto_label, confidence)
4. If original labels exist: confusion matrix (auto vs original), Cohen's kappa
5. **Markdown cell with justification**: labeling approach, what the model was confident/uncertain about

Execute:
```bash
python .claude/skills/eda-generator/scripts/run_notebook.py notebooks/annotation.ipynb
```

### Step 9 — Final summary

```
Annotation complete!
   Model: Claude API (claude-sonnet-4-20250514)
   Labels: World, Sports, Business, Sci/Tech, Politics
   Total labeled: 10,585
   Manually reviewed: 342 (in Label Studio)
   Confidence: mean=0.91
   Disputed remaining: 0

   Saved:
     data/labeled/labeled.parquet
     data/annotation/annotation_spec.md
     data/annotation/quality_metrics.json
     data/annotation/labelstudio_tasks.json
     notebooks/annotation.ipynb (with results)
```

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/auto_label.py` | Classify ALL texts via Claude API, save labeled.parquet |
| `scripts/label_unlabeled.py` | Classify only UNLABELED texts (Option 3), save labeled.parquet |
| `scripts/check_quality.py` | Compute Cohen's kappa, confidence stats, save metrics |
| `scripts/export_ls.py` | Export disputed examples to Label Studio JSON + config XML |
| `scripts/start_ls.py` | Start Label Studio, create project, import tasks, open browser |
| `scripts/import_ls.py` | Merge manual labels from LS export back into labeled.parquet |

## Config

Input dataset: `data/cleaned/cleaned.parquet` (output of Data Detective)
Output dataset: `data/labeled/labeled.parquet` (input for Active Learning Agent)
Artifacts: `data/annotation/` (spec, metrics, LS tasks, LS config, LS export)
Label Studio: `pip install label-studio label-studio-sdk`, runs on localhost:8080
API key: `ANTHROPIC_API_KEY` from `.env`

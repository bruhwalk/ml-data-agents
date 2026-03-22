---
name: model-trainer
description: "Final model training — LLM picks model from AL results, trains on all data, saves model + metrics + report"
user-invocable: true
---

# Model Trainer

Trains the final production model. Claude API analyzes AL experiment results and dataset
to choose the best model, then trains it on all labeled data, evaluates, and saves artifacts.

## Trigger

User asks to train final model, or this skill is called after Active Learner.
Examples:
- "обучи финальную модель"
- "train the final model"
- "запусти обучение"

## Autonomy rules

- **ALL steps are FULLY AUTONOMOUS** — no human-in-the-loop checkpoints.
- Run everything silently, show results at the end.

## Workflow

### Step 1 — Load data and AL results (SILENT)

```python
import pandas as pd
import yaml
from agents.model_trainer_agent import ModelTrainerAgent

df = pd.read_parquet("data/labeled/labeled.parquet")
with open("config.yaml") as f:
    config = yaml.safe_load(f)
task_type = config["task"]["type"]
```

### Step 2 — LLM selects model (SILENT)

Claude API reads AL experiment results (`data/active_learning/`) and dataset stats,
then chooses the best model for training on all data:

```python
agent = ModelTrainerAgent()
recommendation = agent.select_model(df, task_type=task_type)
# → {"model": "logreg_balanced", "reasoning": "..."}
```

### Step 3 — Train and evaluate (SILENT)

Run via script:

```bash
python .claude/skills/model-trainer/scripts/train.py data/labeled/labeled.parquet
```

Or via Python:
```python
result = agent.run(parquet_path="data/labeled/labeled.parquet", task_type=task_type)
```

This:
1. Splits data 80/20 (stratified)
2. Trains TF-IDF + chosen model on train set
3. Evaluates on test set
4. Saves model to `models/final_model.pkl`
5. Saves metrics, confusion matrix, per-class F1 chart
6. Generates MODEL_REPORT.md

### Step 4 — Generate notebook (SILENT)

Create `notebooks/model_training.ipynb` with:
1. Model selection reasoning
2. Training details
3. Confusion matrix visualization
4. Per-class F1 chart
5. Classification report
6. **Markdown cell with analysis**

Execute:
```bash
python .claude/skills/eda-generator/scripts/run_notebook.py notebooks/model_training.ipynb
```

### Step 5 — Summary

```
Final model trained!
   LLM selected: Logistic Regression (balanced)
     Reason: "Severe class imbalance requires balanced weights..."

   Training: 8,468 train / 2,117 test
   Model: TF-IDF (15k features) + LogReg (balanced)

   Metrics:
     Accuracy:        0.9234
     F1 (macro):      0.7891
     F1 (weighted):   0.9198
     Precision:       0.8012
     Recall:          0.7834

   Saved:
     models/final_model.pkl
     models/model_config.json
     data/model/confusion_matrix.png
     data/model/per_class_f1.png
     data/model/metrics.json
     data/model/MODEL_REPORT.md
     notebooks/model_training.ipynb
```

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/train.py` | Full pipeline: LLM model selection + train + evaluate + save |

## Config

Input dataset: `data/labeled/labeled.parquet` (output of Label Master)
AL results: `data/active_learning/` (output of Active Learner)
Output model: `models/final_model.pkl`, `models/model_config.json`
Output artifacts: `data/model/` (metrics, charts, report)
API key: `ANTHROPIC_API_KEY` from `.env`

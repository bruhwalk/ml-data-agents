# ML Data Agents

Мультиагентная система для полного цикла подготовки данных и обучения модели на текстовых данных.
Реализована как набор Python-агентов + Claude Code Skills.

## Быстрый старт

```bash
pip install -r requirements.txt
```

Создать `.env`:
```
ANTHROPIC_API_KEY=sk-...          # обязательно — разметка (Step 3), выбор модели (Steps 4-5)
KAGGLE_USERNAME=your_username     # для загрузки датасетов с Kaggle (Step 1)
KAGGLE_KEY=your_key               # https://www.kaggle.com/settings → API → Create New Token
LABEL_STUDIO_API_KEY=...          # для ручной разметки спорных примеров (Step 3, опционально)
```

Запустить:
```bash
python run_data_pipeline.py "News topic text classification"
```

Или через Claude Code:
```
/data-pipeline классификация новостей по темам
```

---

## Пайплайн

```
Step 1              Step 2              Step 3              Step 4              Step 5
DataCollection  →   DataQuality     →   Annotation      →   ActiveLearning  →   ModelTrainer
⏸ выбор            ⏸ стратегия        ⏸ таксономия        автономно           автономно
  источников          чистки             + Label Studio

combined.parquet    cleaned.parquet     labeled.parquet     REPORT.md           final_model.pkl
                                                            learning_curve.png  MODEL_REPORT.md
                                                                                ↓
                                                                            PIPELINE_REPORT.md
```

⏸ — checkpoint, где пайплайн останавливается и ждёт решения пользователя.

---

## Агенты

### 1. DataCollectionAgent

Сбор данных из нескольких источников с унификацией в единую схему.

| Метод | Описание |
|-------|----------|
| `search_sources(query)` | Поиск датасетов на HuggingFace и Kaggle |
| `validate_source(source)` | Проверка доступности источника |
| `load_dataset(name, source)` | Загрузка с HF / Kaggle |
| `scrape(url, selector)` | Скрапинг веб-страницы |
| `fetch_api(endpoint)` | Запрос к JSON API |
| `fetch_rss(url)` | Парсинг RSS/Atom |
| `merge(sources)` | Объединение в единый DataFrame |
| `run(sources)` | Полный цикл: collect → merge → save |

**Схема данных** — все источники приводятся к единому формату:

| Колонка | Тип | Описание |
|---------|-----|----------|
| `text` | str | Текст |
| `label` | str | Метка класса (может быть пустой) |
| `source` | str | Идентификатор источника |
| `collected_at` | str | Временная метка ISO 8601 |

### 2. DataQualityAgent

Обнаружение и устранение проблем качества данных.

| Метод | Описание |
|-------|----------|
| `detect_issues(df)` | Поиск: пропуски, дубликаты, выбросы, дисбаланс, пустые тексты |
| `fix(df, strategy)` | Применение стратегии чистки |
| `compare(before, after)` | Сравнение метрик до/после |
| `list_strategies()` | Доступные пресеты |

**Стратегии чистки:**

| Стратегия | Пропуски | Дубликаты | Выбросы | Пустые |
|-----------|----------|-----------|---------|--------|
| aggressive | Удалить | Удалить | Удалить (IQR) | Удалить |
| balanced | Заполнить | Удалить | Удалить (z>3) | Удалить |
| conservative | Заполнить | Оставить | Оставить | Оставить |

### 3. AnnotationAgent

Автоматическая разметка через Claude API + экспорт спорных примеров в Label Studio.

| Метод | Описание |
|-------|----------|
| `auto_label(df, labels)` | Zero-shot классификация через Claude API |
| `generate_spec(df, task, labels)` | Генерация спецификации разметки |
| `check_quality(df)` | Cohen's κ, confidence, распределение |
| `export_to_labelstudio(df)` | Экспорт спорных примеров в Label Studio |
| `import_from_labelstudio(df)` | Импорт исправлений из Label Studio |

Добавляет колонки: `auto_label`, `confidence`, `is_disputed`.

### 4. ActiveLearningAgent

Эксперимент по сравнению стратегий выборки для active learning.

| Метод | Описание |
|-------|----------|
| `select_model(df, task_type)` | LLM выбирает модель и seed size |
| `fit(labeled_df)` | Обучение TF-IDF + модель |
| `query(pool_df, strategy)` | Выбор самых информативных примеров |
| `evaluate(test_df)` | Оценка accuracy + F1 |
| `compare_strategies(df)` | Сравнение entropy / margin / random |
| `report(results)` | Генерация отчёта + learning curve |

**Стратегии:** entropy (H(p)), margin (разница top-2), random (baseline).

**Модели:** LogReg, LogReg balanced, SVM (calibrated), Multinomial NB.

### 5. ModelTrainerAgent

Обучение финальной модели на всех данных.

| Метод | Описание |
|-------|----------|
| `select_model(df, al_dir)` | LLM выбирает модель по результатам AL |
| `train(train_df, model_key)` | TF-IDF (15K features) + модель |
| `evaluate(test_df)` | Метрики + confusion matrix + per-class F1 |
| `save_model(model_dir)` | Сохранение в joblib + config |
| `run(parquet_path)` | Полный цикл: select → train → evaluate → save |

---

## Что куда сохраняется

```
ml-data-agents/
│
│── run_data_pipeline.py              ← Standalone скрипт (вместо /data-pipeline)
│── config.yaml                       ← Конфигурация: тип задачи, настройки источников
│── PIPELINE_REPORT.md                ← Итоговый отчёт по всему пайплайну
│
├── agents/                           ← Python-классы агентов (ядро)
│   ├── data_collection_agent.py
│   ├── data_quality_agent.py
│   ├── annotation_agent.py
│   ├── active_learning_agent.py
│   └── model_trainer_agent.py
│
├── data/
│   ├── raw/                          ← Step 1: сырые данные
│   │   ├── combined.parquet              общий датасет
│   │   └── *.parquet                     отдельные источники
│   │
│   ├── eda/                          ← Step 1: EDA-отчёт
│   │   ├── REPORT.md
│   │   └── *.png                         графики (распределения, top words)
│   │
│   ├── cleaned/                      ← Step 2: очищенные данные
│   │   └── cleaned.parquet
│   │
│   ├── detective/                    ← Step 2: отчёт о качестве
│   │   ├── QUALITY_REPORT.md
│   │   ├── problems.json
│   │   └── *.png                         графики (before/after, outliers)
│   │
│   ├── labeled/                      ← Step 3: размеченные данные
│   │   └── labeled.parquet
│   │
│   ├── annotation/                   ← Step 3: артефакты разметки
│   │   ├── annotation_spec.md            спецификация
│   │   ├── quality_metrics.json          метрики (confidence, kappa)
│   │   ├── labelstudio_tasks.json        задачи для Label Studio
│   │   └── labelstudio_config.xml        конфиг Label Studio
│   │
│   ├── active_learning/              ← Step 4: результаты AL
│   │   ├── REPORT.md
│   │   ├── learning_curve.png
│   │   └── history_*.json                история по каждой стратегии
│   │
│   └── model/                        ← Step 5: метрики и графики модели
│       ├── MODEL_REPORT.md
│       ├── metrics.json
│       ├── classification_report.txt
│       ├── confusion_matrix.png
│       └── per_class_f1.png
│
├── models/                           ← Step 5: обученная модель
│   ├── final_model.pkl                   модель + vectorizer (joblib)
│   └── model_config.json                 параметры модели
│
├── notebooks/                        ← Jupyter ноутбуки с результатами
│   ├── eda.ipynb
│   ├── data_quality.ipynb
│   ├── annotation.ipynb
│   └── al_experiment.ipynb
│
└── .claude/skills/                   ← Claude Code Skills (инструкции + скрипты)
    ├── dataset-collector/
    │   ├── SKILL.md
    │   └── scripts/
    │       ├── search_hf.py
    │       ├── search_kaggle.py
    │       └── eda.py
    ├── data-detective/
    │   ├── SKILL.md
    │   └── scripts/
    │       ├── detect.py
    │       ├── fix.py
    │       ├── compare.py
    │       └── visualize.py
    ├── label-master/
    │   ├── SKILL.md
    │   └── scripts/
    │       ├── auto_label.py
    │       ├── check_quality.py
    │       ├── export_ls.py
    │       ├── import_ls.py
    │       ├── label_unlabeled.py
    │       └── start_ls.py
    ├── active-learner/
    │   ├── SKILL.md
    │   └── scripts/
    │       ├── run_al.py
    │       └── visualize.py
    ├── model-trainer/
    │   ├── SKILL.md
    │   └── scripts/
    │       └── train.py
    ├── data-pipeline/
    │   ├── SKILL.md
    │   └── scripts/
    │       └── run_pipeline.py
    ├── eda-generator/
    │   └── SKILL.md
    └── quality-report/
        └── SKILL.md
```

---

## Claude Code

Проект спроектирован для работы с [Claude Code](https://docs.anthropic.com/en/docs/claude-code) — CLI-инструментом Anthropic.
Каждый агент реализован как **Claude Code Skill** — файл `SKILL.md` с инструкциями, которые Claude читает и выполняет.

### Как это работает

1. Пользователь пишет `/data-pipeline классификация новостей` в терминале Claude Code
2. Claude находит `.claude/skills/data-pipeline/SKILL.md` и читает инструкции
3. SKILL.md говорит Claude какие скрипты запускать и где ждать решения пользователя
4. Claude запускает скрипты из `.claude/skills/*/scripts/`, которые импортируют классы из `agents/`
5. На checkpoint'ах Claude показывает результаты и спрашивает пользователя

### Доступные команды

| Команда | Skill | Что делает |
|---------|-------|-----------|
| `/data-pipeline` | data-pipeline | Полный пайплайн (все шаги) |
| `/dataset-collector` | dataset-collector | Поиск и сбор данных |
| `/data-detective` | data-detective | Проверка и чистка данных |
| `/label-master` | label-master | Авторазметка + Label Studio |
| `/active-learner` | active-learner | Эксперимент active learning |
| `/model-trainer` | model-trainer | Обучение финальной модели |
| `/eda-generator` | eda-generator | Генерация EDA-ноутбука |
| `/quality-report` | quality-report | Генерация отчёта о качестве |

### Пример сессии

```
> /data-pipeline классификация новостей по темам

════════════════════════════════════════════════════
  STEP 1: DATA COLLECTION
════════════════════════════════════════════════════

Searching datasets for: классификация новостей по темам

Validated sources:
  1. [hf] fancyzhx/ag_news
  2. [kaggle] ag-news-classification-dataset
  3. [api] Guardian API

Enter source numbers to collect (comma-separated), or 'all':
> all

Collected 10900 rows total.

════════════════════════════════════════════════════
  STEP 2: DATA QUALITY CHECK
════════════════════════════════════════════════════

Detecting issues...
  Duplicates: 315 (2.9%)
  Empty texts: 0

Choose strategy (aggressive / balanced / conservative):
> balanced

Applied 'balanced': 10900 → 10585 rows

════════════════════════════════════════════════════
  STEP 3: AUTO-LABELING
════════════════════════════════════════════════════

Existing labels found: [Business, Sci/Tech, Sports, World]
Proceed with labeling? [Y/n]:
> Y

Auto-labeling 10585 rows...
  Confidence: mean=0.9865
  Disputed: 14 (0.13%)

...steps 4-6 run autonomously...

════════════════════════════════════════════════════
  PIPELINE COMPLETE
════════════════════════════════════════════════════

  Task: классификация новостей по темам
  Accuracy: 0.89, F1: 0.77
  Model: models/final_model.pkl
  Report: PIPELINE_REPORT.md
```

### Отдельные команды

Каждый агент можно запускать отдельно — не обязательно весь пайплайн:

```
> /dataset-collector собери датасет для sentiment analysis отзывов на фильмы

> /data-detective проверь качество данных в data/raw/combined.parquet

> /label-master разметь данные, метки: positive, negative, neutral
```

Claude сам прочитает SKILL.md нужного агента, выполнит шаги, покажет результаты и спросит на checkpoint'ах.

### Без Claude Code

Всё то же самое можно сделать без Claude Code:

| Способ | Команда |
|--------|---------|
| Полный пайплайн | `python run_data_pipeline.py "task description"` |
| Отдельные скрипты | `python .claude/skills/label-master/scripts/auto_label.py data.parquet "Label1,Label2"` |
| Python API | `from agents import AnnotationAgent; agent = AnnotationAgent(); ...` |

---

## Архитектура

```
┌──────────────────────────────────────────────────────────┐
│  Точки входа                                             │
│                                                          │
│  /data-pipeline (Claude Code)    run_data_pipeline.py    │
│         │                               │                │
│         ▼                               ▼                │
│  .claude/skills/*/SKILL.md ──→ .claude/skills/*/scripts/ │
│                                         │                │
│                                         ▼                │
│                                    agents/*.py           │
│                                   (бизнес-логика)        │
│                                         │                │
│                          ┌──────────────┼──────────┐     │
│                          ▼              ▼          ▼     │
│                     data/**        models/**  notebooks/ │
│                    (данные)        (модель)   (ноутбуки) │
└──────────────────────────────────────────────────────────┘
```

3 слоя:
- **Skills** (SKILL.md) — инструкции для Claude, описывают *что делать*
- **Scripts** (scripts/*.py) — CLI-обёртки, запускаются из терминала
- **Agents** (agents/*.py) — классы с бизнес-логикой, переиспользуемые

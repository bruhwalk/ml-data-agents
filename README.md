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

```python
agent = DataCollectionAgent(config="config.yaml")
```

**5 типов источников:**

| Тип | Метод | Пример |
|-----|-------|--------|
| HuggingFace | `load_dataset(name, source="hf")` | `fancyzhx/ag_news` |
| Kaggle | `load_dataset(name, source="kaggle")` | `amananandrai/ag-news-classification-dataset` |
| Web scraping | `scrape(url, selector)` | Любая HTML-страница + CSS-селектор |
| REST API | `fetch_api(endpoint, params)` | Guardian API, NewsAPI и др. |
| RSS/Atom | `fetch_rss(url)` | BBC News, Reddit и др. |

Агент автоматически ищет подходящие датасеты через `search_sources(query)`, валидирует их доступность через `validate_source()`, и объединяет в единый DataFrame.

Все источники приводятся к **единой схеме**:

| Колонка | Тип | Описание |
|---------|-----|----------|
| `text` | str | Текст |
| `label` | str | Метка класса (может быть пустой — разметка на шаге 3) |
| `source` | str | Идентификатор источника (например `huggingface:fancyzhx/ag_news`) |
| `collected_at` | str | Временная метка ISO 8601 |

Настройки в `config.yaml`: лимиты на источник (`max_samples_per_source: 5000`), таймауты, включение/отключение типов источников.

### 2. DataQualityAgent

Обнаружение и устранение проблем качества данных.

```python
agent = DataQualityAgent(text_column="text", label_column="label")
```

**Обнаруживает 5 типов проблем:**
- **Пропущенные значения** — nulls по каждой колонке
- **Дубликаты** — одинаковые тексты
- **Выбросы** — аномальная длина текста (метод IQR: q1 − 1.5·IQR ... q3 + 1.5·IQR)
- **Дисбаланс классов** — отношение крупнейшего к мельчайшему классу (порог > 3.0)
- **Пустые тексты** — пустые строки или только пробелы

**3 готовые стратегии чистки** (`fix(df, strategy)`):

| Стратегия | Пропуски | Дубликаты | Выбросы | Пустые |
|-----------|----------|-----------|---------|--------|
| `aggressive` | Удалить строки | Удалить | Удалить (IQR) | Удалить |
| `balanced` | Заполнить пустой строкой | Удалить | Удалить (z-score > 3) | Удалить |
| `conservative` | Заполнить пустой строкой | Оставить | Оставить | Оставить |

Можно передать свою стратегию как dict:
```python
agent.fix(df, strategy={"missing": "drop", "duplicates": "drop", "outliers": "keep", "empty_texts": "drop"})
```

`compare(df_before, df_after)` — сравнивает метрики до и после чистки: количество строк, пропуски, дубликаты, выбросы, дисбаланс.

### 3. AnnotationAgent

Автоматическая разметка текстов через Claude API с контролем качества и интеграцией с Label Studio.

```python
agent = AnnotationAgent(text_column="text", label_column="label", model="claude-sonnet-4-20250514")
```

**Авторазметка** (`auto_label`) — отправляет тексты в Claude API батчами (по 10), каждый текст обрезается до 500 символов. LLM возвращает метку + confidence (0.0–1.0). Добавляет 3 колонки:

| Колонка | Описание |
|---------|----------|
| `auto_label` | Присвоенная метка |
| `confidence` | Уверенность модели (0.0–1.0) |
| `is_disputed` | `True` если confidence < порога (по умолчанию 0.7) |

**4 варианта использования:**
1. **Разметить всё с нуля** — передать список меток, LLM классифицирует каждый текст
2. **Разметить только неразмеченные** — через `label_unlabeled.py`, существующие метки сохраняются
3. **Разрешить новые метки** — `allow_new_labels=True`, LLM может предложить новые классы (при confidence > 0.9)
4. **Ручная разметка спорных** — экспорт в Label Studio, правка, импорт обратно

**Контроль качества** (`check_quality`):
- Cohen's κ — согласованность с исходными метками (если есть)
- Распределение confidence (mean, median, std)
- Количество и процент спорных примеров (low confidence)
- Распределение меток по классам

**Интеграция с Label Studio:**
- `export_to_labelstudio(df)` — экспорт спорных примеров в JSON (с предзаполненными predictions)
- `generate_ls_config(labels)` — генерация XML-конфига проекта
- `start_ls.py` — запуск Label Studio (находит executable на Windows/Linux)
- `import_from_labelstudio(df)` — импорт ручных исправлений обратно (confidence устанавливается в 1.0)

### 4. ActiveLearningAgent

Эксперимент: сколько размеченных примеров реально нужно для обучения? Сравнивает стратегии умного отбора данных.

```python
agent = ActiveLearningAgent(model="logreg", text_col="text", label_col="auto_label")
```

**LLM выбирает модель и параметры** — `select_model()` отправляет в Claude API статистику датасета (размер, количество классов, дисбаланс, средняя длина текста) и получает рекомендацию.

**4 доступные модели (TF-IDF + sklearn):**

| Ключ | Модель | Когда подходит |
|------|--------|----------------|
| `logreg` | Logistic Regression | Хороший default для текстов |
| `logreg_balanced` | LogReg с `class_weight='balanced'` | Несбалансированные классы |
| `svm` | Linear SVM + Platt scaling | Высокоразмерные данные |
| `nb` | Multinomial Naive Bayes | Маленькие датасеты |

**3 стратегии выборки** (`query(pool_df, strategy)`):

| Стратегия | Формула | Идея |
|-----------|---------|------|
| `entropy` | H(p) = −Σ pᵢ log pᵢ | Берём примеры с максимальной неопределённостью |
| `margin` | p₁ − p₂ (разница top-2) | Берём примеры где модель колеблется между двумя классами |
| `random` | Случайная выборка | Baseline для сравнения |

**Цикл эксперимента** (`compare_strategies`):
1. Фиксированный split: seed (начальная выборка) + pool (нетронутые) + test (20%)
2. Для каждой стратегии: обучение на seed → query самых полезных из pool → добавление → повторение
3. На каждой итерации — свежая модель (не дообучение), оценка accuracy + F1 macro
4. Результат — learning curves для сравнения стратегий

**Генерирует:** `learning_curve.png`, `REPORT.md` с анализом экономии (на сколько % entropy лучше random), `history_*.json` с полной историей.

### 5. ModelTrainerAgent

Обучение финальной модели на всех размеченных данных. LLM сама выбирает оптимальную модель.

```python
agent = ModelTrainerAgent(text_col="text", label_col="auto_label")
```

**LLM выбирает модель** — `select_model()` читает результаты AL-эксперимента (какая стратегия/модель победила), статистику данных, и просит Claude API рекомендовать модель для финального обучения. Это не обязательно та же модель что в AL — LLM может выбрать другую исходя из полных данных.

**Те же 4 модели** что и у AL: `logreg`, `logreg_balanced`, `svm`, `nb` — все поверх TF-IDF (15 000 features, uni+bigrams).

**Полный цикл** (`run()`):
1. LLM анализирует данные + результаты AL → выбирает модель
2. Stratified train/test split (80/20)
3. Обучение TF-IDF + выбранная модель
4. Оценка на тесте: accuracy, F1 macro/weighted, precision, recall
5. Визуализации: confusion matrix, per-class F1 bar chart
6. Сохранение модели в `models/final_model.pkl` (joblib) + `model_config.json`
7. Генерация `MODEL_REPORT.md` с обоснованием выбора и метриками

**Метрики по классам** — для каждого класса отдельно: precision, recall, F1, support.

```python
# Использование обученной модели
import joblib
bundle = joblib.load("models/final_model.pkl")
vectorizer = bundle["vectorizer"]
model = bundle["model"]
prediction = model.predict(vectorizer.transform(["Breaking news: stocks plummet"]))
```

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

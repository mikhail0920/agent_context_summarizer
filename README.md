# Agent Context Summarizer

Лёгкая библиотека экстрактивной суммаризации длинных контекстов LLM-агентов.
Она сжимает диалоги и tool traces, стараясь сохранить не только главную тему,
но и редкие операционные факты: ограничения, решения, ошибки, команды, пути к
файлам, результаты тестов и пользовательские предпочтения.

Проект не требует загрузки нейросетевой модели. По умолчанию используются
лексические признаки и графовая центральность; при необходимости можно передать
внешнюю embedding-функцию.

## Задача

Обычный extractive summarizer часто выбирает предложения, похожие на большую
часть документа. Для агентного контекста этого недостаточно: одна строка с
запретом изменять миграцию или точным текстом ошибки может быть важнее десятков
сообщений на основную тему.

`AgentContextSummarizer` решает две связанные задачи:

1. выбирает ограниченное число информативных предложений;
2. извлекает структурированное состояние агента: задачу, ограничения, решения,
   ошибки, результаты инструментов и другие защищённые факты.

Результат остаётся экстрактивным: библиотека не генерирует новые факты и не
перефразирует исходный текст.

## Как работает алгоритм

Для каждого предложения вычисляются несколько сигналов:

- **graph centrality** — близость к основной теме по PageRank над графом
  похожести предложений;
- **query relevance** — соответствие текущей задаче пользователя;
- **rarity** — бонус фактам, непохожим на остальной контекст;
- **protected anchors** — правила для ограничений, решений, ошибок, тестов,
  команд, файлов, чисел, секретов и предпочтений;
- **recency** — небольшой приоритет более свежей информации;
- **structure** — признаки структурированных сообщений и tool traces.

После объединения оценок алгоритм сначала резервирует часть бюджета для
защищённых фактов, затем заполняет оставшиеся позиции релевантными и
нередундантными предложениями. Выбранные предложения возвращаются в исходном
порядке.

```text
Длинный контекст
      ↓
Разбиение на предложения
      ↓
Centrality + query + rarity + anchors + recency
      ↓
Защищённый бюджет и удаление дубликатов
      ↓
Структурированное состояние + краткий extractive summary
```

## Установка

Требуется Python 3.10 или новее.

```powershell
cd C:\Dev\summarizer
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[test]"
```

У библиотеки нет обязательных runtime-зависимостей вне стандартной библиотеки
Python.

## Использование

```python
from agent_summarizer import AgentContextSummarizer

text = open("long_agent_trace.txt", encoding="utf-8").read()

result = AgentContextSummarizer().summarize(
    text,
    query="исправить тесты, не изменяя миграции",
    max_sentences=10,
)

print(result.text)
print(result.labels)
print(result.compression_ratio)
print(result.memory_units)
```

`SummaryResult` содержит:

- `text` — итоговый текст;
- `sentences` — выбранные предложения и их оценки;
- `all_candidates` — оценки всех предложений;
- `compression_ratio` — отношение длины summary к исходному контексту;
- `memory_units` — извлечённое структурированное состояние;
- `labels` — типы фактов, попавших в summary.

### Семантические embeddings

По умолчанию похожесть считается по лексическим token counts. Для семантической
близости можно передать любую функцию, возвращающую векторы:

```python
from sentence_transformers import SentenceTransformer

from agent_summarizer import AgentContextSummarizer

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
summarizer = AgentContextSummarizer(
    embedder=lambda texts: model.encode(list(texts))
)
```

`sentence-transformers` не входит в зависимости проекта: конкретную embedding-
модель выбирает пользователь библиотеки.

### Настройка весов

```python
from agent_summarizer import AgentContextSummarizer, SummarizerConfig

config = SummarizerConfig(
    max_sentences=12,
    protected_budget_ratio=0.65,
    centrality_weight=0.34,
    query_weight=0.18,
    rarity_weight=0.10,
    anchor_weight=0.25,
    recency_weight=0.08,
    structure_weight=0.05,
)

summarizer = AgentContextSummarizer(config)
```

Файл `utils.py` сохранён как compatibility layer для старых вызовов
`summarize(...)`.

## Эксперименты

В проекте есть два уровня проверки.

### Curated stress fixtures

`benchmarks/real_agent_scenarios.py` содержит длинные сценарии, построенные по
структуре известных семейств задач:

- SWE-bench — исправление issue в репозитории;
- SWE-agent — сжатие траектории с командами и результатами тестов;
- tau-bench — сохранение политики и целевого состояния сервиса;
- ToolBench — порядок API-вызовов и точные параметры инструментов.

Это регрессионные stress fixtures, а не доказательство качества на живых данных.
Для каждого сценария заданы обязательные факты и максимальный compression ratio.

```powershell
python benchmarks/real_agent_scenarios.py
```

### Live Hugging Face benchmark

`benchmarks/live_hf_agent_benchmark.py` загружает реальные публичные строки из:

- `SWE-bench/SWE-bench_Lite`;
- `tuandunghcmut/toolbench-v1`;
- `jkazdan/taubench_traces_training_data`;
- `amityco/tau-bench-retail-train-next-action`.

```powershell
python benchmarks/live_hf_agent_benchmark.py
```

Зафиксированный результат после добавления шума из тех же датасетов:

| Подход | Pass rate | Fact recall | Compression ratio |
| --- | ---: | ---: | ---: |
| **Agent profile** | **36.00%** | **77.46%** | **9.81%** |
| Только centrality | 0.00% | 19.89% | 9.72% |

Главный измеримый результат — защищённые операционные признаки заметно повышают
полноту фактов по сравнению с одной графовой центральностью при сопоставимом
уровне сжатия. При этом pass rate 36% показывает, что некоторые tool/API traces
ещё не укладываются одновременно в бюджет и требования к полноте.

## Тесты

```powershell
pytest -q
```

Тесты проверяют:

- сохранение ограничений, ошибок, решений и пользовательских предпочтений;
- query-aware отбор;
- структурированное состояние агента;
- suppression почти одинаковых предложений;
- качество на curated agent scenarios относительно centrality baseline.

Текущий набор: 8 тестов.

## Структура проекта

```text
summarizer/
├── agent_summarizer/
│   ├── core.py          # scoring, PageRank и выбор предложений
│   ├── features.py      # токены, labels и структурные признаки
│   ├── splitting.py     # разбиение контекста
│   ├── state.py         # извлечение agent state
│   ├── models.py        # конфигурация и dataclasses результата
│   └── evaluation.py    # benchmark dataclasses и метрики
├── benchmarks/
│   ├── real_agent_scenarios.py
│   └── live_hf_agent_benchmark.py
├── tests/
├── utils.py             # совместимость со старым API
├── pyproject.toml
└── LICENSE
```

## Где это можно применять

- сжатие истории перед следующим вызовом LLM-агента;
- долговременная память coding agents;
- сохранение состояния после tool calls;
- подготовка handoff между агентами;
- уменьшение стоимости и latency длинных запросов;
- построение extractive checkpoint перед обрезанием старой истории.

## Ограничения

- Лексические правила зависят от языка и формата входного текста.
- Extractive summary не объединяет несколько разрозненных фактов в новый вывод.
- Большое число важных ограничений может превысить заданный бюджет предложений.
- Live benchmark невелик и не заменяет оценку на реальных production traces.
- Fact recall основан на заранее заданных подстроках и не измеряет смысловую
  эквивалентность во всех формулировках.
- Передача внешнего embedder улучшает семантическую похожесть, но добавляет
  собственные требования к памяти, скорости и воспроизводимости.

Проект завершён как компактная библиотека версии `0.1.0`, но его метрики следует
интерпретировать как результат конкретного benchmark-профиля, а не как
универсальную гарантию сохранения любого важного факта.

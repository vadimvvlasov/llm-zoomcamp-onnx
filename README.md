# LLM Zoomcamp ONNX

RAG-система (Retrieval-Augmented Generation) для работы с FAQ курса LLM Zoomcamp. Использует локальные ONNX-модели для эмбеддингов и поддерживает несколько бэкендов для поиска и LLM.

## Особенности

- **Локальные эмбеддинги** — ONNX-модели без GPU (all-MiniLM-L6-v2, bge-base-en-v1.5)
- **Множественные поисковые бэкенды** — Minsearch (in-memory), SQLite (FTS5), Elasticsearch
- **Несколько LLM-провайдеров** — OpenAI, Ollama, OpenRouter
- **Модульная архитектура** — Protocol-интерфейсы для легкой замены компонентов

## Установка

### Требования

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) (рекомендуется) или pip

### Через uv

```bash
# Клонировать репозиторий
git clone <repo-url>
cd llm-zoomcamp-onnx

# Установить зависимости
uv sync

# Активировать виртуальное окружение
source .venv/bin/activate
```

### Через pip

```bash
pip install -e .
```

## Настройка

Создайте файл `.env` в корне проекта:

```env
# LLM API ключи (выберите нужный)
OPENAI_API_KEY=sk-...
OPENROUTER_API_KEY=sk-or-...

# Опционально: кастомный URL для FAQ данных
FAQ_DATA_URL=https://datatalks.club/faq/json/courses.json
```

## Использование

### Загрузка моделей эмбеддингов

```bash
# Скачать модель по умолчанию (Xenova/all-MiniLM-L6-v2)
python -m src.embed.download

# Скачать другую модель
python -m src.embed.download Xenova/bge-base-en-v1.5

# Указать директорию назначения
python -m src.embed.download sentence-transformers/all-MiniLM-L6-v2 --dest /tmp/models
```

### Использование эмбеддера

```python
from src.embed.embedder import Embedder

# Инициализация (по умолчанию all-MiniLM-L6-v2)
embedder = Embedder()

# Эмбеддинг одного текста
vector = embedder.encode("How to deploy a model?")
print(vector.shape)  # (384,)

# Эмбеддинг батча текстов
vectors = embedder.encode_batch(["Question 1", "Question 2"])
print(vectors.shape)  # (2, 384)
```

### RAG-пайплайн

```python
from src.ingest import FaqHttpLoader, MinsearchIndex
from src.llm import OpenAIClient
from src.rag import RAGBase

# 1. Загрузить FAQ данные
loader = FaqHttpLoader()
docs = loader.load()

# 2. Создать поисковый индекс
index = MinsearchIndex(docs)

# 3. Инициализировать LLM клиент
llm = OpenAIClient()

# 4. Создать RAG пайплайн
rag = RAGBase(
    index=index,
    llm_client=llm,
    llm_model="gpt-4o-mini",
    num_results=5,
)

# 5. Задать вопрос
answer = rag.rag("How do I deploy my model to production?")
print(answer)
```

### Поисковые бэкенды

#### MinsearchIndex (in-memory, TF-IDF)

```python
from src.ingest import MinsearchIndex

index = MinsearchIndex(docs)
results = index.search(
    query="docker deployment",
    num_results=5,
    boost_dict={"question": 3, "answer": 1, "section": 0.5},
    filter_dict={"course": "llm-zoomcamp"},
)
```

#### SqliteIndex (персистентный, FTS5)

```python
from src.ingest import SqliteIndex

# Создать или загрузить индекс
index = SqliteIndex(docs, db_path="faq.db")

# Добавить новые документы
index.add_docs(new_docs)
```

#### ElasticsearchIndex

```python
from src.ingest import ElasticsearchIndex

index = ElasticsearchIndex(
    host="http://localhost:9200",
    index_name="faq"
)
index.index_docs(docs)
```

### LLM-клиенты

#### OpenAI

```python
from src.llm import OpenAIClient

llm = OpenAIClient()
response = llm.complete(
    prompt="Your question",
    instructions="Be helpful and concise",
    model="gpt-4o-mini"
)
```

#### Ollama (локальный)

```python
from src.llm import OllamaClient

llm = OllamaClient(base_url="http://localhost:11434")
response = llm.complete(
    prompt="Your question",
    instructions="Be helpful",
    model="llama3"
)
```

#### OpenRouter

```python
from src.llm import OpenRouterClient

llm = OpenRouterClient()
response = llm.complete(
    prompt="Your question",
    instructions="Be helpful",
    model="openai/gpt-4o-mini"
)
```

## Архитектура

```
src/
├── interfaces.py      # Protocol-контракты (SearchIndex, DataLoader, LLMClient)
├── ingest.py          # Загрузка данных и поисковые индексы
├── llm.py             # LLM-клиенты (OpenAI, Ollama, OpenRouter)
├── rag.py             # Основной RAG-пайплайн
└── embed/
    ├── embedder.py    # ONNX-эмбеддер
    └── download.py    # Загрузка моделей с HuggingFace

models/
└── Xenova/
    ├── all-MiniLM-L6-v2/    # Модель по умолчанию
    └── bge-base-en-v1.5/    # Альтернативная модель
```

### RAG-пайплайн

```
search → build_context → build_prompt → ask
```

1. **search** — поиск релевантных документов по запросу
2. **build_context** — форматирование результатов в контекст
3. **build_prompt** — создание промпта с вопросом и контекстом
4. **ask** — отправка в LLM и получение ответа

## Расширение

### Добавление нового поискового бэкенда

Реализуйте протокол `SearchIndex`:

```python
from src.interfaces import SearchIndex

class MySearchIndex(SearchIndex):
    def search(self, query: str, num_results: int, 
               boost_dict: dict, filter_dict: dict) -> list[dict]:
        # Ваша реализация
        ...
```

### Добавление нового LLM-клиента

Реализуйте протокол `LLMClient`:

```python
from src.interfaces import LLMClient

class MyLLMClient(LLMClient):
    def complete(self, prompt: str, instructions: str, model: str) -> str:
        # Ваша реализация
        ...
```

## Доступные модели эмбеддингов

| Модель | Размерность | Репозиторий |
|--------|-------------|-------------|
| all-MiniLM-L6-v2 | 384 | Xenova/all-MiniLM-L6-v2 |
| bge-base-en-v1.5 | 768 | Xenova/bge-base-en-v1.5 |

## Разработка

### Установка dev-зависимостей

```bash
uv sync --group dev
```

### Запуск Jupyter

```bash
jupyter lab notebooks/
```

## Лицензия

MIT

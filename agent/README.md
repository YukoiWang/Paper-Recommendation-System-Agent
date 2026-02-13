# ML Paper Recommender

Retrieval + QA over ArXiv-style papers: multi-path recall (vector, rule, ItemCF), optional LanceDB backend, and a chat agent backed by an OpenAI-compatible LLM (e.g. DeepSeek).

## Setup

```bash
pip install -r requirements.txt
```

For CPU-only (smaller install):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## Quick run (no API)

```bash
python run_demo.py
```

Uses the built-in corpus, TF-IDF index, and prints a short recommendation list.

## Chat (needs API key)

```bash
export DEEPSEEK_API_KEY=sk-...
python chat_cli.py --api-key $DEEPSEEK_API_KEY
```

With Lance data (local dir, newest-first):

```bash
python chat_cli.py --api-key $DEEPSEEK_API_KEY --papers lance --max-papers 5000 --lance-path ./data/lance_dataset --vector-store lancedb
```

## Data

- **Built-in**: small corpus under `data_loader.py` (no download).
- **Lance**: e.g. [davanstrien/arxiv-cs-papers-lance](https://huggingface.co/davanstrien/arxiv-cs-papers-lance). Download to a local path and pass `--lance-path`. Use `--no-prefer-recent` to load in dataset order instead of by date.

## Project layout

- `models.py` – Paper, UserProfile, RecommendationResult
- `embedder.py` – TF-IDF or SentenceTransformer
- `vector_store.py` – NumpyVectorStore, LanceDBVectorStore
- `cold_start.py` – user vector and trending papers
- `recall_strategies.py` – vector / rule / ItemCF recall and merge
- `agent.py` – RetrievalAgent (index + retrieve_for_user)
- `qa_agent.py` – PaperQAAgent (intent, retrieval, LLM)
- `data_loader.py` – load_builtin_corpus, load_from_lance_hf

## License

MIT

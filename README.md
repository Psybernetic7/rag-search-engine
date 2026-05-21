# RAG Search Engine

A movie search engine built to explore information retrieval techniques — from classic keyword search through hybrid ranking, LLM re-ranking, retrieval-augmented generation, and precision/recall evaluation.

## Features

- **Keyword Search** — BM25 with Porter stemming and stopword removal
- **Semantic Search** — Sentence-transformer embeddings with cosine similarity; supports full-document and chunked matching
- **Hybrid Search** — Two fusion strategies:
  - **Weighted combination** — tune BM25 vs. semantic balance with an `--alpha` parameter
  - **Reciprocal Rank Fusion (RRF)** — rank-based fusion, more robust than score normalization
- **Query Enhancement** — spell correction, query rewriting, and query expansion via Gemini LLM
- **LLM Re-ranking** — three methods applied on top of RRF results:
  - `individual` — one LLM prompt per document, scores 0–10
  - `batch` — single prompt ranks all documents, returns a JSON-ordered list of IDs
  - `cross_encoder` — local `ms-marco-TinyBERT-L2-v2` cross-encoder scores all pairs in one batch
- **Search Evaluation** — per-query precision@k, recall@k, and F1 score against a golden dataset
- **Retrieval-Augmented Generation (RAG)** — four LLM generation modes built on RRF search:
  - `rag` — direct answer from retrieved context
  - `summarize` — multi-document synthesis
  - `citations` — answer with inline `[1]`, `[2]` citations
  - `question` — casual conversational answer

## Project Structure

```
rag-search-engine/
├── cli/
│   ├── lib/
│   │   ├── hybrid_search.py            # Weighted and RRF hybrid search
│   │   ├── semantic_search.py          # Embedding-based search and chunking
│   │   ├── keyword_search.py           # BM25 search
│   │   └── search_utils.py             # Shared utilities
│   ├── hybrid_search_cli.py            # Hybrid search + query enhancement + re-ranking
│   ├── evaluation_cli.py               # Precision@k, recall@k, F1 evaluation
│   ├── augmented_generation_cli.py     # RAG, summarize, citations, question
│   ├── semantic_search_cli.py          # Semantic search and embedding commands
│   └── keyword_search_cli.py          # BM25 keyword search commands
├── data/
│   ├── movies.json                     # Movie dataset
│   └── golden_dataset.json            # Labelled queries for evaluation
├── cache/                              # Auto-generated embeddings and indices
└── README.md
```

## Installation

### Prerequisites

- Python 3.8+
- [uv](https://docs.astral.sh/uv/) package manager
- A Gemini API key (required for query enhancement, LLM re-ranking, and RAG commands)

### Setup

1. Clone the repository
   ```bash
   git clone https://github.com/Psybernetic7/rag-search-engine.git
   cd rag-search-engine
   ```

2. Install dependencies
   ```bash
   uv sync
   ```

3. Create a `.env` file with your Gemini API key
   ```
   GEMINI_API_KEY=your_key_here
   ```

4. Generate the embeddings cache (required before first search)
   ```bash
   uv run cli/semantic_search_cli.py embed_chunks
   ```

## Usage

All commands use `uv run` from the project root.

---

### Hybrid Search

#### Reciprocal Rank Fusion (RRF)

```bash
uv run cli/hybrid_search_cli.py rrf-search "your query" --limit 5
```

Options:

| Flag | Description | Default |
|------|-------------|---------|
| `-k` | RRF k parameter (lower = top results weighted more) | `60` |
| `--limit` | Number of results | `5` |
| `--enhance` | Query enhancement: `spell`, `rewrite`, `expand` | off |
| `--rerank-method` | Re-ranking: `individual`, `batch`, `cross_encoder` | off |
| `--evaluate` | LLM relevance evaluation (0–3 per result) | off |
| `--debug` | Log each pipeline stage | off |

Examples:

```bash
# Basic search
uv run cli/hybrid_search_cli.py rrf-search "family movie about bears" --limit 5

# Spell-correct typos before searching
uv run cli/hybrid_search_cli.py rrf-search "famly movee abut bears" --enhance spell

# Rewrite the query for better recall
uv run cli/hybrid_search_cli.py rrf-search "that bear movie with leo" --enhance rewrite

# Cross-encoder re-ranking (local, no API calls)
uv run cli/hybrid_search_cli.py rrf-search "bear attack survival" --limit 5 --rerank-method cross_encoder

# LLM batch re-ranking (one API call)
uv run cli/hybrid_search_cli.py rrf-search "bear attack survival" --limit 5 --rerank-method batch

# Full pipeline with debug logging
uv run cli/hybrid_search_cli.py rrf-search "bear attack survival" --limit 5 --rerank-method cross_encoder --debug

# Evaluate result quality with an LLM after searching
uv run cli/hybrid_search_cli.py rrf-search "family bear movie" --limit 5 --evaluate
```

#### Weighted Hybrid Search

```bash
uv run cli/hybrid_search_cli.py weighted-search "your query" --alpha 0.5 --limit 5
```

`--alpha` controls the BM25/semantic balance:
- `0.0` = pure semantic, `1.0` = pure BM25, `0.5` = balanced

#### Score Normalization

```bash
uv run cli/hybrid_search_cli.py normalize 0.5 2.3 1.2 0.1
```

---

### Evaluation

Run precision@k, recall@k, and F1 against the golden dataset for every test query:

```bash
uv run cli/evaluation_cli.py --limit 5
```

Output format:

```
k=5

- Query: dangerous bear wilderness survival
  - Precision@5: 1.0000
  - Recall@5: 0.8571
  - F1 Score: 0.9231
  - Retrieved: The Edge, Man in the Wilderness, Claws, Into the Grizzly Maze, Alaska
  - Relevant: Unnatural, Alaska, The Edge, Into the Grizzly Maze, Claws, Man in the Wilderness, The Revenant
```

**Metrics explained:**
- **Precision@k** — fraction of retrieved results that are relevant: `relevant_retrieved / k`
- **Recall@k** — fraction of all relevant documents retrieved: `relevant_retrieved / total_relevant`
- **F1** — harmonic mean of precision and recall: `2 * P * R / (P + R)`

---

### Retrieval-Augmented Generation

All RAG commands perform an RRF search and feed the results to a Gemini LLM.

#### Direct RAG Answer

```bash
uv run cli/augmented_generation_cli.py rag "what movies feature dinosaurs coming back to life"
```

#### Multi-document Summarization

```bash
uv run cli/augmented_generation_cli.py summarize "bear attack survival films" --limit 5
```

#### Citation-aware Answer

```bash
uv run cli/augmented_generation_cli.py citations "what are the best animated bear movies" --limit 5
```

Sources are numbered `[1]`, `[2]` etc. in the response.

#### Conversational Question Answering

```bash
uv run cli/augmented_generation_cli.py question "is there a good comedy with a talking bear"
```

---

### Semantic Search

```bash
# Search by semantic similarity
uv run cli/semantic_search_cli.py search "action movie with police" --limit 5

# Chunked semantic search (sentence-level matching)
uv run cli/semantic_search_cli.py search_chunked "action movie with police" --limit 5

# Regenerate embeddings cache
uv run cli/semantic_search_cli.py embed_chunks
```

### Keyword Search

```bash
uv run cli/keyword_search_cli.py search "your query" --limit 5
```

---

## How It Works

### Search Pipeline

1. **BM25** ranks all documents by keyword relevance (Porter stemming, stopword removal)
2. **Semantic search** ranks all documents by cosine similarity to the query embedding (`all-MiniLM-L6-v2`, 384 dimensions)
3. **RRF fusion** merges both ranked lists: each document's score is `1/(k + rank_bm25) + 1/(k + rank_semantic)`
4. **Re-ranking** (optional) re-scores the top 5× candidates using a cross-encoder or LLM, then truncates to `--limit`

### Re-ranking Methods

| Method | How it works | Speed | Requires API |
|--------|--------------|-------|--------------|
| `individual` | One LLM call per document, scores 0–10 | Slow | Yes |
| `batch` | Single LLM call, returns ranked JSON ID list | Fast | Yes |
| `cross_encoder` | Local `ms-marco-TinyBERT-L2-v2`, all pairs in one batch | Fast | No |

### RAG Pipeline

1. RRF search retrieves the top-k most relevant movie documents
2. Titles and descriptions are formatted as context
3. A Gemini LLM generates a response grounded in the retrieved documents

---

## Technical Details

| Component | Detail |
|-----------|--------|
| Semantic model | `sentence-transformers/all-MiniLM-L6-v2` (384-dim embeddings) |
| Cross-encoder | `cross-encoder/ms-marco-TinyBERT-L2-v2` |
| LLM | Gemini (`gemma-4-31b-it` via Google GenAI SDK) |
| BM25 params | k1=1.5, b=0.75 |
| Similarity metric | Cosine similarity |

### Data Format

```json
{
  "movies": [
    { "id": 1, "title": "Movie Title", "description": "Full description..." }
  ]
}
```

---

## Troubleshooting

**`Module not found`** — run `uv sync` and use `uv run` to execute scripts.

**Embeddings missing** — run `uv run cli/semantic_search_cli.py embed_chunks` to generate the cache.

**BM25 index missing** — the index builds automatically on first use; to force a rebuild: `uv run cli/keyword_search_cli.py build`.

**Gemini 500 errors** — the API occasionally returns transient 500s; the CLI handles them gracefully (scores default to 0). Retry if it happens consistently.

**Cross-encoder GPU errors** — the cross-encoder is loaded with `device="cpu"` by default; no GPU is required.

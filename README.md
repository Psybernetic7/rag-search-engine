# RAG Search Engine

A comprehensive search engine implementation featuring semantic search, keyword-based search, hybrid search, and advanced text chunking capabilities for movie datasets.

## Features

- **Semantic Search**: Find movies based on semantic similarity using transformer-based embeddings
- **Chunked Semantic Search**: Enhanced search using sentence-level chunking for better relevance matching
- **Keyword Search**: Traditional BM25 and TF-IDF based search
- **Hybrid Search**: Combine semantic and keyword search using two strategies:
  - **Weighted Combination**: Tune with alpha parameter to balance keyword vs. semantic relevance
  - **Reciprocal Rank Fusion (RRF)**: Robust ranking-based fusion without score normalization
- **Text Embedding**: Generate embeddings for custom text using pre-trained models
- **Flexible Chunking**: Split text into semantic chunks or word-based chunks with configurable overlap
- **CLI Interface**: Easy-to-use command-line interface for all search operations

## Project Structure

```
rag-search-engine/
├── cli/
│   ├── lib/
│   │   ├── semantic_search.py      # Semantic search and chunking implementation
│   │   ├── keyword_search.py       # BM25 and TF-IDF search
│   │   ├── hybrid_search.py        # Hybrid search combining semantic and keyword
│   │   ├── search_utils.py         # Search utility functions and formatting
│   │   └── ...
│   ├── semantic_search_cli.py      # Semantic search CLI
│   ├── keyword_search_cli.py       # Keyword search CLI
│   ├── hybrid_search_cli.py        # Hybrid search CLI
│   └── ...
├── data/
│   └── movies.json                 # Movie dataset
├── cache/                          # Generated embeddings and metadata (auto-generated)
├── .gitignore
└── README.md
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd rag-search-engine
   ```

2. **Create and activate virtual environment**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Generate embeddings cache** (required before first use)
   ```bash
   cd cli
   python3 semantic_search_cli.py embed_chunks
   ```
   
   This command will:
   - Load the pre-trained sentence transformer model
   - Generate embeddings for all movie chunks
   - Cache the embeddings locally for fast retrieval
   - Create `cache/chunk_embeddings.npy` and `cache/chunk_metadata.json`

## Usage

All commands are run from the `cli/` directory.

### Semantic Search Commands

#### Search all movies by semantic similarity
```bash
python3 semantic_search_cli.py search "your query" --limit 5
```

Example:
```bash
python3 semantic_search_cli.py search "action movie with police" --limit 5
```

#### Search using chunked semantic similarity
For more granular and relevant results by matching against sentence-level chunks:
```bash
python3 semantic_search_cli.py search_chunked "your query" --limit 5
```

Example:
```bash
python3 semantic_search_cli.py search_chunked "action movie with police" --limit 3
```

### Embedding Commands

#### Generate embedding for text
```bash
python3 semantic_search_cli.py embed_text "your text here"
```

#### Generate embedding for a search query
```bash
python3 semantic_search_cli.py embedquery "your query here"
```

### Chunking Commands

#### Semantic chunking (sentence-based)
Splits text into chunks based on sentence boundaries:
```bash
python3 semantic_search_cli.py semantic_chunk "your text here" --max-chunk-size 4 --overlap 1
```

Options:
- `--max-chunk-size`: Maximum number of sentences per chunk (default: 4)
- `--overlap`: Number of sentences to overlap between chunks (default: 0)

Example:
```bash
python3 semantic_search_cli.py semantic_chunk "First sentence. Second sentence. Third sentence." --max-chunk-size 2
```

#### Word-based chunking
Splits text into chunks based on word count:
```bash
python3 semantic_search_cli.py chunk "your text here" --chunk-size 200 --overlap 0
```

Options:
- `--chunk-size`: Number of words per chunk (default: 200)
- `--overlap`: Number of words to overlap between chunks (default: 0)

### Hybrid Search Commands

#### Weighted Hybrid Search
Combines semantic and keyword search with a configurable alpha parameter to balance between the two approaches:

```bash
python3 hybrid_search_cli.py weighted-search "your query" --alpha 0.5 --limit 5
```

Options:
- `--alpha`: Weight for BM25 keyword search (0-1, default 0.5)
  - `0.0` = 100% semantic search
  - `0.5` = 50/50 split
  - `1.0` = 100% keyword search
- `--limit`: Number of results to return (default 5)

Examples:
```bash
# Title search (high alpha for keyword focus)
python3 hybrid_search_cli.py weighted-search "The Lion King" --alpha 0.8

# Conceptual search (low alpha for semantic focus)
python3 hybrid_search_cli.py weighted-search "family movies" --alpha 0.2

# Mixed query (balanced alpha)
python3 hybrid_search_cli.py weighted-search "2015 comedies" --alpha 0.5
```

#### Reciprocal Rank Fusion (RRF) Search
Combines search results using ranking-based fusion instead of score normalization, making it more robust to outliers:

```bash
python3 hybrid_search_cli.py rrf-search "your query" -k 60 --limit 5
```

Options:
- `-k`: RRF k parameter (default 60)
  - Lower values (e.g., 20) give more weight to top-ranked results
  - Higher values (e.g., 100) give more gradual weight distribution
- `--limit`: Number of results to return (default 5)

Example:
```bash
# Standard RRF with default k
python3 hybrid_search_cli.py rrf-search "action movies"

# RRF with higher k for more balanced results
python3 hybrid_search_cli.py rrf-search "family movies" -k 100
```

#### Score Normalization
Normalize a list of scores using min-max normalization:

```bash
python3 hybrid_search_cli.py normalize 0.5 2.3 1.2 0.5 0.1
```

Output:
```
* 0.1818
* 1.0000
* 0.5000
* 0.1818
* 0.0000
```

### Verification Commands

#### Verify model loading
```bash
python3 semantic_search_cli.py verify
```

#### Verify embeddings exist
```bash
python3 semantic_search_cli.py verify_embeddings
```

#### Rebuild chunked embeddings
```bash
python3 semantic_search_cli.py embed_chunks
```

## Search Output Format

### Semantic Search Results
```
1. Movie Title (score: 0.7234)
  Description preview (first 100 characters)...

2. Another Movie (score: 0.6891)
  Another description preview...
```

### Chunked Semantic Search Results
```
1. Movie Title (score: 0.7234)
   Description preview (first 100 characters)...

2. Another Movie (score: 0.6891)
   Another description preview...
```

## Cache Management

The cache directory contains pre-computed embeddings and indices for faster searches:

- **chunk_embeddings.npy**: Sentence-level embeddings for chunked semantic search
- **chunk_metadata.json**: Metadata mapping chunks to documents
- **movie_embeddings.npy**: Full-document embeddings for semantic search
- **index.pkl**: BM25 inverted index for keyword search
- **docmap.pkl**: Document mapping for keyword search
- **term_frequencies.pkl**: Term frequency data for BM25
- **doc_lengths.pkl**: Document length data for BM25

### Regenerating Cache

If you modify the semantic chunking parameters or update the movie dataset, regenerate the cache:

```bash
# Remove old cache
rm -rf cache/chunk_embeddings.npy cache/chunk_metadata.json

# Regenerate
python3 semantic_search_cli.py embed_chunks
```

### Cache Size

The cache files are generated artifacts and are not included in version control (see `.gitignore`). When cloning the repository, run `embed_chunks` to generate them locally.

## How It Works

### Semantic Search

1. **Query Processing**: Your search query is converted to an embedding using a pre-trained transformer model
2. **Similarity Calculation**: Cosine similarity is calculated between the query embedding and all document embeddings
3. **Ranking**: Results are ranked by similarity score and returned in descending order

### Chunked Semantic Search

1. **Document Chunking**: Movie descriptions are split into sentence-level chunks
2. **Query Embedding**: Your query is converted to an embedding
3. **Chunk Matching**: Each chunk's similarity to the query is calculated
4. **Aggregation**: For each movie, the highest-scoring chunk is used as the movie's score
5. **Ranking**: Movies are ranked by their best chunk score

### Text Chunking Strategy

The semantic chunking function handles edge cases:
- Strips leading/trailing whitespace from input
- Treats text without punctuation as a single sentence
- Removes empty chunks after processing
- Supports configurable chunk size and overlap

### Hybrid Search

Two complementary approaches for combining keyword and semantic search:

#### Weighted Combination
1. **BM25 Scoring**: Calculate keyword relevance scores
2. **Semantic Scoring**: Calculate semantic similarity scores
3. **Normalization**: Normalize both score sets to [0, 1] range using min-max normalization
4. **Weighting**: Combine normalized scores using `alpha * bm25 + (1-alpha) * semantic`
5. **Ranking**: Return results sorted by hybrid score in descending order

**Use case**: Works well when you want fine-grained control over keyword vs. semantic balance. Choose alpha based on query type:
- Title searches (high alpha ~0.8): "The Revenant"
- Conceptual searches (low alpha ~0.2): "family movies"
- Mixed searches (balanced alpha ~0.5): "2015 comedies"

#### Reciprocal Rank Fusion (RRF)
1. **BM25 Ranking**: Get BM25 results with positions (rank 1, 2, 3, ...)
2. **Semantic Ranking**: Get semantic search results with positions
3. **RRF Scoring**: Calculate `1 / (k + rank)` for each result in each ranking
4. **Score Aggregation**: Sum RRF scores for documents appearing in both rankings
5. **Ranking**: Return results sorted by combined RRF score in descending order

**Use case**: More robust approach that avoids score normalization issues. Handles outliers and different score distributions well. Tune k parameter:
- Lower k (~20): More weight to top results, steeper drop-off
- Default k (~60): Balanced weighting across results
- Higher k (~100): Gradual decline, more weight to lower-ranked results

## Technical Details

### Models and Algorithms

#### Semantic Search
- **Model**: `sentence-transformers` with `all-MiniLM-L6-v2`
- **Embedding Dimensions**: 384
- **Similarity Metric**: Cosine similarity

#### Keyword Search
- **Algorithm**: BM25 (Best Matching 25)
- **BM25 Parameters**:
  - `k1 = 1.5`: Controls term frequency saturation
  - `b = 0.75`: Controls impact of document length normalization
- **Text Processing**: Porter stemming, stopword removal, punctuation removal

#### Hybrid Search
- **Weighted Combination**: Configurable alpha parameter (0-1)
- **RRF**: Reciprocal Rank Fusion with configurable k parameter

### Data Format

**movies.json** contains an array of movie objects:
```json
{
  "movies": [
    {
      "id": 1,
      "title": "Movie Title",
      "description": "Full movie description..."
    },
    ...
  ]
}
```

## Requirements

See `requirements.txt` for all dependencies. Key packages:
- `sentence-transformers`: For semantic embeddings
- `numpy`: For numerical operations
- `scikit-learn`: For TF-IDF and cosine similarity

## Performance Tips

1. **First Run**: The first `embed_chunks` command will take several minutes as it generates embeddings. Subsequent searches will be fast.
2. **Query Optimization**: Short, specific queries tend to yield better results than long, generic ones
3. **Chunk Size**: Experiment with `--max-chunk-size` when using `semantic_chunk` for different levels of granularity

## Choosing a Search Method

| Query Type | Recommended Method | Reason |
|------------|-------------------|--------|
| **Exact titles** | BM25 (keyword) | Precise matching on movie titles |
| **Conceptual** | Semantic search | "Feel-good movies", "intense dramas" |
| **Mixed/Complex** | Hybrid (weighted or RRF) | Need both keyword and semantic understanding |
| **Unknown intent** | RRF hybrid | Most robust, avoids score normalization issues |
| **Title + concept** | Weighted (alpha 0.5-0.7) | Balance keyword matching with semantic meaning |

### When to Tune Hybrid Parameters

**Weighted Search Alpha**:
- Set to 0.8+ for title-focused searches
- Set to 0.2-0.4 for meaning-focused searches
- Use 0.5 for balanced queries by default

**RRF k Parameter**:
- Use 20-40 if you want top results to dominate
- Use 60 (default) for balanced influence across ranks
- Use 100+ if you want broader coverage of results

## Troubleshooting

### Module not found errors
Ensure you're running commands from the `cli/` directory and the virtual environment is activated.

### Out of memory errors during embedding generation
The embedding process can be memory-intensive. If you encounter errors:
1. Close other applications
2. The process should complete despite memory warnings

### Embeddings not loading
If embeddings seem outdated, regenerate them:
```bash
python3 semantic_search_cli.py embed_chunks
```

### BM25 index not found
If you see "Index files not found" when using hybrid search:
```bash
# The index will auto-build on first use, but you can force rebuild:
python3 keyword_search_cli.py build
```

### Hybrid search returning unexpected results
- For **weighted search**: Try adjusting the `--alpha` parameter to better match your query type
- For **RRF**: Adjust the `-k` parameter - lower k emphasizes top results, higher k broadens influence
- Check that both embeddings and BM25 index are built:
  ```bash
  python3 semantic_search_cli.py embed_chunks
  python3 keyword_search_cli.py build
  ```



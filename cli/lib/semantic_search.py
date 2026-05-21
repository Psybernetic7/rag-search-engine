import json
import re
from pathlib import Path
from sentence_transformers import SentenceTransformer
import numpy as np
from .search_utils import format_search_result

EMBEDDINGS_PATH = Path("cache/movie_embeddings.npy")
DATA_PATH_MOVIES = Path(__file__).parent.parent.parent / "data" / "movies.json"


class SemanticSearch:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.documents = None
        self.documents_map = {}

    def generate_embedding(self, text):
        if text == "" or text.isspace():
            raise ValueError("Text is empty")
        embedding = self.model.encode([text])
        return embedding[0]
    
    def build_embeddings(self, documents):
        self.documents = documents
        for doc in self.documents:
            self.documents_map[doc["id"]] = doc
        texts = [f"{doc['title']}: {doc['description']}" for doc in self.documents]
        self.embeddings = self.model.encode(texts, show_progress_bar=True)
        np.save(EMBEDDINGS_PATH, self.embeddings)
        return self.embeddings

    def search(self, query, limit):
        if self.embeddings is None:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
        query_embedding = self.generate_embedding(query)
        scored = [
            (cosine_similarity(query_embedding, self.embeddings[i]), self.documents[i])
            for i in range(len(self.documents))
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {"score": score, "title": doc["title"], "description": doc["description"]}
            for score, doc in scored[:limit]
        ]

    def load_or_create_embeddings(self, documents):
        self.documents = documents
        for doc in self.documents:
            self.documents_map[doc["id"]] = doc
        if EMBEDDINGS_PATH.exists():
            self.embeddings = np.load(EMBEDDINGS_PATH)
            if len(self.embeddings) == len(documents):
                return self.embeddings
        return self.build_embeddings(documents)


CHUNK_EMBEDDINGS_PATH = Path("cache/chunk_embeddings.npy")
CHUNK_METADATA_PATH = Path("cache/chunk_metadata.json")


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None

    def build_chunk_embeddings(self, documents):
        self.documents = documents
        for doc in self.documents:
            self.documents_map[doc["id"]] = doc

        all_chunks = []
        chunk_metadata = []

        for movie_idx, doc in enumerate(self.documents):
            description = doc.get("description", "")
            if not description or description.isspace():
                continue

            doc_chunks = semantic_chunk(description, max_chunk_size=4, overlap=1)

            total_chunks = len(doc_chunks)
            for chunk_idx, chunk in enumerate(doc_chunks):
                all_chunks.append(chunk)
                chunk_metadata.append({
                    "movie_idx": movie_idx,
                    "chunk_idx": chunk_idx,
                    "total_chunks": total_chunks,
                })

        self.chunk_embeddings = self.model.encode(all_chunks, show_progress_bar=True)
        self.chunk_metadata = chunk_metadata

        cache_dir = Path("cache")
        cache_dir.mkdir(exist_ok=True)

        np.save(CHUNK_EMBEDDINGS_PATH, self.chunk_embeddings)

        with open(CHUNK_METADATA_PATH, "w", encoding="utf-8") as f:
            json.dump({"chunks": chunk_metadata, "total_chunks": len(all_chunks)}, f, indent=2)

        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        for doc in self.documents:
            self.documents_map[doc["id"]] = doc

        if CHUNK_EMBEDDINGS_PATH.exists() and CHUNK_METADATA_PATH.exists():
            self.chunk_embeddings = np.load(CHUNK_EMBEDDINGS_PATH)
            with open(CHUNK_METADATA_PATH, "r", encoding="utf-8") as f:
                self.chunk_metadata = json.load(f)["chunks"]
            return self.chunk_embeddings

        return self.build_chunk_embeddings(documents)

    def search_chunks(self, query: str, limit: int = 10) -> list[dict]:
        if self.chunk_embeddings is None or self.chunk_metadata is None:
            raise ValueError("No chunk embeddings loaded. Call `load_or_create_chunk_embeddings` first.")

        query_embedding = self.generate_embedding(query)

        chunk_scores = []
        for i, chunk_embedding in enumerate(self.chunk_embeddings):
            similarity = cosine_similarity(query_embedding, chunk_embedding)
            metadata = self.chunk_metadata[i]
            chunk_scores.append({
                "chunk_idx": metadata["chunk_idx"],
                "movie_idx": metadata["movie_idx"],
                "score": similarity
            })

        movie_scores = {}
        for chunk_score in chunk_scores:
            movie_idx = chunk_score["movie_idx"]
            if movie_idx not in movie_scores or chunk_score["score"] > movie_scores[movie_idx]:
                movie_scores[movie_idx] = chunk_score["score"]

        sorted_movies = sorted(movie_scores.items(), key=lambda x: x[1], reverse=True)
        top_movies = sorted_movies[:limit]

        results = []
        for movie_idx, score in top_movies:
            doc = self.documents[movie_idx]
            result = format_search_result(
                doc_id=doc["id"],
                title=doc["title"],
                document=doc["description"],
                score=score,
                metadata=doc.get("metadata", {})
            )
            results.append(result)

        return results


def embed_text(text):
    ss = SemanticSearch()
    embedding = ss.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def embed_query_text(query):
    ss = SemanticSearch()
    embedding = ss.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Shape: {embedding.shape}")


def verify_embeddings():
    with open(DATA_PATH_MOVIES, "r", encoding="utf-8") as f:
        documents = json.load(f)["movies"]
    ss = SemanticSearch()
    embeddings = ss.load_or_create_embeddings(documents)
    print(f"Number of docs:   {len(documents)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")


def verify_model():
    ss = SemanticSearch()
    print(f"Model loaded: {ss.model}")
    print(f"Max sequence length: {ss.model.max_seq_length}")

def semantic_chunk(text: str, max_chunk_size: int = 4, overlap: int = 0) -> list[str]:
    text = text.strip()
    if not text:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", text)

    if len(sentences) == 1 and not re.search(r"[.!?]$", text):
        return [text]

    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return []

    step = max(1, max_chunk_size - overlap)
    chunks = []
    i = 0
    while i < len(sentences):
        chunk_sentences = sentences[i:i + max_chunk_size]
        if chunks and len(chunk_sentences) <= overlap:
            break
        chunk = " ".join(chunk_sentences)
        if chunk.strip():
            chunks.append(chunk)
        i += step
    return chunks


def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)
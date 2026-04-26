import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
import numpy as np

EMBEDDINGS_PATH = Path("cache/movie_embeddings.npy")
DATA_PATH_MOVIES = Path(__file__).parent.parent.parent / "data" / "movies.json"


class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
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

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)
import json
import string
from pathlib import Path
from nltk.stem import PorterStemmer
from typing import Dict, Set
import pickle
import sys
from collections import Counter
import math

stemmer = PorterStemmer()

DATA_PATH_MOVIES = Path(__file__).parent.parent.parent / "data" / "movies.json"
DATA_PATH_STOPWORDS = Path(__file__).parent.parent.parent / "data" / "stopwords.txt"

TRANSLATION_TABLE = str.maketrans("", "", string.punctuation)

BM25_K1 = 1.5
BM25_B = 0.75

def load_movies():
    with open(DATA_PATH_MOVIES, "r", encoding="utf-8") as f:
        return json.load(f)

def load_stopwords():
    with open(DATA_PATH_STOPWORDS, "r", encoding="utf-8") as f:
        return f.read().splitlines()

def normalize_filter_stem(text: str, stopwords):
    tokens = text.lower().translate(TRANSLATION_TABLE).split()
    filtered = [t for t in tokens if t not in stopwords]
    return [stemmer.stem(t) for t in filtered]

def search_command(query: str):
    invertedindex = InvertedIndex()

    try:
        invertedindex.load()
    except Exception:
        print( "Error: Index files not found. Please build the index first.")
        sys.exit(1)

    tokens = normalize_filter_stem(text=query, stopwords=invertedindex.stopwords)

    my_set = set()

    for t in tokens:
        doc_ids = invertedindex.get_documents(t)
        my_set.update(doc_ids)
        if len(my_set) == 5:
            break

    for doc_id in my_set:
        par = invertedindex.docmap[doc_id]
        mov_title = par["title"]
        mov_id = par["id"]
        print(f"Title: {mov_title}, ID: {mov_id}")

class InvertedIndex:

    def __init__(self):
        self.index: Dict[str, Set[int]] = {}
        self.docmap: Dict[int, dict] = {}
        self.stopwords = load_stopwords()
        self.term_frequencies: Dict[int, Counter] = {}
        self.doc_lengths: Dict[int, int] = {}
        

    def __add_document(self, doc_id, text):
        tokens = normalize_filter_stem(text, self.stopwords)

        if doc_id not in self.term_frequencies:
                self.term_frequencies[doc_id] = Counter()

        self.term_frequencies[doc_id].update(tokens)
        self.doc_lengths[doc_id] = len(tokens)

        for token in tokens:
            if token not in self.index:
                self.index[token] = set()
            self.index[token].add(doc_id)

            

    def get_documents(self, term):
        tokens = normalize_filter_stem(term, self.stopwords)
        if not tokens:
            return []
        token = tokens[0]
        doc_ids = self.index.get(token, set())
        return sorted(doc_ids)

    def build(self):
        data = load_movies()
        movies_list = data["movies"]

        for m in movies_list:
            doc_id = m["id"]
            text = f"{m['title']} {m['description']}"
            self.__add_document(doc_id, text)
            self.docmap[doc_id] = m

    def save(self):
        cache_dir = Path("cache")
        cache_dir.mkdir(exist_ok=True)

        with open(cache_dir / "index.pkl", "wb") as f:
            pickle.dump(self.index, f)

        with open(cache_dir / "docmap.pkl", "wb") as f:
            pickle.dump(self.docmap, f)

        with open(cache_dir / "term_frequencies.pkl", "wb") as f:
            pickle.dump(self.term_frequencies, f)

        with open(cache_dir / "doc_lengths.pkl", "wb") as f:
            pickle.dump(self.doc_lengths, f)

    def load(self):
        cache_dir = Path("cache")

        index_path = cache_dir / "index.pkl"
        if not index_path.exists():
            raise Exception("File is missing")
        with open(index_path, "rb") as f:
            self.index = pickle.load(f)

        docmap_path = cache_dir / "docmap.pkl"
        if not docmap_path.exists():
            raise Exception("File is missing")
        with open(docmap_path, "rb") as f:
            self.docmap = pickle.load(f)

        frequencies_path = cache_dir / "term_frequencies.pkl"
        if not frequencies_path.exists():
            raise Exception("File is missing")
        with open(frequencies_path, "rb") as f:
            self.term_frequencies = pickle.load(f)

        doc_lengths_path = cache_dir / "doc_lengths.pkl"
        if not doc_lengths_path.exists():
            raise Exception("File is missing")
        with open(doc_lengths_path, "rb") as f:
            self.doc_lengths = pickle.load(f)

    def get_tf(self, doc_id, term):
        tokens = normalize_filter_stem(term, self.stopwords)
        if not tokens:
            return 0
        if len(tokens) > 1:
            raise Exception("term must be single token")
        token = tokens[0]
        counter = self.term_frequencies.get(doc_id, Counter())
        return counter.get(token, 0)
    
    def __get_avg_doc_length(self) -> float:
        if not self.doc_lengths:
            return 0.0
        return sum(self.doc_lengths.values()) / len(self.doc_lengths)

    def get_bm25_tf(self, doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B) -> float:
        tf = self.get_tf(doc_id, term)
        dl = self.doc_lengths.get(doc_id, 0)
        avgdl = self.__get_avg_doc_length()
        length_norm = 1 - b + b * (dl / avgdl) if avgdl > 0 else 1.0
        return (tf * (k1 + 1)) / (tf + k1 * length_norm)

    def bm25(self, doc_id: int, term: str) -> float:
        return self.get_bm25_tf(doc_id, term) * self.get_bm25_idf(term)

    def bm25_search(self, query: str, limit: int = 5):
        tokens = normalize_filter_stem(query, self.stopwords)
        scores: Dict[int, float] = {}
        for doc_id in self.docmap:
            total = sum(self.bm25(doc_id, token) for token in tokens)
            scores[doc_id] = total
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:limit]

    def get_bm25_idf(self, term: str) -> float:
        tokens = normalize_filter_stem(term, self.stopwords)
        if not tokens:
            raise Exception("term must be single token")
        if len(tokens) > 1:
            raise Exception("term must be single token")
        token = tokens[0]
        N = len(self.docmap)
        df = len(self.index.get(token, set()))
        return math.log((N - df + 0.5) / (df + 0.5) + 1)

        

def build_command():
    index = InvertedIndex()
    index.build()
    index.save()

def tf_command(doc_id: int, term: str) -> int:
    index = InvertedIndex()
    index.load()
    return index.get_tf(doc_id, term)
    
def compute_idf(index: InvertedIndex, term: str) -> float:
    tokens = normalize_filter_stem(term, index.stopwords)
    if not tokens:
        print("Error: Empty token set")
        sys.exit(1)
    stemmed_term = tokens[0]
    document_ids_for_term = index.index.get(stemmed_term, set())
    return math.log((len(index.docmap) + 1) / (len(document_ids_for_term) + 1))

def idf_command(term: str):
    index = InvertedIndex()
    index.load()
    return compute_idf(index, term)

def bm25_tf_command(doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B) -> float:
    index = InvertedIndex()
    index.load()
    return index.get_bm25_tf(doc_id, term, k1, b)

def bm25_idf_command(term: str) -> float:
    index = InvertedIndex()
    index.load()
    return index.get_bm25_idf(term)

def bm25search_command(query: str, limit: int = 5):
    index = InvertedIndex()
    index.load()
    results = index.bm25_search(query, limit)
    return [(doc_id, index.docmap[doc_id]["title"], score) for doc_id, score in results]

def tfidf_command(doc_id: int, term: str) -> float:
    index = InvertedIndex()
    index.load()
    tf = index.get_tf(doc_id, term)
    idf = compute_idf(index, term)
    return tf * idf

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
    except Exception as e:
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
        

    def __add_document(self, doc_id, text):
        tokens = normalize_filter_stem(text, self.stopwords)

        if doc_id not in self.term_frequencies:
                self.term_frequencies[doc_id] = Counter()
        
        self.term_frequencies[doc_id].update(tokens)

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

    def get_tf(self, doc_id, term):
        tokens = normalize_filter_stem(term, self.stopwords)
        if not tokens:
            return 0
        if len(tokens) > 1:
            raise Exception("term must be single token")
        token = tokens[0]
        counter = self.term_frequencies.get(doc_id, Counter())
        return counter.get(token, 0)

        

def build_command():
    index = InvertedIndex()
    index.build()
    index.save()

def tf_command(doc_id: int, term: str) -> int:
    index = InvertedIndex()
    index.load()
    return index.get_tf(doc_id, term)
    
def idf_command(term: str):
    index = InvertedIndex()
    index.load()
    tokens = normalize_filter_stem(term, index.stopwords)
    try:
        stemmed_term = tokens[0]
        document_ids_for_term = index.index.get(stemmed_term, set())
    except Exception as e:
        print( "Error: Empty token set")
        sys.exit(1)
    return math.log((len(index.docmap) + 1) / (len(document_ids_for_term) + 1))

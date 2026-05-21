import os
from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch
from .search_utils import format_search_result


class HybridSearch:
    def __init__(self, documents):
        self.documents = documents
        self.documents_map = {doc["id"]: doc for doc in documents}

        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query, limit):
        self.idx.load()
        results = self.idx.bm25_search(query, limit)
        formatted = []
        for doc_id, score in results:
            doc = self.documents_map[doc_id]
            result = format_search_result(
                doc_id=doc["id"],
                title=doc["title"],
                document=doc["description"],
                score=float(score),
                metadata=doc.get("metadata", {})
            )
            formatted.append(result)
        return formatted

    def _semantic_search(self, query, limit):
        return self.semantic_search.search_chunks(query, limit)

    def _normalize_score_list(self, scores):
        if not scores:
            return []

        min_score = min(scores)
        max_score = max(scores)

        if min_score == max_score:
            return [1.0] * len(scores)

        score_range = max_score - min_score
        return [(score - min_score) / score_range for score in scores]

    def weighted_search(self, query, alpha=0.5, limit=5):
        bm25_results = self._bm25_search(query, 500 * limit)
        semantic_results = self._semantic_search(query, 500 * limit)

        bm25_scores = [r["score"] for r in bm25_results]
        normalized_bm25_scores = self._normalize_score_list(bm25_scores)

        semantic_scores = [r["score"] for r in semantic_results]
        normalized_semantic_scores = self._normalize_score_list(semantic_scores)

        bm25_dict = {}
        for i, result in enumerate(bm25_results):
            bm25_dict[result["id"]] = normalized_bm25_scores[i]

        semantic_dict = {}
        for i, result in enumerate(semantic_results):
            semantic_dict[result["id"]] = normalized_semantic_scores[i]

        all_doc_ids = set(bm25_dict.keys()) | set(semantic_dict.keys())

        hybrid_results = []
        for doc_id in all_doc_ids:
            doc = self.documents_map[doc_id]
            bm25_norm = bm25_dict.get(doc_id, 0.0)
            semantic_norm = semantic_dict.get(doc_id, 0.0)
            hybrid = alpha * bm25_norm + (1 - alpha) * semantic_norm

            hybrid_results.append({
                "id": doc_id,
                "title": doc["title"],
                "description": doc["description"],
                "bm25_score": bm25_norm,
                "semantic_score": semantic_norm,
                "hybrid_score": hybrid
            })

        hybrid_results.sort(key=lambda x: x["hybrid_score"], reverse=True)
        return hybrid_results[:limit]

    def rrf_search(self, query, k=60, limit=10):
        bm25_results = self._bm25_search(query, 500 * limit)
        semantic_results = self._semantic_search(query, 500 * limit)

        bm25_ranks = {result["id"]: i + 1 for i, result in enumerate(bm25_results)}
        semantic_ranks = {result["id"]: i + 1 for i, result in enumerate(semantic_results)}

        all_doc_ids = set(bm25_ranks.keys()) | set(semantic_ranks.keys())

        rrf_results = []
        for doc_id in all_doc_ids:
            doc = self.documents_map[doc_id]

            bm25_rank = bm25_ranks.get(doc_id)
            semantic_rank = semantic_ranks.get(doc_id)

            rrf_score_val = 0.0
            if bm25_rank:
                rrf_score_val += 1 / (k + bm25_rank)
            if semantic_rank:
                rrf_score_val += 1 / (k + semantic_rank)

            rrf_results.append({
                "id": doc_id,
                "title": doc["title"],
                "description": doc["description"],
                "bm25_rank": bm25_rank,
                "semantic_rank": semantic_rank,
                "rrf_score": rrf_score_val
            })

        rrf_results.sort(key=lambda x: x["rrf_score"], reverse=True)
        return rrf_results[:limit]

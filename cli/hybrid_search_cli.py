#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from lib.hybrid_search import HybridSearch

DATA_PATH_MOVIES = Path(__file__).parent.parent / "data" / "movies.json"

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser("normalize", help="Normalize a list of scores using min-max normalization")
    normalize_parser.add_argument("scores", type=float, nargs="*", help="Scores to normalize")

    weighted_search_parser = subparsers.add_parser("weighted-search", help="Perform weighted hybrid search")
    weighted_search_parser.add_argument("query", type=str, help="Search query")
    weighted_search_parser.add_argument("--alpha", type=float, default=0.5, help="Weight for BM25 (0-1, default 0.5)")
    weighted_search_parser.add_argument("--limit", type=int, default=5, help="Number of results to return")

    rrf_search_parser = subparsers.add_parser("rrf-search", help="Perform reciprocal rank fusion hybrid search")
    rrf_search_parser.add_argument("query", type=str, help="Search query")
    rrf_search_parser.add_argument("-k", type=int, default=60, help="RRF k parameter (default 60)")
    rrf_search_parser.add_argument("--limit", type=int, default=5, help="Number of results to return")

    args = parser.parse_args()

    match args.command:
        case "normalize":
            if not args.scores:
                return

            min_score = min(args.scores)
            max_score = max(args.scores)

            if min_score == max_score:
                normalized = [1.0] * len(args.scores)
            else:
                score_range = max_score - min_score
                normalized = [(score - min_score) / score_range for score in args.scores]

            for score in normalized:
                print(f"* {score:.4f}")
        case "weighted-search":
            with open(DATA_PATH_MOVIES, "r", encoding="utf-8") as f:
                documents = json.load(f)["movies"]
            hs = HybridSearch(documents)
            results = hs.weighted_search(args.query, args.alpha, args.limit)
            for i, result in enumerate(results, 1):
                print(f"{i}. {result['title']}")
                print(f"  Hybrid Score: {result['hybrid_score']:.3f}")
                print(f"  BM25: {result['bm25_score']:.3f}, Semantic: {result['semantic_score']:.3f}")
                print(f"  {result['description'][:100]}...")
                print()
        case "rrf-search":
            with open(DATA_PATH_MOVIES, "r", encoding="utf-8") as f:
                documents = json.load(f)["movies"]
            hs = HybridSearch(documents)
            results = hs.rrf_search(args.query, args.k, args.limit)
            for i, result in enumerate(results, 1):
                print(f"{i}. {result['title']}")
                print(f"  RRF Score: {result['rrf_score']:.3f}")
                bm25_rank_str = str(result['bm25_rank']) if result['bm25_rank'] else "N/A"
                semantic_rank_str = str(result['semantic_rank']) if result['semantic_rank'] else "N/A"
                print(f"  BM25 Rank: {bm25_rank_str}, Semantic Rank: {semantic_rank_str}")
                print(f"  {result['description'][:100]}...")
                print()
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()

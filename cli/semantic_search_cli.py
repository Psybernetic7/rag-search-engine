#!/usr/bin/env python3

import json
import argparse
from pathlib import Path
from lib.semantic_search import SemanticSearch, verify_model, verify_embeddings, embed_text, embed_query_text

DATA_PATH_MOVIES = Path(__file__).parent.parent / "data" / "movies.json"

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("verify", help="Verify the semantic search model is loaded")
    subparsers.add_parser("verify_embeddings", help="Verify the movie embeddings")

    embed_parser = subparsers.add_parser("embed_text", help="Embed a text string")
    embed_parser.add_argument("text", type=str, help="Text to embed")

    embedquery_parser = subparsers.add_parser("embedquery", help="Embed a search query")
    embedquery_parser.add_argument("query", type=str, help="Query to embed")

    search_parser = subparsers.add_parser("search", help="Search movies by semantic similarity")
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser.add_argument("--limit", type=int, default=5, help="Number of results to return")

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "verify_embeddings":
            verify_embeddings()
        case "embed_text":
            embed_text(args.text)
        case "embedquery":
            embed_query_text(args.query)
        case "search":
            with open(DATA_PATH_MOVIES, "r", encoding="utf-8") as f:
                documents = json.load(f)["movies"]
            ss = SemanticSearch()
            ss.load_or_create_embeddings(documents)
            results = ss.search(args.query, args.limit)
            for i, result in enumerate(results, 1):
                print(f"{i}. {result['title']} (score: {result['score']:.4f})")
                print(f"  {result['description'][:100]}...")
                print()
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
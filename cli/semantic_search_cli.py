#!/usr/bin/env python3

import json
import re
import argparse
from pathlib import Path
from lib import semantic_search
from lib.semantic_search import SemanticSearch, ChunkedSemanticSearch, verify_model, verify_embeddings, embed_text, embed_query_text, semantic_chunk

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

    search_chunked_parser = subparsers.add_parser("search_chunked", help="Search movies using chunked semantic similarity")
    search_chunked_parser.add_argument("query", type=str, help="Search query")
    search_chunked_parser.add_argument("--limit", type=int, default=5, help="Number of results to return")

    subparsers.add_parser("embed_chunks", help="Load or build chunked embeddings for all movies")

    semantic_chunk_parser = subparsers.add_parser("semantic_chunk", help="Split text into sentence-based chunks")
    semantic_chunk_parser.add_argument("text", type=str, help="Text to chunk")
    semantic_chunk_parser.add_argument("--max-chunk-size", type=int, default=4, help="Max sentences per chunk")
    semantic_chunk_parser.add_argument("--overlap", type=int, default=0, help="Number of sentences to overlap between chunks")

    chunk_parser = subparsers.add_parser("chunk", help="Split text into word chunks")
    chunk_parser.add_argument("text", type=str, help="Text to chunk")
    chunk_parser.add_argument("--chunk-size", type=int, default=200, help="Number of words per chunk")
    chunk_parser.add_argument("--overlap", type=int, default=0, help="Number of words to overlap between chunks")

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
        case "search_chunked":
            with open(DATA_PATH_MOVIES, "r", encoding="utf-8") as f:
                documents = json.load(f)["movies"]
            css = ChunkedSemanticSearch()
            css.load_or_create_chunk_embeddings(documents)
            results = css.search_chunks(args.query, args.limit)
            for i, result in enumerate(results, 1):
                print(f"\n{i}. {result['title']} (score: {result['score']:.4f})")
                print(f"   {result['document']}...")
        case "embed_chunks":
            with open(DATA_PATH_MOVIES, "r", encoding="utf-8") as f:
                documents = json.load(f)["movies"]
            css = ChunkedSemanticSearch()
            embeddings = css.load_or_create_chunk_embeddings(documents)
            print(f"Generated {len(embeddings)} chunked embeddings")
        case "semantic_chunk":
            chunks = semantic_chunk(args.text, args.max_chunk_size, args.overlap)
            print(f"Semantically chunking {len(args.text)} characters into {len(chunks)} chunks")
            if chunks:
                for i, chunk in enumerate(chunks, 1):
                    print(f"{i}. {chunk}")
            else:
                print("(empty result)")
        case "chunk":
            words = args.text.split()
            step = max(1, args.chunk_size - args.overlap)
            chunks = []
            i = 0
            while i < len(words):
                chunks.append(" ".join(words[i:i + args.chunk_size]))
                i += step
            print(f"Chunking {len(args.text)} characters")
            for i, chunk in enumerate(chunks, 1):
                print(f"{i}. {chunk}")
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
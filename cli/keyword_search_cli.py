#!/usr/bin/env python3

import argparse
from lib.keyword_search import search_command, build_command, tf_command, idf_command


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # new build command
    subparsers.add_parser("build", help="Build the inverted index")

    tf_parser = subparsers.add_parser("tf", help="Get term frequency")
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Search term")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    idf_parser = subparsers.add_parser("idf", help="Get IDF score")
    idf_parser.add_argument("term", type=str, help="term")

    args = parser.parse_args()

    match args.command:
        case "build":
            build_command()
        case "search":
            print(f"Searching for: {args.query}")
            search_command(query=args.query)
        case "tf":
            tf = tf_command(args.doc_id, args.term)
            print(tf)
        case "idf":
            idf = idf_command(args.term)
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
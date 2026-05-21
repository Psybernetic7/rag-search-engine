#!/usr/bin/env python3

import argparse
import json
import os
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from lib.hybrid_search import HybridSearch

DATA_PATH_MOVIES = Path(__file__).parent.parent / "data" / "movies.json"

load_dotenv()
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")


def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    summarize_parser = subparsers.add_parser(
        "summarize", help="Summarize search results for a query"
    )
    summarize_parser.add_argument("query", type=str, help="Search query to summarize")
    summarize_parser.add_argument(
        "--limit", type=int, default=5, help="Number of results to summarize (default 5)"
    )

    citations_parser = subparsers.add_parser(
        "citations", help="Answer a query with cited sources"
    )
    citations_parser.add_argument("query", type=str, help="Search query")
    citations_parser.add_argument(
        "--limit", type=int, default=5, help="Number of results to retrieve (default 5)"
    )

    question_parser = subparsers.add_parser(
        "question", help="Answer a question conversationally based on search results"
    )
    question_parser.add_argument("question", type=str, help="Question to answer")
    question_parser.add_argument(
        "--limit", type=int, default=5, help="Number of results to retrieve (default 5)"
    )

    args = parser.parse_args()

    match args.command:
        case "rag":
            query = args.query

            with open(DATA_PATH_MOVIES, "r", encoding="utf-8") as f:
                documents = json.load(f)["movies"]

            hs = HybridSearch(documents)
            results = hs.rrf_search(query, k=60, limit=5)

            print("Search Results:")
            for result in results:
                print(f"- {result['title']}")

            docs = "\n".join(
                f"{r['title']}: {r['description']}" for r in results
            )

            prompt = f"""You are a RAG agent for Hoopla, a movie streaming service.
Your task is to provide a natural-language answer to the user's query based on documents retrieved during search.
Provide a comprehensive answer that addresses the user's query.

Query: {query}

Documents:
{docs}

Answer:"""

            client = genai.Client(api_key=GEMINI_API_KEY)
            response = client.models.generate_content(
                model="gemma-4-31b-it",
                contents=prompt,
            )

            print(f"\nRAG Response:\n{response.text.strip()}")
        case "summarize":
            query = args.query

            with open(DATA_PATH_MOVIES, "r", encoding="utf-8") as f:
                documents = json.load(f)["movies"]

            hs = HybridSearch(documents)
            search_results = hs.rrf_search(query, k=60, limit=args.limit)

            print("Search Results:")
            for result in search_results:
                print(f"  - {result['title']}")

            results = "\n".join(
                f"{r['title']}: {r['description']}" for r in search_results
            )

            prompt = f"""Provide information useful to the query below by synthesizing data from multiple search results in detail.

The goal is to provide comprehensive information so that users know what their options are.
Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.

This should be tailored to Hoopla users. Hoopla is a movie streaming service.

Query: {query}

Search results:
{results}

Provide a comprehensive 3–4 sentence answer that combines information from multiple sources:"""

            client = genai.Client(api_key=GEMINI_API_KEY)
            response = client.models.generate_content(
                model="gemma-4-31b-it",
                contents=prompt,
            )

            print(f"\nLLM Summary:\n{response.text.strip()}")
        case "citations":
            query = args.query

            with open(DATA_PATH_MOVIES, "r", encoding="utf-8") as f:
                raw_documents = json.load(f)["movies"]

            hs = HybridSearch(raw_documents)
            results = hs.rrf_search(query, k=60, limit=args.limit)

            print("Search Results:")
            for result in results:
                print(f"  - {result['title']}")

            documents = "\n".join(
                f"[{i}] {r['title']}: {r['description']}"
                for i, r in enumerate(results, 1)
            )

            prompt = f"""Answer the query below and give information based on the provided documents.

The answer should be tailored to users of Hoopla, a movie streaming service.
If not enough information is available to provide a good answer, say so, but give the best answer possible while citing the sources available.

Query: {query}

Documents:
{documents}

Instructions:
- Provide a comprehensive answer that addresses the query
- Cite sources in the format [1], [2], etc. when referencing information
- If sources disagree, mention the different viewpoints
- If the answer isn't in the provided documents, say "I don't have enough information"
- Be direct and informative

Answer:"""

            client = genai.Client(api_key=GEMINI_API_KEY)
            response = client.models.generate_content(
                model="gemma-4-31b-it",
                contents=prompt,
            )

            print(f"\nLLM Answer:\n{response.text.strip()}")
        case "question":
            question = args.question

            with open(DATA_PATH_MOVIES, "r", encoding="utf-8") as f:
                raw_documents = json.load(f)["movies"]

            hs = HybridSearch(raw_documents)
            results = hs.rrf_search(question, k=60, limit=args.limit)

            print("Search Results:")
            for result in results:
                print(f"  - {result['title']}")

            context = "\n".join(
                f"{r['title']}: {r['description']}" for r in results
            )

            prompt = f"""Answer the user's question based on the provided movies that are available on Hoopla, a streaming service.

Question: {question}

Documents:
{context}

Instructions:
- Answer questions directly and concisely
- Be casual and conversational
- Don't be cringe or hype-y
- Talk like a normal person would in a chat conversation

Answer:"""

            client = genai.Client(api_key=GEMINI_API_KEY)
            response = client.models.generate_content(
                model="gemma-4-31b-it",
                contents=prompt,
            )

            print(f"\nAnswer:\n{response.text.strip()}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()

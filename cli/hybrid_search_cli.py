#!/usr/bin/env python3

import argparse
import json
import logging
import os
import time
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from sentence_transformers import CrossEncoder
from lib.hybrid_search import HybridSearch

DATA_PATH_MOVIES = Path(__file__).parent.parent / "data" / "movies.json"

logger = logging.getLogger("hybrid_search_cli")

load_dotenv()
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

def enhance_query_spell(query: str) -> str:
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY environment variable not set")

    client = genai.Client(api_key=GEMINI_API_KEY)

    prompt = f"""Fix any spelling errors in the user-provided movie search query below.
Correct only clear, high-confidence typos. Do not rewrite, add, remove, or reorder words.
Preserve punctuation and capitalization unless a change is required for a typo fix.
If there are no spelling errors, or if you're unsure, output the original query unchanged.
Output only the final query text, nothing else.
User query: "{query}"
"""

    response = client.models.generate_content(
        model="gemma-4-31b-it",
        contents=prompt
    )

    return response.text.strip()

def enhance_query_rewrite(query: str) -> str:
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY environment variable not set")

    client = genai.Client(api_key=GEMINI_API_KEY)

    prompt = f"""Rewrite the user-provided movie search query below to be more specific and searchable.

Consider:
- Common movie knowledge (famous actors, popular films)
- Genre conventions (horror = scary, animation = cartoon)
- Keep the rewritten query concise (under 10 words)
- It should be a Google-style search query, specific enough to yield relevant results
- Don't use boolean logic

Examples:
- "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
- "movie about bear in london with marmalade" -> "Paddington London marmalade"
- "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

If you cannot improve the query, output the original unchanged.
Output only the rewritten query text, nothing else.

User query: "{query}"
"""

    response = client.models.generate_content(
        model="gemma-4-31b-it",
        contents=prompt
    )

    return response.text.strip()

def enhance_query_expand(query: str) -> str:
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY environment variable not set")

    client = genai.Client(api_key=GEMINI_API_KEY)

    prompt = f"""Expand the user-provided movie search query below with related terms.

Add synonyms and related concepts that might appear in movie descriptions.
Keep expansions relevant and focused.
Output only the additional terms; they will be appended to the original query.

Examples:
- "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
- "action movie with bear" -> "action thriller bear chase fight adventure"
- "comedy with bear" -> "comedy funny bear humor lighthearted"

User query: "{query}"
"""

    response = client.models.generate_content(
        model="gemma-4-31b-it",
        contents=prompt
    )

    expanded_terms = response.text.strip()
    return f"{query} {expanded_terms}"

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
    rrf_search_parser.add_argument(
        "--enhance",
        type=str,
        choices=["spell", "rewrite", "expand"],
        help="Query enhancement method",
    )
    rrf_search_parser.add_argument(
        "--rerank-method",
        type=str,
        choices=["individual", "batch", "cross_encoder"],
        help="Re-ranking method to apply after initial RRF search",
    )
    rrf_search_parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging for the search pipeline",
    )
    rrf_search_parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Use an LLM to evaluate the relevance of each result (0-3 scale)",
    )

    args = parser.parse_args()

    if getattr(args, "debug", False):
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[DEBUG] %(message)s"))
        logger.addHandler(handler)

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
            query = args.query
            logger.debug("Original query: '%s'", query)
            if args.enhance == "spell":
                enhanced_query = enhance_query_spell(query)
                if enhanced_query != query:
                    print(f"Enhanced query (spell): '{query}' -> '{enhanced_query}'")
                query = enhanced_query
            elif args.enhance == "rewrite":
                enhanced_query = enhance_query_rewrite(query)
                if enhanced_query != query:
                    print(f"Enhanced query (rewrite): '{query}' -> '{enhanced_query}'")
                query = enhanced_query
            elif args.enhance == "expand":
                enhanced_query = enhance_query_expand(query)
                if enhanced_query != query:
                    print(f"Enhanced query (expand): '{query}' -> '{enhanced_query}'")
                query = enhanced_query

            logger.debug("Query after enhancement: '%s'", query)

            with open(DATA_PATH_MOVIES, "r", encoding="utf-8") as f:
                documents = json.load(f)["movies"]
            hs = HybridSearch(documents)

            fetch_limit = args.limit * 5 if args.rerank_method else args.limit
            results = hs.rrf_search(query, args.k, fetch_limit)

            logger.debug(
                "RRF results (%d): %s",
                len(results),
                ", ".join(f"'{r['title']}' ({r['rrf_score']:.4f})" for r in results),
            )

            if args.rerank_method == "individual":
                print(f"Re-ranking top {args.limit} results using individual method...")
                client = genai.Client(api_key=GEMINI_API_KEY)
                for result in results:
                    description_snippet = result.get("description", "")[:500]
                    prompt = f"""Rate how well this movie matches the search query.

Query: "{query}"
Movie: {result.get("title", "")} - {description_snippet}

Consider:
- Direct relevance to query
- User intent (what they're looking for)
- Content appropriateness

Rate 0-10 (10 = perfect match).
Output ONLY the number in your response, no other text or explanation.

Score:"""
                    try:
                        response = client.models.generate_content(
                            model="gemma-4-31b-it",
                            contents=prompt,
                        )
                        result["llm_score"] = float(response.text.strip())
                    except Exception:
                        result["llm_score"] = 0.0
                    time.sleep(3)

                results.sort(key=lambda x: x["llm_score"], reverse=True)
                results = results[:args.limit]

            elif args.rerank_method == "batch":
                print(f"Re-ranking top {args.limit} results using batch method...")
                client = genai.Client(api_key=GEMINI_API_KEY)
                doc_list_str = "\n".join(
                    f"ID: {r['id']} | {r['title']} - {r['description'][:300]}"
                    for r in results
                )
                prompt = f"""Rank the movies listed below by relevance to the following search query.

Query: "{query}"

Movies:
{doc_list_str}

Return the movie IDs in order of relevance, best match first.

Your response must be a raw JSON array of integers.
Do not wrap the JSON in Markdown. Do not use a ```json code block.
Do not include any explanatory text.

For example:
[75, 12, 34, 2, 1]

Ranking:"""
                try:
                    response = client.models.generate_content(
                        model="gemma-4-31b-it",
                        contents=prompt,
                    )
                    ranked_ids = json.loads(response.text.strip())
                    id_to_rank = {doc_id: rank for rank, doc_id in enumerate(ranked_ids, 1)}
                    for result in results:
                        result["llm_rank"] = id_to_rank.get(result["id"], len(results) + 1)
                except Exception:
                    for i, result in enumerate(results, 1):
                        result["llm_rank"] = i

                results.sort(key=lambda x: x["llm_rank"])
                results = results[:args.limit]

            elif args.rerank_method == "cross_encoder":
                print(f"Re-ranking top {args.limit} results using cross_encoder method...")
                pairs = [
                    [query, f"{r.get('title', '')} - {r.get('description', '')}"]
                    for r in results
                ]
                cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2", device="cpu")
                scores = cross_encoder.predict(pairs)
                for result, score in zip(results, scores):
                    result["ce_score"] = float(score)
                results.sort(key=lambda x: x["ce_score"], reverse=True)
                results = results[:args.limit]

            logger.debug(
                "Final results after re-ranking (%d): %s",
                len(results),
                ", ".join(f"'{r['title']}'" for r in results),
            )

            print(f"Reciprocal Rank Fusion Results for '{query}' (k={args.k}):\n")
            for i, result in enumerate(results, 1):
                print(f"{i}. {result['title']}")
                if args.rerank_method == "individual":
                    print(f"   Re-rank Score: {result['llm_score']:.3f}/10")
                elif args.rerank_method == "batch":
                    print(f"   Re-rank Rank: {result['llm_rank']}")
                elif args.rerank_method == "cross_encoder":
                    print(f"   Cross Encoder Score: {result['ce_score']:.3f}")
                print(f"   RRF Score: {result['rrf_score']:.3f}")
                bm25_rank_str = str(result['bm25_rank']) if result['bm25_rank'] else "N/A"
                semantic_rank_str = str(result['semantic_rank']) if result['semantic_rank'] else "N/A"
                print(f"   BM25 Rank: {bm25_rank_str}, Semantic Rank: {semantic_rank_str}")
                print(f"   {result['description'][:100]}...")
                print()
            if args.evaluate:
                formatted_results = [
                    f"{i}. {r['title']} - {r['description'][:200]}"
                    for i, r in enumerate(results, 1)
                ]
                eval_prompt = f"""Rate how relevant each result is to this query on a 0-3 scale:

Query: "{query}"

Results:
{chr(10).join(formatted_results)}

Scale:
- 3: Highly relevant
- 2: Relevant
- 1: Marginally relevant
- 0: Not relevant

Do NOT give any numbers other than 0, 1, 2, or 3.

Return ONLY the scores in the same order you were given the documents. Return a valid JSON list, nothing else. For example:

[2, 0, 3, 2, 0, 1]"""
                client = genai.Client(api_key=GEMINI_API_KEY)
                try:
                    response = client.models.generate_content(
                        model="gemma-4-31b-it",
                        contents=eval_prompt,
                    )
                    scores = json.loads((response.text or "").strip())
                    print("Evaluation Report:")
                    for i, (result, score) in enumerate(zip(results, scores), 1):
                        print(f"{i}. {result['title']}: {score}/3")
                except Exception as e:
                    print(f"Evaluation failed: {e}")
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()

#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from lib.hybrid_search import HybridSearch

DATA_PATH_MOVIES = Path(__file__).parent.parent / "data" / "movies.json"
DATA_PATH_GOLDEN = Path(__file__).parent.parent / "data" / "golden_dataset.json"


def main():
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()
    limit = args.limit

    with open(DATA_PATH_MOVIES, "r", encoding="utf-8") as f:
        documents = json.load(f)["movies"]

    with open(DATA_PATH_GOLDEN, "r", encoding="utf-8") as f:
        test_cases = json.load(f)["test_cases"]

    hs = HybridSearch(documents)

    print(f"k={limit}\n")

    for test_case in test_cases:
        query = test_case["query"]
        relevant_titles = set(test_case["relevant_docs"])

        results = hs.rrf_search(query, k=60, limit=limit)
        retrieved_titles = [r["title"] for r in results]

        hits = sum(1 for t in retrieved_titles if t in relevant_titles)
        precision = hits / limit if limit > 0 else 0.0
        recall = hits / len(relevant_titles) if relevant_titles else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        print(f"- Query: {query}")
        print(f"  - Precision@{limit}: {precision:.4f}")
        print(f"  - Recall@{limit}: {recall:.4f}")
        print(f"  - F1 Score: {f1:.4f}")
        print(f"  - Retrieved: {', '.join(retrieved_titles)}")
        print(f"  - Relevant: {', '.join(relevant_titles)}")
        print()


if __name__ == "__main__":
    main()

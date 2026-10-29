"""Check the shipped bundle and real API loading path without training a fixture model.

Run with the manifest-compatible Python version and the API/development extras:
    uv run --frozen --extra api python scripts/smoke_serving_bundle.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fastapi.testclient import TestClient

from app.api import create_app
from book_recommender.content_based import book_work_key
from book_recommender.service import RecommenderService


def smoke_bundle(artifact: Path) -> dict[str, object]:
    # This performs checksum, runtime-version, and composite-model validation.
    # A missing/incompatible bundle is an error, never a request to train one.
    service = RecommenderService.load(artifact)
    top_n = 5
    user_id = int(service.ratings.groupby("user_id").size().idxmax())
    query_isbn = str(service.books.iloc[0]["isbn"])
    expected_user = service.recommend_for_user(user_id, top_n=top_n)
    expected_similar = service.recommend_similar(isbn=query_isbn, top_n=top_n)
    if len(expected_user) != top_n or len(expected_similar) != top_n:
        raise AssertionError("Shipped bundle did not return complete smoke recommendation lists.")

    seen_isbns = set(
        service.ratings.loc[service.ratings.user_id == user_id, "isbn"].astype(str)
    )
    seen_works = {
        book_work_key(row.book_title, row.book_author, row.isbn)
        for row in service.books.itertuples(index=False)
        if str(row.isbn) in seen_isbns
    }
    recommended_works = [
        book_work_key(row.book_title, row.book_author, row.isbn)
        for row in expected_user.itertuples(index=False)
    ]
    if len(set(recommended_works)) != top_n or seen_works.intersection(recommended_works):
        raise AssertionError("Shipped bundle repeated or recommended a previously read work.")

    # Use the default loader, not service_loader=lambda _: a synthetic fixture.
    with TestClient(create_app(artifact_path=artifact)) as client:
        ready = client.get("/readyz")
        if ready.status_code != 200:
            raise AssertionError(f"Shipped-bundle API is not ready: {ready.text}")
        if ready.json()["artifact_id"] != service.manifest.artifact_id:
            raise AssertionError("API readiness identifies a different artifact.")
        requests = [
            (f"/v1/recommendations/users/{user_id}", {"top_n": top_n}, expected_user),
            (
                "/v1/recommendations/similar",
                {"isbn": query_isbn, "top_n": top_n},
                expected_similar,
            ),
        ]
        for endpoint, params, expected in requests:
            response = client.get(endpoint, params=params)
            if response.status_code != 200:
                raise AssertionError(f"Shipped-bundle API request failed: {response.text}")
            payload = response.json()
            if payload["artifact_id"] != service.manifest.artifact_id:
                raise AssertionError("API recommendation identifies a different artifact.")
            actual_isbns = [row["isbn"] for row in payload["recommendations"]]
            if actual_isbns != expected.isbn.astype(str).tolist():
                raise AssertionError("API and direct service recommendation order differs.")

    return {
        "status": "passed",
        "artifact_id": service.manifest.artifact_id,
        "books": len(service.books),
        "rating_pairs": len(service.ratings),
        "api_readiness": "ready",
        "recommendations_per_endpoint": top_n,
        "api_service_order_parity": True,
        "training_performed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "models" / "hybrid.joblib",
    )
    args = parser.parse_args()
    print(json.dumps(smoke_bundle(args.artifact)))


if __name__ == "__main__":
    main()

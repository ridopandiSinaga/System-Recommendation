import pandas as pd
from fastapi.testclient import TestClient

from app.api import create_app
from book_recommender.artifacts import fingerprint_dataset, save_model_bundle
from book_recommender.data import clean_books, clean_ratings
from book_recommender.hybrid import HybridRecommender
from book_recommender.service import RecommenderService


def build_service(tmp_path):
    books = clean_books(
        pd.DataFrame(
            [
                {"ISBN": "A", "Book-Title": "Alpha", "Book-Author": "One", "Publisher": "P"},
                {"ISBN": "B", "Book-Title": "Beta", "Book-Author": "One", "Publisher": "P"},
                {"ISBN": "C", "Book-Title": "Gamma", "Book-Author": "Two", "Publisher": "Q"},
            ]
        )
    )
    ratings = clean_ratings(
        pd.DataFrame(
            [
                {"User-ID": 1, "ISBN": "A", "Book-Rating": 10},
                {"User-ID": 1, "ISBN": "B", "Book-Rating": 9},
                {"User-ID": 2, "ISBN": "A", "Book-Rating": 9},
                {"User-ID": 2, "ISBN": "C", "Book-Rating": 8},
            ]
        )
    )
    artifact = tmp_path / "hybrid.joblib"
    model = HybridRecommender(n_neighbors=2).fit(books, ratings)
    save_model_bundle(
        model,
        artifact,
        model_type="hybrid",
        dataset_fingerprint=fingerprint_dataset(books, ratings),
        training_config={"n_neighbors": 2},
    )
    return RecommenderService.load(artifact)


def test_api_health_readiness_and_recommendations(tmp_path):
    service = build_service(tmp_path)
    application = create_app(artifact_path="unused", service_loader=lambda _: service)

    with TestClient(application) as client:
        assert client.get("/healthz").json() == {"status": "alive"}
        ready = client.get("/readyz")
        assert ready.status_code == 200
        assert ready.json()["artifact_id"] == service.manifest.artifact_id

        response = client.get("/v1/recommendations/users/1", params={"top_n": 1})
        assert response.status_code == 200
        assert len(response.json()["recommendations"]) == 1
        assert response.headers["x-request-id"]

        similar = client.get(
            "/v1/recommendations/similar",
            params={"isbn": "A", "top_n": 1},
        )
        assert similar.status_code == 200
        assert similar.json()["recommendations"][0]["isbn"] == "B"
        assert client.get("/metrics").json()["request_count"] >= 4


def test_api_starts_alive_but_not_ready_when_artifact_fails():
    def fail_loader(_):
        raise ValueError("broken artifact")

    application = create_app(artifact_path="missing", service_loader=fail_loader)
    with TestClient(application) as client:
        assert client.get("/healthz").status_code == 200
        ready = client.get("/readyz")
        assert ready.status_code == 503
        assert ready.json() == {"status": "not_ready", "error_code": "MODEL_NOT_READY"}
        model_info = client.get("/model-info")
        assert model_info.status_code == 503
        assert "broken artifact" not in model_info.text
        assert client.get("/v1/recommendations/users/1").status_code == 503


def test_api_rejects_invalid_query_contract(tmp_path):
    service = build_service(tmp_path)
    application = create_app(artifact_path="unused", service_loader=lambda _: service)
    with TestClient(application) as client:
        assert client.get("/v1/recommendations/users/1", params={"top_n": -1}).status_code == 422
        assert client.get("/v1/recommendations/users/0").status_code == 422
        assert client.get("/v1/recommendations/similar").status_code == 422
        assert (
            client.get("/v1/recommendations/similar", params={"title": "   "}).status_code
            == 422
        )
        assert (
            client.get("/v1/recommendations/similar", params={"title": "x" * 501}).status_code
            == 422
        )
        assert (
            client.get(
                "/v1/recommendations/similar",
                params={"title": "Alpha", "isbn": "A"},
            ).status_code
            == 422
        )


def test_api_replaces_untrusted_request_id(tmp_path):
    service = build_service(tmp_path)
    application = create_app(artifact_path="unused", service_loader=lambda _: service)
    with TestClient(application) as client:
        response = client.get("/healthz", headers={"x-request-id": "x" * 200})

    assert response.status_code == 200
    assert response.headers["x-request-id"] != "x" * 200
    assert len(response.headers["x-request-id"]) == 32


def test_api_returns_stable_not_found_contract_without_query_echo(tmp_path):
    service = build_service(tmp_path)
    application = create_app(artifact_path="unused", service_loader=lambda _: service)

    with TestClient(application) as client:
        response = client.get(
            "/v1/recommendations/similar",
            params={"isbn": "secret-missing-isbn"},
        )

    assert response.status_code == 404
    assert response.json() == {"detail": {"error_code": "BOOK_NOT_FOUND"}}
    assert "secret-missing-isbn" not in response.text


def test_api_sanitizes_unexpected_model_errors_and_keeps_request_id(
    monkeypatch,
    tmp_path,
):
    service = build_service(tmp_path)

    def fail_recommendation(**_):
        raise ValueError("secret internal model detail")

    monkeypatch.setattr(service.model.content_model, "recommend_similar", fail_recommendation)
    application = create_app(artifact_path="unused", service_loader=lambda _: service)

    with TestClient(application) as client:
        response = client.get(
            "/v1/recommendations/similar",
            params={"isbn": "A"},
            headers={"x-request-id": "safe-request-123"},
        )

    assert response.status_code == 500
    assert response.json() == {"detail": {"error_code": "INTERNAL_ERROR"}}
    assert "secret internal model detail" not in response.text
    assert response.headers["x-request-id"] == "safe-request-123"

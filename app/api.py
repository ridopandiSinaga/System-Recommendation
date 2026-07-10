from __future__ import annotations

from collections.abc import Callable
from contextlib import asynccontextmanager
import logging
import math
import os
from pathlib import Path
import re
from time import perf_counter
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Path as ApiPath, Query, Request
from fastapi.responses import JSONResponse
import pandas as pd
from pydantic import BaseModel

from book_recommender.observability import RuntimeMetrics, configure_json_logging
from book_recommender.content_based import BookNotFoundError
from book_recommender.service import RecommenderService


DEFAULT_ARTIFACT = Path(__file__).resolve().parents[1] / "models" / "hybrid.joblib"
logger = logging.getLogger("book_recommender.api")
REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,128}$")


class Recommendation(BaseModel):
    isbn: str
    book_title: str
    book_author: str
    publisher: str
    score: float
    source: str | None = None
    reason: str | None = None


class RecommendationResponse(BaseModel):
    artifact_id: str
    recommendations: list[Recommendation]


def create_app(
    *,
    artifact_path: str | Path | None = None,
    service_loader: Callable[[str | Path], RecommenderService] = RecommenderService.load,
) -> FastAPI:
    configure_json_logging()
    configured_path = Path(
        artifact_path or os.getenv("BOOK_RECOMMENDER_ARTIFACT", str(DEFAULT_ARTIFACT))
    )

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        application.state.service = None
        application.state.load_error = None
        application.state.metrics = RuntimeMetrics()
        try:
            application.state.service = service_loader(configured_path)
            logger.info(
                "model_loaded",
                extra={"artifact_id": application.state.service.manifest.artifact_id},
            )
        except Exception as exc:
            # The process must stay alive so orchestration can inspect /healthz and
            # /readyz even when a trusted local artifact is corrupt or incompatible.
            application.state.load_error = f"{type(exc).__name__}: {exc}"
            logger.exception("model_load_failed", extra={"error_type": type(exc).__name__})
        yield

    application = FastAPI(
        title="Book Recommender API",
        version="1.0.0",
        lifespan=lifespan,
    )

    @application.middleware("http")
    async def request_metrics(request: Request, call_next):
        supplied_request_id = request.headers.get("x-request-id", "")
        request_id = (
            supplied_request_id
            if REQUEST_ID_PATTERN.fullmatch(supplied_request_id)
            else uuid4().hex
        )
        started = perf_counter()
        status_code = 500
        try:
            try:
                response = await call_next(request)
                status_code = response.status_code
            except Exception as exc:
                logger.exception(
                    "request_failed",
                    extra={
                        "request_id": request_id,
                        "error_type": type(exc).__name__,
                    },
                )
                response = JSONResponse(
                    status_code=500,
                    content={"detail": {"error_code": "INTERNAL_ERROR"}},
                )
                status_code = 500
            response.headers["x-request-id"] = request_id
            return response
        finally:
            duration_ms = (perf_counter() - started) * 1000
            metrics = getattr(request.app.state, "metrics", None)
            if metrics is not None:
                metrics.observe(duration_ms, status_code)
            logger.info(
                "request_completed",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": status_code,
                    "duration_ms": round(duration_ms, 3),
                },
            )

    @application.get("/healthz")
    def health() -> dict[str, str]:
        return {"status": "alive"}

    @application.get("/readyz")
    def readiness(request: Request):
        service = getattr(request.app.state, "service", None)
        if service is None:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "not_ready",
                    "error_code": "MODEL_NOT_READY",
                },
            )
        return service.model_info()

    @application.get("/model-info")
    def model_info(request: Request):
        return _require_service(request).model_info()

    @application.get("/metrics")
    def metrics(request: Request) -> dict[str, int | float]:
        return request.app.state.metrics.snapshot()

    @application.get(
        "/v1/recommendations/users/{user_id}",
        response_model=RecommendationResponse,
    )
    def recommend_user(
        request: Request,
        user_id: int = ApiPath(ge=1),
        top_n: int = Query(default=10, ge=1, le=100),
    ) -> RecommendationResponse:
        service = _require_service(request)
        recommendations = service.recommend_for_user(user_id=user_id, top_n=top_n)
        return RecommendationResponse(
            artifact_id=service.manifest.artifact_id,
            recommendations=_records(recommendations),
        )

    @application.get(
        "/v1/recommendations/similar",
        response_model=RecommendationResponse,
    )
    def recommend_similar(
        request: Request,
        title: str | None = Query(default=None, min_length=1, max_length=500),
        isbn: str | None = Query(default=None, min_length=1, max_length=64),
        top_n: int = Query(default=10, ge=1, le=100),
    ) -> RecommendationResponse:
        title = title.strip() if title else None
        isbn = isbn.strip() if isbn else None
        if bool(title) == bool(isbn):
            raise HTTPException(status_code=422, detail="Provide exactly one of title or isbn.")
        service = _require_service(request)
        try:
            recommendations = service.recommend_similar(title=title, isbn=isbn, top_n=top_n)
        except BookNotFoundError as exc:
            raise HTTPException(
                status_code=404,
                detail={"error_code": "BOOK_NOT_FOUND"},
            ) from exc
        return RecommendationResponse(
            artifact_id=service.manifest.artifact_id,
            recommendations=_records(recommendations),
        )

    return application


def _require_service(request: Request) -> RecommenderService:
    service = getattr(request.app.state, "service", None)
    if service is None:
        raise HTTPException(
            status_code=503,
            detail={"error_code": "MODEL_NOT_READY"},
        )
    return service


def _records(dataframe) -> list[Recommendation]:
    records = []
    for row in dataframe.itertuples(index=False):
        score = float(row.score)
        if not math.isfinite(score):
            raise RuntimeError("Model produced a non-finite recommendation score.")
        records.append(
            Recommendation(
                isbn=_required_text(row.isbn, "isbn"),
                book_title=_required_text(row.book_title, "book_title"),
                book_author=_required_text(row.book_author, "book_author"),
                publisher=_required_text(row.publisher, "publisher"),
                score=score,
                source=_optional_text(getattr(row, "source", None)),
                reason=_optional_text(getattr(row, "reason", None)),
            )
        )
    return records


def _required_text(value, field_name: str) -> str:
    text = _optional_text(value)
    if not text:
        raise RuntimeError(f"Model produced an empty {field_name} field.")
    return text


def _optional_text(value) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


app = create_app()

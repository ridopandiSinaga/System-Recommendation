from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import logging
from threading import Lock
from typing import Any


class JsonFormatter(logging.Formatter):
    """Small structured formatter with stable fields for container log collection."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for field_name in (
            "request_id",
            "method",
            "path",
            "status_code",
            "duration_ms",
            "artifact_id",
            "error_type",
        ):
            value = getattr(record, field_name, None)
            if value is not None:
                payload[field_name] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_json_logging(level: int = logging.INFO) -> None:
    root = logging.getLogger()
    if any(isinstance(handler.formatter, JsonFormatter) for handler in root.handlers):
        return
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)


@dataclass
class RuntimeMetrics:
    request_count: int = 0
    error_count: int = 0
    total_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    _lock: Lock = field(default_factory=Lock, repr=False)

    def observe(self, duration_ms: float, status_code: int) -> None:
        with self._lock:
            self.request_count += 1
            self.total_latency_ms += float(duration_ms)
            self.max_latency_ms = max(self.max_latency_ms, float(duration_ms))
            if status_code >= 500:
                self.error_count += 1

    def snapshot(self) -> dict[str, int | float]:
        with self._lock:
            average = self.total_latency_ms / self.request_count if self.request_count else 0.0
            return {
                "request_count": self.request_count,
                "error_count": self.error_count,
                "average_latency_ms": round(average, 3),
                "max_latency_ms": round(self.max_latency_ms, 3),
            }

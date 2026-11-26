import json
import re
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from pydantic import ValidationError

from src.agents.schemas import DiagnosticRequest, DiagnosticResult, FaultCodeFinding, SensorFinding
from src.agents.tools import DiagnosticTools
from src.core.settings import Settings
from src.core.tokenizer import tokenize_codes

_KEY_VERSION = 1
_NUMBERS = re.compile(r"(?<![\w.])[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:e[+-]?\d+)?(?![\w.])", re.I)
_COMPARISONS = re.compile(r"<=|>=|!=|==|[<>=%]")
_QUERY_QUALIFIERS = frozenset(
    """above amp amps bar below c celsius critical degc degf f fahrenheit greater higher hz
    k kelvin kw l/min less lower maximum minimum mm/s mpa never no normal not ohm pa psi
    rms rpm volts warning without""".split()
)


def _digest(value: object) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def profile_hash(settings: Settings) -> str:
    profile = settings.model_dump(
        mode="json",
        include={"domain", "retrieval", "llm", "embedding", "reranker"},
        exclude={"llm": {"api_key"}, "embedding": {"api_key"}},
    )
    policy = "mock-extractive" if settings.llm.provider == "mock" else "llm-critic-v3"
    return _digest({"key_version": _KEY_VERSION, "diagnostic_policy": policy, "profile": profile})


def context_hash(request: DiagnosticRequest, tools: DiagnosticTools) -> str:
    findings = tools.evaluate(request)
    sensor_rules = {
        item.sensor.casefold(): item.rule for item in findings if isinstance(item, SensorFinding)
    }
    readings = []
    for label, value in sorted(
        (key.casefold(), value) for key, value in request.sensor_readings.items()
    ):
        rule = sensor_rules.get(label)
        normalized = 0.0 if value == 0 else float(value)
        readings.append((label, normalized.hex(), rule.unit.casefold() if rule else None))
    # Numbers, codes, units, and polarity words constrain reuse even inside unstructured queries.
    details = [
        term
        for term in tokenize_codes(request.query)
        if term in _QUERY_QUALIFIERS or any(char.isdigit() for char in term)
    ]
    return _digest(
        {
            "equipment": (tools.equipment_for(request) or "").casefold(),
            "fault_code": (request.fault_code or "").casefold(),
            "inferred_faults": [
                item.fault_code.casefold()
                for item in findings
                if isinstance(item, FaultCodeFinding)
            ],
            "readings": readings,
            "query_details": details,
            "query_numbers": _NUMBERS.findall(request.query.casefold()),
            "query_comparisons": _COMPARISONS.findall(request.query),
        }
    )


@dataclass(frozen=True)
class CacheScope:
    index_version: str
    profile_hash: str
    context_hash: str

    def as_params(self) -> tuple[str, str, str]:
        return self.index_version, self.profile_hash, self.context_hash


class SQLiteSemanticCache:
    def __init__(self, path: Path, threshold: float = 0.95) -> None:
        if not 0.95 <= threshold <= 1:
            raise ValueError("Cache similarity threshold must be between 0.95 and 1")
        self._path = path
        self._threshold = threshold
        path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as connection, connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS semantic_cache (
                    index_version TEXT NOT NULL, profile_hash TEXT NOT NULL,
                    context_hash TEXT NOT NULL, query_hash TEXT NOT NULL,
                    vector BLOB NOT NULL, result TEXT NOT NULL,
                    PRIMARY KEY (index_version, profile_hash, context_hash, query_hash)
                ) WITHOUT ROWID
                """
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._path, timeout=5)

    def lookup(self, scope: CacheScope, embedding: NDArray[np.float32]) -> DiagnosticResult | None:
        if embedding.ndim != 1:
            raise ValueError("Expected a 1D embedding vector")
        query = embedding.astype(np.float64)
        query_norm = np.linalg.norm(query)
        if not query_norm or not np.isfinite(query_norm):
            return None
        best = None
        best_similarity = self._threshold
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT vector, result FROM semantic_cache
                WHERE index_version = ? AND profile_hash = ? AND context_hash = ?
                ORDER BY query_hash
                """,
                scope.as_params(),
            )
            for raw, payload in rows:
                if len(raw) != embedding.size * np.dtype("<f4").itemsize:
                    continue
                candidate = np.frombuffer(raw, dtype="<f4").astype(np.float64)
                candidate_norm = np.linalg.norm(candidate)
                if not candidate_norm or not np.isfinite(candidate_norm):
                    continue
                similarity = min(float(query @ candidate / (query_norm * candidate_norm)), 1.0)
                if similarity < best_similarity or (
                    best is not None and similarity == best_similarity
                ):
                    continue
                try:
                    response = DiagnosticResult.model_validate_json(payload)
                except ValidationError:
                    continue
                if response.status == "answered" and response.citations:
                    best, best_similarity = response, similarity
        return best

    def store(
        self,
        scope: CacheScope,
        query: str,
        embedding: NDArray[np.float32],
        result: DiagnosticResult,
    ) -> None:
        if embedding.ndim != 1:
            raise ValueError("Expected a 1D embedding vector")
        if result.status != "answered" or not result.citations or not embedding.any():
            return
        query_hash = _digest(" ".join(query.casefold().split()))
        payload = result.model_dump_json(exclude={"trace"})
        with closing(self._connect()) as connection, connection:
            connection.execute(
                """
                INSERT INTO semantic_cache
                    (index_version, profile_hash, context_hash, query_hash, vector, result)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(index_version, profile_hash, context_hash, query_hash)
                DO UPDATE SET vector=excluded.vector, result=excluded.result
                """,
                (*scope.as_params(), query_hash, embedding.astype("<f4").tobytes(), payload),
            )

    def prune_other_versions(self, active_index_version: str) -> None:
        with closing(self._connect()) as connection, connection:
            connection.execute(
                "DELETE FROM semantic_cache WHERE index_version <> ?", (active_index_version,)
            )

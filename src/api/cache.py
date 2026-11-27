import json
import re
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from time import time

import numpy as np
from numpy.typing import NDArray
from pydantic import ValidationError

from src.agents.schemas import DiagnosticRequest, DiagnosticResult, FaultCodeFinding, SensorFinding
from src.agents.tools import RULE_TABLES, DiagnosticTools
from src.core.settings import Settings
from src.core.tokenizer import split_cjk_runs, tokenize_codes

_KEY_VERSION = 1
_NUMBERS = re.compile(r"(?<![\w.])[+\-−]?(?:\d+(?:\.\d+)?|\.\d+)(?:e[+\-−]?\d+)?(?![\w.])", re.I)
_MIXED_NUMBERS = re.compile(
    r"(?<![0-9.])[+\-−]?(?:[0-9]+(?:\.[0-9]+)?|\.[0-9]+)(?:e[+\-−]?[0-9]+)?", re.I
)
_COMPARISONS = re.compile(r"<=|>=|!=|==|[<>=%≤≥≠＜＞]")
_QUERY_QUALIFIERS = frozenset(
    """above amp amps bar below c celsius critical degc degf f fahrenheit greater higher hz
    k kelvin kw l/min less lower maximum minimum mm/s mpa never no normal not ohm pa psi
    rms rpm volts warning without""".split()
)
_CJK_QUALIFIERS = re.compile(
    r"摄氏度|华氏度|大于|小于|高于|低于|超过|达到|等于|正常|异常|预警|告警|报警|严重|临界|最大|最小|不|未|无|℃|℉"
)


def _digest(value: object) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def profile_hash(settings: Settings) -> str:
    profile = settings.model_dump(
        mode="json",
        include={"domain", "retrieval", "llm", "embedding", "reranker"},
        exclude={"llm": {"api_key"}, "embedding": {"api_key"}},
    )
    if settings.rules.source_paths != RULE_TABLES:
        profile["rule_tables"] = settings.rules.source_paths
    policy = "mock-extractive" if settings.llm.provider == "mock" else "llm-critic-v4"
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
    query = request.query.casefold()
    mixed = len(split_cjk_runs(query)) > 1
    context = {
        "equipment": (tools.equipment_for(request) or "").casefold(),
        "fault_code": (request.fault_code or "").casefold(),
        "inferred_faults": [
            item.fault_code.casefold() for item in findings if isinstance(item, FaultCodeFinding)
        ],
        "readings": readings,
        "query_details": details,
        "query_numbers": (_MIXED_NUMBERS if mixed else _NUMBERS).findall(query),
        "query_comparisons": _COMPARISONS.findall(query),
    }
    qualifiers = _CJK_QUALIFIERS.findall(query)
    if qualifiers:
        context["query_cjk_qualifiers"] = qualifiers
    return _digest(context)


@dataclass(frozen=True)
class CacheScope:
    index_version: str
    profile_hash: str
    context_hash: str

    def as_params(self) -> tuple[str, str, str]:
        return self.index_version, self.profile_hash, self.context_hash


class SQLiteSemanticCache:
    def __init__(
        self,
        path: Path,
        threshold: float = 0.95,
        *,
        ttl_seconds: int | None = None,
        max_entries: int | None = 10000,
    ) -> None:
        if not 0.95 <= threshold <= 1:
            raise ValueError("Cache similarity threshold must be between 0.95 and 1")
        if ttl_seconds is not None and ttl_seconds <= 0:
            raise ValueError("Cache TTL must be positive")
        if max_entries is not None and max_entries <= 0:
            raise ValueError("Cache capacity must be positive")
        self._path = path
        self._threshold = threshold
        self._ttl_seconds = ttl_seconds
        self._max_entries = max_entries
        path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as connection, connection:
            connection.execute("PRAGMA journal_mode = WAL").fetchone()
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS semantic_cache (
                    index_version TEXT NOT NULL, profile_hash TEXT NOT NULL,
                    context_hash TEXT NOT NULL, query_hash TEXT NOT NULL,
                    vector BLOB NOT NULL, result TEXT NOT NULL, stored_at REAL NOT NULL DEFAULT 0,
                    PRIMARY KEY (index_version, profile_hash, context_hash, query_hash)
                ) WITHOUT ROWID
                """
            )
            columns = {
                column[1] for column in connection.execute("PRAGMA table_info(semantic_cache)")
            }
            if "stored_at" not in columns:
                connection.execute(
                    "ALTER TABLE semantic_cache ADD COLUMN stored_at REAL NOT NULL DEFAULT 0"
                )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS semantic_cache_age ON semantic_cache(stored_at)"
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._path, timeout=5)
        connection.execute("PRAGMA busy_timeout = 5000")
        return connection

    def _expiry_cutoff(self, now: float) -> float | None:
        return max(0.0, now - self._ttl_seconds) if self._ttl_seconds is not None else None

    def lookup(self, scope: CacheScope, embedding: NDArray[np.float32]) -> DiagnosticResult | None:
        if embedding.ndim != 1:
            raise ValueError("Expected a 1D embedding vector")
        query = embedding.astype(np.float64)
        query_norm = np.linalg.norm(query)
        if not query_norm or not np.isfinite(query_norm):
            return None
        best = None
        best_similarity = self._threshold
        cutoff = self._expiry_cutoff(time())
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT vector, result FROM semantic_cache
                WHERE index_version = ? AND profile_hash = ? AND context_hash = ?
                    AND (? IS NULL OR stored_at > ?)
                ORDER BY query_hash
                """,
                (*scope.as_params(), cutoff, cutoff),
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
            connection.execute("BEGIN IMMEDIATE")
            now = time()
            connection.execute(
                """
                INSERT INTO semantic_cache
                    (index_version, profile_hash, context_hash, query_hash,
                     vector, result, stored_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(index_version, profile_hash, context_hash, query_hash)
                DO UPDATE SET vector=excluded.vector, result=excluded.result,
                    stored_at=excluded.stored_at
                """,
                (*scope.as_params(), query_hash, embedding.astype("<f4").tobytes(), payload, now),
            )
            cutoff = self._expiry_cutoff(now)
            if cutoff is not None:
                connection.execute("DELETE FROM semantic_cache WHERE stored_at <= ?", (cutoff,))
            if self._max_entries is not None:
                connection.execute(
                    """
                    DELETE FROM semantic_cache
                    WHERE (index_version, profile_hash, context_hash, query_hash) IN (
                        SELECT index_version, profile_hash, context_hash, query_hash
                        FROM semantic_cache
                        ORDER BY stored_at DESC, index_version DESC, profile_hash DESC,
                            context_hash DESC, query_hash DESC
                        LIMIT -1 OFFSET ?
                    )
                    """,
                    (self._max_entries,),
                )

    def prune_other_versions(self, active_index_version: str) -> None:
        with closing(self._connect()) as connection, connection:
            connection.execute(
                "DELETE FROM semantic_cache WHERE index_version <> ?", (active_index_version,)
            )

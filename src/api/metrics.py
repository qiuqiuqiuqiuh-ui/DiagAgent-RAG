from bisect import bisect_left
from collections import Counter
from itertools import accumulate
from threading import Lock

from src.api.schemas import HealthResponse

_LATENCY_BOUNDS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60)
_DIAGNOSE_DURATION = "diagagent_diagnose_duration_seconds"
_DIAGNOSIS_STATUSES = ("answered", "insufficient_evidence", "out_of_scope")
_CACHE_OUTCOMES = ("hit", "miss", "bypass")


class ApiMetrics:
    """Process-local metrics. Request labels refer to route templates."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._requests: Counter[tuple[str, int]] = Counter()
        self._diagnoses: Counter[str] = Counter()
        self._cache: Counter[str] = Counter()
        self._retries: Counter[int] = Counter()
        self._duration_buckets = [0] * (len(_LATENCY_BOUNDS) + 1)
        self._duration_sum = 0.0

    def observe_http(self, endpoint: str, status: int, elapsed: float) -> None:
        with self._lock:
            self._requests[endpoint, status] += 1
            if endpoint == "/v1/diagnose" and status == 200:
                self._duration_buckets[bisect_left(_LATENCY_BOUNDS, elapsed)] += 1
                self._duration_sum += elapsed

    def observe_diagnosis(self, status: str, cache_outcome: str, retries: int | None) -> None:
        with self._lock:
            self._diagnoses[status] += 1
            self._cache[cache_outcome] += 1
            if retries is not None:
                self._retries[retries] += 1

    def render(self, health: HealthResponse) -> str:
        with self._lock:
            requests = sorted(self._requests.items())
            diagnoses = self._diagnoses.copy()
            cache = self._cache.copy()
            retries = sorted(self._retries.items())
            buckets = tuple(accumulate(self._duration_buckets))
            duration_sum = self._duration_sum
        matched = cache["hit"] + cache["miss"]
        hit_rate = cache["hit"] / matched if matched else 0.0
        lines = [
            "# TYPE diagagent_http_requests_total counter",
            *(
                f'diagagent_http_requests_total{{endpoint="{endpoint}",status="{status}"}} {count}'
                for (endpoint, status), count in requests
            ),
            "# TYPE diagagent_diagnoses_total counter",
            *(
                f'diagagent_diagnoses_total{{status="{status}"}} {diagnoses[status]}'
                for status in _DIAGNOSIS_STATUSES
            ),
            "# TYPE diagagent_cache_requests_total counter",
            *(
                f'diagagent_cache_requests_total{{outcome="{outcome}"}} {cache[outcome]}'
                for outcome in _CACHE_OUTCOMES
            ),
            "# TYPE diagagent_cache_hit_rate gauge",
            f"diagagent_cache_hit_rate {hit_rate}",
            "# TYPE diagagent_workflow_retries_total counter",
            *(
                f'diagagent_workflow_retries_total{{retries="{count}"}} {total}'
                for count, total in retries
            ),
            f"# TYPE {_DIAGNOSE_DURATION} histogram",
            *(
                f'{_DIAGNOSE_DURATION}_bucket{{le="{upper:g}"}} {count}'
                for upper, count in zip(_LATENCY_BOUNDS, buckets[:-1], strict=True)
            ),
            f'{_DIAGNOSE_DURATION}_bucket{{le="+Inf"}} {buckets[-1]}',
            f"{_DIAGNOSE_DURATION}_sum {duration_sum}",
            f"{_DIAGNOSE_DURATION}_count {buckets[-1]}",
            "# TYPE diagagent_index_chunks gauge",
            f"diagagent_index_chunks {health.chunk_count}",
            "# TYPE diagagent_index_ready gauge",
            f"diagagent_index_ready {int(health.status == 'ok')}",
        ]
        return "\n".join(lines) + "\n"

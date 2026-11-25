from pydantic import BaseModel, ConfigDict

from src.ingestion.schemas import Chunk


class SearchHit(BaseModel):
    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    chunk: Chunk
    score: float
    dense_score: float | None = None
    bm25_score: float | None = None
    rrf_score: float | None = None
    reranker_score: float | None = None

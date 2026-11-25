def resolve_retrieval_query(query: str, refined_query: str | None = None) -> str:
    """Use a nonblank refinement from the critic when it is available."""
    return (refined_query or "").strip() or query.strip()

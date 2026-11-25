from src.core.tokenizer import tokenize_codes


def tokenize_bm25(text: str, *, include_components: bool = False) -> list[str]:
    """Preserve equipment codes and add component terms when indexing documents."""
    return tokenize_codes(text, include_components=include_components)

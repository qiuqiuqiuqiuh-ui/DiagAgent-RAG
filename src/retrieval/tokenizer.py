from src.core.tokenizer import tokenize_codes


def tokenize_bm25(text: str, *, include_components: bool = False) -> list[str]:
    """Keep equipment codes intact and tokenize CJK runs into unigrams and bigrams."""
    return tokenize_codes(text, include_components=include_components)

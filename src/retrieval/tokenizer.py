import re

_TERM = re.compile(r"[^\W_]+(?:[-_./][^\W_]+)*")
_DELIMITER = re.compile(r"[-_./]")


def tokenize_bm25(text: str, *, include_components: bool = False) -> list[str]:
    """Preserve equipment codes and add component terms when indexing documents."""
    terms: list[str] = []
    for term in _TERM.findall(text.casefold()):
        terms.append(term)
        if include_components:
            components = _DELIMITER.split(term)
            if len(components) > 1:
                terms.extend(dict.fromkeys(components))
    return terms

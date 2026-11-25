import re

_TOKEN = re.compile(r"[^\W_]+(?:[-_][^\W_]+)*")
_CODE_TOKEN = re.compile(r"[^\W_]+(?:[-_./][^\W_]+)*")
_DELIMITER = re.compile(r"[-_./]")


def tokenize(text: str) -> list[str]:
    """Case-fold words while keeping hyphenated and underscored equipment codes intact."""
    return _TOKEN.findall(text.casefold())


def tokenize_codes(text: str, *, include_components: bool = False) -> list[str]:
    terms = _CODE_TOKEN.findall(text.casefold())
    if not include_components:
        return terms
    expanded: list[str] = []
    for term in terms:
        expanded.extend(dict.fromkeys([term, *_DELIMITER.split(term)]))
    return expanded


def code_fallback_terms(term: str) -> tuple[str, ...]:
    components = _DELIMITER.split(term)
    if len(components) < 2:
        return ()
    # A family prefix such as VFD alone cannot identify an OC3 fault.
    return tuple(part for part in dict.fromkeys(components) if any(char.isdigit() for char in part))

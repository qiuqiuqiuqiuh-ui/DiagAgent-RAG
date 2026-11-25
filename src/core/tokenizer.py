import re

_TOKEN = re.compile(r"[^\W_]+(?:[-_][^\W_]+)*")


def tokenize(text: str) -> list[str]:
    """Case-fold words while keeping hyphenated and underscored equipment codes intact."""
    return _TOKEN.findall(text.casefold())

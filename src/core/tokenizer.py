import re

_TOKEN = re.compile(r"[^\W_]+(?:[-_][^\W_]+)*")
_CODE_TOKEN = re.compile(r"[^\W_]+(?:[-_./][^\W_]+)*")
_DELIMITER = re.compile(r"[-_./]")
_CJK_RUN = re.compile(
    r"([\u1100-\u11ff\u3005\u3040-\u30ff\u3130-\u318f\u31f0-\u31ff"
    r"\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af\uf900-\ufaff\uff66-\uff9f"
    r"\U00020000-\U0002fa1f\U00030000-\U000323af]+)"
)


def tokenize(text: str) -> list[str]:
    """Case-fold words while keeping hyphenated and underscored equipment codes intact."""
    return _TOKEN.findall(text.casefold())


def split_cjk_runs(text: str) -> list[str]:
    """Return alternating non-CJK fragments and contiguous CJK runs."""
    return _CJK_RUN.split(text)


def tokenize_identifiers(text: str) -> list[str]:
    """Keep exact labels and separate adjacent CJK prose from industrial codes."""
    folded = text.casefold()
    terms = _CODE_TOKEN.findall(folded)
    parts = split_cjk_runs(folded)
    if len(parts) > 1:
        for fragment in parts[::2]:
            terms.extend(_CODE_TOKEN.findall(fragment))
    return list(dict.fromkeys(terms))


def tokenize_codes(text: str, *, include_components: bool = False) -> list[str]:
    terms: list[str] = []
    for index, run in enumerate(split_cjk_runs(text.casefold())):
        if index % 2 == 0:
            terms.extend(_CODE_TOKEN.findall(run))
        else:
            terms.extend(run)
            terms.extend(run[offset : offset + 2] for offset in range(len(run) - 1))
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

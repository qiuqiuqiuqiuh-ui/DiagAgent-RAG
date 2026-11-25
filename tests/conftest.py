import os

import pytest


@pytest.fixture(autouse=True)
def isolate_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in os.environ:
        if key.upper().startswith("DIAGAGENT_"):
            monkeypatch.delenv(key)

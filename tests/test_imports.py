"""The bot imports only what its venv on the PC has: the standard library, numpy, Pillow,
OpenCV (requirements.txt) and the local_game_mcp package (capture and input). 0.25-0.26 read
the map with scipy, which this container has and the PC does not, so both --full runs of
2026-10-29 crashed on the map after the boss [runs 20261029-120329 and -122345]."""

import ast
import sys
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent.parent / "shogunbot"
ALLOWED = {"numpy", "PIL", "cv2", "local_game_mcp"}


def _third_party_imports() -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for p in PACKAGE.rglob("*.py"):
        for node in ast.walk(ast.parse(p.read_text(encoding="utf-8"))):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            for n in names:
                top = n.split(".")[0]
                if top not in sys.stdlib_module_names and top != "shogunbot":
                    out.setdefault(top, set()).add(p.name)
    return out


def test_only_modules_the_pc_venv_has_are_imported():
    extra = {m: sorted(f) for m, f in _third_party_imports().items() if m not in ALLOWED}
    assert extra == {}

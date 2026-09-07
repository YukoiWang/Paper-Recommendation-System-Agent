"""read_fulltext tool (design §5). Cached files only; no live PDF scrape on the request path."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

_DEFAULT_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "fulltext"


def read_fulltext(paper_id: str, cache_dir: Optional[str] = None, max_chars: int = 40000) -> str:
    if not paper_id:
        return ""
    root = Path(cache_dir) if cache_dir else _DEFAULT_DIR
    for name in (f"{paper_id}.txt", f"{paper_id.replace('/', '_')}.txt"):
        path = root / name
        if path.exists():
            return path.read_text(encoding="utf-8", errors="ignore")[:max_chars]
    return ""

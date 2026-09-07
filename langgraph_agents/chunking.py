"""Full-text chunking helpers (design batch 7). Index into a side collection when texts exist."""
from __future__ import annotations

import re
from typing import Dict, List, Tuple

SECTION = re.compile(
    r"(?m)^(abstract|introduction|related work|method|methods|experiment|experiments|"
    r"conclusion|references|1[\s.]+introduction|2[\s.]+)\b",
    re.I,
)


def split_fulltext(text: str, title: str = "", max_tokens: int = 384, overlap: int = 64) -> List[Dict]:
    """Section-aware then sliding window. Token ≈ whitespace/char heuristic."""
    text = (text or "").strip()
    if not text:
        return []
    # drop references block
    text = re.split(r"(?im)^references\s*$", text)[0]
    parts = _by_section(text)
    chunks: List[Dict] = []
    for sec, body in parts:
        for i, window in enumerate(_windows(body, max_tokens, overlap)):
            prefix = f"{title}\nSection: {sec}\n" if title else f"Section: {sec}\n"
            chunks.append({
                "section": sec,
                "window": i,
                "text": (prefix + window).strip(),
            })
    return chunks


def _by_section(text: str) -> List[Tuple[str, str]]:
    idxs = [(m.start(), m.group()) for m in SECTION.finditer(text)]
    if not idxs:
        return [("body", text)]
    out = []
    for i, (start, name) in enumerate(idxs):
        end = idxs[i + 1][0] if i + 1 < len(idxs) else len(text)
        body = text[start:end].strip()
        if len(body) < 40:
            continue
        out.append((name.strip()[:40], body))
    return out or [("body", text)]


def _windows(text: str, max_tokens: int, overlap: int) -> List[str]:
    # approx token = words for latin, chars/2 for CJK
    toks = re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]", text)
    if not toks:
        return [text[: max_tokens * 4]]
    step = max(1, max_tokens - overlap)
    out = []
    for i in range(0, len(toks), step):
        piece = toks[i : i + max_tokens]
        if not piece:
            break
        # naive rejoin
        s = " ".join(piece) if any(c.isascii() for c in piece[:5]) else "".join(piece)
        out.append(s)
        if i + max_tokens >= len(toks):
            break
    return out

# -*- coding: utf-8 -*-
"""
本地缓存：引用列表 + 论文元数据。
供 prefetch 脚本写入，create_listwise2 读取，避免实时 API 调用。
"""
import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional


class PaperCache:
    """SQLite 缓存：refs 表存引用列表，meta 表存论文元数据。"""

    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.cache_dir / "paper_cache.db"
        self._conn: Optional[sqlite3.Connection] = None

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS refs (paper_id TEXT PRIMARY KEY, ref_ids TEXT)"
            )
            self._conn.execute(
                """CREATE TABLE IF NOT EXISTS meta (
                    paper_id TEXT PRIMARY KEY,
                    title TEXT,
                    abstract TEXT,
                    authors TEXT
                )"""
            )
            self._conn.commit()
        return self._conn

    def get_refs(self, paper_id: str) -> Optional[List[str]]:
        """获取论文的引用列表。"""
        row = self._get_conn().execute(
            "SELECT ref_ids FROM refs WHERE paper_id = ?", (paper_id,)
        ).fetchone()
        if row and row[0]:
            try:
                return json.loads(row[0])
            except json.JSONDecodeError:
                return None
        return None

    def set_refs(self, paper_id: str, ref_ids: List[str]) -> None:
        """写入引用列表。"""
        self._get_conn().execute(
            "INSERT OR REPLACE INTO refs (paper_id, ref_ids) VALUES (?, ?)",
            (paper_id, json.dumps(ref_ids)),
        )
        self._get_conn().commit()

    def has_refs(self, paper_id: str) -> bool:
        return self.get_refs(paper_id) is not None

    def get_meta(self, paper_id: str) -> Optional[Dict[str, Any]]:
        """获取论文元数据 {paper_id, title, abstract, authors}。"""
        row = self._get_conn().execute(
            "SELECT title, abstract, authors FROM meta WHERE paper_id = ?",
            (paper_id,),
        ).fetchone()
        if row:
            try:
                authors = json.loads(row[2]) if row[2] else []
                return {
                    "paper_id": paper_id,
                    "title": row[0] or "",
                    "abstract": row[1] or "",
                    "authors": authors,
                }
            except (json.JSONDecodeError, TypeError):
                return None
        return None

    def set_meta(self, paper_id: str, meta: Dict[str, Any]) -> None:
        """写入论文元数据。"""
        authors = meta.get("authors", [])
        if isinstance(authors, list):
            authors_json = json.dumps(authors)
        else:
            authors_json = json.dumps([])
        self._get_conn().execute(
            "INSERT OR REPLACE INTO meta (paper_id, title, abstract, authors) VALUES (?, ?, ?, ?)",
            (
                paper_id,
                meta.get("title", ""),
                meta.get("abstract", ""),
                authors_json,
            ),
        )
        self._get_conn().commit()

    def has_meta(self, paper_id: str) -> bool:
        return self.get_meta(paper_id) is not None

    def refs_count(self) -> int:
        return self._get_conn().execute("SELECT COUNT(*) FROM refs").fetchone()[0]

    def meta_count(self) -> int:
        return self._get_conn().execute("SELECT COUNT(*) FROM meta").fetchone()[0]

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

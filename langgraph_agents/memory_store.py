"""SQLite + JSONL memory: queries, messages, episodes, sessions, exposures (design §11)."""
from __future__ import annotations

import json
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from langgraph_agents.policy import POLICY_VERSIONS, THRESHOLDS

_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT = _ROOT / "data" / "runtime_memory.sqlite"
_EPISODE_JSONL = _ROOT / "logs" / "episodes.jsonl"


def _connect(path: Optional[str] = None) -> sqlite3.Connection:
    p = Path(path or _DEFAULT)
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(p))
    conn.execute(
        """CREATE TABLE IF NOT EXISTS conversations (
            conversation_id TEXT PRIMARY KEY, user_id TEXT, updated_at REAL,
            session_blob TEXT)"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT, role TEXT, content TEXT, episode_id TEXT, ts REAL)"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS queries (
            query_id TEXT PRIMARY KEY, user_id TEXT, conversation_id TEXT, episode_id TEXT,
            raw_query TEXT, topic TEXT, intent TEXT, created_at REAL)"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS episodes (
            episode_id TEXT PRIMARY KEY, query_id TEXT, intent TEXT, intent_source TEXT,
            next_agent TEXT, tool_calls TEXT, failure_type TEXT, recovery TEXT,
            critic TEXT, latency_ms REAL, created_at REAL, policy_versions TEXT)"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS exposures (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT, paper_id TEXT, ts REAL)"""
    )
    try:
        conn.execute("ALTER TABLE conversations ADD COLUMN session_blob TEXT")
    except sqlite3.OperationalError:
        pass
    try:
        conn.execute("ALTER TABLE episodes ADD COLUMN policy_versions TEXT")
    except sqlite3.OperationalError:
        pass
    conn.commit()
    return conn


class MemoryStore:
    def __init__(self, path: Optional[str] = None, jsonl_path: Optional[str] = None):
        self.path = str(path or _DEFAULT)
        self.jsonl_path = Path(jsonl_path) if jsonl_path else _EPISODE_JSONL

    def persist_turn(self, state: Dict[str, Any], latency_ms: float = 0.0) -> str:
        cid = state.get("conversation_id") or "default"
        uid = state.get("user_id") or "anonymous"
        eid = state.get("episode_id") or uuid.uuid4().hex[:16]
        qid = uuid.uuid4().hex[:16]
        wo = state.get("work_order") or {}
        now = time.time()
        versions = state.get("policy_versions") or dict(POLICY_VERSIONS)
        versions = {**versions, "playbook": wo.get("playbook_id") or versions.get("playbook")}
        tools = state.get("researcher_trace") or []
        conn = _connect(self.path)
        try:
            conn.execute(
                """INSERT INTO conversations(conversation_id, user_id, updated_at)
                   VALUES (?,?,?)
                   ON CONFLICT(conversation_id) DO UPDATE SET
                     user_id=excluded.user_id, updated_at=excluded.updated_at""",
                (cid, uid, now),
            )
            raw_q = state.get("user_query") or ""
            conn.execute(
                "INSERT INTO queries VALUES (?,?,?,?,?,?,?,?)",
                (qid, uid, cid, eid, raw_q, wo.get("topic") or "", wo.get("intent") or "", now),
            )
            for msg in (state.get("history") or [])[-2:]:
                conn.execute(
                    "INSERT INTO messages (conversation_id, role, content, episode_id, ts) VALUES (?,?,?,?,?)",
                    (cid, msg.get("role"), (msg.get("content") or "")[:4000], eid, now),
                )
            conn.execute(
                "INSERT OR REPLACE INTO episodes VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    eid, qid, wo.get("intent") or "", wo.get("intent_source") or "",
                    wo.get("next_agent") or "", json.dumps(tools, ensure_ascii=False),
                    state.get("failure_type") or "none", state.get("recovery") or "",
                    state.get("critic_decision") or "", float(latency_ms), now,
                    json.dumps(versions, ensure_ascii=False),
                ),
            )
            conn.commit()
        finally:
            conn.close()
        state["episode_id"] = eid
        self._append_jsonl({
            "episode_id": eid,
            "query_id": qid,
            "user_id": uid,
            "conversation_id": cid,
            "ts": now,
            "intent": wo.get("intent"),
            "intent_source": wo.get("intent_source"),
            "intent_pred_confidence": wo.get("confidence"),
            "next_agent": wo.get("next_agent"),
            "slots": wo.get("slots"),
            "policy_versions": versions,
            "tool_calls": tools,
            "failure_type": state.get("failure_type") or "none",
            "recovery": state.get("recovery") or "",
            "critic": state.get("critic_decision") or "",
            "latency_ms": latency_ms,
        })
        return eid

    def _append_jsonl(self, rec: Dict[str, Any]) -> None:
        try:
            self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            with self.jsonl_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        except Exception:
            pass

    def recent_episodes(self, limit: int = 200) -> List[Dict[str, Any]]:
        conn = _connect(self.path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM episodes ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def record_exposures(self, user_id: str, papers: Sequence[Any]) -> None:
        now = time.time()
        conn = _connect(self.path)
        try:
            for p in papers:
                pid = getattr(p, "paper_id", None)
                if not pid:
                    continue
                conn.execute(
                    "INSERT INTO exposures(user_id, paper_id, ts) VALUES (?,?,?)",
                    (user_id, pid, now),
                )
            conn.commit()
        finally:
            conn.close()

    def recent_exposures(self, user_id: str, days: Optional[int] = None) -> List[str]:
        days = int(days if days is not None else THRESHOLDS.get("exposure_days", 90))
        cutoff = time.time() - days * 86400
        conn = _connect(self.path)
        rows = conn.execute(
            "SELECT DISTINCT paper_id FROM exposures WHERE user_id=? AND ts>=?",
            (user_id, cutoff),
        ).fetchall()
        conn.close()
        return [r[0] for r in rows]

    def save_session(self, conversation_id: str, user_id: str, blob: Dict[str, Any]) -> None:
        conn = _connect(self.path)
        conn.execute(
            "INSERT OR REPLACE INTO conversations(conversation_id, user_id, updated_at, session_blob) VALUES (?,?,?,?)",
            (conversation_id, user_id, time.time(), json.dumps(blob, ensure_ascii=False, default=str)),
        )
        conn.commit()
        conn.close()

    def load_session(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        conn = _connect(self.path)
        row = conn.execute(
            "SELECT session_blob FROM conversations WHERE conversation_id=?",
            (conversation_id,),
        ).fetchone()
        conn.close()
        if not row or not row[0]:
            return None
        try:
            return json.loads(row[0])
        except json.JSONDecodeError:
            return None

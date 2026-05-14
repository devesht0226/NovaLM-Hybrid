from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any


class GenerationDB:
    def __init__(self, db_path: str = "artifacts/db/novalm_hybrid.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS generations (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        mode TEXT NOT NULL,
                        prompt TEXT NOT NULL,
                        generated_text TEXT NOT NULL,
                        contexts_json TEXT NOT NULL,
                        latency_ms REAL NOT NULL,
                        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                    )
                    """
                )
                conn.execute("CREATE INDEX IF NOT EXISTS idx_generations_created_at ON generations(created_at DESC)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_generations_mode ON generations(mode)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_generations_prompt ON generations(prompt)")
                conn.commit()

    def insert_generation(
        self,
        mode: str,
        prompt: str,
        generated_text: str,
        contexts: list[dict[str, Any]],
        latency_ms: float,
    ) -> int:
        with self._lock:
            with self._connect() as conn:
                cur = conn.execute(
                    """
                    INSERT INTO generations (mode, prompt, generated_text, contexts_json, latency_ms)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (mode, prompt, generated_text, json.dumps(contexts), float(latency_ms)),
                )
                conn.commit()
                return int(cur.lastrowid)

    def get_history(
        self,
        limit: int = 10,
        offset: int = 0,
        mode: str | None = None,
        query: str | None = None,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if mode and mode in {"pure", "hybrid"}:
            clauses.append("mode = ?")
            params.append(mode)
        if query:
            clauses.append("(prompt LIKE ? OR generated_text LIKE ?)")
            like = f"%{query}%"
            params.extend([like, like])

        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._lock:
            with self._connect() as conn:
                rows = conn.execute(
                    f"""
                    SELECT id, mode, prompt, generated_text, contexts_json, latency_ms, created_at
                    FROM generations
                    {where_sql}
                    ORDER BY id DESC
                    LIMIT ?
                    OFFSET ?
                    """,
                    (*params, limit, offset),
                ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def count_history(self, mode: str | None = None, query: str | None = None) -> int:
        clauses: list[str] = []
        params: list[Any] = []
        if mode and mode in {"pure", "hybrid"}:
            clauses.append("mode = ?")
            params.append(mode)
        if query:
            clauses.append("(prompt LIKE ? OR generated_text LIKE ?)")
            like = f"%{query}%"
            params.extend([like, like])
        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._lock:
            with self._connect() as conn:
                row = conn.execute(
                    f"""
                    SELECT COUNT(*) as c
                    FROM generations
                    {where_sql}
                    """,
                    params,
                ).fetchone()
        return int(row["c"]) if row is not None else 0

    def get_by_id(self, record_id: int) -> dict[str, Any] | None:
        with self._lock:
            with self._connect() as conn:
                row = conn.execute(
                    """
                    SELECT id, mode, prompt, generated_text, contexts_json, latency_ms, created_at
                    FROM generations
                    WHERE id = ?
                    """,
                    (record_id,),
                ).fetchone()
        if row is None:
            return None
        return self._row_to_dict(row)

    def delete_by_id(self, record_id: int) -> bool:
        with self._lock:
            with self._connect() as conn:
                cur = conn.execute("DELETE FROM generations WHERE id = ?", (record_id,))
                conn.commit()
                return cur.rowcount > 0

    def delete_all(self) -> int:
        with self._lock:
            with self._connect() as conn:
                cur = conn.execute("DELETE FROM generations")
                conn.commit()
                return int(cur.rowcount)

    def export_rows(self, mode: str | None = None, query: str | None = None) -> list[dict[str, Any]]:
        return self.get_history(limit=100000, offset=0, mode=mode, query=query)

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        try:
            parsed_contexts = json.loads(row["contexts_json"] or "[]")
        except Exception:
            parsed_contexts = []
        if not isinstance(parsed_contexts, list):
            parsed_contexts = []
        return {
            "id": int(row["id"]),
            "mode": row["mode"],
            "prompt": row["prompt"],
            "generated_text": row["generated_text"],
            "contexts": parsed_contexts,
            "latency_ms": float(row["latency_ms"]),
            "created_at": row["created_at"],
        }

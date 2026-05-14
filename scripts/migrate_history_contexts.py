from __future__ import annotations

import json
import sqlite3
from pathlib import Path


DB_PATH = Path("artifacts/db/novalm_hybrid.db")


def normalize_contexts(raw: str | None) -> str:
    if not raw:
        return "[]"
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return json.dumps(parsed, ensure_ascii=False)
        return "[]"
    except Exception:
        return "[]"


def main() -> None:
    if not DB_PATH.exists():
        print(f"DB not found at {DB_PATH}, skipping.")
        return

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    rows = cur.execute("SELECT id, contexts_json FROM generations").fetchall()

    updated = 0
    for row in rows:
        normalized = normalize_contexts(row["contexts_json"])
        if normalized != (row["contexts_json"] or ""):
            cur.execute("UPDATE generations SET contexts_json = ? WHERE id = ?", (normalized, row["id"]))
            updated += 1

    conn.commit()
    conn.close()
    print(f"Normalized contexts_json rows: {updated}")


if __name__ == "__main__":
    main()

"""
SQLite table memory: persist table metadata (table name, description, columns, column descriptions).
"""
import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional


class SQLiteTableMemory:
    """
    Store and retrieve table metadata in a SQLite table.
    Each record is a dict: table_name, description, columns (list), column_descriptions (dict).
    """

    def __init__(self, db_path: Optional[str] = None):
        self._path = Path(db_path or ".memory/table_memory.sqlite")
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._init_table()

    def _init_table(self) -> None:
        with sqlite3.connect(self._path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS table_memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    table_name TEXT NOT NULL,
                    description TEXT,
                    columns TEXT,
                    column_descriptions TEXT,
                    created_at DEFAULT CURRENT_TIMESTAMP
                )
            """)

    def _get_conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self._path)

    def save(
        self,
        table_name: str,
        description: str = "",
        columns: Optional[List[str]] = None,
        column_descriptions: Optional[Dict[str, str]] = None,
    ) -> None:
        """Save one table metadata record. columns and column_descriptions default to [] and {}."""
        columns = columns or []
        column_descriptions = column_descriptions or {}
        with self._get_conn() as conn:
            conn.execute(
                """
                INSERT INTO table_memory (table_name, description, columns, column_descriptions)
                VALUES (?, ?, ?, ?)
                """,
                (
                    table_name,
                    description,
                    json.dumps(columns),
                    json.dumps(column_descriptions),
                ),
            )

    def save_from_dict(self, data: Dict[str, Any]) -> None:
        """Save from a single dict with keys: table_name, description, columns, column_descriptions."""
        self.save(
            table_name=data.get("table_name", ""),
            description=data.get("description", ""),
            columns=data.get("columns") or [],
            column_descriptions=data.get("column_descriptions") or {},
        )

    def get(self) -> List[Dict[str, Any]]:
        """Return all stored table metadata as list of dicts."""
        with self._get_conn() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT table_name, description, columns, column_descriptions FROM table_memory ORDER BY id"
            ).fetchall()
        return [
            {
                "table_name": r["table_name"],
                "description": r["description"] or "",
                "columns": json.loads(r["columns"] or "[]"),
                "column_descriptions": json.loads(r["column_descriptions"] or "{}"),
            }
            for r in rows
        ]

    def get_latest(self) -> Optional[Dict[str, Any]]:
        """Return the most recently saved table metadata, or None."""
        all_ = self.get()
        return all_[-1] if all_ else None

    def clear(self) -> None:
        """Remove all table metadata records."""
        with self._get_conn() as conn:
            conn.execute("DELETE FROM table_memory")

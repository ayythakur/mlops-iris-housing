# src/utils/audit.py
import datetime
import json
import os
import sqlite3
from typing import Iterable, Optional

DB_PATH = "logs/predictions.db"
os.makedirs("logs", exist_ok=True)


def _init() -> None:
    """Create the SQLite DB and table if they don't exist."""
    with sqlite3.connect(DB_PATH) as con:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              ts TEXT,
              features TEXT,
              prediction INTEGER,
              probabilities TEXT
            )
            """
        )


# Ensure DB/table exists at import time (idempotent).
_init()


def log_prediction(
    features: Iterable[float],
    prediction: int,
    probabilities: Optional[Iterable[float]],
) -> None:
    """Append one prediction record to SQLite audit log."""
    with sqlite3.connect(DB_PATH) as con:
        con.execute(
            "INSERT INTO predictions (ts, features, prediction, probabilities) "
            "VALUES (?, ?, ?, ?)",
            (
                datetime.datetime.utcnow().isoformat(),
                json.dumps(list(features)),
                int(prediction),
                json.dumps(list(probabilities) if probabilities is not None else None),
            ),
        )

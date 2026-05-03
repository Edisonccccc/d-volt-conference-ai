"""Tiny SQLite-backed store for card records.

Keeps the prototype dependency-light. Swap for Postgres later.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from .models import (
    CardRecord,
    CardStatus,
    CompanyResearch,
    ConversationRecord,
    ConversationStatus,
    ConversationSummary,
    ExtractedCard,
)


DATA_DIR = Path(os.getenv("DATA_DIR", "./data")).resolve()
DB_PATH = Path(os.getenv("DB_PATH", str(DATA_DIR / "app.db"))).resolve()
UPLOAD_DIR = DATA_DIR / "uploads"
REPORT_DIR = DATA_DIR / "reports"

_lock = threading.Lock()


def _ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)


def _connect() -> sqlite3.Connection:
    _ensure_dirs()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _lock, _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cards (
                id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                photo_path TEXT NOT NULL,
                extracted_json TEXT,
                research_json TEXT,
                error TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                started_at TEXT NOT NULL,
                ended_at TEXT,
                card_id TEXT,
                audio_path TEXT,
                transcript TEXT,
                summary_json TEXT,
                error TEXT
            )
            """
        )
        # Migration: add audio_path to existing dbs that predate slice 2.5.
        existing_cols = {
            r["name"]
            for r in conn.execute("PRAGMA table_info(conversations)").fetchall()
        }
        if "audio_path" not in existing_cols:
            conn.execute("ALTER TABLE conversations ADD COLUMN audio_path TEXT")
        conn.commit()


def upload_dir() -> Path:
    _ensure_dirs()
    return UPLOAD_DIR


def report_dir() -> Path:
    _ensure_dirs()
    return REPORT_DIR


def _row_to_record(row: sqlite3.Row) -> CardRecord:
    return CardRecord(
        id=row["id"],
        status=row["status"],
        created_at=row["created_at"],
        photo_path=row["photo_path"],
        extracted=ExtractedCard.model_validate_json(row["extracted_json"])
        if row["extracted_json"]
        else None,
        research=CompanyResearch.model_validate_json(row["research_json"])
        if row["research_json"]
        else None,
        error=row["error"],
    )


def create_card(card_id: str, photo_path: str) -> CardRecord:
    now = datetime.now(timezone.utc).isoformat()
    with _lock, _connect() as conn:
        conn.execute(
            "INSERT INTO cards (id, status, created_at, photo_path) VALUES (?, ?, ?, ?)",
            (card_id, "pending", now, photo_path),
        )
        conn.commit()
    return CardRecord(
        id=card_id, status="pending", created_at=now, photo_path=photo_path
    )


def update_status(card_id: str, status: CardStatus, error: Optional[str] = None) -> None:
    with _lock, _connect() as conn:
        conn.execute(
            "UPDATE cards SET status = ?, error = ? WHERE id = ?",
            (status, error, card_id),
        )
        conn.commit()


def update_extracted(card_id: str, extracted: ExtractedCard) -> None:
    with _lock, _connect() as conn:
        conn.execute(
            "UPDATE cards SET extracted_json = ? WHERE id = ?",
            (extracted.model_dump_json(), card_id),
        )
        conn.commit()


def update_research(card_id: str, research: CompanyResearch) -> None:
    with _lock, _connect() as conn:
        conn.execute(
            "UPDATE cards SET research_json = ? WHERE id = ?",
            (research.model_dump_json(), card_id),
        )
        conn.commit()


def get_card(card_id: str) -> Optional[CardRecord]:
    with _lock, _connect() as conn:
        row = conn.execute("SELECT * FROM cards WHERE id = ?", (card_id,)).fetchone()
    return _row_to_record(row) if row else None


def list_cards(limit: int = 50) -> List[CardRecord]:
    with _lock, _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM cards ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
    return [_row_to_record(r) for r in rows]


# ---------------------------------------------------------------------------
# Conversation CRUD
# ---------------------------------------------------------------------------


def _row_to_conversation(row: sqlite3.Row) -> ConversationRecord:
    # row["audio_path"] is fetched safely even on freshly-migrated old rows
    # because sqlite3.Row supports keys() lookup.
    audio_path = row["audio_path"] if "audio_path" in row.keys() else None
    return ConversationRecord(
        id=row["id"],
        status=row["status"],
        started_at=row["started_at"],
        ended_at=row["ended_at"],
        card_id=row["card_id"],
        audio_path=audio_path,
        transcript=row["transcript"],
        summary=ConversationSummary.model_validate_json(row["summary_json"])
        if row["summary_json"]
        else None,
        error=row["error"],
    )


def create_conversation(
    conv_id: str, card_id: Optional[str] = None
) -> ConversationRecord:
    now = datetime.now(timezone.utc).isoformat()
    with _lock, _connect() as conn:
        conn.execute(
            "INSERT INTO conversations (id, status, started_at, card_id) "
            "VALUES (?, ?, ?, ?)",
            (conv_id, "recording", now, card_id),
        )
        conn.commit()
    return ConversationRecord(
        id=conv_id, status="recording", started_at=now, card_id=card_id
    )


def update_conversation_status(
    conv_id: str, status: ConversationStatus, error: Optional[str] = None
) -> None:
    with _lock, _connect() as conn:
        conn.execute(
            "UPDATE conversations SET status = ?, error = ? WHERE id = ?",
            (status, error, conv_id),
        )
        conn.commit()


def update_conversation_audio(conv_id: str, audio_path: str) -> None:
    with _lock, _connect() as conn:
        conn.execute(
            "UPDATE conversations SET audio_path = ? WHERE id = ?",
            (audio_path, conv_id),
        )
        conn.commit()


def update_conversation_transcript(conv_id: str, transcript: str) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with _lock, _connect() as conn:
        conn.execute(
            "UPDATE conversations SET transcript = ?, ended_at = ? WHERE id = ?",
            (transcript, now, conv_id),
        )
        conn.commit()


def update_conversation_summary(conv_id: str, summary: ConversationSummary) -> None:
    with _lock, _connect() as conn:
        conn.execute(
            "UPDATE conversations SET summary_json = ? WHERE id = ?",
            (summary.model_dump_json(), conv_id),
        )
        conn.commit()


def get_conversation(conv_id: str) -> Optional[ConversationRecord]:
    with _lock, _connect() as conn:
        row = conn.execute(
            "SELECT * FROM conversations WHERE id = ?", (conv_id,)
        ).fetchone()
    return _row_to_conversation(row) if row else None


def list_conversations(limit: int = 50) -> List[ConversationRecord]:
    with _lock, _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM conversations ORDER BY started_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [_row_to_conversation(r) for r in rows]

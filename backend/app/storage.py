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

import uuid

from .models import (
    CardRecord,
    CardStatus,
    CompanyResearch,
    ConversationRecord,
    ConversationStatus,
    ConversationSummary,
    ExtractedCard,
    User,
    UserRole,
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
                error TEXT,
                user_id TEXT
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
                error TEXT,
                user_id TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                email TEXT UNIQUE NOT NULL COLLATE NOCASE,
                password_hash TEXT NOT NULL,
                name TEXT,
                role TEXT NOT NULL DEFAULT 'rep',
                created_at TEXT NOT NULL,
                last_login TEXT
            )
            """
        )

        # ---- migrations: add columns to older DBs ----
        def _has_col(table: str, col: str) -> bool:
            return col in {
                r["name"]
                for r in conn.execute(f"PRAGMA table_info({table})").fetchall()
            }

        if not _has_col("conversations", "audio_path"):
            conn.execute("ALTER TABLE conversations ADD COLUMN audio_path TEXT")
        if not _has_col("cards", "user_id"):
            conn.execute("ALTER TABLE cards ADD COLUMN user_id TEXT")
        if not _has_col("conversations", "user_id"):
            conn.execute("ALTER TABLE conversations ADD COLUMN user_id TEXT")
        if not _has_col("cards", "cost_usd"):
            conn.execute("ALTER TABLE cards ADD COLUMN cost_usd REAL")
        if not _has_col("conversations", "cost_usd"):
            conn.execute("ALTER TABLE conversations ADD COLUMN cost_usd REAL")
        # cost_stt = the Whisper portion of the total. cost_usd - cost_stt
        # = the LLM portion (Claude). Cards never have STT so it's always 0.
        if not _has_col("cards", "cost_stt"):
            conn.execute("ALTER TABLE cards ADD COLUMN cost_stt REAL")
        if not _has_col("conversations", "cost_stt"):
            conn.execute("ALTER TABLE conversations ADD COLUMN cost_stt REAL")
        # Multi-tenant prep: each user belongs to a company. Existing rows
        # default to 'd-volt' for backwards compatibility.
        if not _has_col("users", "company"):
            conn.execute("ALTER TABLE users ADD COLUMN company TEXT")
            conn.execute("UPDATE users SET company = 'd-volt' WHERE company IS NULL")
        # `must_change_password` flag — set when an admin resets someone's
        # password; the user is forced to pick a new one on next login.
        if not _has_col("users", "must_change_password"):
            conn.execute(
                "ALTER TABLE users ADD COLUMN must_change_password INTEGER NOT NULL DEFAULT 0"
            )

        conn.execute("CREATE INDEX IF NOT EXISTS idx_cards_user ON cards(user_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_conv_user  ON conversations(user_id)")

        # One-time wipe of legacy test data (rows with no owner) — but only
        # at true bootstrap, before any user has registered. Once a user
        # exists this becomes a no-op forever, so any future bug that
        # accidentally creates a NULL-user_id row won't be silently deleted.
        # KEEP_LEGACY_DATA=1 still bypasses the wipe entirely if you'd
        # rather assign-then-keep at first deploy.
        user_count_row = conn.execute("SELECT COUNT(*) FROM users").fetchone()
        is_bootstrap = (user_count_row[0] if user_count_row else 0) == 0
        if is_bootstrap and os.getenv("KEEP_LEGACY_DATA") not in ("1", "true", "yes"):
            conn.execute("DELETE FROM cards         WHERE user_id IS NULL")
            conn.execute("DELETE FROM conversations WHERE user_id IS NULL")

        conn.commit()


def upload_dir() -> Path:
    _ensure_dirs()
    return UPLOAD_DIR


def report_dir() -> Path:
    _ensure_dirs()
    return REPORT_DIR


def _row_to_record(row: sqlite3.Row) -> CardRecord:
    keys = row.keys()
    user_id  = row["user_id"]  if "user_id"  in keys else None
    cost_usd = row["cost_usd"] if "cost_usd" in keys else None
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
        user_id=user_id,
        cost_usd=cost_usd,
    )


def create_card(card_id: str, photo_path: str, user_id: Optional[str] = None) -> CardRecord:
    now = datetime.now(timezone.utc).isoformat()
    with _lock, _connect() as conn:
        conn.execute(
            "INSERT INTO cards (id, status, created_at, photo_path, user_id) "
            "VALUES (?, ?, ?, ?, ?)",
            (card_id, "pending", now, photo_path, user_id),
        )
        conn.commit()
    return CardRecord(
        id=card_id, status="pending", created_at=now, photo_path=photo_path,
        user_id=user_id,
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


def add_card_cost(card_id: str, delta_usd: float, *, stt_delta: float = 0.0) -> None:
    """Increment a card's cost_usd by delta_usd and cost_stt by stt_delta."""
    with _lock, _connect() as conn:
        conn.execute(
            "UPDATE cards SET cost_usd = COALESCE(cost_usd, 0) + ?, "
            "cost_stt = COALESCE(cost_stt, 0) + ? WHERE id = ?",
            (float(delta_usd or 0), float(stt_delta or 0), card_id),
        )
        conn.commit()


def add_conversation_cost(
    conv_id: str, delta_usd: float, *, stt_delta: float = 0.0,
) -> None:
    with _lock, _connect() as conn:
        conn.execute(
            "UPDATE conversations SET cost_usd = COALESCE(cost_usd, 0) + ?, "
            "cost_stt = COALESCE(cost_stt, 0) + ? WHERE id = ?",
            (float(delta_usd or 0), float(stt_delta or 0), conv_id),
        )
        conn.commit()


def get_card(card_id: str, user_id: Optional[str] = None) -> Optional[CardRecord]:
    """Fetch one card. If user_id is provided, the card must belong to that
    user. Pass user_id=None for unscoped access (used by manager/admin).
    """
    with _lock, _connect() as conn:
        if user_id is None:
            row = conn.execute("SELECT * FROM cards WHERE id = ?", (card_id,)).fetchone()
        else:
            row = conn.execute(
                "SELECT * FROM cards WHERE id = ? AND user_id = ?",
                (card_id, user_id),
            ).fetchone()
    return _row_to_record(row) if row else None


def list_cards(limit: int = 50, user_id: Optional[str] = None) -> List[CardRecord]:
    with _lock, _connect() as conn:
        if user_id is None:
            rows = conn.execute(
                "SELECT * FROM cards ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM cards WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
                (user_id, limit),
            ).fetchall()
    return [_row_to_record(r) for r in rows]


# ---------------------------------------------------------------------------
# Conversation CRUD
# ---------------------------------------------------------------------------


def _row_to_conversation(row: sqlite3.Row) -> ConversationRecord:
    keys = row.keys()
    audio_path = row["audio_path"] if "audio_path" in keys else None
    user_id    = row["user_id"]    if "user_id"    in keys else None
    cost_usd   = row["cost_usd"]   if "cost_usd"   in keys else None
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
        user_id=user_id,
        cost_usd=cost_usd,
    )


def create_conversation(
    conv_id: str, card_id: Optional[str] = None, user_id: Optional[str] = None,
) -> ConversationRecord:
    now = datetime.now(timezone.utc).isoformat()
    with _lock, _connect() as conn:
        conn.execute(
            "INSERT INTO conversations (id, status, started_at, card_id, user_id) "
            "VALUES (?, ?, ?, ?, ?)",
            (conv_id, "recording", now, card_id, user_id),
        )
        conn.commit()
    return ConversationRecord(
        id=conv_id, status="recording", started_at=now, card_id=card_id,
        user_id=user_id,
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


def get_conversation(
    conv_id: str, user_id: Optional[str] = None
) -> Optional[ConversationRecord]:
    with _lock, _connect() as conn:
        if user_id is None:
            row = conn.execute(
                "SELECT * FROM conversations WHERE id = ?", (conv_id,)
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT * FROM conversations WHERE id = ? AND user_id = ?",
                (conv_id, user_id),
            ).fetchone()
    return _row_to_conversation(row) if row else None


def list_conversations(
    limit: int = 50, user_id: Optional[str] = None
) -> List[ConversationRecord]:
    with _lock, _connect() as conn:
        if user_id is None:
            rows = conn.execute(
                "SELECT * FROM conversations ORDER BY started_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM conversations WHERE user_id = ? ORDER BY started_at DESC LIMIT ?",
                (user_id, limit),
            ).fetchall()
    return [_row_to_conversation(r) for r in rows]


# ---------------------------------------------------------------------------
# User CRUD
# ---------------------------------------------------------------------------


def _row_to_user(row: sqlite3.Row) -> User:
    keys = row.keys()
    company = row["company"] if "company" in keys else None
    must_change = bool(row["must_change_password"]) if "must_change_password" in keys else False
    return User(
        id=row["id"],
        email=row["email"],
        name=row["name"],
        role=row["role"],
        company=company,
        created_at=row["created_at"],
        last_login=row["last_login"],
        must_change_password=must_change,
    )


def create_user(
    email: str,
    password_hash: str,
    name: Optional[str] = None,
    role: UserRole = "rep",
    company: Optional[str] = None,
) -> User:
    user_id = uuid.uuid4().hex
    now = datetime.now(timezone.utc).isoformat()
    with _lock, _connect() as conn:
        conn.execute(
            "INSERT INTO users (id, email, password_hash, name, role, "
            "created_at, company) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (user_id, email.lower().strip(), password_hash, name, role, now,
             (company or "").strip() or None),
        )
        conn.commit()
    return User(
        id=user_id, email=email, name=name, role=role,
        company=company, created_at=now, last_login=None,
    )


def get_user_by_id(user_id: str) -> Optional[User]:
    with _lock, _connect() as conn:
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    return _row_to_user(row) if row else None


def get_user_by_email_with_hash(email: str):
    """Return (user, password_hash) for login verification, or None.

    The hash isn't on the User Pydantic model (so it can't accidentally
    leak through API serialization), so we expose it via this auth-only
    helper.
    """
    with _lock, _connect() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE email = ?", (email.lower().strip(),),
        ).fetchone()
    if row is None:
        return None
    return _row_to_user(row), row["password_hash"]


def update_last_login(user_id: str) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with _lock, _connect() as conn:
        conn.execute(
            "UPDATE users SET last_login = ? WHERE id = ?", (now, user_id),
        )
        conn.commit()


def update_user_password(
    user_id: str, password_hash: str, *, must_change: bool = False,
) -> None:
    """Set a user's password hash and the must_change_password flag."""
    with _lock, _connect() as conn:
        conn.execute(
            "UPDATE users SET password_hash = ?, must_change_password = ? "
            "WHERE id = ?",
            (password_hash, 1 if must_change else 0, user_id),
        )
        conn.commit()


def list_users(limit: int = 100) -> List[User]:
    with _lock, _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM users ORDER BY created_at ASC LIMIT ?", (limit,),
        ).fetchall()
    return [_row_to_user(r) for r in rows]


def delete_user(user_id: str) -> bool:
    with _lock, _connect() as conn:
        cur = conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
        conn.commit()
        return cur.rowcount > 0


def count_users() -> int:
    with _lock, _connect() as conn:
        return conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]


# ---------------------------------------------------------------------------
# Admin stats — team & per-user activity + spend
# ---------------------------------------------------------------------------


def _period_start_iso(period: str) -> Optional[str]:
    """Return an ISO timestamp marking the start of the given period (UTC),
    or None for 'all'. Periods accepted: today | week | month | all."""
    p = (period or "week").lower()
    now = datetime.now(timezone.utc)
    if p == "all":
        return None
    if p == "today":
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    elif p == "month":
        start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    else:
        # week (default) — Monday 00:00 UTC of the current week
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        start = start.fromordinal(start.toordinal() - start.weekday())
        start = datetime(start.year, start.month, start.day, tzinfo=timezone.utc)
    return start.isoformat()


def compute_team_stats(period: str = "week") -> dict:
    """Return team-wide totals + per-user breakdown for the given period.

    Periods: today | week (default) | month | all.
    """
    from_iso = _period_start_iso(period)
    with _lock, _connect() as conn:
        # ---- per-user roll-up of cards ----
        if from_iso:
            cards_rows = conn.execute(
                "SELECT user_id, COUNT(*) AS n, "
                "  COALESCE(SUM(cost_usd), 0) AS spend, "
                "  COALESCE(SUM(cost_stt), 0) AS spend_stt, "
                "  MAX(created_at) AS last "
                "FROM cards WHERE user_id IS NOT NULL AND created_at >= ? "
                "GROUP BY user_id",
                (from_iso,),
            ).fetchall()
        else:
            cards_rows = conn.execute(
                "SELECT user_id, COUNT(*) AS n, "
                "  COALESCE(SUM(cost_usd), 0) AS spend, "
                "  COALESCE(SUM(cost_stt), 0) AS spend_stt, "
                "  MAX(created_at) AS last "
                "FROM cards WHERE user_id IS NOT NULL "
                "GROUP BY user_id"
            ).fetchall()
        cards_by_user = {r["user_id"]: r for r in cards_rows}

        # ---- per-user roll-up of conversations ----
        if from_iso:
            conv_rows = conn.execute(
                "SELECT user_id, COUNT(*) AS n, "
                "  COALESCE(SUM(cost_usd), 0) AS spend, "
                "  COALESCE(SUM(cost_stt), 0) AS spend_stt, "
                "  MAX(started_at) AS last "
                "FROM conversations WHERE user_id IS NOT NULL AND started_at >= ? "
                "GROUP BY user_id",
                (from_iso,),
            ).fetchall()
        else:
            conv_rows = conn.execute(
                "SELECT user_id, COUNT(*) AS n, "
                "  COALESCE(SUM(cost_usd), 0) AS spend, "
                "  COALESCE(SUM(cost_stt), 0) AS spend_stt, "
                "  MAX(started_at) AS last "
                "FROM conversations WHERE user_id IS NOT NULL "
                "GROUP BY user_id"
            ).fetchall()
        conv_by_user = {r["user_id"]: r for r in conv_rows}

        # ---- user list ----
        user_rows = conn.execute(
            "SELECT id, email, name, role, company, created_at, last_login "
            "FROM users ORDER BY created_at ASC"
        ).fetchall()

    per_user = []
    team_cards = 0
    team_convs = 0
    team_spend = 0.0
    team_spend_stt = 0.0
    active_count = 0

    for u in user_rows:
        uid = u["id"]
        cr = cards_by_user.get(uid)
        cv = conv_by_user.get(uid)
        n_cards = cr["n"] if cr else 0
        n_convs = cv["n"] if cv else 0
        spend   = (cr["spend"]     if cr else 0) + (cv["spend"]     if cv else 0)
        spd_stt = (cr["spend_stt"] if cr else 0) + (cv["spend_stt"] if cv else 0)
        # Most recent activity = max of last card and last conversation in period.
        last_active = max(
            (cr["last"] if cr else "") or "",
            (cv["last"] if cv else "") or "",
        ) or None

        if (n_cards + n_convs) > 0:
            active_count += 1

        per_user.append({
            "id":         uid,
            "email":      u["email"],
            "name":       u["name"],
            "role":       u["role"],
            "company":    u["company"] if "company" in u.keys() else None,
            "created_at": u["created_at"],
            "last_login": u["last_login"],
            "last_active": last_active,
            "cards":      n_cards,
            "conversations": n_convs,
            "spend_usd":     round(spend, 6),
            "spend_stt_usd": round(spd_stt, 6),
            "spend_llm_usd": round(spend - spd_stt, 6),
        })

        team_cards     += n_cards
        team_convs     += n_convs
        team_spend     += spend
        team_spend_stt += spd_stt

    return {
        "period": (period or "week").lower(),
        "from":   from_iso,
        "team": {
            "total_users":   len(user_rows),
            "active_users":  active_count,
            "cards":         team_cards,
            "conversations": team_convs,
            "spend_usd":     round(team_spend, 6),
            "spend_stt_usd": round(team_spend_stt, 6),
            "spend_llm_usd": round(team_spend - team_spend_stt, 6),
        },
        "users": per_user,
    }

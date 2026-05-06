"""FastAPI entrypoint for the Conference AI Assistant backend."""

from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import BackgroundTasks, Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# Load .env early so config picks it up.
load_dotenv()

from . import storage  # noqa: E402  (intentional: load_dotenv first)
from .auth import (
    email_allowed,
    generate_temp_password,
    hash_password,
    make_current_user_dep,
    make_jwt,
    require_manager,
    verify_password,
)
from .company_profile import get_or_fetch_company_profile
from .conversation import summarize_conversation
from .models import (
    AuthResponse,
    User,
    UserCreate,
    UserLogin,
)
from .pdf_report import render_conversation_report
from .pipeline import continue_pipeline, run_pipeline
from .transcribe import transcribe_audio


# Build the FastAPI dependency that returns the current User. We inject the
# storage lookup here to avoid a circular import inside auth.py.
current_user = make_current_user_dep(storage.get_user_by_id)


logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger("conference-ai")

# Init the DB at import time so all entrypoints (uvicorn, tests, scripts)
# see the schema before serving any request. CREATE TABLE IF NOT EXISTS makes
# this idempotent.
storage.init_db()

app = FastAPI(title="Conference AI Assistant", version="0.1.0")

# Allow the local web tester (served on a different origin) to call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup() -> None:
    storage.init_db()
    log.info("Backend ready. Data dir: %s", storage.DATA_DIR)


# Mount the web tester so the same uvicorn process can serve it.
# We wrap StaticFiles to set Cache-Control: no-store so iterations show up
# without the user needing to hard-reload.
WEB_DIR = Path(__file__).resolve().parents[2] / "web"


class NoCacheStaticFiles(StaticFiles):
    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        return response


if WEB_DIR.is_dir():
    app.mount(
        "/app",
        NoCacheStaticFiles(directory=str(WEB_DIR), html=True),
        name="web",
    )


@app.get("/")
def root() -> dict:
    return {
        "name": "Conference AI Assistant",
        "version": "0.1.0",
        "docs": "/docs",
        "tester": "/app/" if WEB_DIR.is_dir() else None,
    }


@app.get("/health")
def health() -> dict:
    """Cheap health probe Render pings every minute.

    Returns 200 + a small body. Doesn't touch the DB or any model APIs;
    we want it to succeed even if Anthropic/OpenAI are down so Render
    doesn't pull the service while a vendor is having a bad day.
    """
    return {"status": "ok"}


ALLOWED_IMAGE_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
    "image/heic",
    "image/heif",
}


def _scope_for(user: User) -> Optional[str]:
    """Managers see everyone (None scope); reps see only their own user_id."""
    return None if user.role == "manager" else user.id


@app.post("/cards")
async def create_card_endpoint(
    background_tasks: BackgroundTasks,
    photo: UploadFile = File(...),
    user: User = Depends(current_user),
) -> dict:
    if photo.content_type not in ALLOWED_IMAGE_TYPES and not (
        photo.content_type and photo.content_type.startswith("image/")
    ):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {photo.content_type}",
        )

    card_id = uuid.uuid4().hex[:12]
    suffix = Path(photo.filename or "").suffix.lower() or ".jpg"
    if suffix not in {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"}:
        suffix = ".jpg"

    out = storage.upload_dir() / f"{card_id}{suffix}"
    contents = await photo.read()
    out.write_bytes(contents)

    storage.create_card(card_id, str(out), user_id=user.id)
    background_tasks.add_task(run_pipeline, card_id)

    return {"id": card_id, "status": "pending"}


@app.get("/cards/{card_id}")
def get_card_endpoint(card_id: str, user: User = Depends(current_user)) -> dict:
    record = storage.get_card(card_id, user_id=_scope_for(user))
    if record is None:
        raise HTTPException(status_code=404, detail="Card not found")

    has_pdf = (storage.report_dir() / f"{card_id}.pdf").is_file()
    payload = record.model_dump()
    payload["pdf_url"] = f"/cards/{card_id}/report.pdf" if has_pdf else None
    # Don't leak server-local paths.
    payload.pop("photo_path", None)
    # Costs are manager-only.
    if user.role != "manager":
        payload.pop("cost_usd", None)
    # When this card was paused as a duplicate, attach a thin summary of the
    # original card so the frontend can render the "existing customer" prompt
    # without a second roundtrip.
    if record.status == "duplicate" and record.duplicate_of:
        prior = storage.get_card(record.duplicate_of, user_id=_scope_for(user))
        if prior is not None:
            payload["duplicate_info"] = {
                "id":         prior.id,
                "created_at": prior.created_at,
                "name":       prior.extracted.name    if prior.extracted else None,
                "company":    prior.extracted.company if prior.extracted else None,
                "title":      prior.extracted.title   if prior.extracted else None,
            }
    return payload


@app.get("/cards/{card_id}/report.pdf")
def get_report_pdf(card_id: str, user: User = Depends(current_user)):
    # Auth-scope first so a rep can't download another rep's PDF by guessing the id.
    record = storage.get_card(card_id, user_id=_scope_for(user))
    if record is None:
        raise HTTPException(status_code=404, detail="Not found")
    pdf_path = storage.report_dir() / f"{card_id}.pdf"
    if not pdf_path.is_file():
        raise HTTPException(status_code=404, detail="Report not ready")
    return FileResponse(
        path=str(pdf_path),
        media_type="application/pdf",
        filename=f"sales-brief-{card_id}.pdf",
    )


@app.get("/cards/{card_id}/photo")
def get_photo(card_id: str, user: User = Depends(current_user)):
    record = storage.get_card(card_id, user_id=_scope_for(user))
    if record is None:
        raise HTTPException(status_code=404, detail="Card not found")
    p = Path(record.photo_path)
    if not p.is_file():
        raise HTTPException(status_code=404, detail="Photo missing")
    return FileResponse(path=str(p))


@app.get("/cards/{card_id}/conversations")
def list_card_conversations(
    card_id: str, user: User = Depends(current_user),
) -> list[dict]:
    """Return every conversation linked to this card, newest first.

    Used by the card-detail "Conversations" section so the card profile
    serves as the customer-centric hub: one customer → many conversations.
    Reps see only their own; managers see the full team's.
    """
    # Auth-scope the parent card so reps can't enumerate someone else's
    # contact's conversations.
    if storage.get_card(card_id, user_id=_scope_for(user)) is None:
        raise HTTPException(status_code=404, detail="Card not found")

    out: list[dict] = []
    # Pull conversations belonging to the same scope; filter to this card_id.
    for r in storage.list_conversations(limit=200, user_id=_scope_for(user)):
        if r.card_id != card_id:
            continue
        # First-sentence excerpt from the summary, if available — gives the
        # card section a useful preview without re-fetching the full record.
        excerpt = None
        if r.summary and r.summary.summary:
            text = r.summary.summary.strip()
            # Cheap "first sentence": cut on . / ! / ? or 140 chars.
            for sep in (". ", "! ", "? "):
                if sep in text:
                    text = text.split(sep, 1)[0] + sep.rstrip(" ")
                    break
            excerpt = text[:200]
        row = {
            "id": r.id,
            "started_at": r.started_at,
            "ended_at":   r.ended_at,
            "status":     r.status,
            "has_audio":  bool(r.audio_path),
            "excerpt":    excerpt,
        }
        if user.role == "manager":
            row["cost_usd"] = r.cost_usd
        out.append(row)
    # Sort newest-first by started_at.
    out.sort(key=lambda x: x.get("started_at") or "", reverse=True)
    return out


@app.post("/cards/{card_id}/email")
def email_report(
    card_id: str, payload: dict, user: User = Depends(current_user)
) -> JSONResponse:
    """Stub: returns the would-be email contents without sending."""
    record = storage.get_card(card_id, user_id=_scope_for(user))
    if record is None:
        raise HTTPException(status_code=404, detail="Card not found")
    if record.status != "ready":
        raise HTTPException(status_code=409, detail=f"Not ready (status={record.status})")

    to = (payload or {}).get("to")
    if not to:
        raise HTTPException(status_code=400, detail="Missing 'to' field")

    extracted = record.extracted
    subject = (
        f"Pre-meeting brief: {extracted.name} ({extracted.company})"
        if extracted and (extracted.name or extracted.company)
        else f"Pre-meeting brief {card_id}"
    )
    return JSONResponse(
        {
            "sent": False,
            "reason": "Email sender not configured. Wire SES/SendGrid in app/main.py.",
            "would_send_to": to,
            "subject": subject,
            "pdf_url": f"/cards/{card_id}/report.pdf",
        }
    )


@app.post("/cards/{card_id}/continue")
def continue_card(
    card_id: str,
    background_tasks: BackgroundTasks,
    user: User = Depends(current_user),
) -> dict:
    """Resume a card that was paused at the dedup gate (Plan B: promote-and-delete).

    The fresh extraction on the source row is copied onto its duplicate_of
    target, the source row + its photo/PDF are deleted, and research is
    scheduled on the target. End state: ONE row per (rep, person) with the
    freshest extraction + research + PDF + photo. Costs from both calls
    are summed onto the surviving row so the manager dashboard stays accurate.

    Returns the **target id** the frontend should poll from now on.
    """
    record = storage.get_card(card_id, user_id=_scope_for(user))
    if record is None:
        raise HTTPException(status_code=404, detail="Card not found")
    if record.status != "duplicate":
        raise HTTPException(
            status_code=409,
            detail=f"Card is not waiting on dedup decision (status={record.status})",
        )

    target_id = record.duplicate_of
    target = storage.get_card(target_id) if target_id else None
    if target_id and target is not None:
        # Promote the source's data onto the target, delete the source row.
        # Polling switches to the target id from here on.
        storage.promote_to_target(card_id, target_id)
        storage.delete_card(card_id)
        card_id = target_id
    else:
        # The target was deleted between dedup detection and now (rare race).
        # Fall back to running this card as a fresh, standalone scan.
        log.warning(
            "continue_card: duplicate target %r missing, treating %s as fresh.",
            target_id, card_id,
        )

    storage.update_status(card_id, "researching")
    background_tasks.add_task(continue_pipeline, card_id)
    return {"id": card_id, "status": "researching"}


@app.get("/cards")
def list_cards_endpoint(
    limit: int = 25, user: User = Depends(current_user)
) -> list[dict]:
    out = []
    for r in storage.list_cards(limit=limit, user_id=_scope_for(user)):
        row = {
            "id": r.id,
            "status": r.status,
            "created_at": r.created_at,
            "name": r.extracted.name if r.extracted else None,
            "company": r.extracted.company if r.extracted else None,
            "user_id": r.user_id,
        }
        if user.role == "manager":
            row["cost_usd"] = r.cost_usd
        out.append(row)
    return out


# ===========================================================================
# Conversation endpoints (slice 2)
# ===========================================================================


@app.post("/conversations")
def create_conversation_endpoint(
    payload: Optional[dict] = None, user: User = Depends(current_user),
) -> dict:
    """Start a new conversation. Body: {"card_id": optional str}."""
    card_id = (payload or {}).get("card_id")
    if card_id:
        # Reps can only link their own cards. Managers can link any card.
        if storage.get_card(card_id, user_id=_scope_for(user)) is None:
            raise HTTPException(status_code=404, detail=f"Card {card_id} not found")
    conv_id = uuid.uuid4().hex[:12]
    rec = storage.create_conversation(conv_id, card_id=card_id, user_id=user.id)
    return rec.model_dump()


def _summarize_and_render(conv_id: str) -> None:
    """Shared tail: take the transcript already on the record, summarize, render PDF."""
    rec = storage.get_conversation(conv_id)
    assert rec is not None
    card = storage.get_card(rec.card_id) if rec.card_id else None

    # Resolve the rep's seller-company so the follow-up email can be signed
    # off as the right firm and tailored to their go-to-market.
    seller_company = None
    seller_context = None
    if rec.user_id:
        rep = storage.get_user_by_id(rec.user_id)
        if rep and rep.company:
            seller_company = rep.company
            seller_context = get_or_fetch_company_profile(rep.company)

    storage.update_conversation_status(conv_id, "summarizing")
    summary, summary_cost = summarize_conversation(
        rec.transcript or "",
        card=card,
        seller_company=seller_company,
        company_context=seller_context,
    )
    storage.update_conversation_summary(conv_id, summary)
    storage.add_conversation_cost(conv_id, summary_cost)
    log.info("conv %s summary cost: $%.4f", conv_id, summary_cost)

    rec = storage.get_conversation(conv_id)
    assert rec is not None
    out = storage.report_dir() / f"conv-{conv_id}.pdf"
    render_conversation_report(rec, card, out)
    storage.update_conversation_status(conv_id, "ready")


def _run_summary(conv_id: str) -> None:
    """Background job for the text-transcript path: summarize + render."""
    rec = storage.get_conversation(conv_id)
    if rec is None:
        return
    try:
        _summarize_and_render(conv_id)
    except Exception as exc:  # noqa: BLE001
        log.exception("conversation summary failed for %s", conv_id)
        storage.update_conversation_status(
            conv_id, "error", error=f"{exc.__class__.__name__}: {exc}"
        )


def _run_audio_pipeline(conv_id: str) -> None:
    """Background job for the audio-upload path: transcribe -> summarize -> render."""
    rec = storage.get_conversation(conv_id)
    if rec is None or not rec.audio_path:
        return
    try:
        storage.update_conversation_status(conv_id, "transcribing")
        transcript, transcribe_cost = transcribe_audio(rec.audio_path)
        storage.update_conversation_transcript(conv_id, transcript)
        # The whole transcribe_cost is STT; track it both ways so admin dashboard
        # can break down LLM vs Whisper.
        storage.add_conversation_cost(conv_id, transcribe_cost, stt_delta=transcribe_cost)
        log.info("conv %s whisper cost: $%.4f", conv_id, transcribe_cost)
        _summarize_and_render(conv_id)
    except Exception as exc:  # noqa: BLE001
        log.exception("audio pipeline failed for %s", conv_id)
        storage.update_conversation_status(
            conv_id, "error", error=f"{exc.__class__.__name__}: {exc}"
        )


@app.post("/conversations/{conv_id}/finish")
def finish_conversation(
    conv_id: str,
    payload: dict,
    background_tasks: BackgroundTasks,
    user: User = Depends(current_user),
) -> dict:
    """Submit a text transcript directly (browser-side STT path)."""
    rec = storage.get_conversation(conv_id, user_id=_scope_for(user))
    if rec is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    transcript = (payload or {}).get("transcript", "")
    if not isinstance(transcript, str):
        raise HTTPException(status_code=400, detail="'transcript' must be a string")
    storage.update_conversation_transcript(conv_id, transcript)
    storage.update_conversation_status(conv_id, "summarizing")
    background_tasks.add_task(_run_summary, conv_id)
    return {"id": conv_id, "status": "summarizing"}


_AUDIO_EXTS = {".webm", ".mp4", ".m4a", ".mp3", ".wav", ".ogg", ".oga", ".flac", ".mpga", ".mpeg"}


@app.post("/conversations/{conv_id}/audio")
async def upload_conversation_audio(
    conv_id: str,
    background_tasks: BackgroundTasks,
    audio: UploadFile = File(...),
    user: User = Depends(current_user),
) -> dict:
    """Upload the recorded audio blob, then trigger transcribe -> summarize."""
    rec = storage.get_conversation(conv_id, user_id=_scope_for(user))
    if rec is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # Pick a safe filename suffix from the upload's filename / content type.
    suffix = Path(audio.filename or "").suffix.lower()
    if suffix not in _AUDIO_EXTS:
        ct = (audio.content_type or "").lower()
        if "mp4" in ct or "m4a" in ct or "aac" in ct:
            suffix = ".mp4"
        elif "ogg" in ct:
            suffix = ".ogg"
        elif "wav" in ct:
            suffix = ".wav"
        elif "mpeg" in ct or "mp3" in ct:
            suffix = ".mp3"
        else:
            suffix = ".webm"  # MediaRecorder default in Chromium

    out = storage.upload_dir() / f"conv-{conv_id}{suffix}"
    contents = await audio.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty audio upload")
    out.write_bytes(contents)

    storage.update_conversation_audio(conv_id, str(out))
    storage.update_conversation_status(conv_id, "transcribing")
    background_tasks.add_task(_run_audio_pipeline, conv_id)

    return {
        "id": conv_id,
        "status": "transcribing",
        "size_bytes": len(contents),
        "audio_filename": out.name,
    }


@app.get("/conversations/{conv_id}")
def get_conversation_endpoint(
    conv_id: str, user: User = Depends(current_user),
) -> dict:
    rec = storage.get_conversation(conv_id, user_id=_scope_for(user))
    if rec is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    payload = rec.model_dump()
    has_pdf = (storage.report_dir() / f"conv-{conv_id}.pdf").is_file()
    payload["pdf_url"] = f"/conversations/{conv_id}/report.pdf" if has_pdf else None
    if user.role != "manager":
        payload.pop("cost_usd", None)
    return payload


@app.patch("/conversations/{conv_id}")
def update_conversation_endpoint(
    conv_id: str, payload: dict, user: User = Depends(current_user),
) -> dict:
    """Late-bind a card to a conversation (record-then-scan flow).

    Body: ``{"card_id": "..." | null}``.

    Reps can update conversations they own; managers can update any. The
    target card must also belong to the rep (validated when not a manager).
    If the conversation has already finished (status='ready'), we re-render
    its PDF inline so the brief reflects the new contact. In-flight
    conversations don't need a re-render — the pipeline picks up the new
    card_id when it eventually renders.
    """
    rec = storage.get_conversation(conv_id, user_id=_scope_for(user))
    if rec is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    if "card_id" not in payload:
        return rec.model_dump()  # nothing to change

    card_id = payload.get("card_id")
    if card_id is not None:
        if not isinstance(card_id, str) or not card_id:
            raise HTTPException(
                status_code=400, detail="card_id must be a non-empty string or null",
            )
        # Validate ownership of the target card (managers bypass _scope_for).
        if storage.get_card(card_id, user_id=_scope_for(user)) is None:
            raise HTTPException(status_code=404, detail="Card not found")

    # Apply the update with the same ownership scope as the read.
    storage.update_conversation_card_id(
        conv_id, card_id, user_id=_scope_for(user),
    )

    # If the brief PDF was already produced, regenerate it so the linked
    # card actually shows up in the report. Cheap (no LLM call).
    rec = storage.get_conversation(conv_id, user_id=_scope_for(user))
    if rec is not None and rec.status == "ready":
        try:
            card = storage.get_card(card_id) if card_id else None
            out = storage.report_dir() / f"conv-{conv_id}.pdf"
            render_conversation_report(rec, card, out)
        except Exception as exc:  # noqa: BLE001
            log.warning("conv %s PDF re-render after link failed: %s", conv_id, exc)
            # Don't fail the link — the row was updated, PDF is just stale.

    return rec.model_dump() if rec else {}


@app.get("/conversations/{conv_id}/report.pdf")
def conversation_report_pdf(conv_id: str, user: User = Depends(current_user)):
    rec = storage.get_conversation(conv_id, user_id=_scope_for(user))
    if rec is None:
        raise HTTPException(status_code=404, detail="Not found")
    pdf_path = storage.report_dir() / f"conv-{conv_id}.pdf"
    if not pdf_path.is_file():
        raise HTTPException(status_code=404, detail="Report not ready")
    return FileResponse(
        path=str(pdf_path),
        media_type="application/pdf",
        filename=f"conversation-{conv_id}.pdf",
    )


@app.get("/conversations")
def list_conversations_endpoint(
    limit: int = 25, user: User = Depends(current_user),
) -> list[dict]:
    out = []
    for r in storage.list_conversations(limit=limit, user_id=_scope_for(user)):
        # When manager views team-wide, look up the card without scope so the
        # customer name resolves even for cards belonging to other reps.
        card = storage.get_card(r.card_id) if r.card_id else None
        row = {
            "id": r.id,
            "status": r.status,
            "started_at": r.started_at,
            "ended_at": r.ended_at,
            "card_id": r.card_id,
            "user_id": r.user_id,
            "customer_name": card.extracted.name if card and card.extracted else None,
            "customer_company": card.extracted.company if card and card.extracted else None,
        }
        if user.role == "manager":
            row["cost_usd"] = r.cost_usd
        out.append(row)
    return out


# ===========================================================================
# Auth + admin endpoints (slice 3)
# ===========================================================================


@app.post("/auth/scan-card-for-signup")
async def scan_card_for_signup(
    photo: UploadFile = File(...),
) -> dict:
    """Pre-registration card / badge scan to auto-fill the signup form.

    No auth required. Runs the extraction (Claude Vision, ~$0.005). Photo
    is held in a temp file and deleted immediately — nothing is persisted.
    """
    if photo.content_type and not photo.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Must be an image")
    contents = await photo.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty upload")
    if len(contents) > 25 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image too large (>25 MB)")

    suffix = Path(photo.filename or "").suffix.lower() or ".jpg"
    if suffix not in {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"}:
        suffix = ".jpg"

    import tempfile
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name
    try:
        from .extraction import extract_card
        extracted, _cost = extract_card(tmp_path)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    # Return only the fields the signup form needs.
    return {
        "name":    extracted.name,
        "title":   extracted.title,
        "company": extracted.company,
        "emails":  extracted.emails or [],
    }


@app.post("/auth/register", response_model=AuthResponse)
def register(payload: UserCreate) -> AuthResponse:
    """Self-serve registration. Email domain must be on EMAIL_ALLOWLIST."""
    if not email_allowed(payload.email):
        raise HTTPException(
            status_code=403,
            detail="Email domain not allowed for self-serve signup. "
                   "Ask your admin to invite you.",
        )
    if storage.get_user_by_email_with_hash(payload.email) is not None:
        raise HTTPException(status_code=409, detail="An account with that email already exists.")

    # First user becomes manager automatically (so you, as the first rep
    # to sign up after deploy, get team-wide access without manual SQL).
    role = "manager" if storage.count_users() == 0 else "rep"
    pw_hash = hash_password(payload.password)
    user = storage.create_user(
        email=str(payload.email),
        password_hash=pw_hash,
        name=payload.name,
        role=role,
        company=(payload.company or "").strip() or None,
    )
    return AuthResponse(token=make_jwt(user.id), user=user)


@app.post("/auth/login", response_model=AuthResponse)
def login(payload: UserLogin) -> AuthResponse:
    found = storage.get_user_by_email_with_hash(payload.email)
    if found is None:
        # Same error as wrong password to avoid email-enumeration.
        raise HTTPException(status_code=401, detail="Invalid email or password.")
    user, pw_hash = found
    if not verify_password(payload.password, pw_hash):
        raise HTTPException(status_code=401, detail="Invalid email or password.")
    storage.update_last_login(user.id)
    return AuthResponse(token=make_jwt(user.id), user=user)


@app.get("/auth/me", response_model=User)
def auth_me(user: User = Depends(current_user)) -> User:
    return user


@app.post("/auth/logout")
def logout(user: User = Depends(current_user)) -> dict:
    """Stateless JWT — server has nothing to do. Client just drops the token."""
    return {"ok": True}


@app.get("/admin/users", response_model=list[User])
def admin_list_users(user: User = Depends(current_user)) -> list[User]:
    require_manager(user)
    return storage.list_users(limit=500)


@app.get("/admin/stats")
def admin_stats(period: str = "week", user: User = Depends(current_user)) -> dict:
    """Manager dashboard: team totals + per-user activity & spend.
    Periods: today | week (default) | month | all."""
    require_manager(user)
    if period not in ("today", "week", "month", "all"):
        raise HTTPException(status_code=400, detail="Invalid period")
    return storage.compute_team_stats(period=period)


@app.delete("/admin/users/{user_id}")
def admin_delete_user(user_id: str, user: User = Depends(current_user)) -> dict:
    require_manager(user)
    if user_id == user.id:
        raise HTTPException(status_code=400, detail="Cannot delete yourself.")
    if not storage.delete_user(user_id):
        raise HTTPException(status_code=404, detail="User not found.")
    return {"ok": True}


@app.post("/admin/users/{user_id}/reset-password")
def admin_reset_password(
    user_id: str, user: User = Depends(current_user),
) -> dict:
    """Generate a one-time temp password and force the user to change it on
    next login. Returns the plaintext temp password ONCE so the manager can
    share it with the rep out-of-band (Slack, etc.). Manager-only."""
    require_manager(user)
    target = storage.get_user_by_id(user_id)
    if target is None:
        raise HTTPException(status_code=404, detail="User not found.")
    if target.id == user.id:
        # Managers shouldn't reset themselves through this flow — they should
        # use /auth/change-password directly.
        raise HTTPException(
            status_code=400,
            detail="Use 'Change password' for your own account.",
        )

    temp = generate_temp_password(12)
    storage.update_user_password(user_id, hash_password(temp), must_change=True)
    return {
        "ok": True,
        "user_id": user_id,
        "email": target.email,
        "temp_password": temp,
        "warning": "Share this with the user via a secure channel. They will be required to change it on next login.",
    }


@app.post("/auth/change-password")
def change_password(payload: dict, user: User = Depends(current_user)) -> dict:
    """Logged-in user sets a new password. Clears the must_change_password
    flag if it was set. Used by both the forced-change flow (after admin
    reset) and any voluntary change."""
    new_password = (payload or {}).get("password") or ""
    if len(new_password) < 8:
        raise HTTPException(
            status_code=400,
            detail="Password must be at least 8 characters.",
        )
    if len(new_password) > 200:
        raise HTTPException(status_code=400, detail="Password too long.")
    storage.update_user_password(user.id, hash_password(new_password), must_change=False)
    return {"ok": True}

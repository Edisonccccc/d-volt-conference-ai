"""Glue between FastAPI and the AI calls. Runs in a background task."""

from __future__ import annotations

import logging
import sys
import time
import traceback

from . import storage
from .extraction import extract_card
from .pdf_report import render_report
from .research import research_company


log = logging.getLogger("conference-ai.pipeline")


def _banner(msg: str) -> None:
    """Print a visible banner so it's easy to find each card's run in the log."""
    sep = "─" * 72
    sys.stderr.write(f"\n{sep}\n{msg}\n{sep}\n")
    sys.stderr.flush()


def _research_and_render(card_id: str, t0: float) -> None:
    """Run the back half of the pipeline: research -> PDF -> ready.

    Pulled out as a helper so /cards/{id}/continue can resume here when a
    user chooses 'rescan anyway' on a deduplicated card. `t0` is the
    monotonic start time so the final banner reports total elapsed.
    """
    storage.update_status(card_id, "researching")
    log.info("[%s] step 2/3: research", card_id)
    record = storage.get_card(card_id)
    assert record is not None and record.extracted is not None
    research, research_cost = research_company(record.extracted)
    storage.update_research(card_id, research)
    storage.add_card_cost(card_id, research_cost)
    log.info(
        "[%s] research one_liner=%r (sources=%d, cost $%.4f)",
        card_id, research.one_liner, len(research.sources), research_cost,
    )

    log.info("[%s] step 3/3: render PDF", card_id)
    record = storage.get_card(card_id)
    assert record is not None
    out_path = storage.report_dir() / f"{card_id}.pdf"
    render_report(record, out_path)

    storage.update_status(card_id, "ready")
    elapsed = time.monotonic() - t0
    _banner(f"✓ Card {card_id} ready in {elapsed:.1f}s -> {out_path}")


def run_pipeline(card_id: str) -> None:
    """Run extraction -> [dedup gate] -> research -> PDF for a card.

    If, after extraction, this card looks like a re-scan of a person the
    same rep has already analyzed (matched on normalized name + company),
    we stop at status='duplicate' and let the frontend ask the user
    whether to view the prior result or pay for a fresh research call.
    """
    record = storage.get_card(card_id)
    if record is None:
        log.error("run_pipeline: card %s not found", card_id)
        return

    t0 = time.monotonic()
    _banner(f"▶ Starting pipeline for card {card_id}")

    try:
        storage.update_status(card_id, "extracting")
        log.info("[%s] step 1/3: extract", card_id)
        extracted, extract_cost = extract_card(record.photo_path)
        storage.update_extracted(card_id, extracted)
        storage.add_card_cost(card_id, extract_cost)
        log.info(
            "[%s] extracted: name=%r company=%r (cost $%.4f)",
            card_id, extracted.name, extracted.company, extract_cost,
        )

        # ---- dedup gate: same rep, same (name, company) already on file? ----
        if record.user_id and extracted.name and extracted.company:
            dup = storage.find_duplicate_card(
                record.user_id,
                extracted.name,
                extracted.company,
                exclude_id=card_id,
            )
            if dup is not None:
                storage.mark_duplicate(card_id, dup.id)
                elapsed = time.monotonic() - t0
                log.info(
                    "[%s] duplicate of %s (name=%r company=%r) — pausing pipeline",
                    card_id, dup.id, extracted.name, extracted.company,
                )
                _banner(
                    f"⏸ Card {card_id} paused as duplicate of {dup.id} "
                    f"after {elapsed:.1f}s (saved a research call)"
                )
                return  # frontend will call /continue if user wants to rescan anyway

        _research_and_render(card_id, t0)
    except Exception as exc:  # noqa: BLE001
        elapsed = time.monotonic() - t0
        log.exception("Pipeline failed for card %s after %.1fs", card_id, elapsed)
        storage.update_status(
            card_id,
            "error",
            error=f"{exc.__class__.__name__}: {exc}\n{traceback.format_exc()[-1200:]}",
        )
        _banner(f"✗ Card {card_id} FAILED after {elapsed:.1f}s: {exc}")


def continue_pipeline(card_id: str) -> None:
    """Run research + PDF on a card whose extraction is already on the row.

    Triggered by POST /cards/{id}/continue (rescan-anyway). The endpoint
    has already handled the dedup-target promote and deletion, so by the
    time this runs the card_id is the *surviving* row and just needs the
    back-half of the pipeline.
    """
    record = storage.get_card(card_id)
    if record is None:
        log.error("continue_pipeline: card %s not found", card_id)
        return

    t0 = time.monotonic()
    _banner(f"▶ Resuming pipeline for card {card_id} (rescan-anyway)")
    try:
        _research_and_render(card_id, t0)
    except Exception as exc:  # noqa: BLE001
        elapsed = time.monotonic() - t0
        log.exception("continue_pipeline failed for %s after %.1fs", card_id, elapsed)
        storage.update_status(
            card_id,
            "error",
            error=f"{exc.__class__.__name__}: {exc}\n{traceback.format_exc()[-1200:]}",
        )
        _banner(f"✗ Card {card_id} resume FAILED after {elapsed:.1f}s: {exc}")

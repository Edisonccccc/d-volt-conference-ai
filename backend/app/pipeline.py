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


def run_pipeline(card_id: str) -> None:
    """Run extraction -> research -> PDF for a card. Updates DB at each step."""
    record = storage.get_card(card_id)
    if record is None:
        log.error("run_pipeline: card %s not found", card_id)
        return

    t0 = time.monotonic()
    _banner(f"▶ Starting pipeline for card {card_id}")

    try:
        storage.update_status(card_id, "extracting")
        log.info("[%s] step 1/3: extract", card_id)
        extracted = extract_card(record.photo_path)
        storage.update_extracted(card_id, extracted)
        log.info(
            "[%s] extracted: name=%r company=%r",
            card_id, extracted.name, extracted.company,
        )

        storage.update_status(card_id, "researching")
        log.info("[%s] step 2/3: research", card_id)
        research = research_company(extracted)
        storage.update_research(card_id, research)
        log.info(
            "[%s] research one_liner=%r (sources=%d)",
            card_id, research.one_liner, len(research.sources),
        )

        log.info("[%s] step 3/3: render PDF", card_id)
        record = storage.get_card(card_id)
        assert record is not None
        out_path = storage.report_dir() / f"{card_id}.pdf"
        render_report(record, out_path)

        storage.update_status(card_id, "ready")
        elapsed = time.monotonic() - t0
        _banner(f"✓ Card {card_id} ready in {elapsed:.1f}s -> {out_path}")
    except Exception as exc:  # noqa: BLE001
        elapsed = time.monotonic() - t0
        log.exception("Pipeline failed for card %s after %.1fs", card_id, elapsed)
        storage.update_status(
            card_id,
            "error",
            error=f"{exc.__class__.__name__}: {exc}\n{traceback.format_exc()[-1200:]}",
        )
        _banner(f"✗ Card {card_id} FAILED after {elapsed:.1f}s: {exc}")

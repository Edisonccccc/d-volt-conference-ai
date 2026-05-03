"""Render a sales-ready PDF from an extracted card + research brief."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PIL import Image
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image as RLImage,
    ListFlowable,
    ListItem,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from .models import CardRecord, ConversationRecord


def _styles():
    base = getSampleStyleSheet()
    h1 = ParagraphStyle(
        "H1",
        parent=base["Heading1"],
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#0b3d91"),
        spaceAfter=8,
    )
    h2 = ParagraphStyle(
        "H2",
        parent=base["Heading2"],
        fontSize=13,
        leading=16,
        textColor=colors.HexColor("#0b3d91"),
        spaceBefore=10,
        spaceAfter=4,
    )
    body = ParagraphStyle(
        "Body",
        parent=base["BodyText"],
        fontSize=10.5,
        leading=14,
        alignment=TA_LEFT,
    )
    small = ParagraphStyle(
        "Small",
        parent=body,
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#555555"),
    )
    return h1, h2, body, small


def _scaled_image(path: str, max_w: float, max_h: float) -> Optional[RLImage]:
    try:
        with Image.open(path) as im:
            w, h = im.size
    except Exception:
        return None
    ratio = min(max_w / w, max_h / h)
    return RLImage(path, width=w * ratio, height=h * ratio)


def _bullets(items, body_style) -> ListFlowable:
    return ListFlowable(
        [ListItem(Paragraph(i, body_style), leftIndent=12) for i in items],
        bulletType="bullet",
        leftIndent=14,
    )


def render_report(record: CardRecord, out_path: Path) -> Path:
    """Render the report PDF and return its path."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    h1, h2, body, small = _styles()

    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=LETTER,
        leftMargin=0.7 * inch,
        rightMargin=0.7 * inch,
        topMargin=0.7 * inch,
        bottomMargin=0.7 * inch,
        title="Sales Pre-Meeting Brief",
        author="Conference AI Assistant",
    )

    flowables = []
    extracted = record.extracted
    research = record.research

    title = "Sales Pre-Meeting Brief"
    if extracted and (extracted.name or extracted.company):
        subtitle = " · ".join(
            x for x in [extracted.name, extracted.company] if x
        )
        flowables.append(Paragraph(title, h1))
        flowables.append(Paragraph(subtitle, body))
    else:
        flowables.append(Paragraph(title, h1))
    flowables.append(Paragraph(f"Generated {record.created_at}", small))
    flowables.append(Spacer(1, 8))

    # Two-column block: photo (left) + contact fields (right).
    photo = _scaled_image(record.photo_path, 2.6 * inch, 1.8 * inch)

    def _norm_url(u: Optional[str]) -> Optional[str]:
        if not u:
            return None
        return u if u.lower().startswith(("http://", "https://")) else f"https://{u.lstrip('/')}"

    def _link(url: Optional[str]) -> Optional[str]:
        if not url:
            return None
        href = _norm_url(url)
        return f'<link href="{href}" color="#0b3d91">{url}</link>'

    def _title_display() -> Optional[str]:
        card_t = (extracted.title or "").strip() if extracted else ""
        verified = (research.contact_title_verified or "").strip() if research else ""
        if not verified:
            return card_t or None
        if not card_t:
            return f"{verified} (verified)"
        if card_t.lower() == verified.lower():
            return card_t
        return f"{card_t} (verified: {verified})"

    contact_rows = []
    if extracted:
        # Prefer research-verified URLs when present; render inline links.
        website_val = (research.company_website if research and research.company_website else extracted.website)
        linkedin_val = (research.contact_linkedin if research and research.contact_linkedin else extracted.linkedin)

        for label, val_html in [
            ("Name", extracted.name),
            ("Title", _title_display()),
            ("Company", extracted.company),
            ("Email", ", ".join(extracted.emails) if extracted.emails else None),
            ("Phone", ", ".join(extracted.phones) if extracted.phones else None),
            ("Website", _link(website_val)),
            ("Address", extracted.address),
            ("LinkedIn", _link(linkedin_val)),
        ]:
            if val_html:
                contact_rows.append([
                    Paragraph(f"<b>{label}</b>", body),
                    Paragraph(val_html, body),
                ])
    if not contact_rows:
        contact_rows = [[Paragraph("No fields extracted.", small), ""]]

    contact_table = Table(contact_rows, colWidths=[0.9 * inch, 3.0 * inch])
    contact_table.setStyle(
        TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ("TOPPADDING", (0, 0), (-1, -1), 2),
        ])
    )

    top_block = Table(
        [[photo if photo else Paragraph("(no photo)", small), contact_table]],
        colWidths=[2.7 * inch, 4.3 * inch],
    )
    top_block.setStyle(
        TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ])
    )
    flowables.append(top_block)
    flowables.append(Spacer(1, 8))

    # Research sections.
    flowables.append(Paragraph("Company snapshot", h2))
    if research and research.one_liner:
        flowables.append(Paragraph(research.one_liner, body))
    else:
        flowables.append(Paragraph("Not available.", small))

    if research:
        # ---- snapshot --------------------------------------------------
        snapshot_rows = []
        if research.company_category:
            cat_label = (
                f"<b>{research.company_category}</b>"
                + (
                    f' &nbsp; <font color="#5b6677"><i>{research.category_rationale}</i></font>'
                    if research.category_rationale
                    else ""
                )
            )
            snapshot_rows.append([Paragraph("<b>Category</b>", body), Paragraph(cat_label, body)])
        if research.industry:
            snapshot_rows.append([Paragraph("<b>Industry</b>", body), Paragraph(research.industry, body)])
        if research.estimated_size:
            snapshot_rows.append([Paragraph("<b>Size</b>", body), Paragraph(research.estimated_size, body)])
        if research.products:
            snapshot_rows.append([
                Paragraph("<b>Products</b>", body),
                Paragraph(", ".join(research.products), body),
            ])
        if snapshot_rows:
            t = Table(snapshot_rows, colWidths=[0.9 * inch, 6.1 * inch])
            t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("BOTTOMPADDING", (0, 0), (-1, -1), 1)]))
            flowables.append(Spacer(1, 4))
            flowables.append(t)

        if research.recent_news:
            flowables.append(Paragraph("Recent news", h2))
            flowables.append(_bullets(research.recent_news, body))

        if research.pain_points:
            flowables.append(Paragraph("Likely pain points", h2))
            flowables.append(_bullets(research.pain_points, body))

        if research.opening_questions:
            flowables.append(Paragraph("Suggested opening questions", h2))
            flowables.append(_bullets(research.opening_questions, body))

        if research.sources:
            flowables.append(Paragraph("Sources", h2))
            flowables.append(_bullets(research.sources, small))

    if record.error:
        flowables.append(Spacer(1, 10))
        flowables.append(Paragraph(f"Note: {record.error}", small))

    doc.build(flowables)
    return out_path


# ---------------------------------------------------------------------------
# Conversation summary PDF
# ---------------------------------------------------------------------------


def render_conversation_report(
    conv: ConversationRecord,
    card: Optional[CardRecord],
    out_path: Path,
) -> Path:
    """Render the post-conversation summary PDF and return its path."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    h1, h2, body, small = _styles()

    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=LETTER,
        leftMargin=0.7 * inch,
        rightMargin=0.7 * inch,
        topMargin=0.7 * inch,
        bottomMargin=0.7 * inch,
        title="Sales Conversation Summary",
        author="Conference AI Assistant",
    )

    flowables: list = []
    extracted = card.extracted if card else None
    research = card.research if card else None
    summary = conv.summary

    # ---- Identity block --------------------------------------------------
    customer_line_parts = []
    if extracted and extracted.name:
        customer_line_parts.append(extracted.name)
    if extracted and extracted.company:
        customer_line_parts.append(extracted.company)
    customer_line = " · ".join(customer_line_parts) if customer_line_parts else "Customer (unlinked)"

    flowables.append(Paragraph("Sales Conversation Summary", h1))
    flowables.append(Paragraph(customer_line, body))

    id_rows = [
        [Paragraph("<b>Started</b>", body), Paragraph(conv.started_at, body)],
    ]
    if conv.ended_at:
        id_rows.append([Paragraph("<b>Ended</b>", body), Paragraph(conv.ended_at, body)])
    if extracted:
        if extracted.title:
            id_rows.append([Paragraph("<b>Customer title</b>", body), Paragraph(extracted.title, body)])
        if extracted.emails:
            id_rows.append([Paragraph("<b>Email</b>", body), Paragraph(", ".join(extracted.emails), body)])
    if research and research.company_category:
        id_rows.append([Paragraph("<b>Category</b>", body), Paragraph(research.company_category, body)])

    t = Table(id_rows, colWidths=[1.1 * inch, 5.9 * inch])
    t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("BOTTOMPADDING", (0, 0), (-1, -1), 1)]))
    flowables.append(Spacer(1, 4))
    flowables.append(t)
    flowables.append(Spacer(1, 8))

    # ---- Summary sections ------------------------------------------------
    if summary is None:
        flowables.append(Paragraph("No summary available.", small))
    else:
        flowables.append(Paragraph("Summary", h2))
        flowables.append(Paragraph(summary.summary or "—", body))

        if summary.key_topics:
            flowables.append(Paragraph("Key topics", h2))
            flowables.append(_bullets(summary.key_topics, body))

        if summary.customer_concerns:
            flowables.append(Paragraph("Customer concerns", h2))
            flowables.append(_bullets(summary.customer_concerns, body))

        if summary.commitments:
            flowables.append(Paragraph("Commitments", h2))
            flowables.append(_bullets(summary.commitments, body))

        if summary.next_steps:
            flowables.append(Paragraph("Next steps", h2))
            flowables.append(_bullets(summary.next_steps, body))

        if summary.follow_up_email:
            flowables.append(Paragraph("Draft follow-up email", h2))
            # Preserve line breaks in the email by splitting on newlines.
            for para in summary.follow_up_email.split("\n\n"):
                flowables.append(Paragraph(para.replace("\n", "<br/>"), body))
                flowables.append(Spacer(1, 4))

    # ---- Transcript appendix --------------------------------------------
    if conv.transcript:
        flowables.append(Spacer(1, 12))
        flowables.append(Paragraph("Transcript (verbatim)", h2))
        for para in conv.transcript.split("\n"):
            if para.strip():
                flowables.append(Paragraph(para, small))
                flowables.append(Spacer(1, 2))

    if conv.error:
        flowables.append(Spacer(1, 10))
        flowables.append(Paragraph(f"Note: {conv.error}", small))

    doc.build(flowables)
    return out_path

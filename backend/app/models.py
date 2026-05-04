"""Pydantic models for card extraction, conversation, and users."""

from __future__ import annotations

from typing import List, Optional, Literal
from pydantic import BaseModel, EmailStr, Field


class ExtractedCard(BaseModel):
    """Structured fields extracted from a business card photo."""

    name: Optional[str] = Field(None, description="Person's full name")
    title: Optional[str] = Field(None, description="Job title")
    company: Optional[str] = Field(None, description="Company name")
    emails: List[str] = Field(default_factory=list)
    phones: List[str] = Field(default_factory=list)
    website: Optional[str] = None
    address: Optional[str] = None
    linkedin: Optional[str] = None
    notes: Optional[str] = Field(
        None,
        description="Anything else legible on the card that doesn't fit elsewhere",
    )


class CompanyResearch(BaseModel):
    """Sales-oriented research brief on the customer's company."""

    # --- verified contact links (looked up via web search) -----------------
    contact_linkedin: Optional[str] = Field(
        None,
        description=(
            "The contact's personal LinkedIn profile URL "
            "(linkedin.com/in/...), found via web search."
        ),
    )
    contact_title_verified: Optional[str] = Field(
        None,
        description=(
            "The contact's current title as listed on LinkedIn or the company "
            "site, used to confirm or correct what was on the card."
        ),
    )
    company_website: Optional[str] = Field(
        None, description="The company's official website (homepage URL)."
    )

    # --- company brief ------------------------------------------------------
    one_liner: Optional[str] = None
    company_category: Optional[str] = Field(
        None,
        description=(
            "Where the company sits in d-volt's market. One of: Utility, "
            "Vendor, EPC, Sales Representatives, Distributors, End Users, Other."
        ),
    )
    category_rationale: Optional[str] = Field(
        None,
        description="One short sentence explaining why this category was chosen.",
    )
    industry: Optional[str] = None
    estimated_size: Optional[str] = Field(
        None, description="e.g. '50-200 employees' or 'Public, ~$1B revenue'"
    )
    products: List[str] = Field(default_factory=list)
    recent_news: List[str] = Field(
        default_factory=list,
        description="Bullet-style notes on news from the last ~90 days, with sources where possible.",
    )
    pain_points: List[str] = Field(
        default_factory=list,
        description="Likely pain points framed against d-volt's offering.",
    )
    opening_questions: List[str] = Field(
        default_factory=list,
        description="3 short questions a salesperson can open with.",
    )
    sources: List[str] = Field(default_factory=list)


CardStatus = Literal["pending", "extracting", "researching", "ready", "error"]


class CardRecord(BaseModel):
    id: str
    status: CardStatus
    created_at: str
    photo_path: str
    extracted: Optional[ExtractedCard] = None
    research: Optional[CompanyResearch] = None
    error: Optional[str] = None
    user_id: Optional[str] = None  # owner; nullable on legacy rows
    cost_usd: Optional[float] = Field(
        None,
        description="Sum of vendor charges to produce this card (extraction + research). Manager-only via API.",
    )


# ---------------------------------------------------------------------------
# Conversation (slice 2): live capture + Claude summary
# ---------------------------------------------------------------------------

ConversationStatus = Literal[
    "recording",
    "uploading",
    "transcribing",
    "summarizing",
    "ready",
    "error",
]


class ConversationSummary(BaseModel):
    """Structured summary of a sales conversation."""

    summary: Optional[str] = Field(
        None,
        description="One short paragraph summarizing the conversation.",
    )
    key_topics: List[str] = Field(
        default_factory=list,
        description="Bullet list of the main topics discussed.",
    )
    customer_concerns: List[str] = Field(
        default_factory=list,
        description="Pain points, objections, or concerns the customer raised.",
    )
    commitments: List[str] = Field(
        default_factory=list,
        description=(
            "Concrete commitments either side made during the call "
            "(e.g. 'Sales will send a quote by Friday')."
        ),
    )
    next_steps: List[str] = Field(
        default_factory=list,
        description=(
            "Recommended next steps for the salesperson to move toward a deal."
        ),
    )
    follow_up_email: Optional[str] = Field(
        None,
        description=(
            "A polite, concrete draft follow-up email the salesperson can "
            "send to the customer. Plain text, ~150-250 words, includes "
            "specific points from the conversation."
        ),
    )


class ConversationRecord(BaseModel):
    id: str
    status: ConversationStatus
    started_at: str
    ended_at: Optional[str] = None
    card_id: Optional[str] = None
    audio_path: Optional[str] = None
    transcript: Optional[str] = None
    summary: Optional[ConversationSummary] = None
    error: Optional[str] = None
    user_id: Optional[str] = None  # owner; nullable on legacy rows
    cost_usd: Optional[float] = Field(
        None,
        description="Sum of vendor charges to produce this conversation (Whisper + Claude summary). Manager-only via API.",
    )


# ---------------------------------------------------------------------------
# Users (slice 3)
# ---------------------------------------------------------------------------

UserRole = Literal["rep", "manager"]


class User(BaseModel):
    """User as exposed to the API (no password_hash)."""

    id: str
    email: EmailStr
    name: Optional[str] = None
    role: UserRole = "rep"
    company: Optional[str] = None
    created_at: str
    last_login: Optional[str] = None


class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=200)
    name: Optional[str] = Field(None, max_length=120)
    company: str = Field(min_length=1, max_length=120)


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class AuthResponse(BaseModel):
    token: str
    user: User

"""Core data shapes. Every derived field should be traceable to a source span."""
from __future__ import annotations

from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field


class RawReview(BaseModel):
    """What an ingest adapter emits. Korean text, untouched."""
    id: str
    source: Literal["naver_blog", "naver_cafe", "daum_cafe", "gangnam_unni", "babitalk", "other"]
    url: str
    author: str
    posted_at: str  # ISO date
    clinic_name_raw: str
    clinic_phone: Optional[str] = None
    text_ko: str
    has_receipt: bool = False  # app-level "receipt verified" flag (e.g. 강남언니 영수증 인증)


class Price(BaseModel):
    amount_krw: int = Field(description="Integer KRW. '350' in a review usually means 350만원 = 3,500,000")
    source_span: str = Field(description="Exact substring of the Korean text the price came from")


class Extraction(BaseModel):
    """LLM output schema: translate + extract in one call, with source spans for audit."""
    text_en: str
    procedures: list[str] = Field(description="Canonical procedure_ids from the glossary; 'unknown' if not in glossary")
    surgeon_name_ko: Optional[str] = Field(None, description="Only if literally named in the text, else null")
    prices: list[Price] = []
    sponsored_markers: list[str] = Field([], description="Verbatim Korean phrases implying sponsorship (협찬, 체험단, 원고료, 제공받아)")
    self_paid_markers: list[str] = Field([], description="Verbatim phrases like 내돈내산")
    days_post_op: Optional[int] = None
    sentiment: Literal["positive", "mixed", "negative"]
    complications: list[str] = Field([], description="English, only those explicitly mentioned")
    confidence: float = Field(ge=0, le=1)


class ClinicMatch(BaseModel):
    clinic_id: Optional[str]  # None = abstained -> human queue
    method: Literal["phone", "exact_name", "fuzzy_name", "agent", "abstain"]
    score: float
    candidates: list[str] = []
    evidence: list[str] = []  # agent resolutions must cite what they looked at


class Status(str, Enum):
    published = "published"
    quarantined = "quarantined"   # failed validation; needs human
    duplicate = "duplicate"       # folded into canonical review


class ProcessedReview(BaseModel):
    raw: RawReview
    status: Status
    canonical_id: Optional[str] = None  # if duplicate, points at the kept review
    dup_cluster: list[str] = []
    template_cluster: bool = False      # near-dups across DIFFERENT authors => likely paid campaign
    clinic: Optional[ClinicMatch] = None
    extraction: Optional[Extraction] = None
    trust_score: Optional[float] = None
    trust_reasons: list[str] = []
    issues: list[str] = []
    trace: list[str] = []  # step-by-step provenance log

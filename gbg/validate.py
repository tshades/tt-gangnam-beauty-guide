"""Mechanical checks on LLM output. Anything that fails goes to quarantine, not to the site.

The principle: the model may only assert what it can point to in the source text.
"""
from __future__ import annotations

from .extract import PROCEDURE_IDS, _prices
from .models import Extraction, RawReview

MIN_KRW, MAX_KRW = 50_000, 100_000_000  # below ~5만원 / above 1억 is almost surely a unit error
MIN_CONFIDENCE = 0.6


def validate(raw: RawReview, ex: Extraction) -> list[str]:
    issues = []
    for p in ex.prices:
        if p.source_span not in raw.text_ko:
            issues.append(f"price span not in source: {p.source_span!r}")
        parsed = [x.amount_krw for x in _prices(p.source_span)]
        if parsed and p.amount_krw not in parsed:
            issues.append(f"price {p.amount_krw:,} contradicts its own span {p.source_span!r} (= {parsed[0]:,})")
        if not (MIN_KRW <= p.amount_krw <= MAX_KRW):
            issues.append(f"price out of range (unit error?): {p.amount_krw:,} KRW from {p.source_span!r}")
    if ex.surgeon_name_ko and ex.surgeon_name_ko not in raw.text_ko:
        issues.append(f"surgeon name not in source (hallucinated?): {ex.surgeon_name_ko}")
    for m in ex.sponsored_markers + ex.self_paid_markers:
        if m not in raw.text_ko:
            issues.append(f"marker not in source: {m!r}")
    bad = [p for p in ex.procedures if p not in PROCEDURE_IDS]
    if bad:
        issues.append(f"procedure ids not in glossary: {bad}")
    if ex.confidence < MIN_CONFIDENCE and not ex.text_en.startswith("[offline"):
        issues.append(f"low model confidence {ex.confidence:.2f}")
    return issues
